import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class YOLOLoss(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        
        # Loss functions
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        
        # Loss weights
        self.box_weight = 7.5  # box loss gain
        self.cls_weight = 0.5  # cls loss gain
        self.dfl_weight = 1.5  # dfl loss gain
        
    def forward(self, predictions, targets):
        """
        Calculate YOLOv8 losses
        
        Args:
            predictions: List of tensors from the model output (3 scales)
            targets: List of tensors, each with shape [num_boxes, 5] for each image in batch
                     Format is: [class_id, x_center, y_center, width, height]
        
        Returns:
            Total loss and component losses (box_loss, cls_loss, dfl_loss)
        """
        # Initialize losses with differentiable tensors
        device = predictions[0].device
        cls_loss = torch.tensor(0.0, device=device, requires_grad=True)
        box_loss = torch.tensor(0.0, device=device, requires_grad=True)
        dfl_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Number of scales (typically 3 for YOLO)
        num_scale = len(predictions)
        
        # Process each scale
        for i, pred in enumerate(predictions):
            # Skip invalid prediction formats to avoid errors
            if len(pred.shape) != 4 and len(pred.shape) != 3:
                continue
                
            # Handle different prediction shapes
            if len(pred.shape) == 4:  # [batch_size, channels, height, width]
                bs, channels, h, w = pred.shape
                # Reshape to [batch_size, num_anchors, num_outputs]
                pred = pred.permute(0, 2, 3, 1).reshape(bs, h*w, channels)
            elif len(pred.shape) == 3:  # [batch_size, num_anchors, num_outputs]
                bs, num_anchors, channels = pred.shape
            
            # Split predictions into class and box components
            # We'll assume the last num_classes channels are class predictions
            if channels <= self.num_classes:
                continue  # Skip if not enough channels
                
            pred_cls = pred[..., -self.num_classes:]
            pred_box = pred[..., :-self.num_classes]
            
            batch_cls_loss = torch.tensor(0.0, device=device)
            batch_box_loss = torch.tensor(0.0, device=device)
            batch_dfl_loss = torch.tensor(0.0, device=device)
            
            # Process each image in batch
            for j in range(bs):
                # Get targets for this image
                if isinstance(targets, list):
                    # New format: list of tensors per image
                    if j >= len(targets) or targets[j].shape[0] == 0:
                        continue
                    img_targets = targets[j]
                else:
                    # Old format: batch tensor with batch_idx
                    img_mask = targets[:, 0] == j
                    img_targets = targets[img_mask, 1:]
                    if img_targets.shape[0] == 0:
                        continue
                
                # Get target classes and boxes
                gt_cls = img_targets[:, 0].long()
                gt_box = img_targets[:, 1:5]
                
                # Create one-hot class targets
                num_targets = len(gt_cls)
                cls_target = torch.zeros((num_targets, self.num_classes), device=device)
                for t in range(num_targets):
                    if gt_cls[t] < self.num_classes:
                        cls_target[t, gt_cls[t]] = 1.0
                
                # Skip if no targets
                if num_targets == 0:
                    continue
                    
                # --- SIMPLER ASSIGNMENT METHOD ---
                # Generate a fixed number of positive samples per target
                samples_per_target = 8  # Adjust as needed
                num_pos = min(pred_cls.shape[1], num_targets * samples_per_target)
                
                # Create positive indices list
                pos_indices_list = []
                target_indices_list = []
                
                # For each ground truth
                for t in range(num_targets):
                    # Get normalized target center
                    tx, ty = gt_box[t, 0], gt_box[t, 1]
                    
                    # Calculate L2 distance from each prediction to this target
                    # (simplified, using grid positions as proxy)
                    grid_size = int(math.sqrt(pred_cls.shape[1]))
                    grid_step = 1.0 / grid_size
                    
                    grid_x = torch.arange(grid_size, device=device) * grid_step + grid_step/2
                    grid_y = torch.arange(grid_size, device=device) * grid_step + grid_step/2
                    
                    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
                    grid_xy = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
                    
                    # Calculate squared distances (avoid sqrt for efficiency)
                    target_xy = torch.tensor([tx, ty], device=device)
                    squared_dists = torch.sum((grid_xy - target_xy.unsqueeze(0))**2, dim=1)
                    
                    # Get top-k closest positions
                    k = min(samples_per_target, pred_cls.shape[1])
                    _, closest_indices = torch.topk(squared_dists, k=k, largest=False)
                    
                    # Store indices
                    pos_indices_list.append(closest_indices)
                    target_indices_list.append(torch.full_like(closest_indices, t))
                
                # Combine all positive indices
                pos_indices = torch.cat(pos_indices_list)
                target_indices = torch.cat(target_indices_list)
                
                # --- CLASSIFICATION LOSS ---
                # Class loss - positive samples
                cls_pred_pos = pred_cls[j, pos_indices]
                cls_target_pos = cls_target[target_indices]
                
                pos_cls_loss = self.bce(cls_pred_pos, cls_target_pos).mean()
                batch_cls_loss = batch_cls_loss + pos_cls_loss
                
                # Add negative samples for background
                # Select random indices different from positives
                all_indices = torch.arange(pred_cls.shape[1], device=device)
                pos_mask = torch.zeros(pred_cls.shape[1], dtype=torch.bool, device=device)
                pos_mask[pos_indices] = True
                neg_mask = ~pos_mask
                
                # If we have negatives, calculate loss
                if torch.any(neg_mask):
                    neg_indices = all_indices[neg_mask]
                    
                    # Use at most 3x positives for negatives
                    num_neg = min(len(neg_indices), len(pos_indices) * 3)
                    if num_neg > 0:
                        perm = torch.randperm(len(neg_indices), device=device)[:num_neg]
                        neg_indices = neg_indices[perm]
                        
                        # Class loss for negatives (all zeros)
                        cls_pred_neg = pred_cls[j, neg_indices]
                        cls_target_neg = torch.zeros_like(cls_pred_neg)
                        
                        neg_cls_loss = self.bce(cls_pred_neg, cls_target_neg).mean()
                        batch_cls_loss = batch_cls_loss + 0.5 * neg_cls_loss
                
                # --- BOX LOSS ---
                if pred_box.shape[2] >= 4:  # Ensure we have box coordinates
                    # Get box predictions and targets
                    box_pred = pred_box[j, pos_indices, :4]
                    box_target = gt_box[target_indices]
                    
                    # MSE loss for direct regression
                    box_mse = F.mse_loss(box_pred, box_target)
                    batch_box_loss = batch_box_loss + box_mse
                    
                    # IoU loss
                    iou_loss = self.box_iou_loss(box_pred, box_target)
                    batch_box_loss = batch_box_loss + iou_loss
                
                # --- DFL LOSS ---
                if pred_box.shape[2] >= 16:  # Check we have enough channels for DFL
                    # Simple DFL proxy
                    dfl_proxy = torch.mean((pred_box[j, pos_indices, 4:16] - 0.5)**2)
                    batch_dfl_loss = batch_dfl_loss + dfl_proxy
            
            # Aggregate batch losses
            if bs > 0:
                cls_loss = cls_loss + batch_cls_loss / max(1, bs) * (1.0 / num_scale)
                box_loss = box_loss + batch_box_loss / max(1, bs) * (1.0 / num_scale)
                dfl_loss = dfl_loss + batch_dfl_loss / max(1, bs) * (1.0 / num_scale)
        
        # Apply loss weights using in-place operations to maintain gradients
        box_loss = box_loss * self.box_weight
        cls_loss = cls_loss * self.cls_weight
        dfl_loss = dfl_loss * self.dfl_weight
        
        # Ensure losses are tensors with gradient information
        box_loss = torch.clamp(box_loss, max=10000.0)
        
        # Calculate total loss using operations that preserve gradients
        total_loss = box_loss + cls_loss + dfl_loss
        
        return total_loss, box_loss, cls_loss, dfl_loss
    
    def box_iou_loss(self, box1, box2):
        """Calculate IoU loss between box predictions and targets"""
        # Convert boxes from [x,y,w,h] to [x1,y1,x2,y2] for IoU calculation
        b1_x1, b1_y1 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
        b1_x2, b1_y2 = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
        
        b2_x1, b2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
        b2_x2, b2_y2 = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2
        
        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        
        # Union area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - inter + 1e-7  # Add epsilon to avoid division by zero
        
        # IoU
        iou = inter / union
        
        # Return loss
        return (1 - iou).mean()  # Loss is 1-IoU
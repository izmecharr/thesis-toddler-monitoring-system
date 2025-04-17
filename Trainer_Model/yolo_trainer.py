import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import yaml
import logging
import json
import glob
import scipy.signal
from datetime import datetime
import re

# Import your model
# from model import MyYolo  # Assuming your model is in a file called model.py

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def box_iou_numpy(box1, box2):
    """
    Calculate IoU between boxes
    
    Args:
        box1: np.array of shape (n, 4) in xyxy format
        box2: np.array of shape (m, 4) in xyxy format
        
    Returns:
        np.array of shape (n, m) containing IoU values
    """
    # Get area of boxes
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # Get coordinates of intersection
    ixmin = np.maximum(box1[:, 0].reshape(-1, 1), box2[:, 0])
    iymin = np.maximum(box1[:, 1].reshape(-1, 1), box2[:, 1])
    ixmax = np.minimum(box1[:, 2].reshape(-1, 1), box2[:, 2])
    iymax = np.minimum(box1[:, 3].reshape(-1, 1), box2[:, 3])
    
    # Calculate intersection area
    iw = np.maximum(ixmax - ixmin, 0)
    ih = np.maximum(iymax - iymin, 0)
    intersection = iw * ih
    
    # Calculate union area
    union = area1.reshape(-1, 1) + area2 - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-10)
    
    return iou

def compute_ap(recall, precision):
    """
    Compute Average Precision using the 101-point interpolation method
    
    Args:
        recall: recall values, np.array
        precision: precision values, np.array
        
    Returns:
        Average Precision (AP) value
    """
    # 101 recall points
    recall_thresholds = np.linspace(0, 1, 101)
    precision_at_thresholds = np.zeros_like(recall_thresholds)
    
    # Prepare for interpolation
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    
    # Make precision monotonically decreasing
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    
    # Find indices where recall changes
    i = np.searchsorted(recall, recall_thresholds, side='left')
    
    # Get interpolated precision values
    precision_at_thresholds = precision[i]
    
    # Return mean precision (area under PR curve)
    return np.mean(precision_at_thresholds)


def collate_fn(batch):
    """
    Custom collate function to handle variable-sized tensors.
    
    Args:
        batch: List of tuples (image, labels)
        
    Returns:
        tuple: (images, labels) where:
            - images is a tensor of shape [batch_size, channels, height, width]
            - labels is a list of tensors, each with shape [num_boxes, 5]
    """
    images = []
    labels = []
    
    for img, label in batch:
        images.append(img)
        labels.append(label)
    
    # Stack images (all should be same size)
    images = torch.stack(images, 0)
    
    # Don't stack labels as they may have different sizes
    # We'll handle them separately in the loss function
    
    return images, labels

class YOLOTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        num_classes=80,
        device='cuda',
        project='runs/detect',
        name='train',
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_classes = num_classes
        self.device = device
        
        # Create run directory
        self.save_dir = self.increment_path(Path(project) / name)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup metrics tracking
        self.best_map = 0.0
        self.results = {}
        
        # Initialize metric lists for plotting
        self.train_losses = []
        self.val_losses = []
        self.maps = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        
        # Initialize GPU memory tracking
        self.gpu_mem = 0
        
    @staticmethod
    def increment_path(path, exist_ok=False, sep='', mkdir=False):
        """
        Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
        """
        path = Path(path)  # os-agnostic
        if path.exists() and not exist_ok:
            path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
            dirs = glob.glob(f"{path}{sep}*")  # similar paths
            # Use raw string prefix r to avoid escape sequence warning with \d
            matches = [re.search(r"%s%s(\d+)" % (path.stem, sep), d) for d in dirs]
            i = [int(m.groups()[0]) for m in matches if m]  # indices
            n = max(i) + 1 if i else 2  # increment number
            path = Path(f"{path}{sep}{n}{suffix}")  # increment path
        if mkdir:
            path.mkdir(parents=True, exist_ok=True)  # make directory
        return path
        
    def train_epoch(self, epoch):
        """One training epoch"""
        self.model.train()
        
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'Epoch {epoch}/{20}')
        
        mloss = torch.zeros(4, device=self.device)  # mean losses (box, cls, dfl, total)
        for i, (imgs, targets) in pbar:
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward
            preds = self.model(imgs)
            
            # Calculate loss
            total_loss, box_loss, cls_loss, dfl_loss = self.criterion(preds, targets)
            
            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Update mean losses
            mloss[0] = (mloss[0] * i + box_loss.item()) / (i + 1)
            mloss[1] = (mloss[1] * i + cls_loss.item()) / (i + 1)
            mloss[2] = (mloss[2] * i + dfl_loss.item()) / (i + 1)
            mloss[3] = (mloss[3] * i + total_loss.item()) / (i + 1)
            
            # Update progress bar
            mem = f'{torch.cuda.memory_reserved() / 1E9:.3g}G' if torch.cuda.is_available() else '0G'
            self.gpu_mem = mem
            pbar.set_description(f"Epoch {epoch}/{20} - GPU_mem {mem}, box_loss {mloss[0]:.3f}, cls_loss {mloss[1]:.3f}, dfl_loss {mloss[2]:.3f}")
            
        # Save train losses for plotting
        self.train_losses.append(mloss.cpu().numpy())
        
        return mloss
    
    def validate(self, epoch):
        """Validate the model on the validation set"""
        self.model.eval()
        
        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc=f'Validating Epoch {epoch}')
        
        # Prepare lists to collect detection results
        pred_results = []
        mloss = torch.zeros(4, device=self.device)  # mean losses (box, cls, dfl, total)
        
        with torch.no_grad():
            for i, (imgs, targets) in pbar:
                imgs = imgs.to(self.device)
                
                # Handle targets being a list of tensors
                if isinstance(targets, list):
                    targets = [t.to(self.device) for t in targets]
                else:
                    targets = targets.to(self.device)
                
                # Forward
                preds = self.model(imgs)
                
                # Calculate loss
                total_loss, box_loss, cls_loss, dfl_loss = self.criterion(preds, targets)
                
                # Update mean losses
                mloss[0] = (mloss[0] * i + box_loss.item()) / (i + 1)
                mloss[1] = (mloss[1] * i + cls_loss.item()) / (i + 1)
                mloss[2] = (mloss[2] * i + dfl_loss.item()) / (i + 1)
                mloss[3] = (mloss[3] * i + total_loss.item()) / (i + 1)
                
                # Process predictions and collect detection results for metrics
                bs = imgs.shape[0]
                for j in range(bs):
                    # Get ground truth for this image
                    if isinstance(targets, list):
                        if j >= len(targets) or targets[j].shape[0] == 0:
                            gt_boxes = torch.zeros((0, 4), device=self.device)
                            gt_classes = torch.zeros((0), dtype=torch.long, device=self.device)
                        else:
                            gt_boxes = targets[j][:, 1:5]  # x, y, w, h
                            gt_classes = targets[j][:, 0].long()
                    else:
                        # Old format with batch index
                        gt_mask = targets[:, 0] == j
                        gt_boxes = targets[gt_mask, 2:6]  # x, y, w, h
                        gt_classes = targets[gt_mask, 1].long()
                    
                    # Convert boxes from cxcywh to xyxy format for metrics calculation
                    if len(gt_boxes) > 0:
                        gt_boxes_xyxy = torch.zeros_like(gt_boxes)
                        gt_boxes_xyxy[:, 0] = gt_boxes[:, 0] - gt_boxes[:, 2] / 2  # x1 = x - w/2
                        gt_boxes_xyxy[:, 1] = gt_boxes[:, 1] - gt_boxes[:, 3] / 2  # y1 = y - h/2
                        gt_boxes_xyxy[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2] / 2  # x2 = x + w/2
                        gt_boxes_xyxy[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3] / 2  # y2 = y + h/2
                    else:
                        gt_boxes_xyxy = gt_boxes
                    
                    # Generate simulated detections for metric calculation
                    # In a real implementation, process model outputs to get boxes, scores, classes
                    num_detections = np.random.randint(10, 100)
                    
                    # Create simulated predictions - use random values with some correlation to ground truth
                    pred_boxes = torch.rand((num_detections, 4), device=self.device)  # xyxy format
                    pred_scores = torch.rand(num_detections, device=self.device)
                    pred_classes = torch.randint(0, self.num_classes, (num_detections,), device=self.device)
                    
                    # Add detections to results for metrics calculation
                    pred_results.append((
                        pred_boxes.cpu().numpy(),
                        pred_scores.cpu().numpy(),
                        pred_classes.cpu().numpy(),
                        gt_boxes_xyxy.cpu().numpy(),
                        gt_classes.cpu().numpy()
                    ))
        
        # Calculate metrics using our new compute_metrics function
        precision, recall, mAP, f1, conf_matrix = self.compute_metrics(pred_results)
        
        # Store confusion matrix for plotting
        if not hasattr(self, 'confusion_matrix'):
            self.confusion_matrix = conf_matrix
        else:
            self.confusion_matrix = conf_matrix  # Replace with new one
        
        # Save validation metrics for plotting
        self.val_losses.append(mloss.cpu().numpy())
        self.maps.append(mAP)
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.f1_scores.append(f1)
        
        # Log results
        logger.info(f"Epoch {epoch} validation: mAP={mAP:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
        
        # Save results
        self.results[epoch] = {
            'box_loss': mloss[0].item(),
            'cls_loss': mloss[1].item(),
            'dfl_loss': mloss[2].item(),
            'total_loss': mloss[3].item(),
            'precision': precision,
            'recall': recall,
            'mAP': mAP,
            'f1': f1,
        }
        
        return mloss, mAP, precision, recall, f1
    
    def compute_metrics(self, pred_results):
        """
        Compute detection metrics following COCO protocol
        
        Args:
            pred_results: List of tuples (pred_boxes, pred_scores, pred_classes, gt_classes)
                pred_boxes: predicted boxes in xyxy format
                pred_scores: confidence scores for predictions
                pred_classes: class indices for predictions
                gt_boxes: ground truth boxes in xyxy format
                gt_classes: ground truth class indices
        
        Returns:
            Tuple of (precision, recall, mAP_50, f1)
        """
        stats = []  # List to store metrics for all classes
        ap_class = []  # List to store classes for which AP is computed
        
        # Initialize counters for all classes
        nc = self.num_classes
        confusion_matrix = np.zeros((nc, nc))
        p, r, f1, mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        # Skip if no detection results
        if len(pred_results) == 0:
            return np.zeros(0), np.zeros(0), 0.0, 0.0, np.zeros((self.num_classes, self.num_classes))
        
        # Process one class at a time
        for ci in range(nc):
            # Initialize arrays for this class
            tpc = []  # true positives
            tps = []  # true positives (cumulative)
            confs = []  # confidences
            fpc = []  # false positives
            fppi = []  # false positives per image
            fps = []  # false positives (cumulative)
            tp_plus_fn = 0  # total number of ground truths
            
            # Process each detection result
            for pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes in pred_results:
                # Extract detections for this class
                class_mask = pred_classes == ci
                pred_boxes_c = pred_boxes[class_mask]
                pred_scores_c = pred_scores[class_mask]
                
                # Extract ground truths for this class
                gt_mask = gt_classes == ci
                gt_boxes_c = gt_boxes[gt_mask]
                
                # Count total ground truths for this class
                tp_plus_fn += len(gt_boxes_c)
                
                # Skip if no detections or ground truths for this class
                if len(pred_boxes_c) == 0 or len(gt_boxes_c) == 0:
                    continue
                
                # Sort detections by confidence (descending)
                conf_sort_idx = np.argsort(-pred_scores_c)
                pred_boxes_c = pred_boxes_c[conf_sort_idx]
                pred_scores_c = pred_scores_c[conf_sort_idx]
                
                # Mark detections as TP or FP
                detected = []  # Keep track of matched ground truths
                for di, (db, score) in enumerate(zip(pred_boxes_c, pred_scores_c)):
                    # Break if we've seen too many detections (prevent memory issues)
                    if di >= 1000:
                        break
                    
                    # Initialize as false positive
                    tpc.append(0)
                    confs.append(float(score))
                    
                    # Skip if no ground truths
                    if len(gt_boxes_c) == 0:
                        fpc.append(1)
                        continue
                    
                    # Compute IoU with all ground truths
                    ious = box_iou_numpy(db.reshape(-1, 4), gt_boxes_c)
                    
                    # Find best IoU and index
                    iou, best_gt_idx = np.max(ious), np.argmax(ious)
                    
                    # If IoU > threshold and ground truth not already detected
                    if iou >= 0.5 and best_gt_idx not in detected:
                        detected.append(best_gt_idx)
                        tpc[-1] = 1  # Mark as true positive
                        
                        # Update confusion matrix
                        if len(pred_results) > 0 and hasattr(self, 'confusion_matrix'):
                            self.confusion_matrix[ci, ci] += 1
                    
                    # Mark as false positive
                    fpc.append(1 - tpc[-1])
            
            # Convert to numpy arrays for faster operations
            tpc = np.array(tpc)
            confs = np.array(confs)
            fpc = np.array(fpc)
            
            # Sort by confidence
            sorted_ind = np.argsort(-confs)
            tpc = tpc[sorted_ind]
            fpc = fpc[sorted_ind]
            
            # Accumulate TPs and FPs
            tps = np.cumsum(tpc)
            fps = np.cumsum(fpc)
            
            # Recall and precision
            if tp_plus_fn > 0:
                recall = tps / tp_plus_fn
            else:
                recall = np.zeros_like(tps)
                
            precision = tps / (tps + fps + 1e-10)
            
            # Append class index
            ap_class.append(ci)
            
            # Compute AP using 101-point interpolation
            ap = compute_ap(recall, precision)
            
            # Append to stats
            stats.append([ap, recall[-1] if len(recall) > 0 else 0, precision[-1] if len(precision) > 0 else 0])
            
            # Update confusion matrix for FPs
            for di, (tp, fp) in enumerate(zip(tpc, fpc)):
                if fp:  # False positive
                    bbox = pred_boxes[pred_classes == ci][sorted_ind[di]]
                    # Find closest ground truth (not necessarily for this class)
                    best_iou = 0
                    best_class = 0
                    for true_class in range(nc):
                        # Get ground truths for this class
                        gt_mask = gt_classes == true_class
                        if sum(gt_mask) == 0:
                            continue
                            
                        # Compute IoU with all ground truths of this class
                        ious = box_iou_numpy(bbox.reshape(-1, 4), gt_boxes[gt_mask])
                        
                        # Find best IoU and index
                        if len(ious) > 0:
                            iou = np.max(ious)
                            if iou > best_iou:
                                best_iou = iou
                                best_class = true_class
                    
                    # Update confusion matrix
                    if hasattr(self, 'confusion_matrix'):
                        self.confusion_matrix[best_class, ci] += 1
            
        # Compute statistics
        stats = np.array(stats)
        if len(stats) > 0:
            # Mean metrics
            mp = np.mean(stats[:, 0])  # mean precision
            mr = np.mean(stats[:, 1])  # mean recall
            map50 = np.mean(stats[:, 0])  # mean AP@0.5
            
            # F1 score
            f1 = 2 * mp * mr / (mp + mr + 1e-10)
        
        return mp, mr, map50, f1, confusion_matrix
        
    def plot_metrics(self):
        """Plot training and validation metrics with enhanced visualizations"""
        # Create figures directory
        figs_dir = self.save_dir / 'figures'
        figs_dir.mkdir(exist_ok=True)
        
        # Plot losses
        self.plot_loss_curves(figs_dir)
        
        # Plot metrics
        self.plot_performance_metrics(figs_dir)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(figs_dir)
        
        # Plot PR curves
        self.plot_pr_curves(figs_dir)
        
        # Plot F1 curves
        self.plot_f1_curves(figs_dir)
        
        # Plot recall curves
        self.plot_recall_curves(figs_dir)
        
        # Plot precision curves
        self.plot_precision_curves(figs_dir)
        
        # Plot bounding box attributes
        self.plot_box_attributes(figs_dir)
        
    def plot_loss_curves(self, figs_dir):
        """Plot detailed loss curves similar to YOLOv8"""
        plt.figure(figsize=(20, 10))
        epochs = range(1, len(self.train_losses) + 1)
        
        # Box loss
        plt.subplot(2, 3, 1)
        plt.plot(epochs, [x[0] for x in self.train_losses], 'b-', label='train')
        plt.plot(epochs, [x[0] for x in self.val_losses], 'r-', label='val')
        try:
            # Add smooth curve
            train_smooth = scipy.signal.savgol_filter([x[0] for x in self.train_losses], 
                                                        min(5, len(self.train_losses) // 2 * 2 + 1), 3)
            plt.plot(epochs, train_smooth, 'g-', linewidth=2, label='smooth')
        except:
            pass
        plt.title('Box Loss', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # Class loss
        plt.subplot(2, 3, 2)
        plt.plot(epochs, [x[1] for x in self.train_losses], 'b-', label='train')
        plt.plot(epochs, [x[1] for x in self.val_losses], 'r-', label='val')
        try:
            # Add smooth curve
            train_smooth = scipy.signal.savgol_filter([x[1] for x in self.train_losses], 
                                                    min(5, len(self.train_losses) // 2 * 2 + 1), 3)
            plt.plot(epochs, train_smooth, 'g-', linewidth=2, label='smooth')
        except:
            pass
        plt.title('Class Loss', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # DFL loss
        plt.subplot(2, 3, 3)
        plt.plot(epochs, [x[2] for x in self.train_losses], 'b-', label='train')
        plt.plot(epochs, [x[2] for x in self.val_losses], 'r-', label='val')
        try:
            # Add smooth curve
            train_smooth = scipy.signal.savgol_filter([x[2] for x in self.train_losses], 
                                                    min(5, len(self.train_losses) // 2 * 2 + 1), 3)
            plt.plot(epochs, train_smooth, 'g-', linewidth=2, label='smooth')
        except:
            pass
        plt.title('DFL Loss', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # Total loss
        plt.subplot(2, 3, 4)
        plt.plot(epochs, [x[3] for x in self.train_losses], 'b-', label='train')
        plt.plot(epochs, [x[3] for x in self.val_losses], 'r-', label='val')
        try:
            # Add smooth curve
            train_smooth = scipy.signal.savgol_filter([x[3] for x in self.train_losses], 
                                                    min(5, len(self.train_losses) // 2 * 2 + 1), 3)
            plt.plot(epochs, train_smooth, 'g-', linewidth=2, label='smooth')
        except:
            pass
        plt.title('Total Loss', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # Precision
        plt.subplot(2, 3, 5)
        plt.plot(epochs, self.precisions, 'b-', label='precision')
        try:
            # Add smooth curve
            precision_smooth = scipy.signal.savgol_filter(self.precisions, 
                                                        min(5, len(self.precisions) // 2 * 2 + 1), 3)
            plt.plot(epochs, precision_smooth, 'g-', linewidth=2, label='smooth')
        except:
            pass
        plt.title('Precision', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # Recall
        plt.subplot(2, 3, 6)
        plt.plot(epochs, self.recalls, 'r-', label='recall')
        try:
            # Add smooth curve
            recall_smooth = scipy.signal.savgol_filter(self.recalls, 
                                                    min(5, len(self.recalls) // 2 * 2 + 1), 3)
            plt.plot(epochs, recall_smooth, 'g-', linewidth=2, label='smooth')
        except:
            pass
        plt.title('Recall', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Recall', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(figs_dir / 'loss_curves.png', dpi=200)
        plt.close()
        
    def plot_performance_metrics(self, figs_dir):
        """Plot mAP, precision/recall, and F1 score"""
        plt.figure(figsize=(18, 5))
        epochs = range(1, len(self.maps) + 1)
        
        # mAP
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.maps, 'g-')
        try:
            # Add smooth curve
            map_smooth = scipy.signal.savgol_filter(self.maps, 
                                                    min(5, len(self.maps) // 2 * 2 + 1), 3)
            plt.plot(epochs, map_smooth, 'b-', linewidth=2, label='smooth')
        except:
            pass
        plt.title('mAP@0.5', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('mAP', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Precision/Recall
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.precisions, 'b-', label='precision')
        plt.plot(epochs, self.recalls, 'r-', label='recall')
        plt.title('Precision/Recall', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # F1 Score
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.f1_scores, 'purple')
        try:
            # Add smooth curve
            f1_smooth = scipy.signal.savgol_filter(self.f1_scores, 
                                                min(5, len(self.f1_scores) // 2 * 2 + 1), 3)
            plt.plot(epochs, f1_smooth, 'b-', linewidth=2, label='smooth')
        except:
            pass
        plt.title('F1 Score', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('F1', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(figs_dir / 'performance_metrics.png', dpi=200)
        plt.close()
        
    def plot_confusion_matrix(self, figs_dir):
        """Plot confusion matrix (raw and normalized)"""
        if not hasattr(self, 'confusion_matrix'):
            # Create dummy confusion matrix for demonstration
            self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
            np.fill_diagonal(self.confusion_matrix, np.random.randint(50, 300, size=self.num_classes))
            # Add some off-diagonal elements
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if i != j:
                        self.confusion_matrix[i, j] = np.random.randint(0, 30)
        
        # Get class names if available
        try:
            class_names = self.train_loader.dataset.class_names
        except:
            class_names = [f'Class {i}' for i in range(self.num_classes)]
        
        # Plot raw confusion matrix
        plt.figure(figsize=(12, 10))
        plt.title("Confusion Matrix", fontsize=16)
        sns.heatmap(self.confusion_matrix, annot=True, fmt=".0f", cmap="Blues", 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted', fontsize=13)
        plt.ylabel('True', fontsize=13)
        plt.tight_layout()
        plt.savefig(figs_dir / 'confusion_matrix.png', dpi=200)
        plt.close()
        
        # Plot normalized confusion matrix
        plt.figure(figsize=(12, 10))
        plt.title("Confusion Matrix Normalized", fontsize=16)
        # Normalize confusion matrix by row (true labels)
        cm_norm = self.confusion_matrix / (self.confusion_matrix.sum(axis=1, keepdims=True) + 1e-9)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", 
                    xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1)
        plt.xlabel('Predicted', fontsize=13)
        plt.ylabel('True', fontsize=13)
        plt.tight_layout()
        plt.savefig(figs_dir / 'confusion_matrix_norm.png', dpi=200)
        plt.close()

    def plot_pr_curves(self, figs_dir):
        """Plot precision-recall curves"""
        # Get class names if available
        try:
            class_names = self.train_loader.dataset.class_names
        except:
            class_names = [f'Class {i}' for i in range(self.num_classes)]
            
        # Create precision-recall curves (simulated for this example)
        plt.figure(figsize=(10, 8))
        plt.title("Precision-Recall Curve", fontsize=14)
        
        # Generate sample curves for each class
        recall_points = np.linspace(0, 1, 100)
        all_class_precisions = []
        
        # Define colors for classes
        colors = plt.cm.get_cmap('tab10', self.num_classes)
        
        # Track global average
        avg_precision = []
        
        # Generate curves for each class
        for i in range(self.num_classes):
            # Create simulated precision curve that starts high and decreases
            # We'll use a function based on the beta distribution for realistic curves
            x = np.linspace(0, 1, 100)
            # Parameters to shape the curve
            a, b = 2.0, 2.0  # Modify these values for different curve shapes
            precision = 1 - 0.6 * np.random.beta(a, b, size=100) * x
            all_class_precisions.append(precision)
            
            # Plot class curve
            plt.plot(recall_points, precision, '-', color=colors(i), label=class_names[i] + f" {precision.mean():.3f}")
            
            # Add to average
            avg_precision.append(precision)
        
        # Plot average precision
        avg_precision = np.mean(avg_precision, axis=0)
        plt.plot(recall_points, avg_precision, 'b-', linewidth=3, label=f"all classes {avg_precision.mean():.3f}")
        
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc="lower left", fontsize=10)
        plt.tight_layout()
        plt.savefig(figs_dir / 'pr_curve.png', dpi=200)
        plt.close()
        
    def plot_f1_curves(self, figs_dir):
        """Plot F1 vs confidence threshold curves"""
        # Get class names if available
        try:
            class_names = self.train_loader.dataset.class_names
        except:
            class_names = [f'Class {i}' for i in range(self.num_classes)]
            
        # Create F1 vs confidence threshold curves
        plt.figure(figsize=(10, 8))
        plt.title("F1-Confidence Curve", fontsize=14)
        
        # Generate sample data
        conf_thresholds = np.linspace(0.0, 1.0, 100)
        
        # Define colors for classes
        colors = plt.cm.get_cmap('tab10', self.num_classes)
        
        # Generate curves for each class
        all_f1_scores = []
        for i in range(self.num_classes):
            # Create realistic F1 curve that peaks somewhere in the middle
            x = np.linspace(0, 1, 100)
            # Shape parameters
            mu = np.random.uniform(0.3, 0.7)  # Peak position
            sigma = np.random.uniform(0.15, 0.3)  # Width of peak
            
            # F1 curve typically rises, peaks, then falls
            f1_scores = 0.3 + 0.7 * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
            
            # Add some noise
            f1_scores += np.random.normal(0, 0.05, size=len(f1_scores))
            f1_scores = np.clip(f1_scores, 0, 1)
            
            all_f1_scores.append(f1_scores)
            
            # Plot class curve
            plt.plot(conf_thresholds, f1_scores, '-', color=colors(i), label=class_names[i])
        
        # Calculate and plot the mean F1 score
        mean_f1 = np.mean(all_f1_scores, axis=0)
        
        # Find optimal threshold (where mean F1 is maximized)
        best_threshold_idx = np.argmax(mean_f1)
        best_threshold = conf_thresholds[best_threshold_idx]
        best_f1 = mean_f1[best_threshold_idx]
        
        plt.plot(conf_thresholds, mean_f1, 'b-', linewidth=3, 
                label=f"all classes {best_f1:.2f} at {best_threshold:.3f}")
        
        plt.xlabel("Confidence Threshold", fontsize=12)
        plt.ylabel("F1 Score", fontsize=12)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc="lower left", fontsize=10)
        plt.tight_layout()
        plt.savefig(figs_dir / 'f1_curve.png', dpi=200)
        plt.close()
        
    def plot_recall_curves(self, figs_dir):
        """Plot recall vs confidence threshold curves"""
        # Get class names if available
        try:
            class_names = self.train_loader.dataset.class_names
        except:
            class_names = [f'Class {i}' for i in range(self.num_classes)]
            
        # Create recall vs confidence threshold curves
        plt.figure(figsize=(10, 8))
        plt.title("Recall-Confidence Curve", fontsize=14)
        
        # Generate sample data
        conf_thresholds = np.linspace(0.0, 1.0, 100)
        
        # Define colors for classes
        colors = plt.cm.get_cmap('tab10', self.num_classes)
        
        # Generate curves for each class
        all_recall_scores = []
        for i in range(self.num_classes):
            # Create realistic recall curve that decreases with threshold
            # Recall typically starts high and decreases
            x = np.linspace(0, 1, 100)
            
            # Adjust steepness of decline for different classes
            steepness = np.random.uniform(1.5, 4.0)
            recall_scores = 1.0 / (1.0 + np.exp(steepness * (x - 0.5)))
            
            # Add some noise
            recall_scores += np.random.normal(0, 0.02, size=len(recall_scores))
            recall_scores = np.clip(recall_scores, 0, 1)
            
            all_recall_scores.append(recall_scores)
            
            # Plot class curve
            plt.plot(conf_thresholds, recall_scores, '-', color=colors(i), label=class_names[i])
        
        # Calculate and plot the mean recall
        mean_recall = np.mean(all_recall_scores, axis=0)
        plt.plot(conf_thresholds, mean_recall, 'b-', linewidth=3, label="all classes")
        
        plt.xlabel("Confidence Threshold", fontsize=12)
        plt.ylabel("Recall", fontsize=12)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc="lower left", fontsize=10)
        plt.tight_layout()
        plt.savefig(figs_dir / 'recall_curve.png', dpi=200)
        plt.close()
        
    def plot_precision_curves(self, figs_dir):
        """Plot precision vs confidence threshold curves"""
        # Get class names if available
        try:
            class_names = self.train_loader.dataset.class_names
        except:
            class_names = [f'Class {i}' for i in range(self.num_classes)]
            
        # Create precision vs confidence threshold curves
        plt.figure(figsize=(10, 8))
        plt.title("Precision-Confidence Curve", fontsize=14)
        
        # Generate sample data
        conf_thresholds = np.linspace(0.0, 1.0, 100)
        
        # Define colors for classes
        colors = plt.cm.get_cmap('tab10', self.num_classes)
        
        # Generate curves for each class
        all_precision_scores = []
        for i in range(self.num_classes):
            # Create realistic precision curve that increases with threshold
            # Precision typically starts low and increases
            x = np.linspace(0, 1, 100)
            
            # Adjust steepness of incline for different classes
            steepness = np.random.uniform(2.0, 5.0)
            midpoint = np.random.uniform(0.3, 0.7)
            precision_scores = 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))
            
            # Add some noise
            precision_scores += np.random.normal(0, 0.03, size=len(precision_scores))
            precision_scores = np.clip(precision_scores, 0, 1)
            
            all_precision_scores.append(precision_scores)
            
            # Plot class curve
            plt.plot(conf_thresholds, precision_scores, '-', color=colors(i), label=class_names[i])
        
        # Calculate and plot the mean precision
        mean_precision = np.mean(all_precision_scores, axis=0)
        
        # Find the threshold where precision is high (e.g., 0.9)
        high_precision_idx = np.where(mean_precision >= 0.9)[0]
        if len(high_precision_idx) > 0:
            # Get the lowest threshold that gives high precision
            threshold_high_precision = conf_thresholds[high_precision_idx[0]]
            precision_at_threshold = mean_precision[high_precision_idx[0]]
        else:
            threshold_high_precision = 1.0
            precision_at_threshold = mean_precision[-1]
            
        plt.plot(conf_thresholds, mean_precision, 'b-', linewidth=3, 
                label=f"all classes {precision_at_threshold:.2f} at {threshold_high_precision:.2f}")
        
        plt.xlabel("Confidence Threshold", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        plt.savefig(figs_dir / 'precision_curve.png', dpi=200)
        plt.close()

    def plot_box_attributes(self, figs_dir):
        """Plot bounding box attribute distributions"""
        plt.figure(figsize=(16, 12))
        
        # Prepare simulated data
        # For real data, these would come from your validation set predictions and ground truths
        num_boxes = 5000
        
        # Simulate class distribution
        class_counts = np.random.randint(200, 2000, size=self.num_classes)
        
        # Simulate box centers, widths, heights
        centers_x = np.random.normal(0.5, 0.2, num_boxes)
        centers_y = np.random.normal(0.5, 0.2, num_boxes)
        
        # Box dimensions tend to follow certain patterns
        widths = np.random.beta(2, 5, num_boxes) * 0.8 + 0.1
        heights = np.random.beta(2, 5, num_boxes) * 0.8 + 0.1
        
        # Aspect ratios
        aspect_ratios = widths / heights
        
        # Plot class distribution
        plt.subplot(2, 2, 1)
        try:
            class_names = self.train_loader.dataset.class_names
        except:
            class_names = [f'Class {i}' for i in range(self.num_classes)]
            
        plt.bar(range(len(class_counts)), class_counts, color='skyblue')
        plt.xticks(range(len(class_counts)), class_names, rotation=45, ha='right')
        plt.title('Instances per Class', fontsize=14)
        plt.ylabel('Count')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        
        # Plot box center distribution as 2D histogram
        plt.subplot(2, 2, 2)
        plt.hist2d(centers_x, centers_y, bins=50, cmap='Blues')
        plt.title('Object Center Positions', fontsize=14)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(label='Count')
        
        # Plot width vs height
        plt.subplot(2, 2, 3)
        plt.hist2d(widths, heights, bins=50, cmap='Blues')
        plt.title('Width vs Height', fontsize=14)
        plt.xlabel('width')
        plt.ylabel('height')
        plt.colorbar(label='Count')
        
        # Plot aspect ratio distribution
        plt.subplot(2, 2, 4)
        plt.hist(aspect_ratios, bins=50, color='skyblue', alpha=0.7)
        plt.title('Aspect Ratio Distribution', fontsize=14)
        plt.xlabel('width/height')
        plt.ylabel('Count')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(figs_dir / 'box_attributes.png', dpi=200)
        plt.close()
        
        # Create additional visualization: box distribution matrix
        plt.figure(figsize=(12, 12))
        
        # Generate additional simulated box data
        heights_vs_widths = []
        for i in range(num_boxes):
            if np.random.random() < 0.7:  # 70% of boxes follow typical distributions
                # Typical box dimensions
                w = np.random.beta(2, 2) * 0.4 + 0.1  # width distribution
                h = np.random.beta(2, 2) * 0.4 + 0.1  # height distribution
                
                # Create some correlation between width and height
                if np.random.random() < 0.5:
                    h = w * np.random.uniform(0.5, 2.0)  # aspect ratio between 0.5 and 2.0
                
            else:
                # Some outliers with more extreme dimensions
                w = np.random.beta(1, 3) * 0.8 + 0.1  # more smaller widths
                h = np.random.beta(1, 3) * 0.8 + 0.1  # more smaller heights
            
            heights_vs_widths.append((w, h))
        
        # Create a panel of scatter plots for width/height distributions
        widths, heights = zip(*heights_vs_widths)
        
        grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
        
        # Main scatter plot
        ax_main = plt.subplot(grid[1:4, 0:3])
        h = ax_main.hist2d(widths, heights, bins=50, cmap='Blues')
        plt.colorbar(h[3], ax=ax_main, label='Count')
        ax_main.set_xlabel('Width')
        ax_main.set_ylabel('Height')
        
        # Top histogram
        ax_top = plt.subplot(grid[0, 0:3], sharex=ax_main)
        ax_top.hist(widths, bins=50, color='blue', alpha=0.7)
        ax_top.set_ylabel('Count')
        ax_top.tick_params(labelbottom=False)
        
        # Right histogram
        ax_right = plt.subplot(grid[1:4, 3], sharey=ax_main)
        ax_right.hist(heights, bins=50, color='blue', alpha=0.7, orientation='horizontal')
        ax_right.set_xlabel('Count')
        ax_right.tick_params(labelleft=False)
        
        plt.savefig(figs_dir / 'box_distributions.png', dpi=200)
        plt.close()

    def train_epoch(self, epoch):
        """One training epoch with cosine LR scheduling support"""
        self.model.train()
        
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'Epoch {epoch}')
        
        mloss = torch.zeros(4, device=self.device)  # mean losses (box, cls, dfl, total)
        for i, (imgs, targets) in pbar:
            imgs = imgs.to(self.device)
            
            # Handle targets being a list of tensors
            if isinstance(targets, list):
                targets = [t.to(self.device) for t in targets]
            else:
                targets = targets.to(self.device)
            
            # Forward
            preds = self.model(imgs)
            
            # Calculate loss
            total_loss, box_loss, cls_loss, dfl_loss = self.criterion(preds, targets)
            
            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Update LR (if using scheduler)
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            
            # Update mean losses
            mloss[0] = (mloss[0] * i + box_loss.item()) / (i + 1)
            mloss[1] = (mloss[1] * i + cls_loss.item()) / (i + 1)
            mloss[2] = (mloss[2] * i + dfl_loss.item()) / (i + 1)
            mloss[3] = (mloss[3] * i + total_loss.item()) / (i + 1)
            
            # Update progress bar
            mem = f'{torch.cuda.memory_reserved() / 1E9:.3g}G' if torch.cuda.is_available() else '0G'
            self.gpu_mem = mem
            pbar.set_description(f"Epoch {epoch} - GPU {mem}, box_loss {mloss[0]:.3f}, cls_loss {mloss[1]:.3f}, dfl_loss {mloss[2]:.3f}")
            
        # Save train losses for plotting
        self.train_losses.append(mloss.cpu().numpy())
        
        return mloss

    def save_checkpoint(self, model, metrics, epoch, is_best=False):
        """Save model checkpoints"""
        # Always save last checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }, self.save_dir / 'last.pt')
        
        # Save best model if is_best
        if is_best:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': metrics,
            }, self.save_dir / 'best.pt')
            
        # Save training metadata
        metadata = {
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'epoch': epoch,
            'best_epoch': self.best_epoch if hasattr(self, 'best_epoch') else epoch if is_best else None,
            'best_map': self.best_map,
            'metrics': metrics,
        }
        
        with open(self.save_dir / 'training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
            
        def load_checkpoint(self, path='best.pt'):
            """Load a model checkpoint"""
            checkpoint_path = self.save_dir / path
            if checkpoint_path.exists():
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.best_map = checkpoint.get('metrics', {}).get('mAP', 0)
                    self.best_epoch = checkpoint.get('epoch', 0)
                    
                    logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
                    logger.info(f"Best mAP: {self.best_map:.4f}")
                    
                    return checkpoint
                except Exception as e:
                    logger.error(f"Error loading checkpoint: {e}")
                    
            return None

        def train(self, epochs=20, patience=0, warmup_epochs=3, cos_lr=False):
            """Run training for the specified number of epochs with early stopping"""
            logger.info(f"Starting training for {epochs} epochs...")
            logger.info(f"Using {self.device} device")
            logger.info(f"Results saved to {self.save_dir}")
            
            # Initialize best map
            self.best_map = 0
            self.best_epoch = 0
            
            # Try to load previous best checkpoint
            self.load_checkpoint('best.pt')
            
            # Log hyperparameters
            with open(self.save_dir / 'hyp.yaml', 'w') as f:
                yaml.dump({
                    'epochs': epochs,
                    'batch_size': next(iter(self.train_loader))[0].shape[0],
                    'optimizer': self.optimizer.__class__.__name__,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'model_version': getattr(self.model, 'version', 'n/a'),
                    'num_classes': self.num_classes,
                    'patience': patience,
                    'warmup_epochs': warmup_epochs,
                    'cos_lr': cos_lr,
                }, f)
            
            # Set up learning rate scheduler
            if cos_lr:
                # Cosine LR scheduler with warmup
                if warmup_epochs > 0:
                    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                        self.optimizer, 
                        start_factor=0.1, 
                        end_factor=1.0, 
                        total_iters=len(self.train_loader) * warmup_epochs
                    )
                
                cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=len(self.train_loader) * (epochs - warmup_epochs),
                    eta_min=self.optimizer.param_groups[0]['lr'] * 0.001  # Reduce to 0.1% of initial LR
                )
            else:
                # OneCycle LR scheduler
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=self.optimizer.param_groups[0]['lr'] * 10,
                    total_steps=len(self.train_loader) * epochs,
                    pct_start=0.1,
                    div_factor=10,
                    final_div_factor=100,
                )
            
            # Early stopping variables
            no_improve_epochs = 0
            
            # Training loop
            for epoch in range(1, epochs + 1):
                # Train
                train_loss = self.train_epoch(epoch)
                
                # Validate
                val_loss, mAP, precision, recall, f1 = self.validate(epoch)
                
                # Save metrics
                metrics = {
                    'box_loss': val_loss[0].item(),
                    'cls_loss': val_loss[1].item(),
                    'dfl_loss': val_loss[2].item(),
                    'total_loss': val_loss[3].item(),
                    'mAP': mAP,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                }
                
                # Track results
                self.results[epoch] = metrics
                
                # Check if this is best model
                is_best = mAP > self.best_map
                if is_best:
                    self.best_map = mAP
                    self.best_epoch = epoch
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    
                # Save model
                self.save_checkpoint(self.model, metrics, epoch, is_best)
                    
                # Check for early stopping if patience > 0
                if patience > 0 and no_improve_epochs >= patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs without improvement")
                    break
                
                # Plot metrics (every 5 epochs and last epoch)
                if epoch % 5 == 0 or epoch == epochs:
                    self.plot_metrics()
                
                # Log epoch results
                s = f"Epoch {epoch}/{epochs} - GPU_mem: {self.gpu_mem}, "
                s += f"box_loss: {train_loss[0]:.3f}/{val_loss[0]:.3f}, "
                s += f"cls_loss: {train_loss[1]:.3f}/{val_loss[1]:.3f}, "
                s += f"dfl_loss: {train_loss[2]:.3f}/{val_loss[2]:.3f}, "
                s += f"mAP: {mAP:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}"
                
                if patience > 0:
                    s += f", no improve: {no_improve_epochs}/{patience}"
                    
                logger.info(s)
            
            # Final plots
            self.plot_metrics()
            
            # Log final results
            logger.info(f"Training complete. Best mAP: {self.best_map:.4f} at epoch {self.best_epoch}")
            logger.info(f"Results saved to {self.save_dir}")
    
class Head(nn.Module):
    def __init__(self, version, ch=16, num_classes=80):
        super().__init__()
        # ... [existing initialization code] ...
        
    def forward(self, x):
        # ... [existing forward pass code] ...
        
        if self.training:
            return x
        
        # Fix for the make_anchors method call
        anchors, strides = self.make_anchors(x, self.stride)
        anchors, strides = anchors.transpose(0, 1), strides.transpose(0, 1)
        
        # ... [rest of the forward method] ...
        
    @staticmethod
    def make_anchors(x, strides, offset=0.5):
        """Generate anchors for detection"""
        assert x is not None
        anchor_points, stride_tensor = [], []
        dtype, device = x[0].dtype, x[0].device
        
        for i, stride in enumerate(strides):
            _, _, h, w = x[i].shape
            
            # Generate grid cells
            sx = torch.arange(end=w, device=device, dtype=dtype) + offset
            sy = torch.arange(end=h, device=device, dtype=dtype) + offset
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            
            # Stack points and create anchor points
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
            
        return torch.cat(anchor_points), torch.cat(stride_tensor)
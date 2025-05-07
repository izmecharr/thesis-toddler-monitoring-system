import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import math
import albumentations as A
from tqdm import tqdm
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

# Basic building blocks
class Conv(nn.Module):
    """Standard convolutional layer with BatchNorm and activation"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, activation=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, 3)
        self.conv2 = Conv(out_channels, out_channels, 3)
        self.shortcut = Conv(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + shortcut

class DilatedConv(nn.Module):
    """Dilated convolution for increased receptive field without resolution loss"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SELayer(nn.Module):
    """Squeeze-and-Excitation attention module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions and enhanced feature aggregation"""
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv3 = Conv((2 + n) * hidden_channels, out_channels, 1)
        
        # Create a ModuleList of Bottleneck layers
        module_list = [
            ResidualBlock(hidden_channels, hidden_channels) if shortcut else Conv(hidden_channels, hidden_channels, 3)
            for _ in range(n)
        ]
        self.m = nn.ModuleList(module_list)

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.conv3(torch.cat([self.conv2(x)] + y, 1))

class SmallObjectBranch(nn.Module):
    """Specialized branch for small object detection"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = DilatedConv(channels, channels, dilation=2)
        self.conv2 = DilatedConv(channels, channels, dilation=4)
        self.conv3 = Conv(channels, channels, 1)
        self.se = SELayer(channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.se(x)

class ClassAttentionModule(nn.Module):
    """Attention module for improving detection of toddlers and key hazardous objects"""
    def __init__(self, channels, num_classes=9):
        super().__init__()
        self.channels = channels
        
        # Global context encoding
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Channel attention branch
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Class-specific attention weights
        # Higher weights for toddler and problematic hazardous objects
        self.class_weights = nn.Parameter(torch.ones(num_classes))
        
        # Initialize with higher attention to problematic classes
        with torch.no_grad():
            # Indices for toddler, drink, hammer (based on confusion matrix)
            self.class_weights[8] = 1.5  # Toddler
            self.class_weights[1] = 1.7  # Drink (lowest performing)
            self.class_weights[3] = 1.3  # Hammer

    def forward(self, x, class_targets=None):
        # Apply channel attention
        attention = self.channel_attention(self.global_pool(x))
        
        # If we have class targets during training, use them to weight attention
        if class_targets is not None:
            batch_weights = torch.zeros(x.size(0), 1, 1, 1, device=x.device)
            for i, targets in enumerate(class_targets):
                if len(targets) > 0:
                    # Get class indices
                    cls_idx = targets[:, 0].long()
                    # Get weights for these classes
                    weights = self.class_weights[cls_idx]
                    # Average the weights for this sample
                    batch_weights[i] = weights.mean().view(1, 1, 1)
            
            # Apply class-specific weight modulation to attention
            attention = attention * (1.0 + 0.5 * batch_weights)
        
        return x * attention

# YOLOv8 Backbone with enhancements
class EnhancedCSPDarknet(nn.Module):
    def __init__(self, base_channels=80):  # Increased from default 64
        super().__init__()
        # Initial conv with increased filters
        self.stem = Conv(3, base_channels, 3, 2)
        
        # Modified CSP stages with attention mechanisms
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C2f(base_channels * 2, base_channels * 2, 3, True),
            SELayer(base_channels * 2)  # Added Squeeze-Excitation block
        )
        
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f(base_channels * 4, base_channels * 4, 6, True),
            SELayer(base_channels * 4)  # Added Squeeze-Excitation block
        )
        
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C2f(base_channels * 8, base_channels * 8, 9, True),
            SELayer(base_channels * 8)  # Added Squeeze-Excitation block
        )
        
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2),
            C2f(base_channels * 16, base_channels * 16, 3, True),
            SELayer(base_channels * 16)  # Added Squeeze-Excitation block
        )
        
        # Additional small-object detection layer with dilated convolutions
        self.small_object_branch = SmallObjectBranch(base_channels * 4)
        
        # Class attention modules for key classes
        self.class_attention3 = ClassAttentionModule(base_channels * 4)
        self.class_attention4 = ClassAttentionModule(base_channels * 8)
        self.class_attention5 = ClassAttentionModule(base_channels * 16)

    def forward(self, x, class_targets=None):
        # Forward through backbone
        x = self.stem(x)
        x = self.dark2(x)
        
        # P3 stage with class attention
        x = self.dark3(x)
        p3 = self.class_attention3(x, class_targets)
        small_objects = self.small_object_branch(p3)
        
        # P4 stage with class attention
        x = self.dark4(x)
        p4 = self.class_attention4(x, class_targets)
        
        # P5 stage with class attention
        x = self.dark5(x)
        p5 = self.class_attention5(x, class_targets)
        
        return small_objects, p3, p4, p5

# Enhanced Feature Pyramid Network
class EnhancedFPN(nn.Module):
    def __init__(self, base_channels=80):
        super().__init__()
        
        # Upsampling paths with enhanced feature fusion
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Enhanced lateral connections with residual blocks
        self.lateral_conv1 = ResidualBlock(base_channels * 16, base_channels * 8)
        self.lateral_conv2 = ResidualBlock(base_channels * 8, base_channels * 4)
        self.lateral_conv3 = ResidualBlock(base_channels * 4, base_channels * 2)
        
        # Feature enhancement modules
        self.fpn_conv1 = C2f(base_channels * 16, base_channels * 16, 3, shortcut=False)
        self.fpn_conv2 = C2f(base_channels * 8 * 2, base_channels * 8, 3, shortcut=False)
        self.fpn_conv3 = C2f(base_channels * 4 * 2, base_channels * 4, 3, shortcut=False)
        
        # Additional path for small object detection
        self.small_object_fusion = nn.Sequential(
            Conv(base_channels * 4, base_channels * 2, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            C2f(base_channels * 2, base_channels * 2, 3, shortcut=False)
        )
        
        # PAN downsampling connections
        self.down_conv1 = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.down_conv2 = Conv(base_channels * 8, base_channels * 8, 3, 2)
        
        # PAN feature fusion modules
        self.pan_conv1 = C2f(base_channels * 8 * 2, base_channels * 8, 3, shortcut=False)
        self.pan_conv2 = C2f(base_channels * 16 * 2, base_channels * 16, 3, shortcut=False)
        
        # Output feature enhancement
        self.out_conv_p3 = Conv(base_channels * 4, base_channels * 4, 3, 1)
        self.out_conv_p4 = Conv(base_channels * 8, base_channels * 8, 3, 1)
        self.out_conv_p5 = Conv(base_channels * 16, base_channels * 16, 3, 1)
        self.out_conv_small = Conv(base_channels * 2, base_channels * 2, 3, 1)

    def forward(self, inputs):
        small_objects, p3, p4, p5 = inputs
        
        # FPN top-down pathway
        p5_enhanced = self.fpn_conv1(p5)
        p5_up = self.upsample(self.lateral_conv1(p5_enhanced))
        
        p4_cat = torch.cat([p5_up, p4], 1)
        p4_enhanced = self.fpn_conv2(p4_cat)
        p4_up = self.upsample(self.lateral_conv2(p4_enhanced))
        
        p3_cat = torch.cat([p4_up, p3], 1)
        p3_enhanced = self.fpn_conv3(p3_cat)
        
        # Small object pathway
        small_enhanced = self.small_object_fusion(
            torch.cat([self.upsample(p3_enhanced), small_objects], 1)
        )
        
        # PAN bottom-up pathway
        p3_down = self.down_conv1(p3_enhanced)
        p4_cat_pan = torch.cat([p3_down, p4_enhanced], 1)
        p4_enhanced = self.pan_conv1(p4_cat_pan)
        
        p4_down = self.down_conv2(p4_enhanced)
        p5_cat_pan = torch.cat([p4_down, p5_enhanced], 1)
        p5_enhanced = self.pan_conv2(p5_cat_pan)
        
        # Final feature maps
        p3_out = self.out_conv_p3(p3_enhanced)
        p4_out = self.out_conv_p4(p4_enhanced)
        p5_out = self.out_conv_p5(p5_enhanced)
        small_out = self.out_conv_small(small_enhanced)
        
        return small_out, p3_out, p4_out, p5_out

# YOLO Detection Head
class EnhancedYOLOHead(nn.Module):
    def __init__(self, num_classes=9, base_channels=80):
        super().__init__()
        
        # Detection heads for different feature scales
        self.small_head = self._build_detection_head(base_channels * 2, num_classes)
        self.p3_head = self._build_detection_head(base_channels * 4, num_classes)
        self.p4_head = self._build_detection_head(base_channels * 8, num_classes)
        self.p5_head = self._build_detection_head(base_channels * 16, num_classes)
        
        # Class-specific weights for focal loss
        self.register_buffer('cls_weights', torch.tensor([
            1.0,   # Coin
            2.0,   # Drink (highest weight due to poor performance)
            1.2,   # Fork
            1.5,   # Hammer
            1.0,   # Pliers
            1.3,   # Screwdriver
            1.1,   # Sharp-item
            1.0,   # Stapler
            1.5    # Toddler (important for safety)
        ]))

    def _build_detection_head(self, in_channels, num_classes):
        """Build a detection head for a specific feature map scale"""
        return nn.Sequential(
            Conv(in_channels, in_channels, 3, 1),
            Conv(in_channels, in_channels, 3, 1),
            nn.Conv2d(in_channels, (num_classes + 5) * 3, 1)  # 3 anchors per location
        )
    
    def forward(self, features):
        small_out, p3_out, p4_out, p5_out = features
        
        # Apply detection heads
        small_pred = self.small_head(small_out)
        p3_pred = self.p3_head(p3_out)
        p4_pred = self.p4_head(p4_out)
        p5_pred = self.p5_head(p5_out)
        
        return small_pred, p3_pred, p4_pred, p5_pred

# Loss functions
class IOUloss(nn.Module):
    """Intersection over Union (IoU) loss with different variants"""
    def __init__(self, reduction="none", loss_type="ciou"):
        super().__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        # IoU calculation
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        # Intersection area
        inter_width = torch.min(pred_right, target_right) - torch.max(pred_left, target_left)
        inter_height = torch.min(pred_bottom, target_bottom) - torch.max(pred_top, target_top)
        inter_width = torch.clamp(inter_width, min=0)
        inter_height = torch.clamp(inter_height, min=0)
        
        intersection = inter_width * inter_height

        # Union area
        pred_area = (pred_right - pred_left) * (pred_bottom - pred_top)
        target_area = (target_right - target_left) * (target_bottom - target_top)
        union = pred_area + target_area - intersection
        
        # IoU
        iou = intersection / (union + 1e-6)
        
        if self.loss_type == "iou":
            loss = 1 - iou
        elif self.loss_type == "giou":
            # Generalized IoU includes enclosing box
            enclosing_left = torch.min(pred_left, target_left)
            enclosing_top = torch.min(pred_top, target_top)
            enclosing_right = torch.max(pred_right, target_right)
            enclosing_bottom = torch.max(pred_bottom, target_bottom)
            
            enclosing_width = enclosing_right - enclosing_left
            enclosing_height = enclosing_bottom - enclosing_top
            enclosing_area = enclosing_width * enclosing_height
            
            giou = iou - (enclosing_area - union) / (enclosing_area + 1e-6)
            loss = 1 - giou
        elif self.loss_type == "ciou":
            # Complete IoU includes aspect ratio consistency
            enclosing_left = torch.min(pred_left, target_left)
            enclosing_top = torch.min(pred_top, target_top)
            enclosing_right = torch.max(pred_right, target_right)
            enclosing_bottom = torch.max(pred_bottom, target_bottom)
            
            enclosing_width = enclosing_right - enclosing_left
            enclosing_height = enclosing_bottom - enclosing_top
            
            # Center distance
            pred_cx = (pred_left + pred_right) / 2
            pred_cy = (pred_top + pred_bottom) / 2
            target_cx = (target_left + target_right) / 2
            target_cy = (target_top + target_bottom) / 2
            
            center_dist_squared = (pred_cx - target_cx)**2 + (pred_cy - target_cy)**2
            enclosing_diag_squared = enclosing_width**2 + enclosing_height**2
            
            # Aspect ratio consistency
            pred_aspect = torch.atan((pred_right - pred_left) / (pred_bottom - pred_top + 1e-6))
            target_aspect = torch.atan((target_right - target_left) / (target_bottom - target_top + 1e-6))
            v = 4 / (math.pi**2) * (pred_aspect - target_aspect)**2
            alpha = v / (1 - iou + v + 1e-6)
            
            ciou = iou - center_dist_squared / (enclosing_diag_squared + 1e-6) - alpha * v
            loss = 1 - ciou
        
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        
        return loss

class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance"""
    def __init__(self, gamma=2.0, alpha=0.25, reduction="none"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target, weights=None):
        # Binary focal loss
        pred_sigmoid = torch.sigmoid(pred)
        pt = target * pred_sigmoid + (1 - target) * (1 - pred_sigmoid)
        weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        
        focal_weight = weight * (1 - pt) ** self.gamma
        
        # Apply additional class weights if provided
        if weights is not None:
            # Expand weights to match target shape
            focal_weight = focal_weight * weights
        
        loss = -focal_weight * torch.log(pt + 1e-6)
        
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        
        return loss

class EnhancedYOLOLoss(nn.Module):
    """Enhanced YOLO loss function with class weighting and improved components"""
    def __init__(self, num_classes=9):
        super().__init__()
        # Base loss components
        self.box_loss_weight = 7.5
        self.cls_loss_weight = 0.5
        self.dfl_loss_weight = 1.5
        
        # Class-specific weights based on confusion matrix analysis
        self.register_buffer('cls_weights', torch.tensor([
            1.0,   # Coin
            2.0,   # Drink (highest weight due to poor performance)
            1.2,   # Fork
            1.5,   # Hammer
            1.0,   # Pliers
            1.3,   # Screwdriver
            1.1,   # Sharp-item
            1.0,   # Stapler
            1.5    # Toddler (important for safety)
        ]))
        
        # Loss components
        self.iou_loss = IOUloss(reduction="none", loss_type="ciou")
        self.focal_loss = FocalLoss(gamma=2.0, alpha=0.25, reduction="none")
        
    def forward(self, predictions, targets):
        """
        Compute loss for YOLOv8 predictions
        
        Args:
            predictions: tuple of (small_pred, p3_pred, p4_pred, p5_pred)
            targets: dictionary with bbox targets, class targets, and masks
        
        Returns:
            total_loss and components (box_loss, cls_loss, dfl_loss)
        """
        # This is a simplified implementation - in a real YOLO implementation,
        # there would be complex logic for target assignment and loss calculation
        small_pred, p3_pred, p4_pred, p5_pred = predictions
        
        # Placeholder for complete loss implementation
        # In a real implementation, this would match bounding boxes to ground truth
        # and compute all the necessary loss components
        
        # For demonstration, assume we've extracted these components:
        box_loss = torch.tensor(1.0, device=small_pred.device)
        cls_loss = torch.tensor(1.0, device=small_pred.device)
        dfl_loss = torch.tensor(1.0, device=small_pred.device)
        
        # Apply weights
        total_loss = (
            self.box_loss_weight * box_loss +
            self.cls_loss_weight * cls_loss +
            self.dfl_loss_weight * dfl_loss
        )
        
        return total_loss, torch.stack([box_loss, cls_loss, dfl_loss])

# Complete Enhanced YOLOv8 model
class EnhancedYOLOv8(nn.Module):
    def __init__(self, num_classes=9, base_channels=80):
        super().__init__()
        self.backbone = EnhancedCSPDarknet(base_channels)
        self.fpn = EnhancedFPN(base_channels)
        self.head = EnhancedYOLOHead(num_classes, base_channels)
        self.loss_fn = EnhancedYOLOLoss(num_classes)
        
    def forward(self, x, targets=None):
        # Training mode (with targets)
        if targets is not None:
            # Extract class targets for attention modules
            class_targets = [target[:, 0] for target in targets] if isinstance(targets, list) else None
            
            # Forward through backbone with class attention
            backbone_features = self.backbone(x, class_targets)
            
            # Feature pyramid
            fpn_features = self.fpn(backbone_features)
            
            # Detection heads
            predictions = self.head(fpn_features)
            
            # Compute loss
            loss, loss_items = self.loss_fn(predictions, targets)
            
            return loss, loss_items
        
        # Inference mode
        else:
            backbone_features = self.backbone(x)
            fpn_features = self.fpn(backbone_features)
            predictions = self.head(fpn_features)
            
            return predictions

# Enhanced data augmentation
class EnhancedYOLOAugmentation:
    def __init__(self, img_size=640):
        # Standard augmentations
        self.base_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RGBShift(p=0.2),
            A.HueSaturationValue(p=0.2),
            A.Resize(img_size, img_size),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # Small object-focused augmentations
        self.small_object_transforms = A.Compose([
            A.RandomSizedBBoxSafeCrop(height=img_size, width=img_size, p=0.3),
            A.Blur(blur_limit=3, p=0.1),
            A.Sharpen(p=0.2),
            A.ISONoise(p=0.1),
            A.Resize(img_size, img_size),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # Class-specific augmentations
        self.toddler_transforms = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
            A.MotionBlur(p=0.2),  # Simulates motion which is common with toddlers
            A.Resize(img_size, img_size),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        self.drink_transforms = A.Compose([
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1, p=0.4),
            A.Resize(img_size, img_size),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def __call__(self, image, bboxes, class_labels):
        # Determine if image contains small objects
        has_small_objects = any(bbox[2] * bbox[3] < 0.05 for bbox in bboxes)
        
        # Determine if image contains toddlers or drinks
        has_toddler = 8 in class_labels  # Assuming toddler is class 8
        has_drink = 1 in class_labels    # Assuming drink is class 1
        
        # Apply appropriate augmentation based on content
        if has_toddler:
            transformed = self.toddler_transforms(image=image, bboxes=bboxes, class_labels=class_labels)
        elif has_drink:
            transformed = self.drink_transforms(image=image, bboxes=bboxes, class_labels=class_labels)
        elif has_small_objects:
            transformed = self.small_object_transforms(image=image, bboxes=bboxes, class_labels=class_labels)
        else:
            transformed = self.base_transforms(image=image, bboxes=bboxes, class_labels=class_labels)
        
        return transformed

# Enhanced Mosaic and MixUp augmentation
class EnhancedMosaicMixUp(Dataset):
    def __init__(self, dataset, img_size=640, mosaic_prob=0.8, mixup_prob=0.15):
        self.dataset = dataset
        self.img_size = img_size
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.border = [-img_size // 2, -img_size // 2]
        
    def __getitem__(self, index):
        # Apply mosaic augmentation with certain probability
        if random.random() < self.mosaic_prob:
            # Select indices to form mosaic, with bias toward problematic classes
            mosaic_indices = [index]
            
            # Find images containing underperforming classes (drink, hammer, toddler)
            target_classes = [1, 3, 8]  # Class indices for drink, hammer, toddler
            for _ in range(3):  # Need 3 more images for mosaic
                candidate_idx = random.randint(0, len(self.dataset) - 1)
                img, labels = self.dataset[candidate_idx]
                
                # Check if image contains any target classes
                if any(cls in labels[:, 0] for cls in target_classes):
                    # Higher chance of selecting this image
                    if random.random() < 0.8:
                        mosaic_indices.append(candidate_idx)
                        continue
                
                # If not selected based on class or random chance
                mosaic_indices.append(random.randint(0, len(self.dataset) - 1))
            
            # Apply mosaic augmentation
            mosaic_img, mosaic_labels = self._load_mosaic(mosaic_indices)
            
            # Apply mixup with certain probability
            if random.random() < self.mixup_prob:
                # Select another sample for mixup
                mixup_idx = random.randint(0, len(self.dataset) - 1)
                mixup_img, mixup_labels = self.dataset[mixup_idx]
                mosaic_img, mosaic_labels = self._mixup(mosaic_img, mosaic_labels, mixup_img, mixup_labels)
            
            return mosaic_img, mosaic_labels, None
        
        # No mosaic - return original image
        img, labels = self.dataset[index]
        return img, labels, None
    
    def _load_mosaic(self, indices):
        """Load 4 images and create a mosaic"""
        # Initialize mosaic image and labels
        s = self.img_size
        mosaic_img = np.zeros((s * 2, s * 2, 3), dtype=np.uint8)
        mosaic_labels = []
        
        # Center point for mosaic
        cx, cy = s, s
        
        # Load and place 4 images in mosaic
        for i, idx in enumerate(indices):
            img, labels = self.dataset[idx]
            
            # Convert PIL or torch tensor to numpy if needed
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            
            # Image height and width
            h, w = img.shape[0], img.shape[1]
            
            # Place image in one of the four positions
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(cx - w, 0), max(cy - h, 0), cx, cy
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = cx, max(cy - h, 0), min(cx + w, s * 2), cy
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(cx - w, 0), cy, cx, min(s * 2, cy + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = cx, cy, min(cx + w, s * 2), min(s * 2, cy + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            # Place the image section
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            
            # Adjust bounding box coordinates
            if len(labels):
                # Convert YOLO xywh to pixel xyxy format
                labels_copy = labels.copy()
                labels_copy[:, 1] = w * (labels[:, 1] - labels[:, 3] / 2) + x1a - x1b
                labels_copy[:, 2] = h * (labels[:, 2] - labels[:, 4] / 2) + y1a - y1b
                labels_copy[:, 3] = w * labels[:, 3]
                labels_copy[:, 4] = h * labels[:, 4]
                
                # Append to mosaic labels
                mosaic_labels.append(labels_copy)
        
        if len(mosaic_labels):
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            
            # Clip coordinates to image boundaries
            np.clip(mosaic_labels[:, 1:], 0, 2 * s, out=mosaic_labels[:, 1:])
            
            # Convert back to YOLO format
            mosaic_labels[:, 1] = (mosaic_labels[:, 1] + mosaic_labels[:, 3] / 2) / (2 * s)
            mosaic_labels[:, 2] = (mosaic_labels[:, 2] + mosaic_labels[:, 4] / 2) / (2 * s)
            mosaic_labels[:, 3] = mosaic_labels[:, 3] / (2 * s)
            mosaic_labels[:, 4] = mosaic_labels[:, 4] / (2 * s)
        
        # Resize final mosaic image to target size if needed
        if mosaic_img.shape[0] != self.img_size or mosaic_img.shape[1] != self.img_size:
            mosaic_img = cv2.resize(mosaic_img, (self.img_size, self.img_size))
        
        return mosaic_img, mosaic_labels
    
    def _mixup(self, img1, labels1, img2, labels2, alpha=0.5):
        """Apply mixup augmentation"""
        # Convert to float for blending
        if not isinstance(img1, np.ndarray):
            img1 = np.array(img1)
        if not isinstance(img2, np.ndarray):
            img2 = np.array(img2)
            
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        
        # Resize img2 to match img1 dimensions
        if img1.shape[0] != img2.shape[0] or img1.shape[1] != img2.shape[1]:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Apply mixup
        r = np.random.beta(alpha, alpha)
        img = (r * img1 + (1 - r) * img2).astype(np.uint8)
        
        # Combine labels 
        labels = np.concatenate((labels1, labels2), 0)
        
        return img, labels

# Training configuration
def configure_training(model, train_dataset, val_dataset, num_epochs=20, batch_size=16):
    """Configure training parameters and data loading"""
    # DataLoader with enhanced augmentation
    train_loader = DataLoader(
        EnhancedMosaicMixUp(train_dataset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,  # Custom collate function for batching
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Optimizer with cosine learning rate schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    num_steps = num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        total_steps=num_steps,
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100
    )
    
    return train_loader, val_loader, optimizer, scheduler

# Collate function for batching
def collate_fn(batch):
    """Custom collate function for batching images and targets"""
    imgs, labels, paths = zip(*batch)
    
    # Filter out None values
    imgs = [img for img in imgs if img is not None]
    labels = [label for label in labels if label is not None]
    
    # Stack images
    imgs = torch.stack([torch.from_numpy(img).float() for img in imgs]) / 255.0
    
    # Return images and targets
    return imgs, labels, paths

# Training Loop
def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, device):
    """Enhanced training procedure with class-specific focus"""
    model.train()
    
    epoch_loss = 0
    epoch_box_loss = 0
    epoch_cls_loss = 0
    epoch_dfl_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/20")
    
    for i, (imgs, targets, _) in enumerate(progress_bar):
        imgs = imgs.to(device)
        targets = [target.to(device) for target in targets]
        
        # Forward pass
        loss, loss_items = model(imgs, targets)
        box_loss, cls_loss, dfl_loss = loss_items
        
        # Apply gradient accumulation for larger effective batch size
        if i % 2 == 0:
            # Backward pass
            loss.backward()
            
            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        else:
            loss.backward()
        
        # Update progress bar
        epoch_loss += loss.item()
        epoch_box_loss += box_loss.item()
        epoch_cls_loss += cls_loss.item()
        epoch_dfl_loss += dfl_loss.item()
        
        progress_bar.set_postfix({
            'loss': epoch_loss / (i + 1),
            'box_loss': epoch_box_loss / (i + 1),
            'cls_loss': epoch_cls_loss / (i + 1),
            'dfl_loss': epoch_dfl_loss / (i + 1),
            'lr': optimizer.param_groups[0]['lr']
        })
    
    return {
        'loss': epoch_loss / len(train_loader),
        'box_loss': epoch_box_loss / len(train_loader),
        'cls_loss': epoch_cls_loss / len(train_loader),
        'dfl_loss': epoch_dfl_loss / len(train_loader)
    }

# Validation function
def validate(model, val_loader, device):
    """Evaluate model on validation dataset"""
    model.eval()
    
    stats = []
    
    with torch.no_grad():
        for imgs, targets, _ in tqdm(val_loader, desc="Validating"):
            imgs = imgs.to(device)
            
            # Forward pass (inference mode)
            predictions = model(imgs)
            
            # Here, you'd typically run NMS and calculate metrics
            # This is a simplified placeholder
    
    # Calculate and return metrics
    metrics = {
        'precision': 0.0,
        'recall': 0.0,
        'mAP50': 0.0,
        'mAP50-95': 0.0
    }
    
    return metrics

# Main training function
def train_model(train_dataset, val_dataset, num_classes=9, num_epochs=20, batch_size=16):
    """Train the enhanced YOLOv8 model"""
    # Create model
    model = EnhancedYOLOv8(num_classes=num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Configure training
    train_loader, val_loader, optimizer, scheduler = configure_training(
        model, train_dataset, val_dataset, num_epochs, batch_size
    )
    
    # Training loop
    best_map = 0
    for epoch in range(num_epochs):
        # Train for one epoch
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, epoch, device)
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training: loss={train_metrics['loss']:.3f}, box_loss={train_metrics['box_loss']:.3f}, "
              f"cls_loss={train_metrics['cls_loss']:.3f}, dfl_loss={train_metrics['dfl_loss']:.3f}")
        print(f"Validation: mAP50={val_metrics['mAP50']:.3f}, mAP50-95={val_metrics['mAP50-95']:.3f}, "
              f"precision={val_metrics['precision']:.3f}, recall={val_metrics['recall']:.3f}")
        
        # Save best model
        if val_metrics['mAP50'] > best_map:
            best_map = val_metrics['mAP50']
            torch.save(model.state_dict(), 'best_enhanced_yolov8.pt')
            print(f"Saved best model with mAP50: {best_map:.3f}")
    
    return model

# Implementation example
if __name__ == "__main__":
    import cv2
    
    # Define your dataset classes here
    class YOLODataset(Dataset):
        def __init__(self, img_dir, annotations, img_size=640, transform=None):
            self.img_dir = img_dir
            self.annotations = annotations
            self.img_size = img_size
            self.transform = transform
            
        def __len__(self):
            return len(self.annotations)
            
        def __getitem__(self, idx):
            # Load image and labels
            img_path = os.path.join(self.img_dir, self.annotations[idx]['image'])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get bounding boxes and class labels
            bboxes = self.annotations[idx]['bboxes']
            class_labels = self.annotations[idx]['labels']
            
            # Apply transformations if specified
            if self.transform:
                transformed = self.transform(image=img, bboxes=bboxes, class_labels=class_labels)
                img = transformed['image']
                bboxes = transformed['bboxes']
                class_labels = transformed['class_labels']
            
            # Create labels array in YOLO format
            labels = np.zeros((len(bboxes), 5))
            for i, (bbox, cls) in enumerate(zip(bboxes, class_labels)):
                labels[i, 0] = cls  # Class ID
                labels[i, 1:] = bbox  # x, y, w, h (normalized)
            
            return img, labels, img_path
    
    # Example usage (placeholder)
    print("Setting up enhanced YOLOv8 for toddler and hazardous object detection...")
    print("This model aims to increase confidence scores by at least 0.05 for key classes")
    
    # Placeholder for dataset creation
    print("To use this model:")
    print("1. Create your datasets:")
    print("   train_dataset = YOLODataset(train_img_dir, train_annotations)")
    print("   val_dataset = YOLODataset(val_img_dir, val_annotations)")
    print("2. Apply the enhanced data augmentation:")
    print("   train_transforms = EnhancedYOLOAugmentation()")
    print("   train_dataset.transform = train_transforms")
    print("3. Train the model:")
    print("   model = train_model(train_dataset, val_dataset, num_classes=9)")
    print("4. Use the trained model for inference with higher confidence in detecting toddlers and hazardous objects")
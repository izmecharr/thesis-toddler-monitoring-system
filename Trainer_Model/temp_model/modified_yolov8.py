import torch
import torch.nn as nn
import os
from pathlib import Path
import yaml

from ultralytics.nn.modules import (
    Conv, C2f, SPPF, Concat, Detect,
    DFL, Proto, RepC3, C3
)
from ultralytics.models.yolo.model import DetectionModel
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import initialize_weights
from ultralytics import YOLO

class ResidualC2f(nn.Module):
    """C2f block with residual connection for better gradient flow."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, act=True):
        super().__init__()
        self.c2f = C2f(c1, c2, n, shortcut, g, e, act)
        self.residual = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1, act=act)
        
    def forward(self, x):
        return self.c2f(x) + self.residual(x)

class YOLOv8Enhanced(DetectionModel):
    """Modified YOLOv8 model with enhanced backbone for deeper feature extraction."""
    
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)
        
        if isinstance(cfg, str) and os.path.exists(cfg):
            with open(cfg, 'r') as f:
                self.yaml_dict = yaml.safe_load(f)
            self.yaml = cfg
        elif isinstance(cfg, dict):
            self.yaml_dict = cfg
            self.yaml = 'yolov8n.yaml'
        else:
            self.yaml = cfg
    
    def parse_model(self, d, ch):
        """
        Override the parse_model method to create our enhanced backbone.
        Now with more balanced architecture and residual connections.
        """
        nc, gd, gw, act = (
            self.yaml['nc'],
            self.yaml['depth_multiple'],
            self.yaml['width_multiple'],
            self.yaml['activation']
        )
        
        layers, save, c2 = [], [], ch[-1]
        
        # Initial conv
        layers.append([Conv, [c2, 64, 3, 2, 1, act], 0])  # 0-P1/2
        c2 = 64
        
        # First block - P2/4
        layers.append([Conv, [c2, 128, 3, 2, 1, act], 1])  # 1-P2/4
        c2 = 128
        layers.append([C2f, [c2, c2, 3, 1, act], 2])  # 2
        # Add ONE extra ResidualC2f block for better features
        layers.append([ResidualC2f, [c2, c2, 2, 1, act], 3])  # 3
        
        # Second block - P3/8
        layers.append([Conv, [c2, 256, 3, 2, 1, act], 4])  # 4-P3/8
        c2 = 256
        layers.append([C2f, [c2, c2, 6, 1, act], 5])  # 5
        save.append(5)  # Save P3/8 scale features
        
        # Third block - P4/16
        layers.append([Conv, [c2, 512, 3, 2, 1, act], 6])  # 6-P4/16
        c2 = 512
        layers.append([C2f, [c2, c2, 6, 1, act], 7])  # 7
        # Add ONE extra ResidualC2f block
        layers.append([ResidualC2f, [c2, c2, 3, 1, act], 8])  # 8
        save.append(8)  # Save P4/16 scale features
        
        # Fourth block - P5/32
        layers.append([Conv, [c2, 1024, 3, 2, 1, act], 9])  # 9-P5/32
        c2 = 1024
        layers.append([C2f, [c2, c2, 3, 1, act], 10])  # 10
        
        # SPPF block
        layers.append([SPPF, [c2, c2, 5, act], 11])  # 11
        save.append(11)  # Save P5/32 scale features
        
        # Neck - Upsampling path
        layers.append([nn.Upsample, [None, 2, 'nearest'], 12])  # 12
        layers.append([Concat, [[12, 8]], 13])  # 13 cat backbone P4
        c2 = 1024 + 512
        layers.append([C2f, [c2, 512, 3, 1, act, False], 14])  # 14
        c2 = 512
        
        layers.append([nn.Upsample, [None, 2, 'nearest'], 15])  # 15
        layers.append([Concat, [[15, 5]], 16])  # 16 cat backbone P3
        c2 = 512 + 256
        layers.append([C2f, [c2, 256, 3, 1, act, False], 17])  # 17
        c2 = 256
        
        # Downsampling path
        layers.append([Conv, [c2, 256, 3, 2, 1, act], 18])  # 18
        c2 = 256
        layers.append([Concat, [[18, 14]], 19])  # 19 cat head P4
        c2 = 256 + 512
        layers.append([C2f, [c2, 512, 3, 1, act, False], 20])  # 20
        c2 = 512
        
        layers.append([Conv, [c2, 512, 3, 2, 1, act], 21])  # 21
        c2 = 512
        layers.append([Concat, [[21, 11]], 22])  # 22 cat head P5
        c2 = 512 + 1024
        layers.append([C2f, [c2, 1024, 3, 1, act, False], 23])  # 23
        c2 = 1024
        
        # Detection head
        layers.append([Detect, [nc, [17, 20, 23]], 24])  # Detect(P3, P4, P5)
        
        self.save = save
        return layers
        
    def _forward_once(self, x):
        """Forward pass through the model."""
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

def save_model_with_yaml(model, save_path):
    """Save model with YAML attribute preserved."""
    model_dict = {
        'model': model,
        'yaml_path': model.yaml if hasattr(model, 'yaml') else 'yolov8n.yaml'
    }
    torch.save(model_dict, save_path)
    return save_path

def load_model_with_yaml(model_path):
    """Load model and restore YAML attribute."""
    checkpoint = torch.load(model_path)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model = checkpoint['model']
        yaml_path = checkpoint.get('yaml_path', 'yolov8n.yaml')
        setattr(model, 'yaml', yaml_path)
        return model
    else:
        model = checkpoint
        if not hasattr(model, 'yaml'):
            setattr(model, 'yaml', 'yolov8n.yaml')
        return model

def create_modified_yolov8(size='n', pretrained=False):
    """
    Create a modified YOLOv8 model with enhanced backbone.
    
    Args:
        size (str): Model size - n, s, m, l, or x
        pretrained (bool): Whether to load pretrained weights
        
    Returns:
        YOLOv8Enhanced: Modified YOLOv8 model
    """
    print(f"Creating YOLOv8{size} enhanced model...")
    
    try:
        import pkg_resources
        yaml_path = pkg_resources.resource_filename('ultralytics', f'cfg/models/v8/yolov8{size}.yaml')
        if not os.path.exists(yaml_path):
            yaml_path = pkg_resources.resource_filename('ultralytics', f'models/v8/yolov8{size}.yaml')
            if not os.path.exists(yaml_path):
                yaml_path = f'yolov8{size}.yaml'
                print(f"Using default YAML path: {yaml_path}")
            else:
                print(f"Found YAML at: {yaml_path}")
        else:
            print(f"Found YAML at: {yaml_path}")
    except Exception as e:
        print(f"Error finding YAML path: {e}")
        yaml_path = f'yolov8{size}.yaml'
        print(f"Using default YAML path: {yaml_path}")
    
    model = YOLOv8Enhanced(yaml_path)
    setattr(model, 'yaml', yaml_path)
    
    if pretrained:
        try:
            print("Loading pretrained weights...")
            from ultralytics import YOLO
            
            try:
                original_model = YOLO(f'yolov8{size}.pt')
                print(f"Successfully loaded pretrained YOLOv8{size} model")
                
                pretrained_dict = original_model.model.state_dict()
                model_dict = model.state_dict()
                
                pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                if k in model_dict and v.shape == model_dict[k].shape}
                
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                
                print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} pretrained weights")
            except Exception as e:
                print(f"Error loading pretrained weights via YOLO: {e}")
                print("Starting with random weights initialization")
        except Exception as e:
            print(f"Error during weight transfer: {e}")
            print("Starting with random weights initialization")
    
    return model
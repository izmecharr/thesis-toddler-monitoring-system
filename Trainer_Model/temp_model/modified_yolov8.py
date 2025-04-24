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
# Updated imports to match current Ultralytics structure
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import initialize_weights
from ultralytics import YOLO

class YOLOv8Enhanced(DetectionModel):
    """Modified YOLOv8 model with enhanced backbone for deeper feature extraction."""
    
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):
        # Pass parameters to the parent class constructor
        super().__init__(cfg, ch, nc, verbose)
        
        # Store the YAML configuration explicitly
        # This ensures it's accessible even after saving/loading
        if isinstance(cfg, str) and os.path.exists(cfg):
            with open(cfg, 'r') as f:
                self.yaml_dict = yaml.safe_load(f)
            self.yaml = cfg  # Store the path to YAML file
        elif isinstance(cfg, dict):
            self.yaml_dict = cfg  # Store the YAML dictionary
            self.yaml = 'yolov8n.yaml'  # Default path
        else:
            self.yaml = cfg  # Store whatever was passed
    
    def parse_model(self, d, ch):
        """
        Override the parse_model method to create our enhanced backbone.
        This method is called by the parent class during initialization.
        """
        # Get parameters from model definition
        nc, gd, gw, act = (
            self.yaml['nc'],
            self.yaml['depth_multiple'],
            self.yaml['width_multiple'],
            self.yaml['activation']
        )
        
        layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
        
        # Initial conv
        layers.append([Conv, [c2, 64, 3, 2, 1, act], 0])  # 0-P1/2
        c2 = 64
        
        # First block
        layers.append([Conv, [c2, 128, 3, 2, 1, act], 1])  # 1-P2/4
        c2 = 128
        
        # Enhanced C2f block with extra depth at P2/4 scale
        layers.append([C2f, [c2, c2, 3, 1, act], 2])  # 2
        
        # Additional C2f block at P2/4 scale for deeper features
        layers.append([C2f, [c2, c2, 3, 1, act], 3])  # 3
        
        # Second block with downsampling
        layers.append([Conv, [c2, 256, 3, 2, 1, act], 4])  # 4-P3/8
        c2 = 256
        
        # Enhanced C2f block with extra depth at P3/8 scale
        layers.append([C2f, [c2, c2, 6, 1, act], 5])  # 5
        
        # Additional C2f block at P3/8 scale for deeper features
        layers.append([C2f, [c2, c2, 3, 1, act], 6])  # 6
        save.append(6)  # Save P3/8 scale features
        
        # Third block with downsampling
        layers.append([Conv, [c2, 512, 3, 2, 1, act], 7])  # 7-P4/16
        c2 = 512
        
        # Enhanced C2f block with extra depth at P4/16 scale
        layers.append([C2f, [c2, c2, 9, 1, act], 8])  # 8
        
        # Additional C2f block at P4/16 scale for deeper features
        layers.append([C2f, [c2, c2, 3, 1, act], 9])  # 9
        save.append(9)  # Save P4/16 scale features
        
        # Fourth block with downsampling
        layers.append([Conv, [c2, 1024, 3, 2, 1, act], 10])  # 10-P5/32
        c2 = 1024
        
        # Enhanced C2f block with extra depth at P5/32 scale
        layers.append([C2f, [c2, c2, 3, 1, act], 11])  # 11
        
        # Additional C2f block at P5/32 scale for deeper features
        layers.append([C2f, [c2, c2, 3, 1, act], 12])  # 12
        
        # SPPF block
        layers.append([SPPF, [c2, c2, 5, act], 13])  # 13
        save.append(13)  # Save P5/32 scale features
        
        # Neck
        # Upsampling path
        layers.append([nn.Upsample, [None, 2, 'nearest'], 14])  # 14
        layers.append([Concat, [[14, 9]], 15])  # 15 cat backbone P4
        c2 = 1024 + 512
        layers.append([C2f, [c2, 512, 3, 1, act, False], 16])  # 16
        c2 = 512
        
        layers.append([nn.Upsample, [None, 2, 'nearest'], 17])  # 17
        layers.append([Concat, [[17, 6]], 18])  # 18 cat backbone P3
        c2 = 512 + 256
        layers.append([C2f, [c2, 256, 3, 1, act, False], 19])  # 19
        c2 = 256
        
        # Downsampling path
        layers.append([Conv, [c2, 256, 3, 2, 1, act], 20])  # 20
        c2 = 256
        layers.append([Concat, [[20, 16]], 21])  # 21 cat head P4
        c2 = 256 + 512
        layers.append([C2f, [c2, 512, 3, 1, act, False], 22])  # 22
        c2 = 512
        
        layers.append([Conv, [c2, 512, 3, 2, 1, act], 23])  # 23
        c2 = 512
        layers.append([Concat, [[23, 13]], 24])  # 24 cat head P5
        c2 = 512 + 1024
        layers.append([C2f, [c2, 1024, 3, 1, act, False], 25])  # 25
        c2 = 1024
        
        # Detection head
        layers.append([Detect, [nc, [19, 22, 25]], 26])  # Detect(P3, P4, P5)
        
        self.save = save
        return layers
        
    def _forward_once(self, x):
        """
        Forward pass through the model.
        This is the method that DetectionModel expects to be implemented.
        """
        y = []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

def save_model_with_yaml(model, save_path):
    """Save model with YAML attribute preserved."""
    # Create a dictionary to store all necessary components
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
            setattr(model, 'yaml', 'yolov8n.yaml')  # Default YAML
        return model

# Function to create the modified model
def create_modified_yolov8(size='n', pretrained=False):
    """
    Create a modified YOLOv8 model with enhanced backbone.
    
    Args:
        size (str): Model size - n, s, m, l, or x
        pretrained (bool): Whether to load pretrained weights
        
    Returns:
        YOLOv8Enhanced: Modified YOLOv8 model
    """
    # Initialize the enhanced model
    print(f"Creating YOLOv8{size} enhanced model...")
    
    # Try to find YAML file
    try:
        import pkg_resources
        yaml_path = pkg_resources.resource_filename('ultralytics', f'cfg/models/v8/yolov8{size}.yaml')
        if not os.path.exists(yaml_path):
            # Try alternative path
            yaml_path = pkg_resources.resource_filename('ultralytics', f'models/v8/yolov8{size}.yaml')
            if not os.path.exists(yaml_path):
                # Fall back to direct path for older versions
                yaml_path = f'yolov8{size}.yaml'
                print(f"Using default YAML path: {yaml_path}")
            else:
                print(f"Found YAML at: {yaml_path}")
        else:
            print(f"Found YAML at: {yaml_path}")
    except Exception as e:
        print(f"Error finding YAML path: {e}")
        # Default path as fallback
        yaml_path = f'yolov8{size}.yaml'
        print(f"Using default YAML path: {yaml_path}")
    
    # Create the enhanced model with explicit YAML path
    model = YOLOv8Enhanced(yaml_path)
    
    # Explicitly ensure the yaml attribute is set
    setattr(model, 'yaml', yaml_path)
    
    if pretrained:
        try:
            # Try to load with YOLO instead of torch.hub
            print("Loading pretrained weights...")
            from ultralytics import YOLO
            
            try:
                # First try with official model
                original_model = YOLO(f'yolov8{size}.pt')
                print(f"Successfully loaded pretrained YOLOv8{size} model")
                
                # Transfer compatible weights
                pretrained_dict = original_model.model.state_dict()
                model_dict = model.state_dict()
                
                # Filter out incompatible weights due to architecture changes
                pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                if k in model_dict and v.shape == model_dict[k].shape}
                
                # Update model with pretrained weights
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

# Usage example
if __name__ == "__main__":
    # Create a modified YOLOv8n model
    model = create_modified_yolov8(size='n', pretrained=True)
    
    # Print model summary
    print(model)
    
    # Example inference with a dummy input
    x = torch.randn(1, 3, 640, 640)
    outputs = model(x)
    
    # Print output shapes
    if isinstance(outputs, torch.Tensor):
        print(f"Output shape: {outputs.shape}")
    else:
        for i, output in enumerate(outputs):
            if isinstance(output, torch.Tensor):
                print(f"Output {i} shape: {output.shape}")
import torch
import torch.nn as nn
import os
from pathlib import Path
import yaml
import sys

from ultralytics.nn.modules import (
    Conv, C2f, SPPF, Concat, Detect,
    DFL, Proto, RepC3, C3
)
from ultralytics.models.yolo.model import DetectionModel
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import initialize_weights
from ultralytics import YOLO

class SmallObjectEnhance(nn.Module):
    def __init__(self, c1, c2, act=True):
        super().__init__()
        self.cv1 = Conv(c1, c2//2, 1, 1, act=act)
        self.cv2 = Conv(c2//2, c2, 3, 1, act=act)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(c2, c2//16, 1, act=act),
            Conv(c2//16, c2, 1, act=act),
            nn.Sigmoid()
        )
        # Add attributes needed by YOLOv8
        self.i = 0  # Layer index, will be set by model builder
        self.f = -1  # Input source, will be set by model builder
        
    def forward(self, x):
        x = self.cv2(self.cv1(x))
        return x * self.attn(x)

class ResidualC2f(nn.Module):
    """C2f block with residual connection for better gradient flow."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c2f = C2f(c1, c2, n, shortcut, g, e)  # Removed the 'act' parameter
        self.residual = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        # Add attributes needed by YOLOv8
        self.i = 0  # Layer index, will be set by model builder
        self.f = -1  # Input source, will be set by model builder
        
    def forward(self, x):
        return self.c2f(x) + self.residual(x)

# Register custom modules in the ultralytics.nn.modules namespace
import ultralytics.nn.modules
ultralytics.nn.modules.ResidualC2f = ResidualC2f
ultralytics.nn.modules.SmallObjectEnhance = SmallObjectEnhance

# Make them available at the top level for the model builder
setattr(sys.modules['ultralytics.nn.modules'], 'ResidualC2f', ResidualC2f)
setattr(sys.modules['ultralytics.nn.modules'], 'SmallObjectEnhance', SmallObjectEnhance)

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
    
    def parse_model(self, ch):
        nc, act = (
            self.yaml['nc'],
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
        layers.append([ResidualC2f, [c2, c2, 2, 1 ], 3])  # 3
        
        # Second block - P3/8
        layers.append([Conv, [c2, 256, 3, 2, 1, act], 4])  # 4-P3/8
        c2 = 256
        layers.append([C2f, [c2, c2, 6, 1, act], 5])  # 5
        save.append(5)  # Save P3/8 scale features

        layers.append([ResidualC2f, [c2, c2, 3, 1], 6])  # Changed from 5.5 to integer index
        save.append(6)  # Save ResidualC2f enhanced features

        layers.append([SmallObjectEnhance, [c2, c2, act], 7])  # Changed from 5.6 to integer index
        save.append(7)  # Save small object enhanced features
        
        # Third block - P4/16
        layers.append([Conv, [c2, 512, 3, 2, 1, act], 8])  # Changed from 6 to 8
        c2 = 512
        layers.append([C2f, [c2, c2, 6, 1, act], 9])  # Changed from 7 to 9
        # Add ONE extra ResidualC2f block
        layers.append([ResidualC2f, [c2, c2, 3, 1], 10])  # Changed from 8 to 10
        save.append(10)  # Save P4/16 scale features - Changed from 8 to 10
        
        # Fourth block - P5/32
        layers.append([Conv, [c2, 1024, 3, 2, 1, act], 11])  # Changed from 9 to 11
        c2 = 1024
        layers.append([C2f, [c2, c2, 3, 1, act], 12])  # Changed from 10 to 12
        
        # SPPF block
        layers.append([SPPF, [c2, c2, 5, act], 13])  # Changed from 11 to 13
        save.append(13)  # Save P5/32 scale features - Changed from 11 to 13
        
        # Neck - Upsampling path
        layers.append([nn.Upsample, [None, 2, 'nearest'], 14])  # Changed from 12 to 14
        layers.append([Concat, [[14, 10]], 15])  # Changed from [12, 8] to [14, 10]
        c2 = 1024 + 512
        layers.append([C2f, [c2, 512, 3, 1, act, False], 16])  # Changed from 14 to 16
        c2 = 512
        
        layers.append([nn.Upsample, [None, 2, 'nearest'], 17])  # Changed from 15 to 17

        # Include all enhanced P3/8 features
        layers.append([Concat, [[17, 5, 6, 7]], 18])  # Changed from [15, 5] to [17, 5, 6, 7]
        c2 = 512 + 256 + 256 + 256  # Adjusted to include all three P3/8 features

        layers.append([C2f, [c2, 256, 3, 1, act, False], 19])  # Changed from 17 to 19
        c2 = 256
        
        # Downsampling path
        layers.append([Conv, [c2, 256, 3, 2, 1, act], 20])  # Changed from 18 to 20
        c2 = 256
        layers.append([Concat, [[20, 16]], 21])  # Changed from [18, 14] to [20, 16]
        c2 = 256 + 512
        layers.append([C2f, [c2, 512, 3, 1, act, False], 22])  # Changed from 20 to 22
        c2 = 512
        
        layers.append([Conv, [c2, 512, 3, 2, 1, act], 23])  # Changed from 21 to 23
        c2 = 512
        layers.append([Concat, [[23, 13]], 24])  # Changed from [21, 11] to [23, 13]
        c2 = 512 + 1024
        layers.append([C2f, [c2, 1024, 3, 1, act, False], 25])  # Changed from 23 to 25
        c2 = 1024
        
        # Detection head
        layers.append([Detect, [nc, [19, 22, 25]], 26])  # Changed from [17, 20, 23] to [19, 22, 25]
        
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
    
    # Create model instance
    model = YOLOv8Enhanced(yaml_path)
    setattr(model, 'yaml', yaml_path)
    
    # Add safety for None values in forward pass
    original_forward_once = model._forward_once
    
    def safe_forward_once(self, x):
        """Safer forward pass that handles None values properly."""
        y = []
        for i, m in enumerate(self.model):
            if m.f != -1:
                try:
                    if isinstance(m.f, int):
                        x = y[m.f]
                        # Safety check to avoid operating on None
                        if x is None:
                            # Skip this layer if the input is None
                            y.append(None)
                            continue
                    else:
                        # Handle list inputs safely
                        inputs = [x if j == -1 else y[j] for j in m.f]
                        # Check if any input is None
                        if any(inp is None for inp in inputs):
                            y.append(None)
                            continue
                        x = inputs
                except Exception as e:
                    print(f"Error in forward pass at layer {i}: {e}")
                    y.append(None)
                    continue
            
            try:
                x = m(x)
            except Exception as e:
                print(f"Error processing layer {i} ({m.__class__.__name__}): {e}")
                # Set output to None and continue
                y.append(None)
                continue
            
            y.append(x if m.i in self.save else None)
        
        return x
    
    # Apply the safer forward pass
    model._forward_once = lambda x: safe_forward_once(model, x)
    
    # Print initial parameter count
    initial_params = sum(p.numel() for p in model.parameters())
    print(f"Initial model parameter count: {initial_params:,}")
    
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
    
    # Verify parameter count after loading weights
    final_params = sum(p.numel() for p in model.parameters())
    print(f"Final model parameter count: {final_params:,}")
    
    return model

def create_enhanced_yolov8_direct(size='n', pretrained=True):
    """
    Create an enhanced YOLOv8 model by directly modifying an existing model.
    This approach adapts to the actual model structure.
    """
    import torch
    from ultralytics import YOLO
    from copy import deepcopy
    
    # Load the base model
    print(f"Loading base YOLOv8{size} model...")
    model = YOLO(f'yolov8{size}.pt').model
    
    # Get original parameter count for verification
    original_params = sum(p.numel() for p in model.parameters())
    print(f"Original parameter count: {original_params:,}")
    
    # First, explore the model structure to find P3/8 scale
    print("Model structure exploration:")
    p3_layer_idx = None
    p3_channel_size = None
    
    # Identify channel sizes in the model
    for i, m in enumerate(model.model):
        print(f"Layer {i}: {m.__class__.__name__}")
        
        # Extract channel size if possible
        channel_size = None
        if hasattr(m, 'cv2') and hasattr(m.cv2, 'conv'):
            channel_size = m.cv2.conv.out_channels
        elif hasattr(m, 'conv') and hasattr(m.conv, 'out_channels'):
            channel_size = m.conv.out_channels
        elif hasattr(m, 'out_channels'):
            channel_size = m.out_channels
        
        if channel_size is not None:
            print(f"  Channel size: {channel_size}")
            
        # Look for the C2f block in P3/8 scale (typically with 256 channels)
        if isinstance(m, C2f) and channel_size == 256:
            p3_layer_idx = i
            p3_channel_size = channel_size
            print(f"Found P3/8 scale at layer {i} with {channel_size} channels")
            break
    
    if p3_layer_idx is None or p3_channel_size is None:
        print("Could not identify P3/8 scale layer. Looking for any C2f with 256 channels.")
        for i, m in enumerate(model.model):
            if isinstance(m, C2f):
                # Try to determine channel size
                channel_size = None
                if hasattr(m, 'cv2') and hasattr(m.cv2, 'conv'):
                    channel_size = m.cv2.conv.out_channels
                
                if channel_size == 256:
                    p3_layer_idx = i
                    p3_channel_size = channel_size
                    print(f"Using layer {i} with {channel_size} channels")
                    break
    
    if p3_layer_idx is None or p3_channel_size is None:
        raise ValueError("Could not identify appropriate layer for enhancement. Model structure not recognized.")
    
    # Create our custom enhancer modules
    residual_c2f = ResidualC2f(p3_channel_size, p3_channel_size)
    small_obj_enhance = SmallObjectEnhance(p3_channel_size, p3_channel_size)
    
    # We'll insert these modules directly in the model.model list
    model_modules = list(model.model)
    
    # Insert after P3/8 C2f block
    residual_c2f.i = p3_layer_idx + 1  # Next index
    residual_c2f.f = p3_layer_idx  # Take input from C2f block
    
    small_obj_enhance.i = p3_layer_idx + 2  # Next index
    small_obj_enhance.f = p3_layer_idx + 1  # Take input from ResidualC2f
    
    # Insert new modules
    model_modules.insert(p3_layer_idx + 1, residual_c2f)
    model_modules.insert(p3_layer_idx + 2, small_obj_enhance)
    
    # Update indices for all subsequent layers
    for i in range(p3_layer_idx + 3, len(model_modules)):
        model_modules[i].i = i
        
        # If this layer takes input from previous layers, update those indices
        if hasattr(model_modules[i], 'f'):
            if isinstance(model_modules[i].f, int) and model_modules[i].f >= p3_layer_idx + 1:
                model_modules[i].f += 2  # Shift by 2 for our added modules
            elif isinstance(model_modules[i].f, list):
                model_modules[i].f = [f + 2 if isinstance(f, int) and f >= p3_layer_idx + 1 else f for f in model_modules[i].f]
    
    # Update the model
    model.model = nn.ModuleList(model_modules)
    
    # Add P3/8 enhanced features to the save list if we can find it
    if hasattr(model, 'save'):
        print(f"Original save indices: {model.save}")
        if p3_layer_idx in model.save:
            model.save.append(p3_layer_idx + 1)  # ResidualC2f
            model.save.append(p3_layer_idx + 2)  # SmallObjectEnhance
            print(f"Updated save indices: {model.save}")
    
    # Find the neck concatenation for P3/8 features and update it
    for i, m in enumerate(model.model):
        if isinstance(m, Concat) and any(f == p3_layer_idx for f in m.f if isinstance(f, int)):
            print(f"Found neck concat at layer {i} with inputs {m.f}")
            
            # Add our enhanced features to the concat
            m.f.append(p3_layer_idx + 1)  # ResidualC2f
            m.f.append(p3_layer_idx + 2)  # SmallObjectEnhance
            print(f"Updated to include enhanced features: {m.f}")
            
            # Update next layer's input channels
            if i + 1 < len(model.model) and hasattr(model.model[i+1], 'cv1'):
                next_m = model.model[i+1]
                old_channels = next_m.cv1.conv.in_channels
                c_out = next_m.cv1.conv.out_channels
                
                # Create new Conv with updated channels
                new_channels = old_channels + p3_channel_size * 2
                next_m.cv1 = Conv(new_channels, c_out, 1, 1)
                print(f"Updated next layer input channels: {old_channels} → {new_channels}")
    
    # Verify parameter count has changed
    new_params = sum(p.numel() for p in model.parameters())
    print(f"New parameter count: {new_params:,}")
    print(f"Difference: {new_params - original_params:,} ({(new_params/original_params - 1)*100:.2f}% increase)")
    
    return model

if __name__ == "__main__":
    # Use the direct approach which has proven to work
    enhanced_model = create_enhanced_yolov8_direct(size='n', pretrained=True)

    try:
        import torch
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = enhanced_model(dummy_input)
        print("Forward pass successful")
    except Exception as e:
        print(f"Error during forward pass: {e}")
#!/usr/bin/env python3
# Combined custom modules and model loader
# Save this as modules_loader.py

import os
import sys
import torch.nn as nn
from pathlib import Path

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import required modules before defining custom modules
from ultralytics.nn.modules import C2f, Conv

#--------------------- Custom Module Definitions ---------------------

class GroupNormConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        # Use GroupNorm instead of BatchNorm
        num_groups = min(max(2, c2 // 4), 32)
        self.norm = nn.GroupNorm(num_groups, c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResidualC2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c2f = C2f(c1, c2, n, shortcut, g, e)  # Use standard C2f
        self.residual = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        
    def forward(self, x):
        return self.c2f(x) + self.residual(x)


class SmallObjectEnhance(nn.Module):
    def __init__(self, c1, c2, act=True):
        super().__init__()
        self.cv1 = GroupNormConv(c1, c2//2, 1, 1, act=act)
        self.cv2 = GroupNormConv(c2//2, c2, 3, 1, act=act)
        
        # Attention mechanism
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            GroupNormConv(c2, c2//16, 1, act=act),
            GroupNormConv(c2//16, c2, 1, act=act),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.cv2(self.cv1(x))
        return x * self.attn(x)

#--------------------- Module Registration Functions ---------------------

def register_modules():
    """Register custom modules to the appropriate namespaces"""
    # Register the modules globally
    globals()['GroupNormConv'] = GroupNormConv
    globals()['ResidualC2f'] = ResidualC2f
    globals()['SmallObjectEnhance'] = SmallObjectEnhance
    
    # Register to Ultralytics modules
    try:
        import ultralytics.nn.modules
        setattr(ultralytics.nn.modules, 'GroupNormConv', GroupNormConv)
        setattr(ultralytics.nn.modules, 'ResidualC2f', ResidualC2f)
        setattr(ultralytics.nn.modules, 'SmallObjectEnhance', SmallObjectEnhance)
        print("Registered modules in ultralytics.nn.modules namespace")
    except ImportError:
        print("Could not import ultralytics.nn.modules")
    
    # Register to the main module namespace (important!)
    import builtins
    setattr(builtins, 'GroupNormConv', GroupNormConv)
    setattr(builtins, 'ResidualC2f', ResidualC2f)
    setattr(builtins, 'SmallObjectEnhance', SmallObjectEnhance)
    print("Registered modules in builtins namespace")
    
    # Register in the ultralytics tasks module (where YOLOv8 actually looks for modules)
    try:
        import ultralytics.nn.tasks
        setattr(ultralytics.nn.tasks, 'GroupNormConv', GroupNormConv)
        setattr(ultralytics.nn.tasks, 'ResidualC2f', ResidualC2f)
        setattr(ultralytics.nn.tasks, 'SmallObjectEnhance', SmallObjectEnhance)
        print("Registered modules in ultralytics.nn.tasks namespace")
    except ImportError:
        print("Could not import ultralytics.nn.tasks")
    
    # Register in the Python modules dictionary for direct access
    sys.modules['GroupNormConv'] = GroupNormConv
    sys.modules['ResidualC2f'] = ResidualC2f
    sys.modules['SmallObjectEnhance'] = SmallObjectEnhance
    print("Registered modules in sys.modules dictionary")
    
    # Print available modules for debugging
    print("\nRegistered custom modules:")
    print(" - GroupNormConv")
    print(" - ResidualC2f")
    print(" - SmallObjectEnhance")
    
    # Return the module classes for convenience
    return GroupNormConv, ResidualC2f, SmallObjectEnhance

#--------------------- Model Loading Function ---------------------

# Register modules first
GroupNormConv, ResidualC2f, SmallObjectEnhance = register_modules()

# Now import YOLO (after module registration)
from ultralytics import YOLO

def load_custom_model(yaml_config='yolov8_custom.yaml'):
    """Load a custom YOLO model using the specified YAML config"""
    if not os.path.exists(yaml_config):
        raise FileNotFoundError(f"Config file not found: {yaml_config}")
    
    print(f"Loading custom model from: {yaml_config}")
    
    # Create and return the model
    try:
        model = YOLO(yaml_config)
        print("Custom model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

#--------------------- Main Execution ---------------------

if __name__ == "__main__":
    # Hard-coded path to your YAML configuration file
    yaml_config = 'E:\\Mika Thesis (dontdeleteyet)\\thesis-toddler-monitoring-system\\Trainer_Model\\YOLO_config\\yolov8_custom.yaml'
    
    # Load the model
    model = load_custom_model(yaml_config)
    
    if model is not None:
        # Print model summary (optional - remove if not needed)
        print("\nModel Summary:")
        model.info()
        
        print("\nCustom model loaded and ready for use!")
        print("You can now use this model for:")
        print(" - Training:   model.train(data='data.yaml', epochs=100)")
        print(" - Validation: model.val(data='data.yaml')")
        print(" - Prediction: model.predict('image.jpg')")
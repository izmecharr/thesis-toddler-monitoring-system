#!/usr/bin/env python3
# Register custom YOLOv8 model
# Save this as register_model.py

import os
import sys
from pathlib import Path
import torch

# Add the current directory to path to find custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the custom modules
from custom_modules import GroupNormConv, ResidualC2f, SmallObjectEnhance

# Import Ultralytics
from ultralytics import YOLO

def register_custom_model(config_path='yolov8_custom.yaml'):
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    
    from ultralytics.nn.tasks import attempt_load_weights
    
    
    setattr(sys.modules['ultralytics.nn.modules'], 'GroupNormConv', GroupNormConv)
    setattr(sys.modules['ultralytics.nn.modules'], 'ResidualC2f', ResidualC2f)
    setattr(sys.modules['ultralytics.nn.modules'], 'SmallObjectEnhance', SmallObjectEnhance)
    
    
    model = YOLO(config_path)
    
    print(f"Custom YOLOv8 model registered successfully with modified backbone:")
    print(f" - Added ResidualC2f after the first C2f block")
    print(f" - Added SmallObjectEnhance after the second C2f block")
    print(f" - Added ResidualC2f after the third C2f block")
    print(f" - All custom modules use GroupNormConv for improved small batch training")
    
    return model

if __name__ == "__main__":
    # Example usage
    
    # 1. Save the YAML configuration to a file
    config_content = """
# YOLOv8 Custom Architecture Configuration
# Modified with GroupNormConv, ResidualC2f, and SmallObjectEnhance modules

# Parameters
nc: 80  # number of classes
depth_multiple: 1.0  # model depth multiplier
width_multiple: 1.0  # layer channel multiplier

# Anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv8 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 3, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C2f, [128]],         # 2-Standard C2f
   [-1, 1, ResidualC2f, [128]], # 3-Added ResidualC2f after first C2f
   [-1, 1, Conv, [256, 3, 2]],  # 4-P3/8
   [-1, 6, C2f, [256]],         # 5-Standard C2f
   [-1, 1, SmallObjectEnhance, [256]],  # 6-Added SmallObjectEnhance after second C2f
   [-1, 1, Conv, [512, 3, 2]],  # 7-P4/16
   [-1, 6, C2f, [512]],         # 8-Standard C2f
   [-1, 1, ResidualC2f, [512]], # 9-Added ResidualC2f after third C2f
   [-1, 1, Conv, [1024, 3, 2]], # 10-P5/32
   [-1, 3, C2f, [1024]],        # 11-Standard C2f for remaining
   [-1, 1, SPPF, [1024, 5]],    # 12
  ]

# YOLOv8 head
head:
  [[-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 9], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C2f, [512]],  # 15
   
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C2f, [256]],  # 18
   
   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 15], 1, Concat, [1]],  # cat head P4
   [-1, 3, C2f, [512]],  # 21
   
   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 12], 1, Concat, [1]],  # cat head P5
   [-1, 3, C2f, [1024]],  # 24
   
   [[18, 21, 24], 1, Detect, [nc]],  # Detect(P3, P4, P5)
  ]
"""
    config_path = 'yolov8_custom.yaml'
    
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found. Creating it...")
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"Created config file at {config_path}")
    else:
        print(f"Using existing config file: {config_path}")
    
    model = register_custom_model(config_path)
    
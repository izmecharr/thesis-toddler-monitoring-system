#register_model.py
import os
import sys
from pathlib import Path

# Add the current directory to path to find custom modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import custom modules FIRST, before importing YOLO
from custom_modules import GroupNormConv, ResidualC2f, SmallObjectEnhance

# Now import Ultralytics
from ultralytics import YOLO

def register_custom_model(config_path='yolov8_custom.yaml'):
    """
    Register the custom model with Ultralytics
    
    Args:
        config_path: Path to the custom YAML config file
    
    Returns:
        YOLO model with custom architecture
    """
    # Make sure the YAML config exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # THIS IS CRITICAL - Register the custom modules with Ultralytics BEFORE creating the model
    # We need to make our custom modules available in the modules where YOLO will look for them
    
    # First try the nn.modules namespace (this is the most common location)
    try:
        import ultralytics.nn.modules
        setattr(ultralytics.nn.modules, 'GroupNormConv', GroupNormConv)
        setattr(ultralytics.nn.modules, 'ResidualC2f', ResidualC2f)
        setattr(ultralytics.nn.modules, 'SmallObjectEnhance', SmallObjectEnhance)
        print("Registered custom modules in ultralytics.nn.modules")
    except ImportError:
        print("Could not import ultralytics.nn.modules")
    
    # Also try registering in the global namespace as a backup
    globals()['GroupNormConv'] = GroupNormConv
    globals()['ResidualC2f'] = ResidualC2f
    globals()['SmallObjectEnhance'] = SmallObjectEnhance
    
    # Also make them available in common module locations used by YOLO
    sys.modules['models.common.GroupNormConv'] = GroupNormConv
    sys.modules['models.common.ResidualC2f'] = ResidualC2f
    sys.modules['models.common.SmallObjectEnhance'] = SmallObjectEnhance
    
    # Now try to create the model
    try:
        model = YOLO(config_path)
        print(f"Custom YOLOv8 model registered successfully with modified backbone:")
        print(f" - Added ResidualC2f after the first C2f block")
        print(f" - Added SmallObjectEnhance after the second C2f block")
        print(f" - Added ResidualC2f after the third C2f block")
        return model
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        # Debug information to help identify where modules are expected
        print("\nDebugging module paths:")
        print(f"Current directory: {os.getcwd()}")
        print(f"Python path: {sys.path}")
        raise


if __name__ == "__main__":
    # Check if the config file exists
    config_path = 'E:\\Mika Thesis (dontdeleteyet)\\thesis-toddler-monitoring-system\\Trainer_Model\\YOLO_config\\yolov8_custom.yaml'
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found.")
        print("Please provide the path to your YAML configuration file.")
        sys.exit(1)
        
    # Register the custom model
    try:
        model = register_custom_model(config_path)
        print("Model created successfully!")
    except Exception as e:
        print(f"Failed to create model: {str(e)}")
        import traceback
        traceback.print_exc()
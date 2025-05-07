#train_custom_model.py
import os
import sys
import torch
from pathlib import Path

# Add the current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the register_custom_model function from your existing script
from register_model import register_custom_model

def get_next_dir_number(base_dir):
    """Get the next run number for organizing output directories."""
    import os
    import re
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return 1
        
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    numbers = []
    for d in dirs:
        match = re.search(r'enhanced_n_(\d+)', d)
        if match:
            numbers.append(int(match.group(1)))
    
    if not numbers:
        return 1
        
    return max(numbers) + 1

def train_custom_model():
    """Register and train the custom model using hard-coded parameters"""
    
    # HARD-CODED CONFIGURATION - MODIFY THESE VALUES AS NEEDED
    yaml_config = 'E:\\Mika Thesis (dontdeleteyet)\\thesis-toddler-monitoring-system\\Trainer_Model\\YOLO_config\\yolov8_custom.yaml'
    data_config = 'E:\\Mika Thesis (dontdeleteyet)\\thesis-toddler-monitoring-system\\Thesis_Assets\\baby\\data.yaml'
    epochs = 20                        # Number of training epochs
    batch = 8                          # Batch size
    imgsz = 640                         # Image size
    device = ''                         # Device to use ('' for auto-select, '0' for first GPU)
    workers = 2                         # Number of worker threads
    
    # Get next run number for better organization
    project_dir = "enhanced_yolov8"  # Change to enhanced_yolov8 folder
    next_run_number = get_next_dir_number(project_dir)
    name = f"enhanced_n_{next_run_number}"  # Using the enhanced_n_X format
    
    # Create project directory structure
    os.makedirs(project_dir, exist_ok=True)
    
    # Register the custom model - this also returns the model
    print(f"Registering custom model from {yaml_config}...")
    model = register_custom_model(yaml_config)
    
    if model is None:
        print("Error: Failed to register and create model")
        return None
    
    # Check if data config exists
    if not os.path.exists(data_config):
        print(f"Error: Data config file {data_config} not found!")
        return None
    
    # Start training
    print(f"\nStarting training with the following parameters:")
    print(f" - Model config: {yaml_config}")
    print(f" - Data config: {data_config}")
    print(f" - Epochs: {epochs}")
    print(f" - Batch size: {batch}")
    print(f" - Image size: {imgsz}")
    print(f" - Device: {device}")
    print(f" - Workers: {workers}")
    print(f" - Run name: {name}")
    print(f" - Project directory: {project_dir}")
    
    # Configure advanced training arguments
    training_args = {
        "data": data_config,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "device": device,
        "workers": workers,
        "name": name,
        "project": project_dir,
        "patience": 50,
        "save": True,
        "cache": "disk",
        "amp": True,          # Mixed precision training
        "plots": True,        # Generate plots
        "verbose": True,
        "exist_ok": True,
        "lr0": 0.00005,       # Initial learning rate
        "lrf": 0.005,       # Final learning rate
        "momentum": 0.937,
        "weight_decay": 0.005,
        # "optimizer": "AdamW", # Modern optimizer
        "warmup_epochs": 5,   # Warmup period
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "dropout": 0.9,
        "max_det": 300,
        "iou": 0.6,
        "rect": False,        # Rectangular training
        "single_cls": False,  # Train as multi-class
    }
    
    # Train the model
    try:
        results = model.train(**training_args)
        
        # Get the best model path
        best_model_path = model.best if hasattr(model, "best") else None
        
        if not best_model_path or not os.path.exists(best_model_path):
            # Try to find it in the expected location
            weights_dir = Path(f"{project_dir}/{name}/weights")
            best_model_path = str(weights_dir / 'best.pt')
            
            if not os.path.exists(best_model_path):
                last_model_path = str(weights_dir / 'last.pt')
                if os.path.exists(last_model_path):
                    best_model_path = last_model_path
                    print(f"Best model not found. Using last model: {last_model_path}")
                else:
                    print("No trained model found.")
                    return None
        
        print(f"\nTraining completed successfully!")
        print(f"Results saved to: {os.path.join(project_dir, name)}")
        print(f"Best model path: {best_model_path}")
        
        # Save a copy of the best model in the models directory
        os.makedirs('models', exist_ok=True)
        final_model_path = f"models/enhanced_n_{next_run_number}_best.pt"
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy2(best_model_path, final_model_path)
            print(f"Copied best model to: {final_model_path}")
        
        return model
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the training function
    print("Starting custom YOLOv8 training...")
    model = train_custom_model()
    
    # Optionally validate after training
    if model is not None:
        print("\nRunning validation...")
        model.val()
        print("Validation completed!")
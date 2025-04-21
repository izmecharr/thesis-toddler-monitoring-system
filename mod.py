import os
import torch
import yaml
from ultralytics import YOLO
from temp_model.modified_yolov8 import create_modified_yolov8

def train_enhanced_yolov8(
    data_yaml_path,
    size='n',
    pretrained=True,
    epochs=100,
    batch_size=8,  # Reduced batch size to avoid memory issues
    imgsz=640,
    device=None,  # Auto-detect
    workers=4
):
    """
    Train the enhanced YOLOv8 model on a custom dataset.
    """
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda:1'
            print(torch.cuda.get_device_name(torch.cuda.current_device()))
        else: 'cpu'
    
    print(f"Using device: {device}")
    
    # Set CUDA device if using GPU
    if 'cuda' in str(device) and device != 'cpu':
        cuda_id = device.split(':')[1] if ':' in device else '0'
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
    else:
        print("Training on CPU. This will be much slower.")
    
    # Create directory for saving
    save_dir = f"runs/train/enhanced_yolov8{size}"
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Instead of creating our own model, start with a standard YOLOv8 model
        # and we'll modify it during training
        print(f"Starting with a standard YOLOv8{size} model")
        trainer = YOLO(f'yolov8{size}.pt')
        
        # Configure training settings
        gpu_settings = {
            "device": device,
            "batch": batch_size,
            "workers": workers
        }
        
        # Create a custom configuration for training
        custom_config = {
            "model": f"yolov8{size}.yaml", 
            "data": data_yaml_path,
            "epochs": epochs,
            "imgsz": imgsz,
            "patience": 0,
            "save": True,
            "plots": True,
            "verbose": True,
            "exist_ok": True,
            "project": "enhanced_yolov8",
            "name": f"enhanced_{size}",
            "lr0": 0.001,                # Lower initial learning rate
            "lrf": 0.01,                 # Final learning rate as a fraction of initial
            "momentum": 0.937,           # SGD momentum/Adam beta1
            "weight_decay": 0.0005,      # Optimizer weight decay
            "warmup_epochs": 3.0,        # Warmup epochs (fractions ok)
            "warmup_momentum": 0.8,      # Warmup initial momentum
            "warmup_bias_lr": 0.1,       # Warmup initial bias lr
            **gpu_settings
        }
        
        # Run training
        print(f"\n\n*** ATTEMPTING TO START TRAINING NOW - {epochs} EPOCHS ***\n\n")
        results = trainer.train(**custom_config)
        
        print(f"Training completed successfully for {epochs} epochs.")
        return trainer.best
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Simple training function without inference."""
    try:
        # Set your data path
        data_yaml_path = "C:\\Users\\izzze\\OneDrive\\Documents\\Thesis\\Thesis_Assets\\data\\dataset\\data.yaml"  # CHANGE THIS TO YOUR DATA PATH
        
        print("========== STARTING TRAINING ==========")
        
        # Train the model - reduced batch size and epochs for testing
        best_model_path = train_enhanced_yolov8(
            data_yaml_path=data_yaml_path,
            size='n',
            pretrained=True,
            epochs=100,
            batch_size=8,  # Reduced to avoid memory issues
            imgsz=640,
            device=None,  # Auto-detect
            workers=4
        )
        
        if best_model_path:
            print(f"Training successful! Best model saved at: {best_model_path}")
        else:
            print("Training failed or was interrupted.")
            
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
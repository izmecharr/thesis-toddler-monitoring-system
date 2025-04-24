import os
import torch
import yaml
import multiprocessing
from pathlib import Path
from ultralytics import YOLO
from temp_model.modified_yolov8 import create_modified_yolov8, save_model_with_yaml, load_model_with_yaml

def select_device(device_name=None):
    # Auto-detect device if not specified
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU")
        return 'cpu'
    
    print(f"CUDA available: True | CUDA device count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        current_device_name = torch.cuda.get_device_name(i)
        print(f"CUDA device {i}: {current_device_name}")
        if current_device_name == device_name:
            print(f"Selected {device_name} (CUDA:{i})")
            # Set environment variable to make only this GPU visible
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{i}"
            
            # Since we're making only this GPU visible, it becomes device 0
            print(f"Set CUDA_VISIBLE_DEVICES={i}, GPU is now accessible as device 0")
            return "0"
        
    if device_name == None:
        print("No GPU name provided. Returning CPU")
        return 'cpu'

def train_enhanced_yolov8(
    data_yaml_path,
    size='n',
    pretrained=True,
    epochs=100,
    batch_size=16,
    imgsz=640,
    device=None,
    amp=False,
    workers=4,
    run_name=None
):
    """
    Train an enhanced YOLOv8 model with deeper feature extraction.
    
    Args:
        data_yaml_path: Path to the data YAML file
        size: Model size (n, s, m, l, x)
        pretrained: Whether to load pretrained weights
        epochs: Number of training epochs
        batch_size: Batch size for training
        imgsz: Input image size
        device: Device to use for training (None for auto-detection)
        amp: Whether to use Automatic Mixed Precision
        workers: Number of worker threads for data loading
        
    Returns:
        Path to the best trained models
    """

    if device is None:
        device = select_device('NVIDIA GeForce GTX 1660 Ti with Max-Q Design')
    
    print(f"Using device: {device}")
    
    # Create the enhanced model
    enhanced_model = create_modified_yolov8(size=size, pretrained=pretrained)
    
    # Count parameters to confirm it's the enhanced version
    param_count = sum(p.numel() for p in enhanced_model.parameters())
    print(f"Enhanced model created with {param_count:,} parameters")
    
    # Ensure the YAML attribute is set
    yaml_path = getattr(enhanced_model, 'yaml', f'yolov8{size}.yaml')
    setattr(enhanced_model, 'yaml', yaml_path)
    print(f"Model YAML path: {enhanced_model.yaml}")
    
    # Save model with YAML attribute preserved
    os.makedirs('models', exist_ok=True)
    model_path = f"models/enhanced_yolov8{size}_init.pt"
    save_model_with_yaml(enhanced_model, model_path)
    print(f"Saved initial model to {model_path}")
    
    # Create YOLO model for training
    yolo_model = YOLO(model_path)
    
    # Verify the model still has the enhanced architecture
    yolo_param_count = sum(p.numel() for p in yolo_model.model.parameters())
    print(f"YOLO model loaded with {yolo_param_count:,} parameters")
    print(f"YAML path: {yolo_model.model.yaml if hasattr(yolo_model.model, 'yaml') else 'Not found'}")
    
        # Configure training settings
    training_args = {
        "data": data_yaml_path,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch_size,
        "device": device,
        "workers": workers,
        "patience": 50,
        "save": True,
        "cache": True,
        "amp": amp,
        "plots": True,
        "verbose": True,
        "exist_ok": True,
        "project": "enhanced_yolov8",
        "name": run_name if run_name else f"enhanced_{size}"  # Use provided run_name if available
    }
    
    # Start training
    print(f"\n*** Starting training with enhanced YOLOv8{size} model ***")
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    
    try:
        results = yolo_model.train(**training_args)
        
        # Get path to best model
        best_model_path = yolo_model.best if hasattr(yolo_model, 'best') else None
        
        # If best_model_path is not found, look for it manually
        if not best_model_path or not os.path.exists(best_model_path):
            output_dir = f"enhanced_yolov8/enhanced_{size}"
            best_model_path = os.path.join(output_dir, "weights", "best.pt")
            last_model_path = os.path.join(output_dir, "weights", "last.pt")
            
            if os.path.exists(best_model_path):
                print(f"Best model found at: {best_model_path}")
            elif os.path.exists(last_model_path):
                print(f"Best model not found. Using last model instead: {last_model_path}")
                best_model_path = last_model_path
            else:
                print("No model files found after training.")
                return None
        
        print(f"Training completed successfully. Best model: {best_model_path}")
        return best_model_path
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None

def continue_training(
    model_path,         # Path to your best.pt or last.pt from previous training
    data_yaml_path,
    epochs=100,
    batch_size=16,
    imgsz=640,
    device=None,
    amp=False,
    workers=4
):
    """Continue training from a previously trained model checkpoint while preserving the enhanced architecture."""
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")

    # Auto-detect device
    if device is None:
        device = select_device('NVIDIA GeForce GTX 1660 Ti with Max-Q Design')
    
    print(f"Using device: {device}")
    
    # Verify the model path exists
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist.")
        return None
    
    # Create a fresh instance of the enhanced model
    from temp_model.modified_yolov8 import create_modified_yolov8
    enhanced_model = create_modified_yolov8(size='n', pretrained=False)
    
    # Count parameters to confirm it's the enhanced version
    param_count = sum(p.numel() for p in enhanced_model.parameters())
    print(f"Enhanced model created with {param_count:,} parameters")
    
    # Load weights from checkpoint and ensure YAML attribute
    print(f"Loading weights from checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract state_dict depending on checkpoint format
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            if hasattr(checkpoint['model'], 'state_dict'):
                state_dict = checkpoint['model'].state_dict()
            else:
                state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
    
    # Load weights where shapes match
    enhanced_model_dict = enhanced_model.state_dict()
    matched_weights = {}
    for name, param in state_dict.items():
        if name in enhanced_model_dict and enhanced_model_dict[name].shape == param.shape:
            matched_weights[name] = param
    
    print(f"Loaded {len(matched_weights)}/{len(enhanced_model_dict)} weights from checkpoint")
    enhanced_model.load_state_dict(matched_weights, strict=False)
    
    # Ensure YAML attribute is set
    yaml_path = getattr(enhanced_model, 'yaml', 'yolov8n.yaml')
    setattr(enhanced_model, 'yaml', yaml_path)
    print(f"Model YAML path: {enhanced_model.yaml}")
    
    # Save with YAML attribute preserved
    os.makedirs('models', exist_ok=True)
    temp_path = f"models/enhanced_continue.pt"
    save_model_with_yaml(enhanced_model, temp_path)
    
    # Create YOLO model with our architecture
    yolo_model = YOLO(temp_path)
    
    # Verify the model still has the enhanced architecture
    yolo_param_count = sum(p.numel() for p in yolo_model.model.parameters())
    print(f"YOLO model loaded with {yolo_param_count:,} parameters")
    print(f"YAML path: {yolo_model.model.yaml if hasattr(yolo_model.model, 'yaml') else 'Not found'}")
    
    # Check if we've preserved the enhanced architecture
    expected_params = 3157200  # The specific number for your enhanced model
    if abs(yolo_param_count - expected_params) > 1000:  # Allow small difference due to counting methods
        print(f"WARNING: Parameter count doesn't match expected enhanced model ({expected_params:,})!")
        print(f"Current count: {yolo_param_count:,}")
        proceed = input("Continue anyway? (y/n): ").lower() == 'y'
        if not proceed:
            return None
    
    # Configure training settings
    training_args = {
        "data": data_yaml_path,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch_size,
        "device": device,
        "workers": workers,
        "patience": 50,
        "save": True,
        "cache": True,
        "amp": False,
        "plots": True,
        "verbose": True,
        "exist_ok": True,
        "project": "enhanced_yolov8_continued",
        "name": "continued_training"
    }
    
    # Start training
    print(f"\n*** Continuing training for {epochs} epochs with enhanced architecture ***")
    try:
        results = yolo_model.train(**training_args)
        
        # Get path to best model
        best_model_path = yolo_model.best if hasattr(yolo_model, 'best') else None
        
        # If best_model_path is not found, look for it manually
        if not best_model_path or not os.path.exists(best_model_path):
            output_dir = "enhanced_yolov8_continued/continued_training"
            best_model_path = os.path.join(output_dir, "weights", "best.pt")
            last_model_path = os.path.join(output_dir, "weights", "last.pt")
            
            if os.path.exists(best_model_path):
                print(f"Best model found at: {best_model_path}")
            elif os.path.exists(last_model_path):
                print(f"Best model not found. Using last model instead: {last_model_path}")
                best_model_path = last_model_path
            else:
                print("No model files found after continued training.")
                return None
        
        print(f"Continued training completed successfully. Best model: {best_model_path}")
        return best_model_path
    except Exception as e:
        print(f"Error during continued training: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Replace with your actual data.yaml path
    data_yaml_path = "C:\\Users\\izzze\\OneDrive\\Documents\\GitHub\\thesis-toddler-monitoring-system\\Thesis_Assets\\data\\baby\\data.yaml"
    
    # Function to find the next available directory number
    def get_next_dir_number(base_dir):
        import os
        import re
        
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            return 1
            
        # Get all subdirectories in the base folder
        dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        
        # Extract numbers from directory names
        numbers = []
        for d in dirs:
            match = re.search(r'enhanced_\w+_(\d+)', d)
            if match:
                numbers.append(int(match.group(1)))
        
        # If no matching directories exist, start with 1
        if not numbers:
            return 1
            
        # Return the next number in sequence
        return max(numbers) + 1
    
    # Ask user if they want to start fresh or continue training
    train_option = input("Do you want to (1) start training from scratch or (2) continue from a checkpoint? (1/2): ")
    
    try:
        if train_option == "1":
            # Get next train directory number
            next_number = get_next_dir_number("enhanced_yolov8")
            run_name = f"enhanced_n_{next_number}"
            
            # Train a new enhanced model
            print(f"\n=== Starting new training (run: {run_name}) ===\n")
            best_model_path = train_enhanced_yolov8(
                data_yaml_path=data_yaml_path,
                size='n',
                pretrained=True,
                epochs=100,
                batch_size=16,
                imgsz=640,
                device=None,  # Auto-detect
                amp=False,    # Disable AMP to avoid NaN losses
                workers=4,
                run_name=run_name  # Pass the incremented run name
            )
        elif train_option == "2":
            # Get next continue directory number
            next_number = get_next_dir_number("enhanced_yolov8_continued")
            run_name = f"enhanced_n_{next_number}"
            
            # Continue training from a checkpoint
            checkpoint_path = input("Enter the path to the checkpoint (best.pt or last.pt): ")
            print(f"\n=== Continuing training from {checkpoint_path} (run: {run_name}) ===\n")
            best_model_path = continue_training(
                model_path=checkpoint_path,
                data_yaml_path=data_yaml_path,
                epochs=100,
                batch_size=16,
                imgsz=640,
                device=None,  # Auto-detect
                amp=False,    # Disable AMP to avoid NaN losses
                workers=4,
                run_name=run_name  # Pass the incremented run name
            )
        else:
            print("Invalid option. Please enter 1 or 2.")
            return
        
        if best_model_path:
            print(f"\nTraining complete! Best model saved at: {best_model_path}")
            
            # Run validation on the trained model
            print("\nRunning validation on the trained model...")
            model = YOLO(best_model_path)
            model.val(data=data_yaml_path)
        else:
            print("Training failed or was interrupted.")
            
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Add freeze support for Windows multiprocessing
    multiprocessing.freeze_support()
    main()
import os
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO
from modified_yolov8 import create_enhanced_yolov8, save_model_with_yaml, load_model_with_yaml

def select_device(device_name=None):
    """Auto-detect device if not specified."""
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU")
        return 'cpu'
    
    print(f"CUDA available: True | CUDA device count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        current_device_name = torch.cuda.get_device_name(i)
        print(f"CUDA device {i}: {current_device_name}")
        if device_name and current_device_name == device_name:
            print(f"Selected {device_name} (CUDA:{i})")
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{i}"
            print(f"Set CUDA_VISIBLE_DEVICES={i}, GPU is now accessible as device 0")
            return "0"
    
    # If no specified device found or no device name provided, use the first GPU
    if torch.cuda.device_count() > 0:
        print(f"No specific GPU selected. Using first available GPU: {torch.cuda.get_device_name(0)}")
        return "0"
        
    print("No GPU found. Using CPU")
    return 'cpu'

def prepare_for_training(model, data_yaml_path):
    """
    Convert the enhanced model to YOLO format for training.
    
    In order to train with YOLO's training pipeline, we need to:
    1. Save our enhanced model
    2. Create a standard YOLO model with the same weights
    3. Use YOLO's training methods
    """
    print("Converting enhanced model to YOLO format for training...")
    
    # Save our enhanced model to a temporary file
    os.makedirs('models', exist_ok=True)
    model_path = f"models/enhanced_yolov8_temp.pt"
    save_model_with_yaml(model, model_path)
    
    # The simplest approach: create a new YOLO model with the same size and copy weights
    if hasattr(model, 'base_model'):
        # Determine the model size
        size = 'n'  # Default size
        for s in ['n', 's', 'm', 'l', 'x']:
            if f'yolov8{s}' in str(model.__class__).lower():
                size = s
                break
        
        # Create a fresh YOLO model
        yolo_model = YOLO(f'yolov8{size}.pt')
        
        # Copy weights from our base model to the YOLO model
        with torch.no_grad():
            # Get state dict from our base model
            base_model_state = model.base_model.state_dict()
            
            # Get state dict from YOLO model
            yolo_state = yolo_model.model.state_dict()
            
            # Copy matching parameters
            for name, param in base_model_state.items():
                if name in yolo_state and yolo_state[name].shape == param.shape:
                    yolo_state[name].copy_(param)
            
            # Load the updated state dict back to YOLO model
            yolo_model.model.load_state_dict(yolo_state)
        
        print(f"Created YOLO model with copied weights from enhanced model")
    else:
        # If we can't extract the base model, try loading the entire model directly
        try:
            yolo_model = YOLO(model_path)
        except Exception as e:
            print(f"Error loading enhanced model directly: {e}")
            print("Creating default YOLO model as fallback")
            yolo_model = YOLO('yolov8n.pt')
    
    # Save the YOLO model temporarily so we have a clean checkpoint to start from
    yolo_temp_path = 'models/yolo_temp.pt'
    yolo_model.save(yolo_temp_path)
    
    # Reload to ensure everything is clean
    yolo_model = YOLO(yolo_temp_path)
    
    print("Model prepared for YOLO training")
    return yolo_model, model_path

def train_enhanced_yolov8(
    data_yaml_path,
    size='n',
    pretrained=True,
    epochs=100,
    batch_size=16,
    imgsz=640,
    device=None,
    amp=False,
    workers=8,
    run_name=None,
    learning_rate=0.00005,  # Reduced learning rate for deeper model
    warmup_epochs=5,        # Warmup period for better convergence
    weight_decay=0.0005     # Weight decay for regularization
):
    """
    Train an enhanced YOLOv8 model with deeper feature extraction and improved settings.
    """
    if device is None:
        device = select_device()
    
    print(f"Using device: {device}")
    
    # Create the enhanced model
    enhanced_model = create_enhanced_yolov8(size=size, pretrained=pretrained)
    
    # Count parameters to confirm it's the enhanced version
    param_count = sum(p.numel() for p in enhanced_model.parameters())
    print(f"Enhanced model created with {param_count:,} parameters")
    
    # Save our fully enhanced model for later use
    os.makedirs('models', exist_ok=True)
    enhanced_model_path = f"models/enhanced_yolov8{size}_init.pt"
    save_model_with_yaml(enhanced_model, enhanced_model_path)
    print(f"Saved initial enhanced model to {enhanced_model_path}")
    
    # Prepare for training with YOLO framework (extracts base model)
    yolo_model, base_model_path = prepare_for_training(enhanced_model, data_yaml_path)
    
    # Configure improved training settings
    training_args = {
        "data": data_yaml_path,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch_size,
        "device": device,
        "workers": workers,
        "patience": 50,
        "save": True,
        "cache": "disk",
        "amp": amp,
        "plots": True,
        "verbose": True,
        "exist_ok": True,
        "project": "enhanced_yolov8",
        "name": run_name if run_name else f"enhanced_{size}",
        
        # Improved hyperparameters
        "lr0": learning_rate,          # Initial learning rate
        "lrf": 0.01,                   # Final learning rate (lr0 * lrf)
        "momentum": 0.937,             # SGD momentum
        "weight_decay": weight_decay,  # Weight decay
        "warmup_epochs": warmup_epochs,# Warmup epochs
        "warmup_momentum": 0.8,        # Warmup initial momentum
        "warmup_bias_lr": 0.1,         # Warmup initial bias lr
        "box": 7.5,                    # Box loss gain
        "cls": 0.5,                    # Class loss gain
        "dfl": 1.5,                    # DFL loss gain
        "mosaic": 1.0,                 # Image mosaic
        "mixup": 0.1,                  # Image mixup
        "copy_paste": 0.3,             # Copy-paste augmentation
        "degrees": 10.0,               # Image rotation (+/- deg)
        "translate": 0.1,              # Image translation (+/- fraction)
        "scale": 0.9,                  # Image scale (+/- gain)
        "shear": 2.0,                  # Image shear (+/- deg)
        "perspective": 0.0001,         # Image perspective
        "flipud": 0.5,                 # Image flip up-down
        "fliplr": 0.5,                 # Image flip left-right
        "hsv_h": 0.015,                # Image HSV-Hue augmentation
        "hsv_s": 0.7,                  # Image HSV-Saturation augmentation
        "hsv_v": 0.4,                  # Image HSV-Value augmentation
        "label_smoothing": 0.1,        # Label smoothing epsilon
        "dropout": 0.05                # Dropout rate for regularization
    }
    
    # Small object specific augmentations (using only valid parameters)
    training_args.update({
        "max_det": 300,                # Increase maximum detections
        "iou": 0.6,                    # IoU threshold for evaluation
        "kobj": 1.0                    # Object loss gain (replaces obj_pw)
    })
    
    # Start training
    print(f"\n*** Starting training with enhanced YOLOv8{size} model ***")
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    
    try:
        results = yolo_model.train(**training_args)
        
        # Get path to best model
        best_model_path = yolo_model.best if hasattr(yolo_model, 'best') else None
        
        if not best_model_path or not os.path.exists(best_model_path):
            output_dir = f"enhanced_yolov8/{run_name if run_name else f'enhanced_{size}'}"
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
        
        # Convert the trained base model back to our enhanced format
        try:
            print("Converting trained model back to enhanced format...")
            
            # Create a new enhanced model with the same architecture
            new_enhanced = create_enhanced_yolov8(size=size, pretrained=False)
            
            # Load the trained weights into the base model
            trained_model = YOLO(best_model_path)
            
            # Copy weights from the trained YOLO model to our enhanced model's base
            with torch.no_grad():
                # Get state dict from the trained model
                trained_state = trained_model.model.state_dict()
                
                # Get state dict from our base model
                base_state = new_enhanced.base_model.state_dict()
                
                # Copy matching parameters
                for name, param in trained_state.items():
                    if name in base_state and base_state[name].shape == param.shape:
                        base_state[name].copy_(param)
                
                # Load the updated state dict back to our base model
                new_enhanced.base_model.load_state_dict(base_state)
            
            # Save the final enhanced model
            final_path = f"models/enhanced_yolov8{size}_final.pt"
            save_model_with_yaml(new_enhanced, final_path)
            print(f"Saved final enhanced model to: {final_path}")
            
            return final_path
        except Exception as e:
            print(f"Warning: Error converting trained model to enhanced format: {e}")
            print(f"Returning standard trained model path: {best_model_path}")
            return best_model_path
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None

def continue_training(
    model_path,
    data_yaml_path,
    epochs=100,
    batch_size=16,
    imgsz=640,
    device=None,
    amp=False,
    workers=8,
    run_name=None,
    learning_rate=0.0001,  # Lower learning rate for fine-tuning
    warmup_epochs=3
):
    """Continue training from a previously trained checkpoint with optimized parameters."""
    if device is None:
        device = select_device()
    
    print(f"Using device: {device}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist.")
        return None
    
    # Try to load as an enhanced model
    try:
        print(f"Loading model from {model_path} as enhanced model...")
        enhanced_model = load_model_with_yaml(model_path)
        print("Successfully loaded as enhanced model")
    except Exception as e:
        print(f"Error loading as enhanced model: {e}")
        print("Attempting to load as standard YOLO model...")
        
        try:
            # Create a new enhanced model
            size = 'n'  # Default
            if 'yolov8s' in model_path.lower():
                size = 's'
            elif 'yolov8m' in model_path.lower():
                size = 'm'
            elif 'yolov8l' in model_path.lower():
                size = 'l'
            elif 'yolov8x' in model_path.lower():
                size = 'x'
            
            enhanced_model = create_enhanced_yolov8(size=size, pretrained=False)
            
            # Load the standard YOLO model
            yolo_model = YOLO(model_path)
            
            # Copy weights from the YOLO model to our enhanced model's base
            with torch.no_grad():
                # Get state dict from the YOLO model
                yolo_state = yolo_model.model.state_dict()
                
                # Get state dict from our base model
                base_state = enhanced_model.base_model.state_dict()
                
                # Copy matching parameters
                for name, param in yolo_state.items():
                    if name in base_state and base_state[name].shape == param.shape:
                        base_state[name].copy_(param)
                
                # Load the updated state dict back to our base model
                enhanced_model.base_model.load_state_dict(base_state)
            
            print(f"Created enhanced model from standard YOLO model")
        except Exception as e2:
            print(f"Error creating enhanced model: {e2}")
            print("Using standard YOLO model directly")
            return None
    
    # Prepare for training with YOLO framework
    yolo_model, base_model_path = prepare_for_training(enhanced_model, data_yaml_path)
    
    # Configure training settings for fine-tuning
    training_args = {
        "data": data_yaml_path,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch_size,
        "device": device,
        "workers": workers,
        "patience": 30,
        "save": True,
        "cache": "disk",
        "amp": amp,
        "plots": True,
        "verbose": True,
        "exist_ok": True,
        "project": "enhanced_yolov8_continued",
        "name": run_name if run_name else f"continued_training",
        
        # Fine-tuning hyperparameters
        "lr0": learning_rate,
        "lrf": 0.1,
        "warmup_epochs": warmup_epochs,
        "weight_decay": 0.0005,
        "box": 5.0,                    # Lower box loss gain for fine-tuning
        "cls": 0.5,
        "dfl": 1.0,
        "mosaic": 0.5,                 # Reduce augmentation for fine-tuning
        "mixup": 0.05,
        "copy_paste": 0.1,
        "label_smoothing": 0.05,
        "dropout": 0.05,               # Lower dropout for fine-tuning
        "scale": 0.8
    }
    
    # Small object specific fine-tuning
    training_args.update({
        "max_det": 300,                # Increase maximum detections
        "nms_time_threshold": 50,      # Time threshold for NMS
        "iou": 0.6,                    # IoU threshold for evaluation
        "cls_pw": 1.0,                 # cls BCELoss positive_weight
        "obj_pw": 1.0,                 # obj BCELoss positive_weight
        "iou_t": 0.6                   # IoU training threshold
    })
    
    # Start training
    print(f"\n*** Continuing training for {epochs} epochs with enhanced architecture ***")
    try:
        results = yolo_model.train(**training_args)
        
        # Get path to best model
        best_model_path = yolo_model.best if hasattr(yolo_model, 'best') else None
        
        if not best_model_path or not os.path.exists(best_model_path):
            output_dir = f"enhanced_yolov8_continued/{run_name if run_name else 'continued_training'}"
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
        
        print(f"Continued training completed successfully. Best model: {best_model_path}")
        
        # Convert the trained base model back to our enhanced format
        try:
            print("Converting trained model back to enhanced format...")
            
            # Create a new enhanced model with the same architecture
            new_enhanced = create_enhanced_yolov8(size=size, pretrained=False)
            
            # Load the trained weights into the base model
            trained_model = YOLO(best_model_path)
            
            # Copy weights from the trained YOLO model to our enhanced model's base
            with torch.no_grad():
                # Get state dict from the trained model
                trained_state = trained_model.model.state_dict()
                
                # Get state dict from our base model
                base_state = new_enhanced.base_model.state_dict()
                
                # Copy matching parameters
                for name, param in trained_state.items():
                    if name in base_state and base_state[name].shape == param.shape:
                        base_state[name].copy_(param)
                
                # Load the updated state dict back to our base model
                new_enhanced.base_model.load_state_dict(base_state)
            
            # Save the final enhanced model
            final_path = f"models/enhanced_yolov8_continued_final.pt"
            save_model_with_yaml(new_enhanced, final_path)
            print(f"Saved final enhanced model to: {final_path}")
            
            return final_path
        except Exception as e:
            print(f"Warning: Error converting trained model to enhanced format: {e}")
            print(f"Returning standard trained model path: {best_model_path}")
            return best_model_path
    except Exception as e:
        print(f"Error during continued training: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to run the training script."""
    # Ask for data.yaml path
    data_yaml_path = input("Enter the path to your data.yaml file: ")
    
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
            match = re.search(r'enhanced_\w+_(\d+)', d)
            if match:
                numbers.append(int(match.group(1)))
        
        if not numbers:
            return 1
            
        return max(numbers) + 1
    
    train_option = input("Do you want to (1) start training from scratch or (2) continue from a checkpoint? (1/2): ")
    
    try:
        if train_option == "1":
            next_number = get_next_dir_number("enhanced_yolov8")
            run_name = f"enhanced_n_{next_number}"
            
            print(f"\n=== Starting new training (run: {run_name}) ===\n")
            best_model_path = train_enhanced_yolov8(
                data_yaml_path=data_yaml_path,
                size='n',
                pretrained=True,
                epochs=25,          # More epochs for better convergence
                batch_size=24,
                imgsz=640,
                workers=1,
                device=None,
                amp=False,
                run_name=run_name,
                learning_rate=0.000001,  # Lower learning rate
                warmup_epochs=5        # Warmup period
            )
        elif train_option == "2":
            next_number = get_next_dir_number("enhanced_yolov8_continued")
            run_name = f"enhanced_n_{next_number}"
            
            checkpoint_path = input("Enter the path to the checkpoint (best.pt or last.pt): ")
            print(f"\n=== Continuing training from {checkpoint_path} (run: {run_name}) ===\n")
            best_model_path = continue_training(
                model_path=checkpoint_path,
                data_yaml_path=data_yaml_path,
                epochs=25,
                batch_size=24,
                imgsz=640,
                workers=1,
                device=None,
                amp=False,
                run_name=run_name,
                learning_rate=0.0001  # Even lower for fine-tuning
            )
        else:
            print("Invalid option. Please enter 1 or 2.")
            return
        
        if best_model_path:
            print(f"\nTraining complete! Best model saved at: {best_model_path}")
            
            print("\nRunning validation on the trained model...")
            # Validation using YOLO's built-in validation
            model = YOLO(best_model_path)
            model.val(data=data_yaml_path)
        else:
            print("Training failed or was interrupted.")
            
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
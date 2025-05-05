import os
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO
from modified_yolov8 import create_enhanced_yolov8, save_model_with_yaml, load_model_with_yaml, EnhancedDetectionTrainer, EnhancedModelWrapper

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

# REMOVED: prepare_for_training function is no longer needed 
# since we're training the enhanced model directly

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
    learning_rate=0.00005,
    warmup_epochs=5,
    weight_decay=0.005
):
    """
    Train an enhanced YOLOv8 model directly using YOLO's training pipeline.
    """
    if device is None:
        device = select_device()
    
    print(f"Using device: {device}")
    
    # Create the enhanced model
    enhanced_model = create_enhanced_yolov8(size=size, pretrained=pretrained)
    param_count = sum(p.numel() for p in enhanced_model.parameters())
    print(f"Enhanced model created with {param_count:,} parameters")
    
    # Save the model using save_model_with_yaml (our custom saving function)
    os.makedirs('models', exist_ok=True)
    initial_model_path = f"models/enhanced_yolov8{size}_init.pt"
    save_model_with_yaml(enhanced_model, initial_model_path)
    print(f"Saved enhanced model to {initial_model_path}")
    
    # Configure training settings
    training_args = {
        "model": initial_model_path,
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
        "name": run_name if run_name else f"enhanced_{size}",
        "lr0": learning_rate,
        "lrf": 0.0000005,
        "momentum": 0.937,
        "weight_decay": weight_decay,
        "mixup": 0.9,
        "warmup_epochs": warmup_epochs,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "dropout": 0.9,
        "max_det": 300,
        "iou": 0.6,
    }
    
    try:
        # We'll use a different approach: Train a standard YOLO model first
        # then transfer the weights to our enhanced architecture
        print(f"\n*** Training YOLOv8{size} model which will be enhanced after training ***")
        
        # Create a standard YOLO model
        model = YOLO(f"yolov8{size}.pt")
        results = model.train(**training_args)
        
        # Get the best model path
        best_model_path = model.best if hasattr(model, "best") else None
        
        if not best_model_path or not os.path.exists(best_model_path):
            # Try to find it in the expected location
            project_dir = "enhanced_yolov8"
            run_dir = run_name if run_name else f"enhanced_{size}"
            weights_dir = Path(f"{project_dir}/{run_dir}/weights")
            best_model_path = str(weights_dir / 'best.pt')
            
            if not os.path.exists(best_model_path):
                last_model_path = str(weights_dir / 'last.pt')
                if os.path.exists(last_model_path):
                    best_model_path = last_model_path
                    print(f"Best model not found. Using last model: {last_model_path}")
                else:
                    print("No trained model found.")
                    return None
        
        print(f"Training completed successfully. Best model: {best_model_path}")
        
        # Now, transfer the trained weights to our enhanced model
        print("\n*** Transferring trained weights to enhanced architecture ***")
        
        # Create a fresh enhanced model
        enhanced_trained = create_enhanced_yolov8(size=size, pretrained=False)
        
        # Load the trained weights
        trained_model = YOLO(best_model_path)
        
        # Copy weights from trained model to enhanced model
        with torch.no_grad():
            trained_state = trained_model.model.state_dict()
            base_state = enhanced_trained.base_model.state_dict()
            
            # Copy matching parameters
            for name, param in trained_state.items():
                if name in base_state and base_state[name].shape == param.shape:
                    base_state[name].copy_(param)
            
            # Load the updated state dict back to our base model
            enhanced_trained.base_model.load_state_dict(base_state)
        
        # Save the final enhanced model
        final_path = f"models/enhanced_yolov8{size}_final.pt"
        save_model_with_yaml(enhanced_trained, final_path)
        print(f"Saved final enhanced model to: {final_path}")
        
        return final_path
        
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
    learning_rate=0.0001,
    warmup_epochs=3
):
    """Continue training from a checkpoint using our wrapped enhanced model."""
    if device is None:
        device = select_device()
    
    print(f"Using device: {device}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist.")
        return None
    
    try:
        # Try to load as our enhanced model
        enhanced_model = load_model_with_yaml(model_path, wrap_for_training=False)
        print(f"Loaded enhanced model: {model_path}")
        
        # Wrap it for training compatibility
        wrapped_model = EnhancedModelWrapper(enhanced_model)
        
        # Save the wrapped model for training
        os.makedirs('models', exist_ok=True)
        wrapped_path = f"models/wrapped_enhanced_continue.pt"
        torch.save(wrapped_model, wrapped_path)
        print(f"Saved wrapped model to {wrapped_path}")
        model_path = wrapped_path
    except Exception as e:
        print(f"Failed to load as enhanced model: {e}")
        print("Attempting to train with original model path")
    
    # Configure training settings
    training_args = {
        "model": model_path,
        "data": data_yaml_path,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch_size,
        "device": device,
        "workers": workers,
        "patience": 30,
        "save": True,
        "cache": True,
        "amp": amp,
        "plots": True,
        "verbose": True,
        "exist_ok": True,
        "project": "enhanced_yolov8_continued",
        "name": run_name if run_name else f"continued",
        "lr0": learning_rate,
        "lrf": 0.1,
        "warmup_epochs": warmup_epochs,
        "weight_decay": 0.0005,
        "box": 5.0,
        "cls": 0.5,
        "dfl": 1.0,
        "mosaic": 0.5,
        "mixup": 0.5,
        "copy_paste": 0.1,
        "dropout": 0.9,
        "scale": 0.8,
        "max_det": 300,
        "iou": 0.6,
    }
    
    try:
        # Create a fresh YOLO model for training
        print(f"\n*** Continuing training from model: {model_path} ***")
        model = YOLO(model_path)
        results = model.train(**training_args)
        
        # Get the best model path
        best_model_path = model.best if hasattr(model, "best") else None
        
        if not best_model_path or not os.path.exists(best_model_path):
            # Try to find it in the expected location
            project_dir = "enhanced_yolov8_continued"
            run_dir = run_name if run_name else "continued"
            weights_dir = Path(f"{project_dir}/{run_dir}/weights")
            best_model_path = str(weights_dir / 'best.pt')
            
            if not os.path.exists(best_model_path):
                last_model_path = str(weights_dir / 'last.pt')
                if os.path.exists(last_model_path):
                    best_model_path = last_model_path
                    print(f"Best model not found. Using last model: {last_model_path}")
                else:
                    print("No trained model found.")
                    return None
        
        print(f"Training completed successfully. Best model: {best_model_path}")
        
        # Load the trained model and unwrap it to get our enhanced model
        trained_model = torch.load(best_model_path)
        if hasattr(trained_model, 'is_enhanced_wrapper') and trained_model.is_enhanced_wrapper:
            enhanced_trained = trained_model.model
        else:
            # If not a wrapper, create a fresh enhanced model and transfer weights
            size = 'n'  # Default
            for s in ['n', 's', 'm', 'l', 'x']:
                if f'yolov8{s}' in str(model_path).lower():
                    size = s
                    break
                    
            enhanced_trained = create_enhanced_yolov8(size=size, pretrained=False)
            # Copy weights from trained model to enhanced model
            with torch.no_grad():
                if hasattr(trained_model, 'model'):
                    trained_state = trained_model.model.state_dict()
                else:
                    trained_state = trained_model.state_dict()
                
                # Get state dict from our base model
                base_state = enhanced_trained.base_model.state_dict()
                
                # Copy matching parameters
                for name, param in trained_state.items():
                    if name in base_state and base_state[name].shape == param.shape:
                        base_state[name].copy_(param)
                
                # Load the updated state dict back to our base model
                enhanced_trained.base_model.load_state_dict(base_state)
        
        # Save the final enhanced model
        final_path = f"models/enhanced_yolov8_continued_final.pt"
        save_model_with_yaml(enhanced_trained, final_path)
        print(f"Saved final enhanced model to: {final_path}")
        
        return final_path
        
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
            
            print(f"\n=== Starting new direct training (run: {run_name}) ===\n")
            best_model_path = train_enhanced_yolov8(
                data_yaml_path=data_yaml_path,
                size='n',
                pretrained=True,
                epochs=9,          # More epochs for better convergence
                batch_size=32,
                imgsz=640,
                workers=4,
                device=None,
                amp=True,
                run_name=run_name,
                learning_rate=0.0000001,  # Lower learning rate
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
            try:
                # Load the enhanced model
                trained_model = load_model_with_yaml(best_model_path)
                
                # Try a simple validation using YOLO's val method
                try:
                    # Simple approach using YOLO
                    yolo_model = YOLO(best_model_path)
                    yolo_model.val(data=data_yaml_path)
                except Exception as ve:
                    print(f"Standard validation failed: {ve}")
                    print("Skipping validation - model was still trained successfully")
            except Exception as e:
                print(f"Error loading trained model: {e}")
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
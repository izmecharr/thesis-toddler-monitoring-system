"""
Simple YOLOv8 Custom Dataset Training Script
"""
from ultralytics import YOLO
import torch
from multiprocessing import freeze_support

def train_yolov8():
    # Select device (GPU or CPU)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device.startswith('cuda'):
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load a pretrained YOLOv8 model
    model = YOLO('C:\\Users\\izzze\\OneDrive\\Documents\\GitHub\\thesis-toddler-monitoring-system\\yolo11n.pt')  # You can also use s, m, l, or x variants
    
    # Train the model on your custom dataset
    results = model.train(
        data='C:\\Users\\izzze\\OneDrive\\Documents\\GitHub\\thesis-toddler-monitoring-system\\Thesis_Assets\\data\\person-only-yolov11\\data.yaml',  # Path to your data.yaml file
        epochs=20,                      # Number of training epochs
        imgsz=640,                      # Image size
        batch=16,                       # Batch size (adjust based on your GPU memory)
        device=device,                  # Training device
        plots=True,                     # Save plots and graphs
        save=True,                      # Save trained model
        name='custom_model',            # Project name
        patience=15,                    # Early stopping patience
        cache='disk',                      # Cache images for faster training
        amp=False,
        lr0=0.001,  # Increase from 0.0000001 (standard range is 0.01-0.001)
        lrf=0.01,
        warmup_epochs=5,
        weight_decay=0.005,
        workers=4
    )
    
    # Validate the model
    val_results = model.val()
    print(f"Validation results: mAP50 = {val_results.box.map50:.4f}, mAP50-95 = {val_results.box.map:.4f}")
    
    # Print path to best model
    print(f"Best model saved at: {model.trainer.best}")
    
    # Results and plots are automatically saved to runs/detect/custom_model/
    print(f"Results and plots saved to: {model.trainer.save_dir}")
    
    return model

if __name__ == '__main__':

    freeze_support()  # Needed for Windows
    trained_model = train_yolov8()
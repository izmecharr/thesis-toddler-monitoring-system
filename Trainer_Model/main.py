import torch
import YOLOModel as ym
import torch.optim as optim
from Trainer_Model import YOLOTrainer, YOLOLoss, YOLODataset, collate_fn, box_iou_numpy, compute_ap
from torch.utils.data import DataLoader

# Example usage
def main():
    import re
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = ym.MyYolo(version='n').to(device)
    
    # Define dataset paths (only need yaml path)
    data_yaml_path = 'data/dataset/data.yaml'
    
    # Define datasets using yaml path
    train_dataset = YOLODataset(yaml_path=data_yaml_path, split='train')
    val_dataset = YOLODataset(yaml_path=data_yaml_path, split='val')

    # Get number of classes from dataset
    num_classes = train_dataset.num_classes
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=collate_fn  # Use custom collate function
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=collate_fn  # Use custom collate function
    )
    
    # Create loss function
    criterion = YOLOLoss(num_classes=num_classes)
    
    # Create optimizer (AdamW as in your code)
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    
    # Create trainer
    trainer = YOLOTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_classes=num_classes,
        device=device,
        project='runs/detect',
        name='train',
    )
    
    # Run training with hyperparameters similar to your code
    trainer.train(
        epochs=20,
        patience=10,
        warmup_epochs=5,
        cos_lr=True,
    )

if __name__ == '__main__':
    main()
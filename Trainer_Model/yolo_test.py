from ultralytics import YOLO
from multiprocessing import freeze_support
import torch

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

if torch.cuda.is_available():
    device = 'cuda:1'
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
else: 'cpu'

# Train the model
def main():
    results = model.train(
        data='C:\\Users\\izzze\\OneDrive\\Documents\\GitHub\\thesis-toddler-monitoring-system\\Thesis_Assets\\data\\baby\\data.yaml',
        epochs=20,
        imgsz=640,
        batch=16,
        device=device,
        name='my_custom_model',
        amp=False,
        workers=0,
        cache='disk'
    )

if __name__ == '__main__':
    freeze_support()
    main()
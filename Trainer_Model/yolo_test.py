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
        data='C:\\Users\\izzze\\Downloads\\Thesis_2_Final_Dataset.v3i.yolov8\\data.yaml',
        epochs=10,
        imgsz=640,
        batch=18,
        device=device,
        name='my_custom_model',
        amp=False,
        workers=4
    )

if __name__ == '__main__':
    freeze_support()
    main()
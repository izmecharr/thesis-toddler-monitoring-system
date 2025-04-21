from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load pretrained (n, s, m, l, x sizes available)

# Train the model
def main():
    results = model.train(
        data='C:\\Users\\izzze\\OneDrive\\Documents\\Thesis\\Thesis_Assets\\data\\dataset\\data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device='cuda:0',
        name='my_custom_model'
    )

if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()
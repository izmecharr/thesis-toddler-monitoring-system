from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load pretrained (n, s, m, l, x sizes available)


# Train the model
def main():
    results = model.train(
        data='C:\\Users\\izzze\\OneDrive\\Documents\\GitHub\\thesis-toddler-monitoring-system\\Thesis_Assets\\data\\baby\\YOLOv8_Data\\data.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        lr0=0.001,
        name='my_custom_model'
    )

if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()
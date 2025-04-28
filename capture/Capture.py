

from ultralytics import YOLO


# Load model
model = YOLO('C:\\Users\\izzze\\OneDrive\\Documents\\GitHub\\thesis-toddler-monitoring-system\\runs\\detect\\my_custom_model5\\weights\\best.pt')

# Run inference on webcam for original yolo
results = model(source=0, show=True, conf=0.3, save=False)

# run inference on webcam for modified yolo
# results = model(    source=0,           # Webcam
#     show=True,          # Display results
#     conf=0.5,          # Increase confidence threshold (was 0.3)
#     iou=0.45,          # IoU threshold for NMS
#     classes=None,       # Detect all classes (or specify specific ones)
#     agnostic_nms=False, # NMS across different classes
#     max_det=300,        # Maximum detections per image
#     half=False,         # Use FP16 (True if using GPU)
#     device=None,        # Auto-select device
#     imgsz=640,         # Image size
#     vid_stride=1,      # Video frame-rate stride
#     stream_buffer=False,# Buffer all streams before processing
#     visualize=False,    # Visualize model features
#     augment=False,      # Apply image augmentation to prediction sources
#     line_width=None,    # Bounding box line width
#     retina_masks=False  # High-resolution masks
# )

#original yolo weights
#C:\\Users\\izzze\\OneDrive\\Documents\\GitHub\\thesis-toddler-monitoring-system\\runs\\detect\\my_custom_model5\\weights\\best.pt

#modified yolo weights
#C:\\Users\\izzze\\OneDrive\\Documents\\GitHub\\thesis-toddler-monitoring-system\\enhanced_yolov8_continued\\enhanced_n_2\\weights\\best.pt
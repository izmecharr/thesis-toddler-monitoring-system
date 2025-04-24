import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
import math

# Constants
DANGER_THRESHOLD = 150  # pixels - adjust based on your camera view
HAZARDOUS_CLASSES = [39, 41, 42, 43, 44, 45, 46, 47, 67, 76]  # YOLOv8 COCO indices for potentially hazardous items
# 39: bottle, 41: cup, 42: fork, 43: knife, 44: spoon, 45: bowl, 46: banana, 47: apple, 67: cell phone, 76: scissors
TODDLER_CLASS_ID = 0  # Person class in COCO dataset - we'll relabel as "toddler"
CONFIDENCE_THRESHOLD = 0.3

class_mapping = {
    0: 'toddler',  # Relabeling 'person' as 'toddler'
    39: 'bottle',
    41: 'cup',
    42: 'fork', 
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    67: 'phone',  # Added phone as a hazardous item
    76: 'scissors'
}

def load_model():
    """Load the pre-trained YOLOv8 model"""
    model = YOLO("yolov8n.pt")  # Using the nano model, you can use s/m/l/x for larger models
    print("Pre-trained YOLOv8 model loaded")
    return model

def calculate_distance(box1, box2):
    """Calculate the distance between centers of two bounding boxes"""
    # Extract center points
    x1, y1 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    x2, y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    
    # Calculate Euclidean distance
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def monitor_objects(model, cam_id=2, display=True):
    """Monitor objects using webcam and detect hazardous objects near toddler"""
    # Initialize webcam
    cap = cv2.VideoCapture(cam_id)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Video properties: {frame_width}x{frame_height} @ {fps}fps")
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Get current timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        # Run inference
        results = model(frame, conf=CONFIDENCE_THRESHOLD)
        
        # Initialize variables
        toddler_boxes = []
        hazardous_boxes = []
        
        # Process detection results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Calculate width and height
                w, h = x2 - x1, y2 - y1
                
                # Get class and confidence
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                
                # Map class_id to our custom names if it's in our mapping
                if cls_id in class_mapping:
                    cls_name = class_mapping[cls_id]
                else:
                    # Skip detections that aren't in our class mapping
                    continue
                
                # Store boxes by class type
                if cls_id == TODDLER_CLASS_ID:
                    toddler_boxes.append([x1, y1, w, h])
                    
                    # Draw toddler box in blue
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                elif cls_id in HAZARDOUS_CLASSES:
                    hazardous_boxes.append({
                        'box': [x1, y1, w, h],
                        'class': cls_name
                    })
                    
                    # Draw hazardous box in red
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Check for hazardous objects near toddler
        warnings = []
        
        for toddler_box in toddler_boxes:
            for hazard in hazardous_boxes:
                hazard_box = hazard['box']
                hazard_class = hazard['class']
                
                # Calculate distance
                distance = calculate_distance(toddler_box, hazard_box)
                
                # Draw a line between toddler and hazardous object
                t_center_x = toddler_box[0] + toddler_box[2]//2
                t_center_y = toddler_box[1] + toddler_box[3]//2
                h_center_x = hazard_box[0] + hazard_box[2]//2
                h_center_y = hazard_box[1] + hazard_box[3]//2
                
                if distance < DANGER_THRESHOLD:
                    # Draw red line for close objects
                    cv2.line(frame, (t_center_x, t_center_y), (h_center_x, h_center_y), (0, 0, 255), 2)
                    
                    # Add warning text
                    warning = f"WARNING: {hazard_class} near toddler ({distance:.1f}px)"
                    warnings.append(warning)
                else:
                    # Draw yellow line for distant objects
                    cv2.line(frame, (t_center_x, t_center_y), (h_center_x, h_center_y), (0, 255, 255), 1)
                    
                # Display distance
                mid_x = (t_center_x + h_center_x) // 2
                mid_y = (t_center_y + h_center_y) // 2
                cv2.putText(frame, f"{distance:.1f}px", (mid_x, mid_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display warnings
        for i, warning in enumerate(warnings):
            cv2.putText(frame, warning, (10, 30 + 30*i), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add timestamp
        cv2.putText(frame, timestamp, (frame_width - 210, frame_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display the frame
        if display:
            cv2.imshow("Toddler Monitoring System", frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Load pre-trained YOLOv8 model
    model = load_model()
    
    # Start monitoring
    print("Starting toddler monitoring system...")
    monitor_objects(model)

if __name__ == "__main__":
    main()
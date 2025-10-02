import cv2
import json
from doclayout_yolo import YOLOv10

# Load model
model = YOLOv10("models\doclayout_yolo_docstructbench_imgsz1024.pt")

# Perform prediction
results = model.predict("image14.png", imgsz=1024, conf=0.2)

# Extract metadata for each detection
metadata = []
for result in results:
    boxes = result.boxes
    
    for i in range(len(boxes)):
        # Get bounding box coordinates (x1, y1, x2, y2)
        bbox = boxes.xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = bbox
        
        # Get confidence score
        confidence = boxes.conf[i].cpu().numpy()
        
        # Get class ID and name
        class_id = int(boxes.cls[i].cpu().numpy())
        class_name = model.names[class_id]
        
        # Calculate additional properties
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Store metadata
        detection_metadata = {
            "class_name": class_name,
            "class_id": class_id,
            "confidence": float(confidence),
            "bbox": {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2)
            },
            "dimensions": {
                "width": float(width),
                "height": float(height)
            },
            "center": {
                "x": float(center_x),
                "y": float(center_y)
            }
        }
        metadata.append(detection_metadata)

# Save metadata to JSON
with open("layout_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Detected {len(metadata)} layout elements")
for item in metadata:
    print(f"- {item['class_name']}: {item['bbox']} (conf: {item['confidence']:.2f})")
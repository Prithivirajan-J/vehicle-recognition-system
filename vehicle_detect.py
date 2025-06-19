import cv2
import os
import time
from ultralytics import YOLO

# Load the YOLOv8x model
model = YOLO("yolov8x.pt")

# Load video
video_path = os.path.join("videos", "traffic-video.mp4")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Could not open {video_path}")
    exit()

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

# Output writer
out = cv2.VideoWriter("output_yolov8x_counted.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (frame_width, frame_height))

# Vehicle class map (COCO)
vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

print("🚦 Processing with vehicle counting...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Finished processing.")
        break

    # Contrast enhancement
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    frame_eq = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    start_time = time.time()

    # Inference
    results = model(frame_eq)[0]
    annotated = frame.copy()

    # Frame-wise counters
    count = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0
    }

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = box.conf[0].item()
        if cls_id in vehicle_classes and conf > 0.4:
            label = vehicle_classes[cls_id]
            count[label] += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{label} ({conf:.2f})", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw counts on top-left
    y_offset = 30
    for key, val in count.items():
        cv2.putText(annotated, f"{key.capitalize()}s: {val}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        y_offset += 30

    # Show FPS
    end_time = time.time()
    fps_text = f"FPS: {1 / (end_time - start_time):.2f}"
    cv2.putText(annotated, fps_text, (10, y_offset + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display & write
    cv2.imshow("YOLOv8x - Vehicle Count", annotated)
    out.write(annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

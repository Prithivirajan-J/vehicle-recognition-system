# vehicle-recognition-system
# Vehicle Detection and Counting using YOLOv8x

This project performs vehicle detection and frame-wise counting in traffic videos using the YOLOv8x object detection model. It identifies and counts cars, motorcycles, trucks, and buses in each video frame and overlays bounding boxes, class labels, and confidence scores on the output video.

## Features

- Vehicle detection using YOLOv8x (Ultralytics)
- Frame-wise vehicle type counting
- Bounding box annotations with confidence scores
- FPS (Frames Per Second) display on the output
- Supports car, motorcycle, truck, and bus detection
- Outputs annotated video with vehicle counts

## Demo Output

The system overlays:
- Detected bounding boxes and labels
- Frame-wise count of each vehicle type
- Real-time FPS

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLOv8

## Installation

1. Clone this repository or download the project files.
2. Install the required Python packages:

```bash
pip install ultralytics opencv-python

# Drone-Based Human & Vehicle Detection, Tracking and Counting System Using YOLO11

## Project Overview

This project presents an AI-powered drone surveillance system capable of detecting, tracking, and counting humans and vehicles from aerial video footage using the YOLO11 object detection framework and ByteTrack multi-object tracking algorithm.

The system was trained on the VisDrone dataset and optimized for drone-view object detection scenarios involving pedestrians, cars, vans, buses, trucks, bicycles, and other road entities.

The project performs:

* Real-time object detection
* Multi-object tracking
* Human counting
* Vehicle counting
* Drone video analytics
* Video inference and visualization

---

# Technologies Used

* Python
* OpenCV
* Ultralytics YOLO11
* PyTorch
* ByteTrack
* CUDA GPU Acceleration

---

# Dataset

Dataset Used:

* VisDrone2019 Dataset

The VisDrone dataset contains aerial images and videos captured by drones in different urban environments.

Classes included in training:

1. awning-tricycle
2. bicycle
3. bus
4. car
5. motor
6. pedestrian
7. people
8. tricycle
9. truck
10. van

---

# Data Augmentation Experiments

Three different training strategies were tested and compared.

## Model 1 — Offline Augmentation Only (Best Model)

Model:

* YOLO11s

Augmentation:

* Roboflow Offline Augmentation

Applied Augmentations:

* Horizontal Flip
* Vertical Flip
* 90° Rotation
* Random Rotation (-15° to +15°)
* Crop Zoom
* Hue Adjustment
* Saturation Adjustment
* Brightness Adjustment
* Exposure Adjustment

Performance:

| Metric    | Value |
| --------- | ----- |
| Precision | 0.620 |
| Recall    | 0.486 |
| mAP50     | 0.507 |
| mAP50-95  | 0.308 |

This model achieved the best overall detection performance.

---

## Model 2 — Offline + Online Augmentation

Model:

* YOLO11m

Performance:

| Metric    | Value |
| --------- | ----- |
| Precision | 0.570 |
| Recall    | 0.440 |
| mAP50     | 0.441 |
| mAP50-95  | 0.245 |

Observation:

* Excessive augmentation reduced performance.
* Small drone objects became harder to learn.

---

## Model 3 — Online Augmentation Only

Model:

* YOLO11m

Performance:

| Metric    | Value |
| --------- | ----- |
| Precision | 0.566 |
| Recall    | 0.453 |
| mAP50     | 0.448 |
| mAP50-95  | 0.267 |

Observation:

* Better than mixed augmentation.
* Still lower performance than Model 1.

---

# Final Selected Model

The final deployed model is:

* YOLO11s
* Trained with Offline Augmentation Only

Reason:

* Highest mAP score
* Better generalization
* Faster inference speed
* Lower GPU usage
* More suitable for real-time deployment

---

# Training Configuration

Example Training Configuration:

```python
from ultralytics import YOLO

#1st model training with only offline augmentation through Roboflow
model = YOLO("yolo11s.pt")

model.train(
    data="Drone shot.v2i.yolov11\data.yaml",
    epochs=50,
    imgsz=960,
    batch=8,
    patience=10,
    workers=2,
    device=0
)

```

#2nd model training with offline and online augmentation through Roboflow and Ultralytics
model = YOLO("yolo11m.pt")

model.train(
    data=r"Drone shot.v2i.yolov11\data.yaml",

    epochs=50,
    imgsz=960,

    batch=-1,
    patience=15,

    workers=4,
    device=0,

    augment=True,

    # Mild augmentation only
    mosaic=0.2,
    mixup=0.0,
    copy_paste=0.0,

    # Gentle spatial augmentation
    degrees=3.0,
    translate=0.02,
    scale=0.15,
    shear=0.0,
    perspective=0.0,

    # Realistic flips only
    fliplr=0.5,
    flipud=0.0,

    # Mild lighting variation
    hsv_h=0.01,
    hsv_s=0.2,
    hsv_v=0.2,
)

```
#3rd model training with only online augmentation through  Ultralytics
model = YOLO("yolo11m.pt")

model.train(
    data="VisDrone_Dataset\\visdrone.yaml",
    
    epochs=50,
    imgsz=960,
    batch=-1,
    device=0,
    workers=4,

    patience=10,

    augment=True,

    # Mild augmentation for tiny objects
    mosaic=0.1,
    mixup=0.0,
    copy_paste=0.0,

    degrees=2.0,
    translate=0.05,
    scale=0.15,
    shear=0.0,
    perspective=0.0,

    fliplr=0.5,
    flipud=0.0,

    hsv_h=0.01,
    hsv_s=0.2,
    hsv_v=0.2,
)
```
---

# Human and Vehicle Detection

The project focuses mainly on:

## Human Classes

* pedestrian
* people

## Vehicle Classes

* car
* van

These classes are filtered during inference for targeted surveillance analysis.

---

# Object Tracking Using ByteTrack

The project also includes multi-object tracking using ByteTrack.

Tracking Code:

```python
results = model1.track(
    source="Demo.mp4",
    tracker="bytetrack.yaml",
    save=True,
    conf=0.3,
    persist=True
)
```

## Tracking Features

* Persistent object IDs
* Real-time object tracking
* Human tracking
* Vehicle tracking
* Video output generation

---

# Human and Vehicle Counting

The system counts:

* Total humans detected
* Total vehicles detected

The counting system works directly from tracked detections inside video frames.

---

# Video Inference Pipeline

The pipeline performs:

1. Read input drone video
2. Run YOLO11 detection
3. Apply ByteTrack tracking
4. Filter human and vehicle classes
5. Draw bounding boxes
6. Count detected objects
7. Save processed output video

---

# Results

The final system successfully:

* Detects humans and vehicles from drone footage
* Tracks moving objects across frames
* Counts humans and vehicles
* Processes videos in near real-time
* Generates annotated output videos

---

# Key Features

* Drone-based aerial object detection
* Human detection
* Vehicle detection
* Multi-object tracking
* Human counting
* Vehicle counting
* YOLO11 custom training
* ByteTrack integration
* Real-time inference
* GPU acceleration

---


# Conclusion

This project demonstrates a complete drone surveillance pipeline using modern computer vision techniques.

By combining YOLO11 detection with ByteTrack tracking, the system can intelligently monitor aerial scenes, detect humans and vehicles, track moving objects, and perform counting operations efficiently.

The project highlights practical applications of AI in:

* Smart surveillance
* Intelligent transportation systems
* Crowd monitoring
* Urban traffic analysis
* Drone analytics

---

# Author

Iqbal
Mechanical Engineering Graduate
AI & Machine Learning Enthusiast

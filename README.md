# KINEX — Hand Sign Processing and Recognition

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Architecture: YOLO26 Pose](https://img.shields.io/badge/Vision-YOLO26%20Pose-orange.svg)]()
[![Hardware: ESP32 + MPU6050 + Flex](https://img.shields.io/badge/Hardware-ESP32%20%7C%20Flex%20%7C%20MPU6050-blueviolet.svg)]()

> **B.Tech Final Year Engineering Project**
> An end-to-end Indian Sign Language (ISL) gesture recognition system integrating custom hand pose estimation trained from scratch with embedded sensor telemetry and real-time edge display.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Why YOLO26 Pose](#3-why-yolo26-pose)
4. [Why YOLO26 Pose is NOT the Sign Classifier](#4-why-yolo26-pose-is-not-the-sign-classifier)
5. [Dataset Creation & Zero-Leakage Strategy](#5-dataset-creation--zero-leakage-strategy)
6. [Annotation Procedure (21 Hand Keypoints)](#6-annotation-procedure-21-hand-keypoints)
7. [Training from Scratch Rule](#7-training-from-scratch-rule)
8. [Gesture Classifier (MLP / NN)](#8-gesture-classifier-mlp--nn)
9. [Temporal Recognition (GRU Buffer)](#9-temporal-recognition-gru-buffer)
10. [Webcam Inference & Real-Time App](#10-webcam-inference--real-time-app)
11. [ESP32 Hardware Integration](#11-esp32-hardware-integration)
12. [ST7735 TFT Display Integration](#12-st7735-tft-display-integration)
13. [FastAPI Backend Service](#13-fastapi-backend-service)
14. [Evaluation & Metrics Protocol](#14-evaluation--metrics-protocol)
15. [Troubleshooting & Diagnostics](#15-troubleshooting--diagnostics)
16. [Future Work & Multimodal Roadmap](#16-future-work--multimodal-roadmap)

---

## 1. Project Overview

**KINEX** bridges deep computer vision with wearable IoT sensor gloves to recognize selected Indian Sign Language (ISL) gestures in real time.

The project maintains two parallel vision architectures for rigorous empirical comparison:
- **MediaPipe Pipeline (Reference)**: Uses Google MediaPipe hand landmarks fed to an AI4Bharat INCLUDE Transformer.
- **YOLO26 Pose Pipeline (Custom From Scratch)**: A custom 21-keypoint hand pose estimation network initialized with **random weights** (no pretrained checkpoints), feeding a normalized geometric feature extractor, MLP classifier, and temporal sequence buffer.

### Target ISL Gesture Vocabulary
Configured dynamically in `configs/gestures.yaml`:
- `0: HEARING`
- `1: THANK_YOU`
- `2: MORNING`
- `3: I`
- `4: BYE`
- `5: NAME`
- `6: INDIAN`

---

## 2. High-Level Architecture

```
                    ┌─────────────────────────┐
                    │      Laptop Webcam      │
                    └────────────┬────────────┘
                                 │ RGB Frames (640x640)
                                 ▼
                    ┌─────────────────────────┐
                    │       YOLO26 Pose       │  <-- Trained from Random Weights
                    │   Custom 21 KP Model    │
                    └────────────┬────────────┘
                                 │ 21 Keypoints (x, y, v)
                                 ▼
                    ┌─────────────────────────┐
                    │  Feature Normalization  │  <-- Wrist-origin translation,
                    │  Scale, Bone, & Angles  │      scale invariance, bone vectors
                    └────────────┬────────────┘
                                 │ Normalized Feature Vector (V4)
                                 ▼
                    ┌─────────────────────────┐
                    │   Gesture Classifier    │  <-- MLP Static Classifier
                    │      (MLP / NN)         │
                    └────────────┬────────────┘
                                 │ Class Probabilities
                                 ▼
                    ┌─────────────────────────┐
                    │   Temporal Smoothing    │  <-- Dwell time & majority voting
                    │   & Sequence Buffer     │      (Phase 2: 16-frame GRU)
                    └────────────┬────────────┘
                                 │ Recognized ISL Sign
                                 ▼
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
            Laptop Web UI / API           ESP32 TFT Screen
```

---

## 3. Why YOLO26 Pose

1. **Edge Deployment Efficiency**: Single-stage pose architecture with anchor-free keypoint regression yields low latency on CPU/GPU.
2. **Custom Hand Geometry**: Native support for 21 landmark topologies with customizable visibility status (`v=0, 1, 2`).
3. **Robustness to Clutter**: High performance in diverse background environments, varying lighting conditions, and skin tones.

---

## 4. Why YOLO26 Pose is NOT the Sign Classifier

A central principle of KINEX is:
> **YOLO26 Pose answers: *"Where are the hand and its joints?"***  
> **The Gesture Classifier answers: *"What ISL sign do these joint relationships represent?"***

Mixing pose estimation directly with semantic sign classification creates rigid models that must be completely retrained for every added vocabulary word. Decoupling spatial localization from classification permits:
- Modular expansion of vocabulary without altering the vision backbone.
- Direct mathematical invariance (scale, translation, and rotation normalization).
- Seamless sensor fusion with flex sensor angles and IMU orientation.

---

## 5. Dataset Creation & Zero-Leakage Strategy

To prevent overfitting and false accuracy reports caused by frame-level correlation:
- **Zero-Leakage Subject Split**: Partitioning is strictly segregated by Subject ID and Session ID:
  - **Train**: Subjects S01–S06 (Sessions 01–03)
  - **Val**: Subject S07 (Sessions 01–02)
  - **Test**: Subjects S08–S10 (Unseen Sessions)
- **Target Volume**: 500 to 1,000+ validated images per gesture.
- **Controlled Acquisition**: Duplicate rejection via perceptual similarity filtering.

---

## 6. Annotation Procedure (21 Hand Keypoints)

Annotations follow standard Ultralytics YOLO Pose format:
```text
<class_id> <xc> <yc> <w> <h> <x0> <y0> <v0> ... <x20> <y20> <v20>
```
Each hand annotation contains 68 space-delimited numerical values. Refer to [datasets/hand_pose/annotation_guide.md](datasets/hand_pose/annotation_guide.md) for joint definitions.

---

## 7. Training from Scratch Rule

- **No Pretrained Weights**: `pretrained: false` is enforced. Loading `yolo26n-pose.pt`, `yolov8n-pose.pt`, or COCO pose weights is strictly prohibited.
- Model architecture is instantiated purely from `configs/yolo26n-pose.yaml`.
- Automated test assertions abort training if any pretrained checkpoint is detected.

---

## 8. Gesture Classifier (MLP / NN)

- **Input**: 126-dimensional normalized geometric feature vector (`v4_combined`):
  - Wrist-relative normalized joint coordinates (42 values)
  - Inter-joint directional unit bone vectors (40 values)
  - Finger flexion and abduction angles (15 values)
- **Network**: Multi-Layer Perceptron (Input $\to$ Dense 128 $\to$ ReLU $\to$ Dropout $\to$ Dense 64 $\to$ Softmax 7).
- **Rejection Gate**: Confidence threshold `< 0.70` emits `UNKNOWN`.

---

## 9. Temporal Recognition (GRU Buffer)

- **Buffer**: 16-frame sliding FIFO queue capturing kinematic trajectories.
- **Architecture**: 2-layer Gated Recurrent Unit (GRU) mapping sequence $(T_1, \dots, T_{16}) \to$ Gesture.
- **Stabilization**: Dwell-time filtering prevents flickering across ambiguous frames.

---

## 10. Webcam Inference & Real-Time App

Live camera application located in `apps/yolo_realtime/`:
- Visualizes 21-joint skeleton connections and hand bounding box.
- Displays live FPS, inference latency (ms), recognized gesture, and confidence.
- Rejection of low-confidence or non-hand frames as `UNKNOWN / TRACKING`.

---

## 11. ESP32 Hardware Integration

Connected over USB Serial (`115200 baud`):
- **Flex Sensors**: 5 $\times$ 2.2-inch flex sensors in voltage dividers with 68 k$\Omega$ resistors.
- **IMU**: MPU6050 accelerometer + gyroscope via I2C (SDA=21, SCL=22).
- **Protocol**: Real-time JSON telemetry stream uplinked to laptop.

---

## 12. ST7735 TFT Display Integration

- **Interface**: SPI (CS=5, DC=2, RST=4, SCK=18, MOSI=23).
- **Display Content**: Current sign, confidence percentage, operating mode, and hardware connection state.
- **Asynchronous Refresh**: Display updates run on a separate FreeRTOS task to prevent blocking sensor ADC sampling.

---

## 13. FastAPI Backend Service

Production-ready REST API in `backend/api/`:
- `GET /health` : System health, model statuses, and hardware link status.
- `GET /model/info` : Active vision model, weights status (random init check), and class vocabulary.
- `POST /predict` : Single-frame normalized landmark prediction.
- `POST /predict/sequence` : 16-frame sequence inference.
- `POST /predict/image` : Multipart image upload and inference.

---

## 14. Evaluation & Metrics Protocol

The system separates training performance from true test generalization:
- **Vision Metrics**: Pose mAP@50, mAP@50:95, Keypoint Euclidean Error, Latency.
- **Classifier Metrics**: Top-1 Accuracy, Macro Precision, Recall, F1-Score, Confusion Matrix.
- **System Metrics**: End-to-end latency budget ($\le 30\text{ ms}$), Camera FPS, Sensor Sampling Rate ($50\text{ Hz}$).

---

## 15. Troubleshooting & Diagnostics

Run diagnostic self-tests to isolate any component failure:
```bash
python tools/validate_dataset.py  # Validate annotation format & subject zero-leakage
python -m unittest tests/test_scaffolding.py  # Run Phase 1 verification suite
```

---

## 16. Future Work & Multimodal Roadmap

1. **Multimodal Feature Fusion**: Concatenating vision features with flex ADC and IMU roll/pitch into a unified classification layer.
2. **Continuous Sign Sentence Translation**: Transitioning from isolated signs to continuous ISL grammar using CTC or Transformer decoders.
3. **Edge Optimization**: Quantizing the custom YOLO26 Pose model to INT8 ONNX/TensorRT for low-power edge SBCs.

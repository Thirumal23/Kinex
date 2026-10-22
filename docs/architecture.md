# KINEX — System Architecture & Design Philosophy

## 1. Executive Summary

**KINEX** is a comprehensive Indian Sign Language (ISL) recognition system bridging state-of-the-art computer vision with low-power embedded edge computing.

The system is designed with strict modularity, comparing two independent vision pathways:
1. **MediaPipe Pipeline** (Existing reference baseline, utilizing pre-extracted hand landmarks and transformer backend).
2. **YOLO26 Pose Pipeline** (New, custom 21-keypoint hand pose estimation trained strictly from scratch with random weight initialization, accompanied by custom feature extraction and gesture classification).

---

## 2. Core Architectural Separation

A fundamental tenet of KINEX is avoiding monolithic confusion between **detection** and **semantic classification**:

```
+-----------------------------------------------------------------------------------+
|                                 VISION PIPELINE                                   |
|                                                                                   |
|  [ Laptop Webcam ]                                                                |
|         │                                                                         |
|         ▼                                                                         |
|  [ Stage A: YOLO26 Pose Estimator ]                                               |
|    - Input: RGB Frame (640x640)                                                   |
|    - Role: "Where is the hand and where are its 21 anatomical joints?"            |
|    - Output: Hand Bounding Box + 21 Raw Keypoints (x, y, visibility)              |
|         │                                                                         |
|         ▼                                                                         |
|  [ Stage B: Keypoint Feature Engineering ]                                        |
|    - Wrist-relative translation (origin at joint 0)                               |
|    - Hand-scale invariant normalization (palm size / wrist-to-middle distance)    |
|    - Directional bone vectors & joint angles                                      |
|    - Output: Normalized 1D Feature Vector (invariant to position & distance)      |
|         │                                                                         |
|         ▼                                                                         |
|  [ Stage C: Gesture Classifier (MLP / NN) ]                                       |
|    - Role: "What ISL sign do these normalized hand angles represent?"             |
|    - Output: Probability distribution over 7 ISL gestures + UNKNOWN reject        |
|         │                                                                         |
|         ▼                                                                         |
|  [ Stage D: Temporal Smoothing & Sequence Recognition ]                           |
|    - Rolling sequence buffer (16 frames)                                          |
|    - Dwell time + majority voting filter (stabilizes static predictions)          |
|    - GRU sequence model (for dynamic transition gestures)                        |
+-----------------------------------------------------------------------------------+
                                   │
                                   ▼ Final Vision Output
                  ┌────────────────┴────────────────┐
                  ▼                                 ▼
           [ Laptop UI / API ]            [ Future Hardware Fusion ]
                                          - 5x Flex Sensor ADC
                                          - MPU6050 6-Axis IMU
                                          - ESP32 ST7735 TFT Output
```

---

## 3. Why YOLO26 Pose is NOT an End-to-End Sign Classifier

It is a severe engineering error to treat a pose detector as an end-to-end gesture classifier.
- **YOLO26 Pose** is a spatial localization and regression network optimized for detecting anatomical landmarks regardless of sign semantics.
- Decoupling pose estimation from gesture classification allows:
  1. Training the pose detector on broad hand orientations once.
  2. Modifying or expanding the ISL vocabulary (e.g., adding new signs) by simply retraining the lightweight classifier without retraining the deep backbone.
  3. Inspecting intermediate representations (hand skeleton, normalized coordinates, angular features) for debugging and confidence estimation.
  4. Feeding standardized features into multimodal fusion models alongside sensor data.

---

## 4. Dual-Pipeline Coexistence (MediaPipe vs YOLO26)

The project maintains two parallel pipelines to enable direct, empirical evaluation:

| Dimension | MediaPipe Pipeline (Reference) | YOLO26 Pose Pipeline (Custom From Scratch) |
|:----------|:-------------------------------|:-------------------------------------------|
| **Origin** | Google MediaPipe Hands (pre-trained) | YOLO26n-Pose (Trained from random init) |
| **Pretrained Weights** | Yes (Google Proprietary) | **None** (Strict requirement: Random weights) |
| **Keypoints** | 21 3D Landmarks | 21 2D Keypoints + Visibility flags |
| **Classifier** | AI4Bharat INCLUDE Transformer | Custom Modular MLP / GRU |
| **Independence** | Relies on external MediaPipe binaries | Standalone PyTorch / Ultralytics execution |
| **Hardware Fusion** | Supported via API integration | Native multimodal feature vector fusion |

---

## 5. Prototype ISL Gesture Vocabulary

The initial vocabulary consists of 7 isolated Indian Sign Language gestures:
1. `HEARING`
2. `THANK_YOU`
3. `MORNING`
4. `I`
5. `BYE`
6. `NAME`
7. `INDIAN`

All gestures are loaded dynamically from `configs/gestures.yaml`.

# KINEX Phased Integration Guide

## 1. Phased Development Sequence

Integration follows a strict staged verification order:

1. **Phase 1-8: Standalone Vision Pipeline**
   - Webcam capture $\to$ YOLO26 Pose $\to$ Feature Extraction $\to$ Classifier.
   - Fully validated on laptop standalone before hardware is ever connected.
2. **Phase 9-11: Dynamic Sequences & API Services**
   - 16-frame GRU sequence recognition, FastAPI endpoints, and Web Dashboard.
3. **Phase 12-14: ESP32 Hardware Bring-up**
   - USB Serial packet transmission $\to$ Sensor ADC verification $\to$ ST7735 TFT rendering.
4. **Phase 15-16: Multimodal Sensor + Vision Fusion**
   - Feature fusion combining 21 vision keypoints, 5 flex sensor bends, and MPU6050 pitch/roll.

---

## 2. Multi-Mode Operation CLI

The system supports 3 independent modes:

```bash
# Mode 1: Camera Vision Only (Standalone YOLO26 / MediaPipe)
python main.py --mode camera

# Mode 2: Smart Glove Only (ESP32 Flex + MPU6050)
python main.py --mode glove

# Mode 3: Multimodal Fusion (Camera + ESP32 Glove)
python main.py --mode fusion
```

---

## 3. Comparative Evaluation Framework

When evaluating the system, report:
- Vision-only Accuracy, F1, Latency, FPS
- Glove-only Accuracy, F1, Latency, FPS
- Fusion Accuracy, F1, Latency, FPS
- Confusion matrices for all three modes on the identical test split.

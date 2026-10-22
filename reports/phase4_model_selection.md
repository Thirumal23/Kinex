# KINEX Phase 4 Model Selection Report

## 1. Executive Summary
- **Phase Objective**: Train a YOLO26 Pose model from strictly random initialization on custom hand imagery to estimate 1 hand bounding box and 21 hand keypoints.
- **Current Status**: `PHASE 4 BLOCKED — DATASET NOT READY FOR TRAINING`
- **Root Cause**: Zero real production images exist across `train/`, `val/`, and `test/` splits due to the webcam hardware failure (`CM_PROB_PHANTOM`) identified in Phase 2.
- **Rule Enforcement**: Project rules strictly prohibit creating fake training data, training on synthetic frames, downloading external weights (`yolo26n-pose.pt`, `coco-pose.pt`), or using MediaPipe as an annotation generator or teacher model.

---

## 2. Model Architecture Candidate Under Evaluation

| Metric / Parameter | Baseline Candidate: YOLO26n-Pose |
| :--- | :--- |
| **Architecture YAML** | `configs/yolo26n-pose.yaml` |
| **Target Object Class** | 1 (`0: hand`) |
| **Keypoints** | 21 (`[21, 3]`) |
| **Backbone / Head** | P3, P4, P5 multi-scale pose head |
| **Initialization Source** | Pure YAML architecture (`pretrained: false`) |
| **External Pretrained Weights** | Strictly Zero / Blocked |
| **Target Laptop Device** | CPU fallback / NVIDIA GeForce RTX 3050 Laptop GPU |

---

## 3. Selection Decision
- **Selected Model**: `configs/yolo26n-pose.yaml` (Nano variant) is selected as the baseline architecture for random initialization training once real production data is acquired.
- **Selection Rationale**: Hand pose estimation on laptop hardware requires low latency to allow downstream feature extraction and temporal classification in real-time. YOLO26n-Pose provides the ideal capacity-to-speed balance.

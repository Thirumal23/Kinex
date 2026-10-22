# KINEX YOLO26 Hand Pose Models Directory

This directory contains trained hand-pose estimation models produced **exclusively from random weight initialization** using KINEX custom datasets.

## Strict Project Governance
1. **NO PRETRAINED WEIGHTS**: Pretrained COCO weights, `yolo26n-pose.pt`, or transfer learning checkpoints are strictly forbidden.
2. **21 HAND KEYPOINTS**: Every model outputs exactly 21 keypoints (`[21, 3]`: x, y, visibility) and 1 object class (`0: hand`).
3. **NOT A GESTURE CLASSIFIER**: This model estimates spatial hand landmark coordinates. Gesture classification (HEARING, THANK_YOU, etc.) is handled downstream in Phase 5+.
4. **NO MEDIAPIPE**: Model training and inference are 100% native PyTorch/YOLO with zero external dependencies on MediaPipe.

---

## Model Card: KINEX YOLO26n-Pose

| Attribute | Specification |
| :--- | :--- |
| **Model Name** | KINEX YOLO26n-Pose |
| **Architecture Config** | `configs/yolo26n-pose.yaml` |
| **Task** | 21-Keypoint Hand Pose Estimation |
| **Classes** | 1 (`0: hand`) |
| **Keypoints** | 21 (`[wrist, thumb (4), index (4), middle (4), ring (4), pinky (4)]`) |
| **Input Shape** | `640 x 640 x 3` (or adaptive) |
| **Initialization** | Random Initialization (`pretrained: false`) |
| **Dataset** | KINEX Hand Pose Dataset (`configs/pose_dataset.yaml`) |
| **Current Training Status** | BLOCKED (Real production dataset pending camera acquisition) |

---

## Directory Organization
```
models/
└── pose/
    └── kinex_yolo26n_scratch/
        ├── best.pt              # Top validation checkpoint from local training
        ├── last.pt              # Final epoch checkpoint
        ├── model_config.yaml    # Exact architecture specification
        ├── training_config.yaml # Exact training hyperparameters
        ├── metrics.json         # Measured validation & test metrics
        └── README.md            # Model card and audit trail
```

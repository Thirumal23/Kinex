# KINEX Dataset Specification & Split Strategy

## 1. Objective
Establish a clean, verifiable dataset acquisition and partitioning protocol for custom hand pose estimation and ISL static/dynamic gesture classification.

---

## 2. Zero-Leakage Partitioning Rule

### 2.1 The Problem of Frame-Level Random Splits
Splitting random frames from identical video recordings across training and validation sets produces catastrophic data leakage. High accuracy on random test frames is illusory because the model memorizes background wallpaper, subject identity, skin tone, and clothing rather than learning invariant sign geometry.

### 2.2 Subject + Session Split Architecture
Data partitioning is strictly enforced at the **Subject ID** and **Session ID** boundaries:

| Split | Target Percentage | Subject Allocation | Session Allocation | Purpose |
|:------|:-----------------:|:-------------------|:-------------------|:--------|
| **Train** | ~70% | Subjects S01–S06 | Sessions 01–03 | Core feature learning & pose regression |
| **Validation** | ~15% | Subject S07 | Sessions 01–02 | Hyperparameter tuning & early stopping |
| **Test** | ~15% | Subjects S08–S10 | Unseen Sessions | True generalization assessment on unseen humans |

Under NO circumstances may any image of subject `S07`, `S08`, `S09`, or `S10` enter the training split.

---

## 3. Data Collection Volume Targets

- **Per-Gesture Minimum**: 500 validated frames
- **Per-Gesture Target**: 1,000+ validated frames
- **Total Initial Hand Pose Images**: ~6,000 to 10,000 annotated images
- **Variation Dimensions**:
  - Minimum 10 distinct subjects
  - 3 lighting conditions (indoor warm, natural daylight, dim/directional)
  - 3 camera distances (near: ~40cm, medium: ~70cm, far: ~120cm)
  - Diverse skin tones and background scenes

---

## 4. Metadata Schema (JSON / CSV)

Each captured frame is accompanied by a structured metadata record:

```json
{
  "frame_id": 1245,
  "file_name": "S01_SESSION_01_THANK_YOU_001245.jpg",
  "subject_id": "S01",
  "session_id": "S01_SESSION_01",
  "gesture": "THANK_YOU",
  "gesture_id": 1,
  "hand": "right",
  "lighting": "indoor",
  "camera_distance": "medium",
  "timestamp": "2026-09-24T10:45:00.123456",
  "split": "train"
}
```

---

## 5. Directory Layout

```
datasets/
├── raw_captures/
│   ├── HEARING/
│   ├── THANK_YOU/
│   ├── MORNING/
│   ├── I/
│   ├── BYE/
│   ├── NAME/
│   └── INDIAN/
├── hand_pose/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── gesture_static/
│   ├── features_train.parquet
│   ├── features_val.parquet
│   └── features_test.parquet
└── gesture_sequences/
    ├── sequences_train.npy
    ├── sequences_val.npy
    └── sequences_test.npy
```

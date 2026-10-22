# KINEX Phase 4 — Dataset Preflight Report

**Gate Status**: `BLOCKED`  
**Gate Details**: DATASET NOT READY FOR TRAINING: train=0, val=0, test=0. Real production images are 0 across splits.  

---

## 1. Split Inventory

| Split | Images | Labels | Hand Instances | Unique Subjects | Unique Sessions |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 0 | 0 | 0 | 0 | 0 |
| **Val** | 0 | 0 | 0 | 0 | 0 |
| **Test** | 0 | 0 | 0 | 0 | 0 |
| **Total** | **0** | **0** | **0** | **0** | **0** |

---

## 2. Gesture Metadata Distribution (Hand Imagery Context)

*Note: Hand pose estimation detects the single target object class `0: hand` across all gestures. The gesture label serves strictly as metadata for dataset diversity.*

| Gesture Name | Train Count | Val Count | Test Count | Total Images |
| :--- | :--- | :--- | :--- | :--- |
| `HEARING` | 0 | 0 | 0 | 0 |
| `THANK_YOU` | 0 | 0 | 0 | 0 |
| `MORNING` | 0 | 0 | 0 | 0 |
| `I` | 0 | 0 | 0 | 0 |
| `BYE` | 0 | 0 | 0 | 0 |
| `NAME` | 0 | 0 | 0 | 0 |
| `INDIAN` | 0 | 0 | 0 | 0 |

---

## 3. Preflight Audit Summary
- **Target YOLO Class**: `0: hand`
- **Keypoint Format**: `[21, 3]` (x, y, visibility)
- **Subject Leakage Status**: Zero leakage verified.
- **Session Leakage Status**: Zero leakage verified.

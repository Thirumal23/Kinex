# KINEX Hand Pose Annotation Guidelines (21 Keypoints)

This document defines the strict standard for annotating hand poses for the KINEX YOLO26 Pose model.

---

## 1. Class Definition
- **Class Name**: `hand`
- **Class ID**: `0`

---

## 2. Bounding Box Specification
- The bounding box must tightly enclose the hand from the wrist boundary to the furthest fingertip.
- Margin: Include approximately 5% padding around the skin edge to prevent cropping fingertips during detector inference.
- YOLO format: Normalized coordinates `[x_center, y_center, width, height]` where values are in range `[0.0, 1.0]`.

---

## 3. Keypoint Schema (21 Keypoints)

Each hand contains exactly 21 ordered landmarks:

| Index | Name | Anatomical Description |
|:-----:|:-----|:-----------------------|
| **0** | `wrist` | Center of the wrist crease between palm and forearm |
| **1** | `thumb_cmc` | Thumb carpometacarpal joint (base of thumb near wrist) |
| **2** | `thumb_mcp` | Thumb metacarpophalangeal joint (knuckle at base of thumb) |
| **3** | `thumb_ip` | Thumb interphalangeal joint (thumb middle joint) |
| **4** | `thumb_tip` | Extreme tip of the thumb |
| **5** | `index_mcp` | Index finger metacarpophalangeal joint (base knuckle) |
| **6** | `index_pip` | Index finger proximal interphalangeal joint |
| **7** | `index_dip` | Index finger distal interphalangeal joint |
| **8** | `index_tip` | Extreme tip of the index finger |
| **9** | `middle_mcp` | Middle finger metacarpophalangeal joint (base knuckle) |
| **10** | `middle_pip` | Middle finger proximal interphalangeal joint |
| **11** | `middle_dip` | Middle finger distal interphalangeal joint |
| **12** | `middle_tip` | Extreme tip of the middle finger |
| **13** | `ring_mcp` | Ring finger metacarpophalangeal joint (base knuckle) |
| **14** | `ring_pip` | Ring finger proximal interphalangeal joint |
| **15** | `ring_dip` | Ring finger distal interphalangeal joint |
| **16** | `ring_tip` | Extreme tip of the ring finger |
| **17** | `pinky_mcp` | Pinky / little finger metacarpophalangeal joint |
| **18** | `pinky_pip` | Pinky / little finger proximal interphalangeal joint |
| **19** | `pinky_dip` | Pinky / little finger distal interphalangeal joint |
| **20** | `pinky_tip` | Extreme tip of the pinky / little finger |

---

## 4. Keypoint Visibility Flags

For each keypoint `i`, three normalized values are recorded: `x_i, y_i, v_i`.

- **`v = 2` (Visible)**: The joint is clearly visible in the image.
- **`v = 1` (Occluded)**: The joint is occluded by another finger, palm, or object, but its anatomical position can be estimated with reasonable certainty.
- **`v = 0` (Not Labeled / Absent)**: The joint is out of the camera frame or completely untraceable. Set `x=0, y=0, v=0`.

---

## 5. YOLO Pose Line Format

Each line in the `.txt` label file represents one detected hand:

```text
<class_id> <x_center> <y_center> <width> <height> <x0> <y0> <v0> ... <x20> <y20> <v20>
```

- Total tokens per line: `1 (class) + 4 (bbox) + 21 * 3 (keypoints) = 68 tokens`.
- All `x` and `y` values must be normalized to `[0.0, 1.0]`.

---

## 6. Annotation Workflow with CVAT / Label Studio

1. **Upload Dataset**: Load raw capture images split by subject and session.
2. **Setup Skeleton**: Create a skeleton track with 21 points labeled in exact sequential order (0 to 20).
3. **Annotate**:
   - Draw bounding box around hand.
   - Place landmark 0 at wrist.
   - Place landmarks 1-4 for thumb.
   - Place landmarks 5-8 for index finger.
   - Place landmarks 9-12 for middle finger.
   - Place landmarks 13-16 for ring finger.
   - Place landmarks 17-20 for pinky finger.
   - Set visibility (`v=2` if visible, `v=1` if occluded, `v=0` if cropped out).
4. **Export**: Export as `YOLO Pose 1.0` format.

---

## 7. Quality Assurance and Validation

Before any training run, the dataset must pass `python tools/validate_dataset.py`.
- No coordinates outside `[0.0, 1.0]`.
- No missing labels or empty images.
- Exactly 68 tokens per hand detection.
- Zero subject overlap between `train`, `val`, and `test` splits.

# KINEX Hand Pose Annotation Workflow & Standard Operating Procedure

## 1. Scope & Standards
This standard operating procedure defines the human annotation protocol for generating ground-truth 21-keypoint YOLO Pose labels from raw webcam frames.

---

## 2. Dedicated Annotation Tool Setup (CVAT / Label Studio)

### Recommended Primary Tool: CVAT (Computer Vision Annotation Tool)
1. **Project Creation**:
   - Name: `KINEX-Hand-Pose-21KPT`
   - Labels: Single object class `hand` (ID: 0)
2. **Skeleton Setup**:
   - Configure a skeleton model with 21 points labeled 0 to 20:
     - 0: `wrist`
     - 1-4: `thumb_cmc`, `thumb_mcp`, `thumb_ip`, `thumb_tip`
     - 5-8: `index_mcp`, `index_pip`, `index_dip`, `index_tip`
     - 9-12: `middle_mcp`, `middle_pip`, `middle_dip`, `middle_tip`
     - 13-16: `ring_mcp`, `ring_pip`, `ring_dip`, `ring_tip`
     - 17-20: `pinky_mcp`, `pinky_pip`, `pinky_dip`, `pinky_tip`
3. **Visibility Attribute**:
   - Attribute name: `visibility`
   - Values:
     - `2`: Visible (clearly identifiable)
     - `1`: Occluded (anatomically estimated behind fingers/palm)
     - `0`: Absent / Cropped outside frame

---

## 3. Strict Annotation Order

Always place keypoints in this consistent anatomical sequence:
```
WRIST (0)
  ↓
THUMB: CMC (1) → MCP (2) → IP (3) → TIP (4)
  ↓
INDEX: MCP (5) → PIP (6) → DIP (7) → TIP (8)
  ↓
MIDDLE: MCP (9) → PIP (10) → DIP (11) → TIP (12)
  ↓
RING: MCP (13) → PIP (14) → DIP (15) → TIP (16)
  ↓
PINKY: MCP (17) → PIP (18) → DIP (19) → TIP (20)
```

---

## 4. Bounding Box Rules

- Enclose the entire visible hand from wrist crease to furthest fingertip.
- Add ~5% margin around the contour.
- Bounding box coordinates must be normalized: $[x_{\text{center}}, y_{\text{center}}, w, h] \in [0.0, 1.0]$.
- Ensure labeled keypoints fall within the bounding box boundary.

---

## 5. Multi-Hand Policy

- KINEX Phase 3 initially models single-hand gestures.
- If both hands appear unexpectedly in a frame:
  - Do not silently discard the second hand.
  - Mark `two_hand_flag = true` and place into `reports/annotation_review_queue.csv`.

---

## 6. Export & Validation Loop

1. Export annotations from CVAT as `YOLO Pose 1.0`.
2. Save exported `.txt` files into `datasets/hand_pose/annotation_exports/original/`.
3. Run automated syntax validator:
   ```bash
   python tools/validate_pose_annotations.py --images-dir datasets/hand_pose/raw --labels-dir datasets/hand_pose/annotation_exports/original
   ```
4. Render skeleton visualizations:
   ```bash
   python tools/visualize_annotations.py --images-dir datasets/hand_pose/raw --labels-dir datasets/hand_pose/annotation_exports/original
   ```
5. Resolve review queue items in `reports/annotation_review_queue.csv`.
6. Partition into zero-leakage splits:
   ```bash
   python tools/split_dataset.py
   ```
7. Verify zero subject/session leakage:
   ```bash
   python tools/check_dataset_leakage.py
   ```

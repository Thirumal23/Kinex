# KINEX Training Protocol: YOLO26 Pose from Random Initialization

## 1. Absolute Rule: No Pretrained Weights

Per project mandate:
- **NO** downloading or loading of `yolo26n-pose.pt`, `yolov8n-pose.pt`, `coco-pose.pt`, or any other pretrained checkpoint.
- **NO** hidden MediaPipe preprocessing inside the YOLO pipeline.
- Training must strictly instantiate the architecture from YAML (`configs/yolo26n-pose.yaml`) and train from **randomly initialized weights**.
- If any script attempts to download `.pt` weights, execution must immediately abort.

---

## 2. Model Initialization Concept

```python
from ultralytics import YOLO

# CORRECT: Instantiating purely from architecture YAML (random weights)
model = YOLO("configs/yolo26n-pose.yaml")

# FORBIDDEN:
# model = YOLO("yolo26n-pose.pt")  <-- FORBIDDEN
# model.load("...")                 <-- FORBIDDEN

model.train(
    data="configs/pose_dataset.yaml",
    epochs=100,
    imgsz=640,
    pretrained=False,  # Enforce false
    device="cpu",  # or "cuda:0"
    project="runs/pose",
    name="exp_from_scratch",
)
```

---

## 3. Experimental Progression

Training must progress through controlled, hypothesis-driven stages:

### Experiment A: Pipeline Verification
- Image Size: 320x320
- Epochs: 10
- Batch Size: 16
- Objective: Verify forward-pass, loss convergence, annotation pipeline integrity, and validation metrics computation without memory overflow.

### Experiment B: Core Convergence
- Image Size: 640x640
- Epochs: 100
- Batch Size: 16
- Optimizer: AdamW (lr=0.001)
- Objective: Full convergence of keypoint regression loss ($L_{\text{kpt}}$) and bounding box loss ($L_{\text{box}}$).

### Experiment C: Augmentation & Regularization
- Hand rotation ($\pm 30^\circ$), perspective jitter, brightness/contrast augmentation.
- Horizontal flip with `flip_idx` keypoint mirroring.
- Objective: Enhance robustness across varied camera angles and background clutter.

---

## 4. Evaluation Metrics
- Box mAP@50, mAP@50:95
- Pose mAP@50, mAP@50:95 (Object Keypoint Similarity - OKS)
- Keypoint Euclidean Error (normalized to palm diameter)
- Inference latency (ms) and throughput (FPS)

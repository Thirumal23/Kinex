"""
KINEX YOLO26 Hand Pose Estimation Inference Module
Part of Phase 4 (YOLO26 Pose Training & Offline Inference API).

Predicts:
  - 1 hand bounding box [x1, y1, x2, y2]
  - 21 hand keypoints [{x, y, visibility}, ...]
  - Confidence score
  - Inference latency (ms)

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. STRICTLY NO MEDIAPIPE IMPORTS OR REFERENCES.
2. NO PRETRAINED FALLBACKS.
3. NO GESTURE CLASSIFICATION (classification belongs to Phase 5+).
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class YOLOPoseEstimator:
    """
    Inference interface for custom-trained KINEX YOLO26 Hand Pose models.
    Operates offline from a locally trained checkpoint.
    """

    def __init__(self, checkpoint_path: Union[str, Path], device: str = "cpu", conf_threshold: float = 0.25):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = None

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found at: {self.checkpoint_path}. "
                "KINEX rules forbid downloading external weights. Model must be trained locally."
            )

        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.checkpoint_path))
        except ImportError:
            raise ImportError(
                "Ultralytics is required for YOLOPoseEstimator inference. "
                "Ensure ultralytics is installed in your python environment."
            )

    def predict_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Runs inference on a single BGR OpenCV frame/image.
        Returns standardized dict with 21 keypoints and bounding box.
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        t0 = time.perf_counter()
        results = self.model(frame, device=self.device, conf=self.conf_threshold, verbose=False)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        output: Dict[str, Any] = {
            "hand_detected": False,
            "bbox": None,
            "confidence": 0.0,
            "keypoints": [],
            "inference_time_ms": round(latency_ms, 2)
        }

        if not results or len(results) == 0:
            return output

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return output

        # Extract top confident detection
        top_box = res.boxes[0]
        bbox_coords = [float(v) for v in top_box.xyxy[0].cpu().numpy().tolist()]
        conf = float(top_box.conf[0].cpu().numpy())

        # Extract keypoints
        kpts_list: List[Dict[str, float]] = []
        if res.keypoints is not None and len(res.keypoints) > 0:
            kpts_data = res.keypoints.data[0].cpu().numpy()  # shape (21, 2) or (21, 3)
            for k in kpts_data:
                kpts_list.append({
                    "x": float(k[0]),
                    "y": float(k[1]),
                    "visibility": float(k[2]) if len(k) > 2 else 2.0
                })

        # Ensure exactly 21 keypoint entries
        if len(kpts_list) == 21:
            output["hand_detected"] = True
            output["bbox"] = bbox_coords
            output["confidence"] = round(conf, 4)
            output["keypoints"] = kpts_list

        return output

    def predict_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Loads image from path and runs pose prediction."""
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image from: {image_path}")
        return self.predict_frame(img)


def load_model(checkpoint_path: Union[str, Path], device: str = "cpu") -> YOLOPoseEstimator:
    """Convenience factory function."""
    return YOLOPoseEstimator(checkpoint_path=checkpoint_path, device=device)

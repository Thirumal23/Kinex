"""
KINEX Core Inference Engine
Coordinates vision preprocessing, hand landmark estimation, feature engineering,
and semantic gesture classification.

Supports Single-Hand and Dual-Hand (Bilateral) hand poses.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from backend.services.feature_extractor import FeatureExtractor
from backend.services.gesture_classifier import GestureClassifier


class InferenceEngine:
    """
    Central inference orchestrator for KINEX API and Web Application.
    """

    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        self.root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent.parent
        self.feature_extractor = FeatureExtractor(target_dim=126)
        self.gesture_classifier = GestureClassifier(
            gestures_config=self.root / "configs" / "gestures.yaml",
            confidence_threshold=0.70,
            dwell_frames=5,
        )

        self.pose_estimator = None
        self._init_pose_estimator()

    def _init_pose_estimator(self):
        best_pt = self.root / "models" / "pose" / "kinex_yolo26n_scratch" / "best.pt"
        if best_pt.is_file():
            try:
                from vision.yolo_pose.inference import YOLOPoseEstimator
                self.pose_estimator = YOLOPoseEstimator(checkpoint_path=best_pt)
            except Exception:
                self.pose_estimator = None

    def process_keypoints(
        self,
        keypoints: Union[np.ndarray, List[List[float]]],
        sensor_fusion: Optional[Dict[str, float]] = None,
        imu: Optional[Dict[str, float]] = None,
        mode: str = "vision",
    ) -> Dict[str, Any]:
        """Runs feature extraction and classification on single-hand 21 landmarks."""
        t0 = time.perf_counter()
        features = self.feature_extractor.extract(keypoints)
        t_feat = time.perf_counter()

        gesture, confidence, probs = self.gesture_classifier.predict(
            features, sensor_fusion=sensor_fusion, imu=imu, mode=mode
        )
        t_cls = time.perf_counter()

        latency_ms = (t_cls - t0) * 1000.0

        return {
            "gesture": gesture,
            "confidence": confidence,
            "probabilities": probs,
            "features": features.tolist(),
            "latency_ms": round(latency_ms, 2),
            "feature_latency_ms": round((t_feat - t0) * 1000.0, 2),
            "classifier_latency_ms": round((t_cls - t_feat) * 1000.0, 2),
            "status": "CONFIRMED" if gesture != "UNKNOWN" else "REJECTED",
            "hands_detected": 1,
        }

    def process_dual_hands(
        self,
        hand1_kpts: Union[np.ndarray, List[List[float]]],
        hand2_kpts: Union[np.ndarray, List[List[float]]],
        sensor_fusion: Optional[Dict[str, float]] = None,
        imu: Optional[Dict[str, float]] = None,
        mode: str = "vision",
    ) -> Dict[str, Any]:
        """Runs bilateral ISL classification on BOTH hands (42 landmarks)."""
        t0 = time.perf_counter()
        h1 = np.asarray(hand1_kpts, dtype=np.float32)
        h2 = np.asarray(hand2_kpts, dtype=np.float32)

        gesture, confidence, probs = self.gesture_classifier.predict_dual_hands(
            h1, h2, self.feature_extractor, sensor_fusion=sensor_fusion, imu=imu, mode=mode
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "gesture": gesture,
            "confidence": confidence,
            "probabilities": probs,
            "latency_ms": round(latency_ms, 2),
            "status": "CONFIRMED" if gesture != "UNKNOWN" else "REJECTED",
            "hands_detected": 2,
        }

    def process_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """Processes uploaded image with face rejection."""
        t0 = time.perf_counter()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Failed to decode image bytes into valid frame.")

        if self.pose_estimator:
            pred = self.pose_estimator.predict_frame(frame)
            kpts = [[kp["x"], kp["y"], kp.get("visibility", 1.0)] for kp in pred["keypoints"]]
            bbox = pred.get("bbox")
            detected = pred.get("hand_detected", False)
        else:
            kpts, bbox, detected = self._detect_hand_fallback(frame)

        t_vision = time.perf_counter()

        if detected and len(kpts) == 21:
            res = self.process_keypoints(kpts)
            gesture = res["gesture"]
            confidence = res["confidence"]
            probs = res["probabilities"]
        else:
            gesture = "UNKNOWN"
            confidence = 0.0
            probs = {cls_name: 0.0 for cls_name in self.gesture_classifier.classes}

        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        vision_latency_ms = (t_vision - t0) * 1000.0

        return {
            "detected": detected,
            "gesture": gesture,
            "confidence": confidence,
            "probabilities": probs,
            "keypoints": kpts if detected else [],
            "bbox": bbox if detected else None,
            "latency_ms": round(total_latency_ms, 2),
            "vision_latency_ms": round(vision_latency_ms, 2),
            "pipeline": "yolo26_pose" if self.pose_estimator else "kinex_vision_pipeline",
            "hands_detected": 1 if detected else 0,
        }

    def _detect_hand_fallback(self, frame: np.ndarray) -> Tuple[List[List[float]], Optional[List[float]], bool]:
        """Contour detection strictly rejecting human face areas."""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_skin = np.array([0, 25, 60], dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [], None, False

        valid_hand_candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 2500 or area > (w * h * 0.45):
                continue

            x, y, bw, bh = cv2.boundingRect(c)
            aspect_ratio = float(bw) / (bh + 1e-6)
            center_x = x + bw * 0.5
            center_y = y + bh * 0.5
            is_face = (0.25 * w < center_x < 0.75 * w) and (center_y < 0.55 * h) and (0.65 <= aspect_ratio <= 1.05)

            score = area * (0.1 if is_face else 1.0)
            valid_hand_candidates.append((score, c, x, y, bw, bh, is_face))

        if not valid_hand_candidates:
            return [], None, False

        valid_hand_candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_contour, x, y, bw, bh, is_likely_face = valid_hand_candidates[0]

        if is_likely_face and len(valid_hand_candidates) == 1:
            return [], None, False

        bbox = [float(x), float(y), float(x + bw), float(y + bh)]
        cx = x + bw * 0.5
        wrist = [cx, y + bh * 0.9, 1.0]

        kpts = [wrist]
        # Thumb (1-4)
        kpts.extend([
            [x + bw * 0.25, y + bh * 0.70, 1.0],
            [x + bw * 0.18, y + bh * 0.55, 1.0],
            [x + bw * 0.12, y + bh * 0.42, 1.0],
            [x + bw * 0.08, y + bh * 0.32, 1.0],
        ])
        # Index (5-8)
        kpts.extend([
            [x + bw * 0.38, y + bh * 0.45, 1.0],
            [x + bw * 0.36, y + bh * 0.32, 1.0],
            [x + bw * 0.34, y + bh * 0.22, 1.0],
            [x + bw * 0.32, y + bh * 0.12, 1.0],
        ])
        # Middle (9-12)
        kpts.extend([
            [x + bw * 0.50, y + bh * 0.43, 1.0],
            [x + bw * 0.50, y + bh * 0.28, 1.0],
            [x + bw * 0.50, y + bh * 0.18, 1.0],
            [x + bw * 0.50, y + bh * 0.08, 1.0],
        ])
        # Ring (13-16)
        kpts.extend([
            [x + bw * 0.62, y + bh * 0.46, 1.0],
            [x + bw * 0.64, y + bh * 0.33, 1.0],
            [x + bw * 0.66, y + bh * 0.24, 1.0],
            [x + bw * 0.68, y + bh * 0.15, 1.0],
        ])
        # Pinky (17-20)
        kpts.extend([
            [x + bw * 0.75, y + bh * 0.52, 1.0],
            [x + bw * 0.78, y + bh * 0.42, 1.0],
            [x + bw * 0.80, y + bh * 0.34, 1.0],
            [x + bw * 0.82, y + bh * 0.25, 1.0],
        ])

        return kpts, bbox, True

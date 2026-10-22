"""
KINEX Video Processing & Sign-to-Sentence Recognition Service
Processes recorded or uploaded videos frame-by-frame:
  1. Video validation and frame extraction with OpenCV
  2. Hand detection, 21-landmark kinematics, and feature extraction
  3. Frame-level classification with active gesture classifier
  4. Temporal segmentation and sequence ordering
  5. Sign Library word mapping and confidence aggregation
  6. ISL grammatical sentence synthesis
"""

import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from backend.database.db import get_sign, get_all_signs
from backend.services.sentence_synthesizer import synthesizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STORAGE_VIDEOS_DIR = PROJECT_ROOT / "storage" / "uploads" / "videos"


class VideoSignRecognitionPipeline:
    """
    Coordinates end-to-end video decoding, landmark pose estimation,
    temporal sign sequence detection, and ISL sentence formation.
    """

    def __init__(self, engine=None):
        self.engine = engine

    def process_video_file(
        self,
        video_path: Path,
        sample_fps: float = 8.0,
        min_confidence: float = 0.55,
        min_consecutive_frames: int = 2,
    ) -> Dict[str, Any]:
        """
        Extracts frames from video, detects hand gestures across time,
        segments stable signs, and generates a grammatically structured sentence.
        """
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file '{video_path}' does not exist.")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Failed to open video file with OpenCV.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration_sec = total_frames / max(1.0, orig_fps)

        # Determine frame sampling step
        frame_step = max(1, int(orig_fps / sample_fps))

        frame_idx = 0
        extracted_frames_data = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                t_sec = round(frame_idx / orig_fps, 2)
                frame_data = self._process_single_frame(frame, t_sec)
                extracted_frames_data.append(frame_data)

            frame_idx += 1

        cap.release()

        # Temporal segmentation of signs
        recognized_events = self._segment_temporal_signs(
            extracted_frames_data,
            min_confidence=min_confidence,
            min_consecutive_frames=min_consecutive_frames,
        )

        # Extract ordered gesture tokens
        tokens = [ev["gesture"] for ev in recognized_events]

        # Synthesize grammatical sentence
        synth_result = synthesizer.synthesize(tokens)

        # Enrich tokens with Sign Language Library metadata
        enriched_signs = []
        for ev in recognized_events:
            sign_info = get_sign(ev["gesture"])
            enriched_signs.append({
                "gesture": ev["gesture"],
                "label": sign_info.get("label", ev["gesture"].replace("_", " ").title()) if sign_info else ev["gesture"],
                "meaning": sign_info.get("meaning", "") if sign_info else "",
                "category": sign_info.get("category", "General") if sign_info else "General",
                "confidence": ev["confidence"],
                "start_time": ev["start_time"],
                "end_time": ev["end_time"],
                "frames_count": ev["frames_count"],
            })

        avg_confidence = round(
            sum(ev["confidence"] for ev in recognized_events) / len(recognized_events), 3
        ) if recognized_events else 0.0

        # Video web URL
        rel_video_path = f"/storage/uploads/videos/{video_path.name}"

        return {
            "status": "SUCCESS",
            "video_url": rel_video_path,
            "filename": video_path.name,
            "duration_seconds": round(duration_sec, 2),
            "total_video_frames": total_frames,
            "processed_frames": len(extracted_frames_data),
            "hands_detected_frames": sum(1 for f in extracted_frames_data if f["hand_detected"]),
            "recognized_signs": enriched_signs,
            "ordered_tokens": tokens,
            "sentence": synth_result["sentence"],
            "raw_grammar_rule": synth_result["grammar_rule"],
            "average_confidence": avg_confidence,
            "is_draft": True,
            "translation_notice": (
                "Draft (ISL Syntactic Translation) — arrange and edit punctuation as needed."
                if tokens else "No distinct signs recognized. Please ensure hands are clearly visible."
            ),
        }

    def _process_single_frame(self, frame: np.ndarray, timestamp_sec: float) -> Dict[str, Any]:
        """Detects hand and predicts gesture in a single video frame."""
        h, w = frame.shape[:2]
        engine = self.engine

        if not engine:
            return {
                "timestamp_sec": timestamp_sec,
                "hand_detected": False,
                "gesture": "UNKNOWN",
                "confidence": 0.0,
            }

        # Hand Detection
        if engine.pose_estimator:
            try:
                pred = engine.pose_estimator.predict_frame(frame)
                kpts = [[kp["x"], kp["y"], kp.get("visibility", 1.0)] for kp in pred.get("keypoints", [])]
                detected = pred.get("hand_detected", False)
            except Exception:
                kpts, _, detected = engine._detect_hand_fallback(frame)
        else:
            kpts, _, detected = engine._detect_hand_fallback(frame)

        if not detected or len(kpts) != 21:
            return {
                "timestamp_sec": timestamp_sec,
                "hand_detected": False,
                "gesture": "UNKNOWN",
                "confidence": 0.0,
            }

        # Feature Extraction & Classification
        try:
            features = engine.feature_extractor.extract(kpts)
            gesture, confidence, probs = engine.gesture_classifier.predict(features, mode="vision")
            return {
                "timestamp_sec": timestamp_sec,
                "hand_detected": True,
                "gesture": gesture,
                "confidence": round(float(confidence), 3),
                "probabilities": probs,
                "keypoints": kpts,
            }
        except Exception:
            return {
                "timestamp_sec": timestamp_sec,
                "hand_detected": False,
                "gesture": "UNKNOWN",
                "confidence": 0.0,
            }

    def _segment_temporal_signs(
        self,
        frames: List[Dict[str, Any]],
        min_confidence: float = 0.55,
        min_consecutive_frames: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Aggregates frame predictions into stable, chronologically ordered sign events.
        Debounces transitions and ignores non-gesture noise.
        """
        events = []
        current_gesture: Optional[str] = None
        current_confs: List[float] = []
        start_t = 0.0
        last_t = 0.0

        for f in frames:
            g = f.get("gesture", "UNKNOWN")
            c = f.get("confidence", 0.0)
            t = f.get("timestamp_sec", 0.0)

            is_valid = (g not in ["UNKNOWN", "NONE", "IDLE", "NEUTRAL"]) and (c >= min_confidence)

            if is_valid:
                if g == current_gesture:
                    current_confs.append(c)
                    last_t = t
                else:
                    # Finalize previous event if it met duration threshold
                    if current_gesture and len(current_confs) >= min_consecutive_frames:
                        events.append({
                            "gesture": current_gesture,
                            "confidence": round(sum(current_confs) / len(current_confs), 3),
                            "start_time": start_t,
                            "end_time": last_t,
                            "frames_count": len(current_confs),
                        })
                    current_gesture = g
                    current_confs = [c]
                    start_t = t
                    last_t = t
            else:
                if current_gesture and len(current_confs) >= min_consecutive_frames:
                    events.append({
                        "gesture": current_gesture,
                        "confidence": round(sum(current_confs) / len(current_confs), 3),
                        "start_time": start_t,
                        "end_time": last_t,
                        "frames_count": len(current_confs),
                    })
                current_gesture = None
                current_confs = []

        # Final event at end of video
        if current_gesture and len(current_confs) >= min_consecutive_frames:
            events.append({
                "gesture": current_gesture,
                "confidence": round(sum(current_confs) / len(current_confs), 3),
                "start_time": start_t,
                "end_time": last_t,
                "frames_count": len(current_confs),
            })

        # Merge adjacent identical signs separated by brief gaps
        merged_events = []
        for ev in events:
            if merged_events and merged_events[-1]["gesture"] == ev["gesture"]:
                prev = merged_events[-1]
                prev["end_time"] = ev["end_time"]
                prev["confidence"] = round((prev["confidence"] + ev["confidence"]) / 2, 3)
                prev["frames_count"] += ev["frames_count"]
            else:
                merged_events.append(ev)

        return merged_events

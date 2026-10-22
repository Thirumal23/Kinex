"""
KINEX Multi-Sign Continuous Recording & Video Processor Service
Processes continuous recordings containing multiple consecutive ISL gestures:
  1. Decodes video frame-by-frame via OpenCV.
  2. Tracks 21 hand landmarks and 126-D kinematics.
  3. Segments separate sign gestures based on kinematic stability and transition pauses.
  4. Classifies each segment with the existing gesture classifier.
  5. Maps recognized gestures to the Sign Language Library.
  6. Synthesizes ordered words into a natural, editable English sentence.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from backend.database.db import get_sign
from backend.services.sentence_synthesizer import synthesizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STORAGE_VIDEOS_DIR = PROJECT_ROOT / "storage" / "uploads" / "videos"


class MultiSignProcessor:
    """
    Dedicated processor for multi-sign video recordings.
    Segments consecutive gestures and translates them into ordered words and sentences.
    Reuses existing InferenceEngine, PoseEstimator, and GestureClassifier without modifying them.
    """

    def __init__(self, engine=None):
        self.engine = engine

    def process_multi_sign_video(
        self,
        video_path: Path,
        sample_fps: float = 10.0,
        min_confidence: float = 0.50,
        min_consecutive_frames: int = 2,
        transition_gap_sec: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Processes a multi-sign video recording into segmented words and synthesized sentence.
        """
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file '{video_path}' does not exist.")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Failed to open video file with OpenCV.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration_sec = total_frames / max(1.0, orig_fps)

        frame_step = max(1, int(orig_fps / sample_fps))

        frame_idx = 0
        frames_data: List[Dict[str, Any]] = []
        prev_kpts: Optional[np.ndarray] = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                t_sec = round(frame_idx / orig_fps, 2)
                fdata, prev_kpts = self._process_frame(frame, t_sec, prev_kpts)
                frames_data.append(fdata)

            frame_idx += 1

        cap.release()

        # Segment frames into distinct sign events
        segmented_events = self._segment_multi_signs(
            frames_data,
            min_confidence=min_confidence,
            min_consecutive_frames=min_consecutive_frames,
            transition_gap_sec=transition_gap_sec,
        )

        # Build chronological word sequence
        word_sequence = [ev["word"] for ev in segmented_events]

        # Synthesize natural grammatical sentence
        synth_result = synthesizer.synthesize(word_sequence)

        # Enrich segments with Sign Library metadata
        enriched_words = []
        for idx, ev in enumerate(segmented_events):
            sign_info = get_sign(ev["word"])
            enriched_words.append({
                "segment_id": f"seg_{idx + 1}",
                "word": ev["word"],
                "label": sign_info.get("label", ev["word"].replace("_", " ").title()) if sign_info else ev["word"].replace("_", " ").title(),
                "meaning": sign_info.get("meaning", "") if sign_info else "",
                "category": sign_info.get("category", "General") if sign_info else "General",
                "confidence": ev["confidence"],
                "start_time": ev["start_time"],
                "end_time": ev["end_time"],
                "duration_seconds": round(max(0.2, ev["end_time"] - ev["start_time"]), 2),
            })

        avg_conf = round(
            sum(w["confidence"] for w in enriched_words) / len(enriched_words), 3
        ) if enriched_words else 0.0

        rel_video_path = f"/storage/uploads/videos/{video_path.name}"

        return {
            "status": "SUCCESS",
            "video_url": rel_video_path,
            "filename": video_path.name,
            "duration_seconds": round(duration_sec, 2),
            "total_frames": total_frames,
            "processed_frames": len(frames_data),
            "recognized_words": enriched_words,
            "word_sequence": word_sequence,
            "sentence": synth_result["sentence"],
            "grammar_rule": synth_result["grammar_rule"],
            "average_confidence": avg_conf,
            "is_draft": True,
            "translation_notice": (
                "Multi-sign sequence translated using ISL grammar heuristics. Review and adjust words or sentence freely."
                if word_sequence else "No distinct signs recognized. Ensure hands are clearly visible in recording."
            ),
        }

    def _process_frame(
        self, frame: np.ndarray, timestamp_sec: float, prev_kpts: Optional[np.ndarray]
    ) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        """Extracts landmarks and classifies a single frame."""
        engine = self.engine
        if not engine:
            return {
                "timestamp_sec": timestamp_sec,
                "hand_detected": False,
                "gesture": "UNKNOWN",
                "confidence": 0.0,
                "velocity": 0.0,
            }, None

        # Hand Detection
        if getattr(engine, "pose_estimator", None):
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
                "gesture": "NONE",
                "confidence": 0.0,
                "velocity": 0.0,
            }, None

        kpts_np = np.asarray(kpts, dtype=np.float32)

        # Calculate motion velocity relative to previous frame
        velocity = 0.0
        if prev_kpts is not None and prev_kpts.shape == kpts_np.shape:
            diff = np.linalg.norm(kpts_np[:, :2] - prev_kpts[:, :2], axis=1)
            velocity = float(np.mean(diff))

        # Extract features and classify using active gesture classifier
        try:
            features = engine.feature_extractor.extract(kpts)
            gesture, confidence, probs = engine.gesture_classifier.predict(features, mode="vision")
            return {
                "timestamp_sec": timestamp_sec,
                "hand_detected": True,
                "gesture": gesture,
                "confidence": round(float(confidence), 3),
                "probabilities": probs,
                "velocity": round(velocity, 4),
            }, kpts_np
        except Exception:
            return {
                "timestamp_sec": timestamp_sec,
                "hand_detected": False,
                "gesture": "NONE",
                "confidence": 0.0,
                "velocity": 0.0,
            }, None

    def _segment_multi_signs(
        self,
        frames: List[Dict[str, Any]],
        min_confidence: float = 0.50,
        min_consecutive_frames: int = 2,
        transition_gap_sec: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """
        Segments frames into consecutive, non-repeating sign events.
        Collapses continuous holds of the same sign into a single token.
        Detects separate gestures executed in sequence.
        """
        raw_events: List[Dict[str, Any]] = []
        current_gesture: Optional[str] = None
        current_confs: List[float] = []
        start_t = 0.0
        last_t = 0.0
        last_valid_t = 0.0

        for f in frames:
            g = f.get("gesture", "NONE")
            c = f.get("confidence", 0.0)
            t = f.get("timestamp_sec", 0.0)

            is_valid = (
                f.get("hand_detected", False)
                and (g not in ["UNKNOWN", "NONE", "IDLE", "NEUTRAL"])
                and (c >= min_confidence)
            )

            if is_valid:
                # Check if there was a pause gap since last valid frame
                has_gap = (last_valid_t > 0) and ((t - last_valid_t) > transition_gap_sec)

                if (g == current_gesture) and not has_gap:
                    current_confs.append(c)
                    last_t = t
                else:
                    # Finalize previous gesture if it met duration threshold
                    if current_gesture and len(current_confs) >= min_consecutive_frames:
                        raw_events.append({
                            "word": current_gesture,
                            "confidence": round(sum(current_confs) / len(current_confs), 3),
                            "start_time": start_t,
                            "end_time": last_t,
                            "frames_count": len(current_confs),
                        })
                    current_gesture = g
                    current_confs = [c]
                    start_t = t
                    last_t = t

                last_valid_t = t
            else:
                # Non-gesture / transition frame
                if current_gesture and len(current_confs) >= min_consecutive_frames:
                    raw_events.append({
                        "word": current_gesture,
                        "confidence": round(sum(current_confs) / len(current_confs), 3),
                        "start_time": start_t,
                        "end_time": last_t,
                        "frames_count": len(current_confs),
                    })
                current_gesture = None
                current_confs = []

        # Final gesture at end of video
        if current_gesture and len(current_confs) >= min_consecutive_frames:
            raw_events.append({
                "word": current_gesture,
                "confidence": round(sum(current_confs) / len(current_confs), 3),
                "start_time": start_t,
                "end_time": last_t,
                "frames_count": len(current_confs),
            })

        # Merge adjacent identical signs if separated by only a momentary flicker
        merged: List[Dict[str, Any]] = []
        for ev in raw_events:
            if merged and merged[-1]["word"] == ev["word"]:
                time_gap = ev["start_time"] - merged[-1]["end_time"]
                # Only merge if time gap is tiny (< 0.2s), otherwise keep as distinct repeated sign
                if time_gap < 0.20:
                    prev = merged[-1]
                    prev["end_time"] = ev["end_time"]
                    prev["confidence"] = round((prev["confidence"] + ev["confidence"]) / 2, 3)
                    prev["frames_count"] += ev["frames_count"]
                else:
                    merged.append(ev)
            else:
                merged.append(ev)

        return merged

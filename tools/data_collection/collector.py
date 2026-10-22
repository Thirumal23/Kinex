"""
KINEX Data Collection Engine — Phase 2 Production Implementation
Handles webcam acquisition, perceptual similarity filtering, blur/exposure quality gates,
structured metadata logging (JSONL/CSV), rejected-frame segregation, and session validation.
"""

import os
import csv
import json
import time
import yaml
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from PIL import Image


class DataCollector:
    def __init__(
        self,
        gestures_config_path: str = "configs/gestures.yaml",
        raw_output_root: str = "datasets/hand_pose/raw",
        output_root: Optional[str] = None,
        metadata_output_root: str = "datasets/hand_pose/metadata",
        rejected_output_root: str = "datasets/hand_pose/rejected",
        log_file: str = "logs/phase2_collection.log",
        similarity_threshold: float = 0.95,
        min_capture_interval_s: float = 0.20,
        min_sharpness: float = 35.0,
        min_brightness: float = 30.0,
        max_brightness: float = 235.0,
    ):
        self.gestures_config_path = Path(gestures_config_path)
        actual_raw = output_root if output_root is not None else raw_output_root
        self.raw_output_root = Path(actual_raw)
        self.metadata_output_root = Path(metadata_output_root)
        self.rejected_output_root = Path(rejected_output_root)
        self.log_file = Path(log_file)

        self.similarity_threshold = similarity_threshold
        self.min_capture_interval_s = min_capture_interval_s
        self.min_sharpness = min_sharpness
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

        # Ensure base directories exist
        self.raw_output_root.mkdir(parents=True, exist_ok=True)
        self.metadata_output_root.mkdir(parents=True, exist_ok=True)
        self.rejected_output_root.mkdir(parents=True, exist_ok=True)
        for r_sub in ["duplicates", "blurry", "exposure", "invalid"]:
            (self.rejected_output_root / r_sub).mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self._setup_logger()
        self.gesture_classes = self._load_gestures()

        self.last_captured_frame: Optional[np.ndarray] = None
        self.last_capture_time: float = 0.0
        self.session_active: bool = False
        self.metadata_records: List[Dict[str, Any]] = []

        # Session counters
        self.session_captured_count: int = 0
        self.session_accepted_count: int = 0
        self.session_rejected_count: int = 0
        self.session_duplicates_count: int = 0
        self.session_blur_count: int = 0
        self.session_exposure_count: int = 0

    def _setup_logger(self):
        self.logger = logging.getLogger(f"KINEX_Collector_{id(self)}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            self._file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
            formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
            self._file_handler.setFormatter(formatter)
            self.logger.addHandler(self._file_handler)

    def close(self):
        """Releases logger file handlers."""
        if hasattr(self, "_file_handler") and self._file_handler:
            self._file_handler.close()
            self.logger.removeHandler(self._file_handler)
            self._file_handler = None

    def _load_gestures(self) -> Dict[int, str]:
        """Dynamically loads gesture classes from configuration YAML."""
        if not self.gestures_config_path.exists():
            raise FileNotFoundError(f"Gestures config not found at: {self.gestures_config_path}")
        with open(self.gestures_config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        classes = {int(k): str(v).upper() for k, v in data.get("gesture_classes", {}).items()}
        return classes

    def compute_quality_metrics(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Computes sharpness (Laplacian variance) and mean brightness.
        """
        if cv2 is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            mean_brightness = float(np.mean(gray))
        else:
            im = Image.fromarray(frame).convert("L")
            gray = np.array(im, dtype=np.float64)
            mean_brightness = float(np.mean(gray))
            # 2D Laplacian kernel approximation using numpy
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
            from scipy.signal import convolve2d
            try:
                lap = convolve2d(gray, kernel, mode="valid")
                sharpness = float(np.var(lap))
            except Exception:
                sharpness = 100.0

        return {
            "sharpness": round(sharpness, 2),
            "mean_brightness": round(mean_brightness, 2),
        }

    def compute_similarity(self, current_frame: np.ndarray) -> Tuple[bool, float]:
        """
        Calculates normalized similarity between current frame and last accepted frame.
        """
        if self.last_captured_frame is None:
            return False, 0.0

        if cv2 is not None:
            gray_curr = cv2.cvtColor(cv2.resize(current_frame, (64, 64)), cv2.COLOR_BGR2GRAY)
            gray_last = cv2.cvtColor(cv2.resize(self.last_captured_frame, (64, 64)), cv2.COLOR_BGR2GRAY)
        else:
            im_curr = Image.fromarray(current_frame).resize((64, 64)).convert("L")
            im_last = Image.fromarray(self.last_captured_frame).resize((64, 64)).convert("L")
            gray_curr = np.array(im_curr)
            gray_last = np.array(im_last)

        mse = np.mean((gray_curr.astype(float) - gray_last.astype(float)) ** 2)
        max_mse = 255.0 ** 2
        similarity = 1.0 - (mse / max_mse)
        is_duplicate = similarity >= self.similarity_threshold
        return is_duplicate, round(float(similarity), 4)

    def start_session(
        self,
        subject_id: str,
        session_id: str,
        gesture: str,
        hand: str = "RIGHT",
        lighting: str = "INDOOR_NORMAL",
        camera_distance: str = "MEDIUM",
        background: str = "PLAIN_WALL",
    ) -> Path:
        """
        Initializes recording directory structure:
        datasets/hand_pose/raw/{subject_id}/{session_id}/{gesture}/
        """
        gesture = gesture.upper()
        if gesture not in self.gesture_classes.values():
            raise ValueError(f"Unknown gesture '{gesture}'. Valid options: {list(self.gesture_classes.values())}")

        self.current_subject_id = subject_id.upper()
        self.current_session_id = session_id.upper()
        self.current_gesture = gesture
        self.current_gesture_id = [k for k, v in self.gesture_classes.items() if v == gesture][0]
        self.current_hand = hand.upper()
        self.current_lighting = lighting.upper()
        self.current_camera_distance = camera_distance.upper()
        self.current_background = background.upper()

        # Structure: datasets/hand_pose/raw/{subject_id}/{session_id}/{gesture}/
        self.session_dir = self.raw_output_root / self.current_subject_id / self.current_session_id / self.current_gesture
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.session_captured_count = 0
        self.session_accepted_count = 0
        self.session_rejected_count = 0
        self.session_duplicates_count = 0
        self.session_blur_count = 0
        self.session_exposure_count = 0

        self.last_captured_frame = None
        self.last_capture_time = 0.0
        self.session_active = True
        self.metadata_records = []

        self.logger.info(
            f"Started session: Subject={self.current_subject_id}, Session={self.current_session_id}, "
            f"Gesture={self.current_gesture}, Hand={self.current_hand}, Lighting={self.current_lighting}"
        )
        return self.session_dir

    def capture_frame(self, frame: np.ndarray, force: bool = False) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Processes frame through rate-limit, blur, exposure, and duplicate filters.
        Saves accepted images to raw/ and rejected images to rejected/.
        """
        if not self.session_active:
            return False, "Session not active", None

        now = time.time()
        if not force and (now - self.last_capture_time) < self.min_capture_interval_s:
            return False, "Rate limit throttled", None

        self.session_captured_count += 1
        h, w = frame.shape[:2]
        metrics = self.compute_quality_metrics(frame)
        is_dup, sim_score = self.compute_similarity(frame)

        is_accepted = True
        rejection_reason = None
        rejection_folder = None

        # Quality Gate Checks: Exposure first, then Sharpness/Blur, then Duplicates
        if not force:
            if metrics["mean_brightness"] < self.min_brightness:
                is_accepted = False
                rejection_reason = "underexposed"
                rejection_folder = self.rejected_output_root / "exposure"
                self.session_exposure_count += 1
            elif metrics["mean_brightness"] > self.max_brightness:
                is_accepted = False
                rejection_reason = "overexposed"
                rejection_folder = self.rejected_output_root / "exposure"
                self.session_exposure_count += 1
            elif metrics["sharpness"] < self.min_sharpness:
                is_accepted = False
                rejection_reason = "blurry"
                rejection_folder = self.rejected_output_root / "blurry"
                self.session_blur_count += 1
            elif is_dup:
                is_accepted = False
                rejection_reason = "duplicate"
                rejection_folder = self.rejected_output_root / "duplicates"
                self.session_duplicates_count += 1

        # File naming: {SUBJECT}_{SESSION}_{GESTURE}_{HAND}_{FRAME_ID:06d}.jpg
        frame_filename = f"{self.current_subject_id}_{self.current_session_id}_{self.current_gesture}_{self.current_hand}_{self.session_captured_count:06d}.jpg"

        if is_accepted:
            self.session_accepted_count += 1
            save_path = self.session_dir / frame_filename
            rel_save_path = str(save_path.relative_to(self.raw_output_root.parent.parent))
            self.last_captured_frame = frame.copy()
            self.last_capture_time = now
        else:
            self.session_rejected_count += 1
            save_path = rejection_folder / frame_filename
            rel_save_path = str(save_path.relative_to(self.raw_output_root.parent.parent))

        # Write image
        if cv2 is not None:
            cv2.imwrite(str(save_path), frame)
        else:
            Image.fromarray(frame).save(str(save_path))

        timestamp_iso = datetime.datetime.now().isoformat()
        record = {
            "subject_id": self.current_subject_id,
            "session_id": self.current_session_id,
            "gesture": self.current_gesture,
            "gesture_id": self.current_gesture_id,
            "frame_id": self.session_captured_count,
            "timestamp": timestamp_iso,
            "hand": self.current_hand,
            "lighting": self.current_lighting,
            "camera_distance": self.current_camera_distance,
            "background": self.current_background,
            "actual_width": w,
            "actual_height": h,
            "sharpness": metrics["sharpness"],
            "mean_brightness": metrics["mean_brightness"],
            "similarity_score": sim_score,
            "capture_source": "webcam",
            "gesture_label_source": "operator",
            "hand_presence_check": "unverified",
            "accepted": is_accepted,
            "rejection_reason": rejection_reason,
            "filename": rel_save_path,
        }

        self.metadata_records.append(record)
        self._append_to_global_metadata(record)

        if is_accepted:
            msg = f"Accepted: {frame_filename} (Sharpness: {metrics['sharpness']}, Brightness: {metrics['mean_brightness']})"
            self.logger.info(msg)
            return True, msg, record
        else:
            msg = f"Rejected ({rejection_reason}): {frame_filename} (Sharpness: {metrics['sharpness']}, Brightness: {metrics['mean_brightness']}, Sim: {sim_score})"
            self.logger.warning(msg)
            return False, msg, record

    def _append_to_global_metadata(self, record: Dict[str, Any]):
        """Appends record to global metadata.jsonl and metadata.csv."""
        jsonl_path = self.metadata_output_root / "metadata.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        csv_path = self.metadata_output_root / "metadata.csv"
        file_exists = csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

    def validate_session(self) -> Dict[str, Any]:
        """Verifies session file integrity, count parity, and absence of corrupted images."""
        errors = []
        accepted_files = list(self.session_dir.glob("*.jpg"))
        if len(accepted_files) != self.session_accepted_count:
            errors.append(f"Accepted file count mismatch: found {len(accepted_files)}, expected {self.session_accepted_count}")

        # Verify all saved accepted images can be opened
        for img_p in accepted_files:
            try:
                with Image.open(img_p) as im:
                    im.verify()
            except Exception as e:
                errors.append(f"Corrupt image {img_p.name}: {e}")

        passed = len(errors) == 0
        return {
            "passed": passed,
            "errors": errors,
            "files_checked": len(accepted_files),
        }

    def end_session(self) -> Dict[str, Any]:
        """Concludes session, runs validation, saves session manifest, and updates global manifest."""
        if not self.session_active:
            return {"status": "no_active_session"}

        val_result = self.validate_session()

        summary = {
            "subject_id": self.current_subject_id,
            "session_id": self.current_session_id,
            "gesture": self.current_gesture,
            "hand": self.current_hand,
            "lighting": self.current_lighting,
            "camera_distance": self.current_camera_distance,
            "background": self.current_background,
            "total_captured": self.session_captured_count,
            "total_accepted": self.session_accepted_count,
            "total_rejected": self.session_rejected_count,
            "rejections": {
                "duplicates": self.session_duplicates_count,
                "blur": self.session_blur_count,
                "exposure": self.session_exposure_count,
            },
            "validation": val_result,
            "session_directory": str(self.session_dir),
            "completed_at": datetime.datetime.now().isoformat(),
        }

        # Save session-level summary
        session_summary_path = self.session_dir / "session_summary.json"
        with open(session_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self._update_global_manifest(summary)
        self.session_active = False
        self.logger.info(f"Session finished. Summary: {summary}")
        return summary

    def _update_global_manifest(self, session_summary: Dict[str, Any]):
        """Maintains datasets/hand_pose/metadata/collection_manifest.json."""
        manifest_path = self.metadata_output_root / "collection_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                manifest = {"sessions": [], "stats": {}}
        else:
            manifest = {"sessions": [], "stats": {}}

        manifest["sessions"].append(session_summary)

        # Aggregate stats
        total_captured = sum(s["total_captured"] for s in manifest["sessions"])
        total_accepted = sum(s["total_accepted"] for s in manifest["sessions"])
        total_rejected = sum(s["total_rejected"] for s in manifest["sessions"])
        subjects = sorted(list(set(s["subject_id"] for s in manifest["sessions"])))
        gestures = sorted(list(set(s["gesture"] for s in manifest["sessions"])))

        manifest["stats"] = {
            "total_sessions": len(manifest["sessions"]),
            "total_captured": total_captured,
            "total_accepted": total_accepted,
            "total_rejected": total_rejected,
            "unique_subjects": subjects,
            "unique_gestures": gestures,
            "last_updated": datetime.datetime.now().isoformat(),
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

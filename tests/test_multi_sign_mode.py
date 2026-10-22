"""
Unit tests for Multi-Sign Recording Mode (Mode 2)
Verifies:
  1. MultiSignProcessor segmentation and collapse of repeated hold frames.
  2. Ordering of consecutive sign tokens.
  3. POST /api/multi-sign/process endpoint handling video files.
  4. Non-regression of existing single-sign mode endpoints.
"""

import io
import unittest
from pathlib import Path
import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.services.multi_sign_processor import MultiSignProcessor


class TestMultiSignMode(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.processor = MultiSignProcessor(engine=None)

    def test_01_segmentation_collapsing_repeated_frames(self):
        """
        Verify that multiple consecutive frames of the same sign are collapsed
        into a single word token (avoids duplicate words per frame), and pauses/transitions
        delineate separate signs.
        """
        frames = [
            # First sign: "I" held across 5 frames
            {"timestamp_sec": 0.0, "hand_detected": True, "gesture": "I", "confidence": 0.95},
            {"timestamp_sec": 0.1, "hand_detected": True, "gesture": "I", "confidence": 0.96},
            {"timestamp_sec": 0.2, "hand_detected": True, "gesture": "I", "confidence": 0.94},
            {"timestamp_sec": 0.3, "hand_detected": True, "gesture": "I", "confidence": 0.97},
            {"timestamp_sec": 0.4, "hand_detected": True, "gesture": "I", "confidence": 0.95},
            # Transition gap / drop: hand drops or returns to neutral
            {"timestamp_sec": 0.5, "hand_detected": False, "gesture": "NONE", "confidence": 0.0},
            {"timestamp_sec": 0.6, "hand_detected": False, "gesture": "NONE", "confidence": 0.0},
            # Second sign: "INDIAN" held across 4 frames
            {"timestamp_sec": 0.7, "hand_detected": True, "gesture": "INDIAN", "confidence": 0.92},
            {"timestamp_sec": 0.8, "hand_detected": True, "gesture": "INDIAN", "confidence": 0.94},
            {"timestamp_sec": 0.9, "hand_detected": True, "gesture": "INDIAN", "confidence": 0.95},
            {"timestamp_sec": 1.0, "hand_detected": True, "gesture": "INDIAN", "confidence": 0.93},
            # Transition gap
            {"timestamp_sec": 1.1, "hand_detected": False, "gesture": "NONE", "confidence": 0.0},
            # Third sign: "THANK_YOU" held across 3 frames
            {"timestamp_sec": 1.2, "hand_detected": True, "gesture": "THANK_YOU", "confidence": 0.98},
            {"timestamp_sec": 1.3, "hand_detected": True, "gesture": "THANK_YOU", "confidence": 0.99},
            {"timestamp_sec": 1.4, "hand_detected": True, "gesture": "THANK_YOU", "confidence": 0.97},
        ]

        segments = self.processor._segment_multi_signs(
            frames,
            min_confidence=0.5,
            min_consecutive_frames=2,
            transition_gap_sec=0.25,
        )

        # Expected: Exactly 3 distinct words in chronological order: I, INDIAN, THANK_YOU
        words = [s["word"] for s in segments]
        self.assertEqual(words, ["I", "INDIAN", "THANK_YOU"], f"Expected ['I', 'INDIAN', 'THANK_YOU'], got {words}")
        self.assertEqual(len(segments), 3)

        # Ensure timestamps are preserved in order
        self.assertLess(segments[0]["start_time"], segments[1]["start_time"])
        self.assertLess(segments[1]["start_time"], segments[2]["start_time"])

    def test_02_distinct_repeated_signs_preserved_if_separated_by_pause(self):
        """
        Verify that if a user performs the same sign twice with a clear pause/drop between them,
        both instances are preserved in order (e.g. BYE ... pause ... BYE).
        """
        frames = [
            # First BYE
            {"timestamp_sec": 0.0, "hand_detected": True, "gesture": "BYE", "confidence": 0.95},
            {"timestamp_sec": 0.1, "hand_detected": True, "gesture": "BYE", "confidence": 0.95},
            # Pause of 0.6 seconds
            {"timestamp_sec": 0.2, "hand_detected": False, "gesture": "NONE", "confidence": 0.0},
            {"timestamp_sec": 0.4, "hand_detected": False, "gesture": "NONE", "confidence": 0.0},
            {"timestamp_sec": 0.6, "hand_detected": False, "gesture": "NONE", "confidence": 0.0},
            {"timestamp_sec": 0.8, "hand_detected": False, "gesture": "NONE", "confidence": 0.0},
            # Second BYE
            {"timestamp_sec": 0.9, "hand_detected": True, "gesture": "BYE", "confidence": 0.95},
            {"timestamp_sec": 1.0, "hand_detected": True, "gesture": "BYE", "confidence": 0.95},
        ]

        segments = self.processor._segment_multi_signs(
            frames,
            min_confidence=0.5,
            min_consecutive_frames=2,
            transition_gap_sec=0.25,
        )

        words = [s["word"] for s in segments]
        self.assertEqual(words, ["BYE", "BYE"], f"Expected ['BYE', 'BYE'], got {words}")

    def test_03_multi_sign_process_endpoint_validation(self):
        """
        Verify that POST /api/multi-sign/process checks file format and accepts video uploads.
        """
        # Unsupported file extension
        fake_txt = io.BytesIO(b"Hello world")
        res_bad = self.client.post(
            "/api/multi-sign/process",
            files={"file": ("test.txt", fake_txt, "text/plain")}
        )
        self.assertEqual(res_bad.status_code, 400)

        # Create temporary valid MP4 video
        tmp_dir = Path("storage/test_artifacts")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        test_video_path = tmp_dir / "test_multi_sign.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(test_video_path), fourcc, 10.0, (320, 240))
        for _ in range(20):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        with open(test_video_path, "rb") as f:
            res = self.client.post(
                "/api/multi-sign/process",
                files={"file": ("test_multi_sign.mp4", f, "video/mp4")}
            )

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("recognized_words", data)
        self.assertIn("word_sequence", data)
        self.assertIn("sentence", data)
        self.assertIn("video_url", data)

        # Cleanup
        try:
            test_video_path.unlink()
        except Exception:
            pass

    def test_04_existing_mode_unaffected(self):
        """
        Verify Mode 1 (Existing Sign Recognition endpoints & model health) continues to work.
        """
        # Health check
        res_health = self.client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json()["status"], "healthy")

        # Single sign prediction endpoint
        res_pred = self.client.post("/predict", json={
            "mode": "vision",
            "features": [0.0] * 126
        })
        self.assertEqual(res_pred.status_code, 200)
        self.assertIn("gesture", res_pred.json())
        self.assertIn("confidence", res_pred.json())


if __name__ == "__main__":
    unittest.main()

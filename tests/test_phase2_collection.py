#!/usr/bin/env python3
"""
Unit and Integration Test Suite for Phase 2: Live Webcam Collection Tool & Data Gathering
Verifies collector quality gates, duplicate filtering, file writing, metadata logging,
session validation, and manifest generation.

Usage:
    python -m unittest tests/test_phase2_collection.py
"""

import os
import sys
import glob
import shutil
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.data_collection.collector import DataCollector
from tools.generate_contact_sheets import create_contact_sheet
from tools.generate_phase2_report import generate_report


class TestPhase2Collection(unittest.TestCase):
    def setUp(self):
        self.root = PROJECT_ROOT
        self.test_raw = self.root / "datasets" / "hand_pose" / "test_raw"
        self.test_meta = self.root / "datasets" / "hand_pose" / "test_meta"
        self.test_rej = self.root / "datasets" / "hand_pose" / "test_rej"
        self.test_log = self.root / "logs" / "test_collection.log"

        self.collector = DataCollector(
            gestures_config_path=str(self.root / "configs" / "gestures.yaml"),
            raw_output_root=str(self.test_raw),
            metadata_output_root=str(self.test_meta),
            rejected_output_root=str(self.test_rej),
            log_file=str(self.test_log),
            similarity_threshold=0.95,
            min_capture_interval_s=0.05,
            min_sharpness=30.0,
            min_brightness=20.0,
            max_brightness=240.0,
        )

    def tearDown(self):
        # Close file handlers before unlinking
        if hasattr(self, "collector"):
            self.collector.close()

        # Clean up temporary test directories
        for p in [self.test_raw, self.test_meta, self.test_rej]:
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
        if self.test_log.exists():
            try:
                self.test_log.unlink(missing_ok=True)
            except Exception:
                pass

    def test_01_session_initialization_and_directory_structure(self):
        """Verify directory structure follows datasets/hand_pose/raw/{subject_id}/{session_id}/{gesture}."""
        session_dir = self.collector.start_session(
            subject_id="S01",
            session_id="SESSION_01",
            gesture="THANK_YOU",
            hand="RIGHT",
            lighting="INDOOR_NORMAL",
            camera_distance="MEDIUM",
            background="PLAIN_WALL",
        )
        self.assertTrue(session_dir.exists())
        expected_rel = Path("S01") / "SESSION_01" / "THANK_YOU"
        self.assertTrue(str(session_dir).endswith(str(expected_rel)))

    def test_02_deterministic_filename_generation(self):
        """Verify filenames follow {SUBJECT}_{SESSION}_{GESTURE}_{HAND}_{FRAME_ID:06d}.jpg."""
        self.collector.start_session(
            subject_id="S02",
            session_id="SESSION_03",
            gesture="MORNING",
            hand="LEFT",
        )
        # Create clear frame
        frame = np.full((240, 320, 3), 120, dtype=np.uint8)
        # Draw high contrast lines for sharpness
        frame[50:70, :, :] = 255
        frame[100:120, :, :] = 0

        saved, reason, record = self.collector.capture_frame(frame, force=True)
        self.assertTrue(saved)
        self.assertIn("S02_SESSION_03_MORNING_LEFT_000001.jpg", record["filename"])

    def test_03_duplicate_frame_rejection(self):
        """Verify identical or near-identical frames are routed to rejected/duplicates/."""
        self.collector.start_session(
            subject_id="S01",
            session_id="SESSION_01",
            gesture="HEARING",
        )
        frame1 = np.full((240, 320, 3), 120, dtype=np.uint8)
        frame1[40:80, :, :] = 250

        # Frame 1: Accepted
        s1, r1, rec1 = self.collector.capture_frame(frame1, force=True)
        self.assertTrue(s1)

        # Frame 2: Identical, should be rejected as duplicate
        import time
        time.sleep(0.06)
        s2, r2, rec2 = self.collector.capture_frame(frame1.copy(), force=False)
        self.assertFalse(s2)
        self.assertEqual(rec2["rejection_reason"], "duplicate")

        # Verify duplicate file was saved in rejected/duplicates/
        dup_files = list((self.test_rej / "duplicates").glob("*.jpg"))
        self.assertGreater(len(dup_files), 0)

    def test_04_quality_gate_blur_and_exposure(self):
        """Verify blurry and extreme exposure frames are routed to rejected folders."""
        self.collector.start_session(
            subject_id="S01",
            session_id="SESSION_01",
            gesture="BYE",
        )

        # Extremely dark frame (brightness < min_brightness)
        dark_frame = np.full((240, 320, 3), 5, dtype=np.uint8)
        s_dark, r_dark, rec_dark = self.collector.capture_frame(dark_frame, force=False)
        self.assertFalse(s_dark)
        self.assertEqual(rec_dark["rejection_reason"], "underexposed")

        # Uniform gray frame (sharpness == 0 < min_sharpness)
        flat_frame = np.full((240, 320, 3), 128, dtype=np.uint8)
        import time
        time.sleep(0.06)
        s_flat, r_flat, rec_flat = self.collector.capture_frame(flat_frame, force=False)
        self.assertFalse(s_flat)
        self.assertEqual(rec_flat["rejection_reason"], "blurry")

    def test_05_metadata_jsonl_and_csv_generation(self):
        """Verify global metadata.jsonl and metadata.csv receive exact records."""
        self.collector.start_session(
            subject_id="S03",
            session_id="SESSION_01",
            gesture="NAME",
            hand="RIGHT",
        )
        frame = np.full((240, 320, 3), 110, dtype=np.uint8)
        frame[20:60, :, :] = 240
        self.collector.capture_frame(frame, force=True)
        summary = self.collector.end_session()

        jsonl_path = self.test_meta / "metadata.jsonl"
        csv_path = self.test_meta / "metadata.csv"
        manifest_path = self.test_meta / "collection_manifest.json"

        self.assertTrue(jsonl_path.exists())
        self.assertTrue(csv_path.exists())
        self.assertTrue(manifest_path.exists())

        # Check JSONL
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)

    def test_06_contact_sheet_generation(self):
        """Verify contact sheet grid generates without error."""
        # Create 4 dummy images
        img_paths = []
        dummy_dir = self.test_raw / "dummy_images"
        dummy_dir.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            im = Image.new("RGB", (320, 240), color=(i * 50, 100, 150))
            p = dummy_dir / f"test_{i:02d}.jpg"
            im.save(p)
            img_paths.append(p)

        out_sheet = self.root / "reports" / "contact_sheets" / "test_sheet.jpg"
        success = create_contact_sheet(img_paths, out_sheet, title="TEST SHEET", cols=2, max_images=4)
        self.assertTrue(success)
        self.assertTrue(out_sheet.exists())
        out_sheet.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

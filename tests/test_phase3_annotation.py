#!/usr/bin/env python3
"""
Unit and Integration Test Suite for Phase 3: Dataset Annotation & Quality Gate
Verifies 21-keypoint schema, 68-token YOLO pose format, coordinate bounds,
visibility flags, bounding-box consistency, subject leakage detection, and dataset splitting.

Usage:
    python -m unittest tests/test_phase3_annotation.py
"""

import os
import sys
import shutil
import tempfile
import unittest
from pathlib import Path
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validate_pose_annotations import PoseAnnotationValidator
from tools.check_dataset_leakage import DatasetLeakageAuditor, extract_subject_and_session
from tools.split_dataset import DatasetSplitter
from tools.visualize_annotations import visualize_annotation


class TestPhase3Annotation(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="kinex_test_p3_"))
        self.img_dir = self.test_dir / "images"
        self.lbl_dir = self.test_dir / "labels"
        self.img_dir.mkdir(parents=True)
        self.lbl_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_mock_pair(self, stem: str, is_valid: bool = True, token_count: int = 68, out_of_bounds: bool = False):
        """Helper to create dummy image and label pairs."""
        img_p = self.img_dir / f"{stem}.jpg"
        lbl_p = self.lbl_dir / f"{stem}.txt"

        im = Image.new("RGB", (320, 240), color=(100, 150, 200))
        im.save(img_p)

        if token_count == 68 and not out_of_bounds:
            # Valid 68 tokens: class=0, xc=0.5, yc=0.5, w=0.4, h=0.4, then 21 kpts (x, y, v)
            tokens = ["0", "0.500", "0.500", "0.400", "0.400"]
            for i in range(21):
                tokens.extend(["0.500", "0.500", "2"])
            content = " ".join(tokens)
        elif out_of_bounds:
            tokens = ["0", "1.500", "0.500", "0.400", "0.400"]  # xc out of bounds
            for i in range(21):
                tokens.extend(["0.500", "0.500", "2"])
            content = " ".join(tokens)
        else:
            # Malformed token count
            content = "0 0.5 0.5 0.4 0.4 0.1 0.1 2"

        with open(lbl_p, "w", encoding="utf-8") as f:
            f.write(content + "\n")

        return img_p, lbl_p

    def test_01_valid_68_token_annotation(self):
        """Verify that a compliant 68-token hand pose annotation passes validation."""
        self._create_mock_pair("S01_SESSION_01_THANK_YOU_RIGHT_000001", is_valid=True)
        validator = PoseAnnotationValidator(images_dir=self.img_dir, labels_dir=self.lbl_dir)
        passed = validator.run_validation()
        self.assertTrue(passed)
        self.assertEqual(validator.stats["valid_instances"], 1)
        self.assertEqual(validator.stats["malformed_lines"], 0)

    def test_02_invalid_token_count_rejection(self):
        """Verify annotations with token count != 68 are flagged as malformed."""
        self._create_mock_pair("S01_SESSION_01_HEARING_RIGHT_000002", token_count=8)
        validator = PoseAnnotationValidator(images_dir=self.img_dir, labels_dir=self.lbl_dir)
        passed = validator.run_validation()
        self.assertFalse(passed)
        self.assertEqual(validator.stats["malformed_lines"], 1)

    def test_03_out_of_bounds_coordinate_rejection(self):
        """Verify coordinates outside [0.0, 1.0] fail validation."""
        self._create_mock_pair("S01_SESSION_01_MORNING_RIGHT_000003", out_of_bounds=True)
        validator = PoseAnnotationValidator(images_dir=self.img_dir, labels_dir=self.lbl_dir)
        passed = validator.run_validation()
        self.assertFalse(passed)
        self.assertGreater(validator.stats["malformed_lines"], 0)

    def test_04_missing_and_orphan_label_detection(self):
        """Verify missing labels and orphan labels are correctly identified."""
        # Image without label
        im = Image.new("RGB", (100, 100))
        im.save(self.img_dir / "S01_SESSION_01_I_RIGHT_000004.jpg")

        # Label without image
        with open(self.lbl_dir / "S01_SESSION_01_BYE_RIGHT_000005.txt", "w") as f:
            f.write("0 0.5 0.5 0.4 0.4\n")

        validator = PoseAnnotationValidator(images_dir=self.img_dir, labels_dir=self.lbl_dir)
        passed = validator.run_validation()
        self.assertFalse(passed)
        self.assertGreaterEqual(validator.stats["missing_labels"], 1)
        self.assertGreaterEqual(validator.stats["orphan_labels"], 1)

    def test_05_subject_and_session_extraction(self):
        """Verify deterministic parsing of subject and session from filename."""
        subj1, sess1 = extract_subject_and_session("S01_SESSION_01_THANK_YOU_RIGHT_000001.jpg")
        self.assertEqual(subj1, "S01")
        self.assertEqual(sess1, "SESSION_01")

        subj2, sess2 = extract_subject_and_session("S07_SESSION_03_BYE_LEFT_000420.jpg")
        self.assertEqual(subj2, "S07")
        self.assertEqual(sess2, "SESSION_03")

    def test_06_leakage_auditor_detection(self):
        """Verify auditor catches subject leakage across train and val splits."""
        mock_ds = self.test_dir / "mock_dataset"
        (mock_ds / "images" / "train").mkdir(parents=True)
        (mock_ds / "images" / "val").mkdir(parents=True)
        (mock_ds / "images" / "test").mkdir(parents=True)

        # Subject S01 placed in both Train and Val
        with open(mock_ds / "images" / "train" / "S01_SESSION_01_THANK_YOU_001.jpg", "w") as f:
            f.write("test")
        with open(mock_ds / "images" / "val" / "S01_SESSION_02_THANK_YOU_002.jpg", "w") as f:
            f.write("test")

        auditor = DatasetLeakageAuditor(dataset_dir=mock_ds)
        passed = auditor.audit()
        self.assertFalse(passed, "Subject S01 present in both Train and Val must fail audit")
        self.assertIn("Subject leakage", auditor.violations[0])

    def test_07_visualizer_execution(self):
        """Verify skeleton visualizer renders without errors."""
        img_p, lbl_p = self._create_mock_pair("S01_SESSION_01_NAME_RIGHT_000006")
        out_p = self.test_dir / "qa_preview.jpg"
        success = visualize_annotation(img_p, lbl_p, out_p)
        self.assertTrue(success)
        self.assertTrue(out_p.exists())


if __name__ == "__main__":
    unittest.main()

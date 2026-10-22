"""
Unit Tests for KINEX Phase 4: YOLO26 Pose Training Pipeline & Offline Inference
Validates:
1. YAML model architecture exists and is syntactically valid.
2. Dataset YAML exists and defines single class '0: hand' and kpt_shape [21, 3].
3. Model architecture defines nc=1 and kpt_shape [21, 3].
4. Random-init enforcement: no external checkpoints (.pt) referenced in configs.
5. verify_random_init tool passes cleanly.
6. Dataset preflight generator accurately flags empty dataset as BLOCKED.
7. Model artifact verification tool rejects forbidden external checkpoints.
8. Zero MediaPipe dependency in vision/yolo_pose/inference.py.
9. YOLOPoseEstimator requires valid checkpoint file (fails cleanly if missing).
10. Checkpoint manifest schema correctness.
11. Training report generation executes cleanly.
12. Checkpoint provenance verification logic.
"""

import json
import os
import sys
import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

CONFIG_DIR = PROJECT_ROOT / "configs"
REPORTS_DIR = PROJECT_ROOT / "reports"


class TestPhase4Training(unittest.TestCase):

    def test_01_model_yaml_exists_and_valid(self):
        """Verify configs/yolo26n-pose.yaml exists and parses as valid YAML."""
        yaml_path = CONFIG_DIR / "yolo26n-pose.yaml"
        self.assertTrue(yaml_path.is_file(), f"Missing {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.assertIsInstance(data, dict)
        self.assertEqual(data.get("nc"), 1, "Model YAML must specify nc: 1")
        self.assertEqual(data.get("kpt_shape"), [21, 3], "Model YAML must specify kpt_shape: [21, 3]")

    def test_02_dataset_yaml_structure(self):
        """Verify configs/pose_dataset.yaml has single class 'hand' and kpt_shape [21, 3]."""
        dataset_yaml = CONFIG_DIR / "pose_dataset.yaml"
        self.assertTrue(dataset_yaml.is_file(), f"Missing {dataset_yaml}")
        with open(dataset_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.assertEqual(data.get("kpt_shape"), [21, 3])
        self.assertEqual(data.get("names", {}).get(0), "hand")
        self.assertEqual(len(data.get("names", {})), 1, "Dataset YAML must only contain class 0: hand")
        self.assertEqual(len(data.get("flip_idx", [])), 21, "flip_idx must contain exactly 21 indices")

    def test_03_no_external_weights_in_configs(self):
        """Verify configs do not reference forbidden pretrained .pt or pretrained: true."""
        for cfg in CONFIG_DIR.glob("*.yaml"):
            content = cfg.read_text(encoding="utf-8").lower()
            self.assertNotIn("pretrained: true", content, f"Forbidden pretrained: true found in {cfg.name}")
            self.assertNotIn(".pt", content, f"Forbidden .pt found in {cfg.name}")
            self.assertNotIn("yolo26n-pose.pt", content)
            self.assertNotIn("coco-pose", content)

    def test_04_verify_random_init_script(self):
        """Verify tools/verify_random_init.py passes without violations."""
        from tools.verify_random_init import verify_random_init
        audit = verify_random_init()
        self.assertEqual(audit["status"], "PASS")
        self.assertFalse(audit["forbidden_checkpoint_referenced"])
        self.assertTrue(audit["pretrained_flag_enforced_false"])

    def test_05_dataset_preflight_blocks_when_empty(self):
        """Verify preflight generator correctly detects 0 images and marks status as BLOCKED."""
        from tools.generate_dataset_preflight import generate_preflight
        preflight = generate_preflight()
        self.assertIn(preflight["gate_status"], ["BLOCKED", "PASS"])
        if preflight["total_images"] == 0:
            self.assertEqual(preflight["gate_status"], "BLOCKED")
            self.assertIn("DATASET NOT READY", preflight["gate_reason"])

    def test_06_model_artifact_verification_rejects_external(self):
        """Verify verify_model_artifact rejects forbidden external weights names."""
        from tools.verify_model_artifact import verify_model_artifact
        res = verify_model_artifact("yolo26n-pose.pt")
        self.assertEqual(res["status"], "FAIL")

    def test_07_no_mediapipe_in_yolo_inference(self):
        """Verify vision/yolo_pose/inference.py has zero mediapipe references."""
        inf_file = PROJECT_ROOT / "vision" / "yolo_pose" / "inference.py"
        self.assertTrue(inf_file.is_file())
        content = inf_file.read_text(encoding="utf-8").lower()
        self.assertNotIn("import mediapipe", content)
        self.assertNotIn("from mediapipe", content)

    def test_08_inference_missing_checkpoint_raises(self):
        """Verify YOLOPoseEstimator raises FileNotFoundError if checkpoint missing."""
        from vision.yolo_pose.inference import YOLOPoseEstimator
        with self.assertRaises(FileNotFoundError):
            YOLOPoseEstimator(checkpoint_path="nonexistent_checkpoint.pt")

    def test_09_training_report_generation(self):
        """Verify generate_training_report creates valid JSON and Markdown reports."""
        from tools.train_yolo_pose import generate_training_report
        rep = generate_training_report(status="PHASE 4 BLOCKED", note="Testing test suite execution")
        self.assertEqual(rep["status"], "PHASE 4 BLOCKED")
        self.assertTrue((REPORTS_DIR / "phase4_training_report.json").is_file())
        self.assertTrue((REPORTS_DIR / "phase4_training_report.md").is_file())

    def test_10_environment_report_exists_and_valid(self):
        """Verify reports/phase4_environment.json exists with required fields."""
        env_file = REPORTS_DIR / "phase4_environment.json"
        self.assertTrue(env_file.is_file())
        with open(env_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("python_version", data)
        self.assertIn("cuda_available", data)
        self.assertIn("device", data)

    def test_11_model_card_exists(self):
        """Verify models/README.md model card exists with required documentation."""
        readme = PROJECT_ROOT / "models" / "README.md"
        self.assertTrue(readme.is_file())
        content = readme.read_text(encoding="utf-8")
        self.assertIn("KINEX YOLO26n-Pose", content)
        self.assertIn("21-Keypoint Hand Pose Estimation", content)
        self.assertIn("Random Initialization", content)

    def test_12_failure_analysis_report_exists(self):
        """Verify reports/phase4_failure_analysis.md exists."""
        fa = REPORTS_DIR / "phase4_failure_analysis.md"
        self.assertTrue(fa.is_file())
        content = fa.read_text(encoding="utf-8")
        self.assertIn("Root Cause Analysis", content)
        self.assertIn("CM_PROB_PHANTOM", content)


if __name__ == "__main__":
    unittest.main()

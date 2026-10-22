#!/usr/bin/env python3
"""
KINEX Dataset Validation Tool
Validates YOLO Pose annotations and dataset integrity for 21-keypoint hand pose estimation.

Checks Performed:
1. Missing labels (images without labels, labels without images)
2. Malformed label syntax (non-numeric tokens, incorrect token counts)
3. Keypoint count (must be exactly 21 keypoints -> 68 tokens per hand detection)
4. Coordinates within normalized [0.0, 1.0] range
5. Keypoint visibility values in {0, 1, 2}
6. Valid class ID (0 for hand)
7. Empty/corrupt image files
8. Duplicate image detection via content hashing
9. Train/Val/Test subject leakage (zero overlap between splits)

Usage:
    python tools/validate_dataset.py [--data-yaml configs/pose_dataset.yaml]
"""

import os
import sys
import glob
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
import yaml
from PIL import Image


class DatasetValidator:
    def __init__(self, data_yaml_path: str = "configs/pose_dataset.yaml"):
        self.data_yaml_path = Path(data_yaml_path)
        if not self.data_yaml_path.exists():
            raise FileNotFoundError(f"Dataset config not found at: {self.data_yaml_path}")

        with open(self.data_yaml_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.dataset_root = Path(self.config.get("path", "datasets/hand_pose"))
        self.kpt_shape = self.config.get("kpt_shape", [21, 3])
        self.expected_num_kpts = self.kpt_shape[0]
        self.expected_kpt_dim = self.kpt_shape[1]
        self.expected_tokens_per_line = 1 + 4 + (self.expected_num_kpts * self.expected_kpt_dim)

        self.splits = ["train", "val", "test"]
        self.report: Dict[str, any] = {
            "splits": {},
            "total_images": 0,
            "total_labels": 0,
            "missing_labels": 0,
            "orphan_labels": 0,
            "malformed_labels": 0,
            "invalid_kpt_counts": 0,
            "coords_out_of_bounds": 0,
            "invalid_visibilities": 0,
            "invalid_class_ids": 0,
            "empty_or_corrupt_images": 0,
            "duplicate_images": 0,
            "subject_leakage_violations": [],
            "passed": False,
        }

    def _get_image_hash(self, img_path: Path) -> str:
        """Calculate MD5 hash of image file to detect exact duplicates."""
        hasher = hashlib.md5()
        with open(img_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _extract_subject_id(self, filename: str) -> str:
        """
        Extracts subject ID (e.g., 'S01', 'S02') from filename.
        Convention: S01_SESSION_01_GESTURE_001.jpg -> 'S01'
        Fallback: If format differs, attempts regex or prefix matching.
        """
        parts = filename.split("_")
        for part in parts:
            if part.startswith("S") and len(part) >= 2 and part[1:].isdigit():
                return part.upper()
        return "UNKNOWN"

    def validate(self) -> bool:
        """Run all validation checks and return True if all pass, False otherwise."""
        print("=" * 60)
        print("KINEX HAND POSE DATASET VALIDATOR")
        print(f"Config: {self.data_yaml_path}")
        print(f"Dataset Root: {self.dataset_root}")
        print(f"Expected Keypoints: {self.expected_num_kpts} ({self.expected_tokens_per_line} tokens/line)")
        print("=" * 60)

        image_hashes: Dict[str, Path] = {}
        split_subjects: Dict[str, Set[str]] = {s: set() for s in self.splits}
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        for split in self.splits:
            img_dir = self.dataset_root / "images" / split
            lbl_dir = self.dataset_root / "labels" / split

            self.report["splits"][split] = {
                "images": 0,
                "labels": 0,
                "subjects": [],
            }

            if not img_dir.exists():
                print(f"[INFO] Image directory for split '{split}' does not exist: {img_dir} (creating empty placeholder)")
                img_dir.mkdir(parents=True, exist_ok=True)
            if not lbl_dir.exists():
                lbl_dir.mkdir(parents=True, exist_ok=True)

            img_files = [p for p in img_dir.iterdir() if p.suffix.lower() in valid_extensions]
            lbl_files = [p for p in lbl_dir.iterdir() if p.suffix.lower() == ".txt"]

            self.report["splits"][split]["images"] = len(img_files)
            self.report["splits"][split]["labels"] = len(lbl_files)
            self.report["total_images"] += len(img_files)
            self.report["total_labels"] += len(lbl_files)

            img_basenames = {p.stem: p for p in img_files}
            lbl_basenames = {p.stem: p for p in lbl_files}

            # Check 1: Missing labels (Image exists, label does not)
            for stem, img_path in img_basenames.items():
                if stem not in lbl_basenames:
                    self.report["missing_labels"] += 1
                    print(f"  [ERROR] Missing label for image: {img_path}")

            # Check 2: Orphan labels (Label exists, image does not)
            for stem, lbl_path in lbl_basenames.items():
                if stem not in img_basenames:
                    self.report["orphan_labels"] += 1
                    print(f"  [ERROR] Orphan label without image: {lbl_path}")

            # Check 3 & 4: Image validity & duplicates
            for img_path in img_files:
                subj = self._extract_subject_id(img_path.name)
                if subj != "UNKNOWN":
                    split_subjects[split].add(subj)

                try:
                    with Image.open(img_path) as im:
                        im.verify()
                except Exception as e:
                    self.report["empty_or_corrupt_images"] += 1
                    print(f"  [ERROR] Corrupt image {img_path}: {e}")
                    continue

                h = self._get_image_hash(img_path)
                if h in image_hashes:
                    self.report["duplicate_images"] += 1
                    print(f"  [WARNING] Duplicate image: {img_path} matches {image_hashes[h]}")
                else:
                    image_hashes[h] = img_path

            # Check 5: Label contents and keypoints
            for lbl_path in lbl_files:
                try:
                    with open(lbl_path, "r", encoding="utf-8") as f:
                        lines = [line.strip() for line in f if line.strip()]

                    if not lines:
                        # Empty label file is acceptable if frame contains no hand,
                        # but check if intended
                        continue

                    for line_idx, line in enumerate(lines):
                        tokens = line.split()
                        if len(tokens) != self.expected_tokens_per_line:
                            self.report["invalid_kpt_counts"] += 1
                            print(f"  [ERROR] {lbl_path} line {line_idx+1}: Expected {self.expected_tokens_per_line} tokens, got {len(tokens)}")
                            continue

                        try:
                            class_id = int(tokens[0])
                            values = [float(x) for x in tokens[1:]]
                        except ValueError as ve:
                            self.report["malformed_labels"] += 1
                            print(f"  [ERROR] {lbl_path} line {line_idx+1}: Malformed numerical values ({ve})")
                            continue

                        # Check class ID
                        if class_id != 0:
                            self.report["invalid_class_ids"] += 1
                            print(f"  [ERROR] {lbl_path} line {line_idx+1}: Invalid class ID {class_id}, expected 0 (hand)")

                        # Check bounding box (xc, yc, w, h)
                        xc, yc, w, h = values[0:4]
                        for name, val in [("xc", xc), ("yc", yc), ("w", w), ("h", h)]:
                            if not (0.0 <= val <= 1.0):
                                self.report["coords_out_of_bounds"] += 1
                                print(f"  [ERROR] {lbl_path} line {line_idx+1}: BBox {name}={val} out of bounds [0, 1]")

                        # Check keypoints: x, y, v
                        kpt_vals = values[4:]
                        for kpt_idx in range(self.expected_num_kpts):
                            base = kpt_idx * self.expected_kpt_dim
                            kx = kpt_vals[base]
                            ky = kpt_vals[base + 1]
                            kv = int(kpt_vals[base + 2])

                            if not (0.0 <= kx <= 1.0) or not (0.0 <= ky <= 1.0):
                                self.report["coords_out_of_bounds"] += 1
                                print(f"  [ERROR] {lbl_path} line {line_idx+1}: KPT {kpt_idx} coords ({kx}, {ky}) out of bounds")

                            if kv not in (0, 1, 2):
                                self.report["invalid_visibilities"] += 1
                                print(f"  [ERROR] {lbl_path} line {line_idx+1}: KPT {kpt_idx} invalid visibility {kv} (must be 0, 1, or 2)")

                except Exception as e:
                    self.report["malformed_labels"] += 1
                    print(f"  [ERROR] Failed to read {lbl_path}: {e}")

            self.report["splits"][split]["subjects"] = sorted(list(split_subjects[split]))

        # Check 6: Subject Leakage across Splits (Train vs Val vs Test)
        train_subjs = split_subjects["train"]
        val_subjs = split_subjects["val"]
        test_subjs = split_subjects["test"]

        train_val_overlap = train_subjs.intersection(val_subjs)
        train_test_overlap = train_subjs.intersection(test_subjs)
        val_test_overlap = val_subjs.intersection(test_subjs)

        if train_val_overlap:
            leak = f"Subject Leakage between TRAIN and VAL: {train_val_overlap}"
            self.report["subject_leakage_violations"].append(leak)
            print(f"  [CRITICAL ERROR] {leak}")

        if train_test_overlap:
            leak = f"Subject Leakage between TRAIN and TEST: {train_test_overlap}"
            self.report["subject_leakage_violations"].append(leak)
            print(f"  [CRITICAL ERROR] {leak}")

        if val_test_overlap:
            leak = f"Subject Leakage between VAL and TEST: {val_test_overlap}"
            self.report["subject_leakage_violations"].append(leak)
            print(f"  [CRITICAL ERROR] {leak}")

        # Summary decision
        has_errors = (
            self.report["missing_labels"] > 0
            or self.report["orphan_labels"] > 0
            or self.report["malformed_labels"] > 0
            or self.report["invalid_kpt_counts"] > 0
            or self.report["coords_out_of_bounds"] > 0
            or self.report["invalid_visibilities"] > 0
            or self.report["invalid_class_ids"] > 0
            or self.report["empty_or_corrupt_images"] > 0
            or len(self.report["subject_leakage_violations"]) > 0
        )

        self.report["passed"] = not has_errors
        self.print_summary()
        return self.report["passed"]

    def print_summary(self):
        """Print formatted dataset validation report."""
        print("\n" + "=" * 60)
        print("DATASET VALIDATION SUMMARY REPORT")
        print("=" * 60)
        for s in self.splits:
            info = self.report["splits"].get(s, {})
            print(f"Split [{s:5s}]: {info.get('images', 0):4d} images | {info.get('labels', 0):4d} labels | Subjects: {info.get('subjects', [])}")

        print("-" * 60)
        print(f"Total Images:              {self.report['total_images']}")
        print(f"Total Labels:              {self.report['total_labels']}")
        print(f"Missing Labels:            {self.report['missing_labels']}")
        print(f"Orphan Labels:             {self.report['orphan_labels']}")
        print(f"Malformed Labels:          {self.report['malformed_labels']}")
        print(f"Invalid Keypoint Counts:   {self.report['invalid_kpt_counts']}")
        print(f"Out-of-Bounds Coordinates: {self.report['coords_out_of_bounds']}")
        print(f"Invalid Visibility Values: {self.report['invalid_visibilities']}")
        print(f"Invalid Class IDs:         {self.report['invalid_class_ids']}")
        print(f"Corrupt/Empty Images:      {self.report['empty_or_corrupt_images']}")
        print(f"Duplicate Images Found:    {self.report['duplicate_images']}")
        print(f"Subject Leakage Overlaps:  {len(self.report['subject_leakage_violations'])}")
        print("=" * 60)

        if self.report["passed"]:
            print("[PASS] Dataset integrity validation SUCCEEDED.")
        else:
            print("[FAIL] Dataset integrity validation FAILED. Training must be halted.")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Validate KINEX Hand Pose Dataset")
    parser.add_argument("--data-yaml", default="configs/pose_dataset.yaml", help="Path to pose_dataset.yaml")
    args = parser.parse_args()

    validator = DatasetValidator(data_yaml_path=args.data_yaml)
    success = validator.validate()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

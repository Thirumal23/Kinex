#!/usr/bin/env python3
"""
KINEX Hand Pose Annotation Validator (21 Keypoints)
Performs strict syntax, topology, coordinate boundary, visibility,
and bounding-box consistency validation on YOLO Pose annotation files.

Expected Format per Hand Instance (68 tokens):
  <class_id=0> <xc> <yc> <w> <h> <x0> <y0> <v0> ... <x20> <y20> <v20>

Usage:
    python tools/validate_pose_annotations.py [--dataset-dir datasets/hand_pose]
"""

import os
import sys
import csv
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PoseAnnotationValidator:
    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        expected_kpts: int = 21,
        review_queue_path: Path = PROJECT_ROOT / "reports" / "annotation_review_queue.csv",
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.expected_kpts = expected_kpts
        self.expected_tokens = 1 + 4 + (expected_kpts * 3)  # 68 tokens
        self.review_queue_path = review_queue_path
        self.review_queue_path.parent.mkdir(parents=True, exist_ok=True)

        self.stats = {
            "images_found": 0,
            "labels_found": 0,
            "valid_instances": 0,
            "missing_labels": 0,
            "orphan_labels": 0,
            "malformed_lines": 0,
            "invalid_class_ids": 0,
            "invalid_token_counts": 0,
            "coords_out_of_bounds": 0,
            "invalid_visibilities": 0,
            "nan_or_inf_values": 0,
            "invalid_bbox_dimensions": 0,
            "bbox_keypoint_mismatches": 0,
            "corrupt_images": 0,
            "review_queue_count": 0,
            "passed": False,
        }
        self.review_queue: List[Dict[str, Any]] = []

    def validate_instance(self, line: str, line_no: int, label_path: Path) -> Tuple[bool, str, Dict[str, Any]]:
        tokens = line.strip().split()
        if len(tokens) != self.expected_tokens:
            return False, f"Expected {self.expected_tokens} tokens, got {len(tokens)}", {}

        # Parse numerical values
        try:
            class_id = int(tokens[0])
            numeric_vals = [float(x) for x in tokens[1:]]
        except ValueError as ve:
            return False, f"Non-numeric field encountered ({ve})", {}

        # Check for NaN / Inf
        for idx, val in enumerate(numeric_vals):
            if math.isnan(val) or math.isinf(val):
                return False, f"NaN or Inf detected at token position {idx + 1}", {}

        # Check class ID (hand = 0)
        if class_id != 0:
            return False, f"Invalid class ID {class_id}, expected 0 (hand)", {}

        # Bounding box [xc, yc, w, h]
        xc, yc, w, h = numeric_vals[:4]
        if w <= 0.0 or h <= 0.0:
            return False, f"Invalid bbox dimensions w={w}, h={h} (must be > 0)", {}

        for name, val in [("xc", xc), ("yc", yc), ("w", w), ("h", h)]:
            if not (0.0 <= val <= 1.0):
                return False, f"Bbox {name}={val} out of bounds [0.0, 1.0]", {}

        # Keypoints: [x, y, v] for 21 points
        kpt_vals = numeric_vals[4:]
        kpts = []
        visible_kpts_outside_box = 0

        # Calculate bounding box extents [xmin, ymin, xmax, ymax]
        xmin = xc - (w / 2.0)
        ymin = yc - (h / 2.0)
        xmax = xc + (w / 2.0)
        ymax = yc + (h / 2.0)
        box_margin = 0.10  # 10% tolerance for fingertips extending near box edge

        for i in range(self.expected_kpts):
            kx = kpt_vals[i * 3]
            ky = kpt_vals[i * 3 + 1]
            kv = int(kpt_vals[i * 3 + 2])

            if not (0.0 <= kx <= 1.0) or not (0.0 <= ky <= 1.0):
                return False, f"Keypoint {i} coords ({kx}, {ky}) out of bounds [0.0, 1.0]", {}

            if kv not in (0, 1, 2):
                return False, f"Keypoint {i} invalid visibility {kv} (must be 0, 1, or 2)", {}

            kpts.append((kx, ky, kv))

            # Quality check: If keypoint is marked visible (v > 0), verify it lies reasonably within or near bbox
            if kv > 0:
                if (kx < (xmin - box_margin)) or (kx > (xmax + box_margin)) or \
                   (ky < (ymin - box_margin)) or (ky > (ymax + box_margin)):
                    visible_kpts_outside_box += 1

        details = {
            "class_id": class_id,
            "bbox": (xc, yc, w, h),
            "keypoints": kpts,
            "kpts_outside_box": visible_kpts_outside_box,
        }

        # Soft outlier detection: if >3 keypoints are significantly outside the bounding box
        if visible_kpts_outside_box >= 3:
            return True, f"WARNING: {visible_kpts_outside_box} keypoints located far outside bbox", details

        return True, "OK", details

    def run_validation(self) -> bool:
        valid_img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        img_files = {p.stem: p for p in self.images_dir.rglob("*") if p.suffix.lower() in valid_img_exts}
        lbl_files = {p.stem: p for p in self.labels_dir.rglob("*.txt")}

        self.stats["images_found"] = len(img_files)
        self.stats["labels_found"] = len(lbl_files)

        # 1. Missing labels (image exists without label)
        for stem, img_path in img_files.items():
            if stem not in lbl_files:
                self.stats["missing_labels"] += 1
                self.review_queue.append({
                    "error_code": "E009",
                    "file": str(img_path),
                    "reason": "Missing label file for image",
                })

        # 2. Orphan labels (label exists without image)
        for stem, lbl_path in lbl_files.items():
            if stem not in img_files:
                self.stats["orphan_labels"] += 1
                self.review_queue.append({
                    "error_code": "E011",
                    "file": str(lbl_path),
                    "reason": "Orphan label file without corresponding image",
                })

        # 3. Image integrity check
        for stem, img_path in img_files.items():
            try:
                with Image.open(img_path) as im:
                    im.verify()
            except Exception as e:
                self.stats["corrupt_images"] += 1
                self.review_queue.append({
                    "error_code": "E011",
                    "file": str(img_path),
                    "reason": f"Corrupt image file: {e}",
                })

        # 4. Label contents validation
        for stem, lbl_path in lbl_files.items():
            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip()]

                if not lines:
                    # Empty label file
                    continue

                for line_idx, line in enumerate(lines):
                    is_valid, msg, details = self.validate_instance(line, line_idx + 1, lbl_path)
                    if not is_valid:
                        self.stats["malformed_lines"] += 1
                        self.review_queue.append({
                            "error_code": "E010",
                            "file": f"{lbl_path.name}:{line_idx + 1}",
                            "reason": msg,
                        })
                    else:
                        self.stats["valid_instances"] += 1
                        if "WARNING" in msg:
                            self.stats["bbox_keypoint_mismatches"] += 1
                            self.review_queue.append({
                                "error_code": "E001",
                                "file": f"{lbl_path.name}:{line_idx + 1}",
                                "reason": msg,
                            })

            except Exception as e:
                self.stats["malformed_lines"] += 1
                self.review_queue.append({
                    "error_code": "E011",
                    "file": str(lbl_path),
                    "reason": f"Failed to read label file: {e}",
                })

        # Write Review Queue CSV
        self.stats["review_queue_count"] = len(self.review_queue)
        with open(self.review_queue_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["error_code", "file", "reason"])
            writer.writeheader()
            for row in self.review_queue:
                writer.writerow(row)

        has_critical_errors = (
            self.stats["missing_labels"] > 0
            or self.stats["orphan_labels"] > 0
            or self.stats["malformed_lines"] > 0
            or self.stats["corrupt_images"] > 0
        )

        self.stats["passed"] = not has_critical_errors
        self.print_report()
        return self.stats["passed"]

    def print_report(self):
        print("\n" + "=" * 60)
        print("KINEX HAND POSE ANNOTATION VALIDATION REPORT")
        print("=" * 60)
        print(f"Images Evaluated:          {self.stats['images_found']}")
        print(f"Labels Evaluated:          {self.stats['labels_found']}")
        print(f"Valid Hand Instances:      {self.stats['valid_instances']}")
        print(f"Missing Labels:            {self.stats['missing_labels']}")
        print(f"Orphan Labels:             {self.stats['orphan_labels']}")
        print(f"Malformed Lines:           {self.stats['malformed_lines']}")
        print(f"Corrupt Images:            {self.stats['corrupt_images']}")
        print(f"Bbox-Keypoint Mismatches:  {self.stats['bbox_keypoint_mismatches']}")
        print(f"Review Queue Items:        {self.stats['review_queue_count']}")
        print("-" * 60)
        if self.stats["passed"]:
            print("[PASS] Pose annotation syntax & consistency validation PASSED.")
        else:
            print("[FAIL] Pose annotation validation FAILED. Quality gates not met.")
        print(f"Review Queue saved to: {self.review_queue_path}")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Validate KINEX Hand Pose Annotations")
    parser.add_argument("--images-dir", default="datasets/hand_pose/images", help="Root directory containing images")
    parser.add_argument("--labels-dir", default="datasets/hand_pose/labels", help="Root directory containing labels")
    args = parser.parse_args()

    validator = PoseAnnotationValidator(
        images_dir=Path(args.images_dir),
        labels_dir=Path(args.labels_dir),
    )
    success = validator.run_validation()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

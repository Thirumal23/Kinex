#!/usr/bin/env python3
"""
KINEX Phase 3 Annotation & Quality Gate Reporting Engine
Evaluates dataset readiness, annotation statistics, quality gates, and subject leakage.

Outputs:
  reports/phase3_annotation_report.json
  reports/phase3_annotation_report.csv
  reports/phase3_annotation_report.md
  datasets/hand_pose/dataset_manifest.json
"""

import os
import sys
import csv
import json
import datetime
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validate_pose_annotations import PoseAnnotationValidator
from tools.check_dataset_leakage import DatasetLeakageAuditor


def generate_phase3_report():
    dataset_root = PROJECT_ROOT / "datasets" / "hand_pose"
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw_images = list((dataset_root / "raw").rglob("*.jpg"))
    train_imgs = list((dataset_root / "images" / "train").glob("*.jpg"))
    val_imgs = list((dataset_root / "images" / "val").glob("*.jpg"))
    test_imgs = list((dataset_root / "images" / "test").glob("*.jpg"))

    train_lbls = list((dataset_root / "labels" / "train").glob("*.txt"))
    val_lbls = list((dataset_root / "labels" / "val").glob("*.txt"))
    test_lbls = list((dataset_root / "labels" / "test").glob("*.txt"))

    # Run Validator
    validator = PoseAnnotationValidator(
        images_dir=dataset_root / "images",
        labels_dir=dataset_root / "labels",
    )
    val_passed = validator.run_validation()

    # Run Leakage Auditor
    auditor = DatasetLeakageAuditor(dataset_dir=dataset_root)
    leakage_passed = auditor.audit()

    total_images_in_splits = len(train_imgs) + len(val_imgs) + len(test_imgs)
    total_labels_in_splits = len(train_lbls) + len(val_lbls) + len(test_lbls)

    # Determine Phase 3 Status
    if len(raw_images) == 0 and total_images_in_splits == 0:
        status = "BLOCKED — NO REAL PRODUCTION IMAGES AVAILABLE"
    elif not val_passed or not leakage_passed:
        status = "FAILED — QUALITY GATES NOT SATISFIED"
    elif total_images_in_splits > 0 and val_passed and leakage_passed:
        status = "COMPLETE — DATASET READY FOR YOLO26 TRAINING"
    else:
        status = "IN PROGRESS"

    report_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "phase3_status": status,
        "data_availability": {
            "raw_production_images": len(raw_images),
            "images_in_splits": total_images_in_splits,
            "labels_in_splits": total_labels_in_splits,
        },
        "quality_gates": {
            "IMAGE_INTEGRITY": "PASS" if validator.stats["corrupt_images"] == 0 else "FAIL",
            "ANNOTATION_FORMAT": "PASS" if validator.stats["malformed_lines"] == 0 else "FAIL",
            "KEYPOINT_COUNT": "PASS" if validator.stats["invalid_token_counts"] == 0 else "FAIL",
            "COORDINATE_RANGE": "PASS" if validator.stats["coords_out_of_bounds"] == 0 else "FAIL",
            "VISIBILITY_VALUES": "PASS" if validator.stats["invalid_visibilities"] == 0 else "FAIL",
            "IMAGE_LABEL_MATCH": "PASS" if (validator.stats["missing_labels"] == 0 and validator.stats["orphan_labels"] == 0) else "FAIL",
            "SUBJECT_LEAKAGE": "PASS" if len(auditor.violations) == 0 else "FAIL",
            "SESSION_LEAKAGE": "PASS" if len(auditor.violations) == 0 else "FAIL",
            "MANUAL_QA_QUEUE": f"{len(validator.review_queue)} flagged instances",
        },
        "split_counts": {
            "train": {"images": len(train_imgs), "labels": len(train_lbls)},
            "val": {"images": len(val_imgs), "labels": len(val_lbls)},
            "test": {"images": len(test_imgs), "labels": len(test_lbls)},
        },
        "review_queue_count": len(validator.review_queue),
    }

    # 1. JSON Report
    json_path = reports_dir / "phase3_annotation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # 2. CSV Report
    csv_path = reports_dir / "phase3_annotation_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Gate", "Status"])
        for g, s in report_data["quality_gates"].items():
            writer.writerow([g, s])

    # 3. Markdown Report
    md_path = reports_dir / "phase3_annotation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# KINEX — Phase 3 Annotation & Quality Gate Report\n\n")
        f.write(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Status: `{status}`\n\n")
        f.write("### 1. Data Availability & Split Partition Summary\n\n")
        f.write(f"- **Raw Production Images:** {len(raw_images)}\n")
        f.write(f"- **Train Partition:** {len(train_imgs)} images, {len(train_lbls)} labels\n")
        f.write(f"- **Validation Partition:** {len(val_imgs)} images, {len(val_lbls)} labels\n")
        f.write(f"- **Test Partition:** {len(test_imgs)} images, {len(test_lbls)} labels\n\n")

        f.write("### 2. Quality Gate Audit Checklist\n\n")
        f.write("| Quality Gate | Result | Evaluation Criteria |\n")
        f.write("|:---|:---:|:---|\n")
        for gate, res in report_data["quality_gates"].items():
            f.write(f"| **{gate}** | `{res}` | Evaluated by automated validator |\n")
        f.write("\n")

        f.write("### 3. Review Queue\n\n")
        f.write(f"Total instances flagged for human QA: **{len(validator.review_queue)}**\n")
        f.write(f"Review queue file: `{reports_dir / 'annotation_review_queue.csv'}`\n")

    # 4. Dataset Manifest
    manifest_path = dataset_root / "dataset_manifest.json"
    dataset_manifest = {
        "dataset_name": "KINEX Hand Pose Dataset",
        "version": "v1.0-annotation",
        "created_at": datetime.datetime.now().isoformat(),
        "total_images": total_images_in_splits,
        "total_labels": total_labels_in_splits,
        "keypoint_count": 21,
        "keypoint_dimensions": 3,
        "classes": {0: "hand"},
        "splits": {
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs),
        },
        "quality_status": status,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(dataset_manifest, f, indent=2)

    print(f"[PHASE 3 REPORT] Generated:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")
    print(f"  - {manifest_path}")
    return report_data


def main():
    generate_phase3_report()


if __name__ == "__main__":
    main()

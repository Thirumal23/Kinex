#!/usr/bin/env python3
"""
KINEX Zero-Leakage Dataset Splitter
Partitions annotated hand images and YOLO pose labels into train, val, and test splits
based strictly on Subject and Session IDs defined in configs/split.yaml.

Outputs:
  datasets/hand_pose/images/{train, val, test}/
  datasets/hand_pose/labels/{train, val, test}/
  reports/split_manifest.json

Usage:
    python tools/split_dataset.py [--split-config configs/split.yaml]
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Set
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.check_dataset_leakage import extract_subject_and_session


class DatasetSplitter:
    def __init__(
        self,
        split_config_path: Path = PROJECT_ROOT / "configs" / "split.yaml",
        raw_images_dir: Path = PROJECT_ROOT / "datasets" / "hand_pose" / "raw",
        annotations_dir: Path = PROJECT_ROOT / "datasets" / "hand_pose" / "annotation_exports" / "validated",
        dataset_root: Path = PROJECT_ROOT / "datasets" / "hand_pose",
        manifest_out_path: Path = PROJECT_ROOT / "reports" / "split_manifest.json",
    ):
        self.split_config_path = split_config_path
        self.raw_images_dir = raw_images_dir
        self.annotations_dir = annotations_dir
        self.dataset_root = dataset_root
        self.manifest_out_path = manifest_out_path
        self.manifest_out_path.parent.mkdir(parents=True, exist_ok=True)

        self._load_split_config()

    def _load_split_config(self):
        if not self.split_config_path.exists():
            raise FileNotFoundError(f"Split config not found at: {self.split_config_path}")
        with open(self.split_config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.train_subjects = set(cfg.get("train_subjects", []))
        self.val_subjects = set(cfg.get("val_subjects", []))
        self.test_subjects = set(cfg.get("test_subjects", []))

        # Check configuration for internal subject overlap
        if self.train_subjects.intersection(self.val_subjects) or \
           self.train_subjects.intersection(self.test_subjects) or \
           self.val_subjects.intersection(self.test_subjects):
            raise ValueError("FATAL: configs/split.yaml contains subject overlap across splits!")

    def execute_split(self) -> Dict[str, Any]:
        """Copies or links matching image-label pairs into their respective split directories."""
        valid_img_exts = {".jpg", ".jpeg", ".png"}
        raw_imgs = {p.stem: p for p in self.raw_images_dir.rglob("*") if p.suffix.lower() in valid_img_exts}
        labels = {p.stem: p for p in self.annotations_dir.rglob("*.txt")}

        # Ensure split target directories exist
        for s in ["train", "val", "test"]:
            (self.dataset_root / "images" / s).mkdir(parents=True, exist_ok=True)
            (self.dataset_root / "labels" / s).mkdir(parents=True, exist_ok=True)

        split_counts = {"train": 0, "val": 0, "test": 0, "unassigned": 0}
        split_manifest = {
            "strategy": "SUBJECT_AND_SESSION_BASED",
            "train": {"subjects": sorted(list(self.train_subjects)), "count": 0, "files": []},
            "val": {"subjects": sorted(list(self.val_subjects)), "count": 0, "files": []},
            "test": {"subjects": sorted(list(self.test_subjects)), "count": 0, "files": []},
        }

        # Pair validated labels with raw images
        paired_stems = set(labels.keys()).intersection(set(raw_imgs.keys()))
        print(f"[SPLITTER] Found {len(paired_stems)} annotated image-label pairs.")

        for stem in sorted(paired_stems):
            img_path = raw_imgs[stem]
            lbl_path = labels[stem]

            subj, sess = extract_subject_and_session(stem)

            target_split = None
            if subj in self.train_subjects:
                target_split = "train"
            elif subj in self.val_subjects:
                target_split = "val"
            elif subj in self.test_subjects:
                target_split = "test"
            else:
                split_counts["unassigned"] += 1
                continue

            dest_img = self.dataset_root / "images" / target_split / img_path.name
            dest_lbl = self.dataset_root / "labels" / target_split / lbl_path.name

            shutil.copy2(img_path, dest_img)
            shutil.copy2(lbl_path, dest_lbl)

            split_counts[target_split] += 1
            split_manifest[target_split]["count"] += 1
            split_manifest[target_split]["files"].append(stem)

        with open(self.manifest_out_path, "w", encoding="utf-8") as f:
            json.dump(split_manifest, f, indent=2)

        print("[SPLITTER] Split completed successfully:")
        print(f"  - Train: {split_counts['train']} pairs")
        print(f"  - Val:   {split_counts['val']} pairs")
        print(f"  - Test:  {split_counts['test']} pairs")
        print(f"  - Unassigned (Excluded): {split_counts['unassigned']}")
        print(f"  Manifest written to: {self.manifest_out_path}")
        return split_manifest


def main():
    parser = argparse.ArgumentParser(description="Split KINEX dataset into zero-leakage partitions")
    parser.add_argument("--split-config", default="configs/split.yaml", help="Path to split.yaml")
    args = parser.parse_args()

    splitter = DatasetSplitter(split_config_path=Path(args.split_config))
    splitter.execute_split()


if __name__ == "__main__":
    main()

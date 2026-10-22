#!/usr/bin/env python3
"""
KINEX Dataset Leakage Audit Tool
Strictly verifies that no subject or session appears across more than one split partition (Train, Val, Test).

Criteria:
  Train Subjects ∩ Val Subjects = ∅
  Train Subjects ∩ Test Subjects = ∅
  Val Subjects ∩ Test Subjects = ∅

Usage:
    python tools/check_dataset_leakage.py [--dataset-dir datasets/hand_pose]
"""

import os
import sys
import re
import argparse
from pathlib import Path
from typing import Dict, Set, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def extract_subject_and_session(filename: str) -> Tuple[str, str]:
    """
    Extracts (subject_id, session_id) from standard KINEX filenames.
    Format: S01_SESSION_01_THANK_YOU_RIGHT_000001.jpg -> ('S01', 'SESSION_01')
    """
    parts = filename.split("_")
    subject = "UNKNOWN"
    session = "UNKNOWN"

    for idx, part in enumerate(parts):
        if re.match(r"^S\d+$", part.upper()):
            subject = part.upper()
        if part.upper() == "SESSION" and idx + 1 < len(parts):
            session = f"SESSION_{parts[idx + 1].upper()}"

    # Fallback to direct token matching
    if session == "UNKNOWN":
        for part in parts:
            if "SESSION" in part.upper():
                session = part.upper()

    return subject, session


class DatasetLeakageAuditor:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.splits = ["train", "val", "test"]
        self.split_subjects: Dict[str, Set[str]] = {s: set() for s in self.splits}
        self.split_sessions: Dict[str, Set[str]] = {s: set() for s in self.splits}
        self.split_file_counts: Dict[str, int] = {s: 0 for s in self.splits}
        self.violations: List[str] = []

    def audit(self) -> bool:
        valid_exts = {".jpg", ".jpeg", ".png", ".txt"}

        for split in self.splits:
            img_split_dir = self.dataset_dir / "images" / split
            lbl_split_dir = self.dataset_dir / "labels" / split

            files = []
            if img_split_dir.exists():
                files.extend([p for p in img_split_dir.iterdir() if p.suffix.lower() in valid_exts])
            if lbl_split_dir.exists():
                files.extend([p for p in lbl_split_dir.iterdir() if p.suffix.lower() == ".txt"])

            unique_stems = set(p.stem for p in files)
            self.split_file_counts[split] = len(unique_stems)

            for stem in unique_stems:
                subj, sess = extract_subject_and_session(stem)
                if subj != "UNKNOWN":
                    self.split_subjects[split].add(subj)
                if sess != "UNKNOWN":
                    self.split_sessions[split].add(f"{subj}_{sess}")

        # Check Subject Leakage
        train_sub = self.split_subjects["train"]
        val_sub = self.split_subjects["val"]
        test_sub = self.split_subjects["test"]

        tv_leak = train_sub.intersection(val_sub)
        tt_leak = train_sub.intersection(test_sub)
        vt_leak = val_sub.intersection(test_sub)

        if tv_leak:
            self.violations.append(f"Subject leakage between TRAIN and VAL: {sorted(list(tv_leak))}")
        if tt_leak:
            self.violations.append(f"Subject leakage between TRAIN and TEST: {sorted(list(tt_leak))}")
        if vt_leak:
            self.violations.append(f"Subject leakage between VAL and TEST: {sorted(list(vt_leak))}")

        # Check Session Leakage
        train_sess = self.split_sessions["train"]
        val_sess = self.split_sessions["val"]
        test_sess = self.split_sessions["test"]

        tv_sess = train_sess.intersection(val_sess)
        tt_sess = train_sess.intersection(test_sess)
        vt_sess = val_sess.intersection(test_sess)

        if tv_sess:
            self.violations.append(f"Session leakage between TRAIN and VAL: {sorted(list(tv_sess))}")
        if tt_sess:
            self.violations.append(f"Session leakage between TRAIN and TEST: {sorted(list(tt_sess))}")
        if vt_sess:
            self.violations.append(f"Session leakage between VAL and TEST: {sorted(list(vt_sess))}")

        passed = len(self.violations) == 0
        self.print_report(passed)
        return passed

    def print_report(self, passed: bool):
        print("\n" + "=" * 60)
        print("KINEX ZERO-LEAKAGE PARTITION AUDIT REPORT")
        print("=" * 60)
        for s in self.splits:
            print(f"Split [{s:5s}]: {self.split_file_counts[s]:4d} instances | Subjects: {sorted(list(self.split_subjects[s]))}")

        print("-" * 60)
        if passed:
            print("[PASS] ZERO LEAKAGE DETECTED across Train, Val, and Test partitions.")
        else:
            print("[CRITICAL FAILURE] LEAKAGE DETECTED:")
            for v in self.violations:
                print(f"  - {v}")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Audit KINEX Dataset for Subject/Session Leakage")
    parser.add_argument("--dataset-dir", default="datasets/hand_pose", help="Path to hand_pose dataset directory")
    args = parser.parse_args()

    auditor = DatasetLeakageAuditor(dataset_dir=Path(args.dataset_dir))
    success = auditor.audit()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

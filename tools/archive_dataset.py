#!/usr/bin/env python3
"""
KINEX Raw Dataset Backup Archiver
Creates timestamped zip backups of datasets/hand_pose/raw and metadata.
Output: archives/kinex_phase2_<timestamp>.zip
"""

import os
import sys
import zipfile
import datetime
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def archive_dataset(output_dir: str = "archives"):
    out_dir = PROJECT_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"kinex_phase2_{timestamp}.zip"
    zip_path = out_dir / zip_name

    targets = [
        PROJECT_ROOT / "datasets" / "hand_pose" / "raw",
        PROJECT_ROOT / "datasets" / "hand_pose" / "metadata",
        PROJECT_ROOT / "configs",
    ]

    total_files = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for target in targets:
            if not target.exists():
                continue
            for file_path in target.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(PROJECT_ROOT)
                    zf.write(file_path, arcname)
                    total_files += 1

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[ARCHIVE] Successfully created backup: {zip_path}")
    print(f"          Total files included: {total_files} ({size_mb:.2f} MB)")
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Archive KINEX raw dataset")
    parser.add_argument("--output-dir", default="archives", help="Directory to store zip backup")
    args = parser.parse_args()

    archive_dataset(output_dir=args.output_dir)


if __name__ == "__main__":
    main()

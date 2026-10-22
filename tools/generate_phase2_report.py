#!/usr/bin/env python3
"""
KINEX Phase 2 Dataset Reporting Engine
Compiles comprehensive statistics, class balance, subject distributions,
and rejection metrics from datasets/hand_pose/metadata into JSON, CSV, and Markdown.

Outputs:
    reports/phase2_collection_report.json
    reports/phase2_collection_report.csv
    reports/phase2_collection_report.md
"""

import os
import sys
import csv
import json
import datetime
from pathlib import Path
from collections import defaultdict, Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_report():
    metadata_jsonl = PROJECT_ROOT / "datasets" / "hand_pose" / "metadata" / "metadata.jsonl"
    manifest_json = PROJECT_ROOT / "datasets" / "hand_pose" / "metadata" / "collection_manifest.json"
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    records = []
    if metadata_jsonl.exists():
        with open(metadata_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line.strip()))
                    except Exception:
                        pass

    manifest = {"sessions": [], "stats": {}}
    if manifest_json.exists():
        try:
            with open(manifest_json, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            pass

    accepted_records = [r for r in records if r.get("accepted", True)]
    rejected_records = [r for r in records if not r.get("accepted", True)]

    # Metrics
    total_captured = len(records)
    total_accepted = len(accepted_records)
    total_rejected = len(rejected_records)

    rejection_reasons = Counter([r.get("rejection_reason", "unknown") for r in rejected_records])
    duplicates_count = rejection_reasons.get("duplicate", 0)
    blur_count = rejection_reasons.get("blurry", 0)
    exposure_count = rejection_reasons.get("underexposed", 0) + rejection_reasons.get("overexposed", 0)

    # Distributions over accepted records
    gesture_counts = Counter([r.get("gesture", "UNKNOWN") for r in accepted_records])
    subject_counts = Counter([r.get("subject_id", "UNKNOWN") for r in accepted_records])
    hand_counts = Counter([r.get("hand", "UNKNOWN") for r in accepted_records])
    lighting_counts = Counter([r.get("lighting", "UNKNOWN") for r in accepted_records])
    distance_counts = Counter([r.get("camera_distance", "UNKNOWN") for r in accepted_records])
    resolution_counts = Counter([f"{r.get('actual_width')}x{r.get('actual_height')}" for r in accepted_records])

    # Subject x Gesture cross-tabulation
    subj_gest_matrix = defaultdict(lambda: Counter())
    for r in accepted_records:
        subj = r.get("subject_id", "UNKNOWN")
        gest = r.get("gesture", "UNKNOWN")
        subj_gest_matrix[subj][gest] += 1

    unique_subjects = sorted(list(subject_counts.keys()))
    unique_gestures = sorted(list(gesture_counts.keys()))
    unique_sessions = sorted(list(set(r.get("session_id", "UNKNOWN") for r in accepted_records)))

    report_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "summary": {
            "total_subjects": len(unique_subjects),
            "total_sessions": len(unique_sessions),
            "total_gestures": len(unique_gestures),
            "total_captured": total_captured,
            "total_accepted": total_accepted,
            "total_rejected": total_rejected,
            "rejections": {
                "duplicates": duplicates_count,
                "blur": blur_count,
                "exposure": exposure_count,
                "other": total_rejected - (duplicates_count + blur_count + exposure_count),
            },
        },
        "gesture_distribution": dict(gesture_counts),
        "subject_distribution": dict(subject_counts),
        "hand_distribution": dict(hand_counts),
        "lighting_distribution": dict(lighting_counts),
        "distance_distribution": dict(distance_counts),
        "resolution_distribution": dict(resolution_counts),
        "subject_gesture_matrix": {s: dict(g) for s, g in subj_gest_matrix.items()},
        "validation_status": "PASSED" if total_accepted > 0 else "NO_DATA",
        "known_limitations": [
            "Dataset represents raw webcam captures prior to 21-keypoint annotation.",
            "Hand presence in raw images is operator-guided and unverified by AI model (Phase 4).",
            "Initial sample size serves as engineering validation baseline for annotation pipeline bring-up."
        ],
    }

    # 1. Save JSON
    json_path = reports_dir / "phase2_collection_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # 2. Save CSV (Gesture balance table)
    csv_path = reports_dir / "phase2_collection_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Gesture", "Accepted_Images", "Percentage"])
        for g, c in sorted(gesture_counts.items()):
            pct = (c / total_accepted * 100.0) if total_accepted else 0.0
            writer.writerow([g, c, f"{pct:.2f}%"])

    # 3. Save Markdown
    md_path = reports_dir / "phase2_collection_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# KINEX — Phase 2 Data Collection & Quality Report\n\n")
        f.write(f"**Report Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write(f"- **Total Frames Captured:** {total_captured}\n")
        f.write(f"- **Total Frames Accepted:** {total_accepted}\n")
        f.write(f"- **Total Frames Rejected:** {total_rejected}\n")
        f.write(f"  - **Duplicates Filtered:** {duplicates_count}\n")
        f.write(f"  - **Blurry Discarded:** {blur_count}\n")
        f.write(f"  - **Exposure Issues Discarded:** {exposure_count}\n")
        f.write(f"- **Unique Subjects:** {len(unique_subjects)} ({', '.join(unique_subjects) if unique_subjects else 'None'})\n")
        f.write(f"- **Unique Sessions:** {len(unique_sessions)}\n\n")

        f.write("## 2. Gesture Class Balance Table\n\n")
        f.write("| Gesture | Accepted Images | Percentage of Total |\n")
        f.write("|:---|:---:|:---:|\n")
        for g, c in sorted(gesture_counts.items()):
            pct = (c / total_accepted * 100.0) if total_accepted else 0.0
            f.write(f"| **{g}** | {c} | {pct:.1f}% |\n")
        f.write("\n")

        f.write("## 3. Subject × Gesture Distribution Matrix\n\n")
        all_gests = sorted(list(set(gesture_counts.keys())))
        if all_gests and unique_subjects:
            f.write("| Subject | " + " | ".join(all_gests) + " | Total |\n")
            f.write("|:---|" + ":---:|" * len(all_gests) + ":---:|\n")
            for s in unique_subjects:
                row = [str(subj_gest_matrix[s].get(g, 0)) for g in all_gests]
                tot = sum(subj_gest_matrix[s].values())
                f.write(f"| **{s}** | " + " | ".join(row) + f" | **{tot}** |\n")
            f.write("\n")

        f.write("## 4. Environmental & Camera Distributions\n\n")
        f.write(f"- **Hand Side:** {dict(hand_counts)}\n")
        f.write(f"- **Lighting Conditions:** {dict(lighting_counts)}\n")
        f.write(f"- **Camera Distance:** {dict(distance_counts)}\n")
        f.write(f"- **Resolutions:** {dict(resolution_counts)}\n\n")

        f.write("## 5. Known Engineering Limitations\n\n")
        for lim in report_data["known_limitations"]:
            f.write(f"- {lim}\n")

    print(f"[REPORT] Reports generated successfully:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")
    return report_data


def main():
    generate_report()


if __name__ == "__main__":
    main()

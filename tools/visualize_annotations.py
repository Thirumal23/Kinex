#!/usr/bin/env python3
"""
KINEX Hand Pose Annotation Visualizer
Renders hand bounding boxes, 21 keypoint landmarks with finger color groupings,
skeleton kinematic bones, and visibility status onto image frames for QA verification.

Saves rendered samples to: reports/annotation_previews/{gesture}/

Usage:
    python tools/visualize_annotations.py [--images-dir ...] [--labels-dir ...]
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 21 Hand Keypoint Topology Connections
HAND_SKELETON_EDGES = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm arch
    (5, 9), (9, 13), (13, 17),
]

# Color Palette (RGB)
PALETTE = {
    "bbox": (245, 169, 127),      # Orange/Peach
    "wrist": (255, 255, 255),     # White
    "thumb": (250, 179, 135),     # Coral/Yellow
    "index": (166, 227, 161),     # Green
    "middle": (137, 220, 235),    # Cyan
    "ring": (137, 180, 250),      # Blue
    "pinky": (203, 166, 247),     # Lavender/Magenta
    "palm": (186, 194, 222),      # Gray/Lavender
    "occluded": (243, 139, 168),  # Red/Pink
}


def get_kpt_color(kpt_idx: int) -> Tuple[int, int, int]:
    if kpt_idx == 0:
        return PALETTE["wrist"]
    elif 1 <= kpt_idx <= 4:
        return PALETTE["thumb"]
    elif 5 <= kpt_idx <= 8:
        return PALETTE["index"]
    elif 9 <= kpt_idx <= 12:
        return PALETTE["middle"]
    elif 13 <= kpt_idx <= 16:
        return PALETTE["ring"]
    else:
        return PALETTE["pinky"]


def visualize_annotation(image_path: Path, label_path: Path, output_path: Path) -> bool:
    """Renders 21-kpt skeleton onto the image and saves the preview."""
    try:
        im = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Could not open image {image_path}: {e}")
        return False

    w, h = im.size
    draw = ImageDraw.Draw(im)

    if not label_path.exists():
        print(f"[WARN] No label found for {image_path.name}")
        return False

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        tokens = line.split()
        if len(tokens) != 68:
            continue

        numeric = [float(x) for x in tokens[1:]]
        xc, yc, bw, bh = numeric[:4]
        # Pixel coordinates
        xmin = int((xc - bw / 2.0) * w)
        ymin = int((yc - bh / 2.0) * h)
        xmax = int((xc + bw / 2.0) * w)
        ymax = int((yc + bh / 2.0) * h)

        # Draw Bounding Box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=PALETTE["bbox"], width=2)
        draw.text((xmin + 4, max(ymin - 16, 4)), "hand", fill=PALETTE["bbox"])

        # Parse Keypoints
        kpts = []
        for i in range(21):
            kx = int(numeric[4 + i * 3] * w)
            ky = int(numeric[4 + i * 3 + 1] * h)
            kv = int(numeric[4 + i * 3 + 2])
            kpts.append((kx, ky, kv))

        # Draw Skeleton Bones
        for p1, p2 in HAND_SKELETON_EDGES:
            x1, y1, v1 = kpts[p1]
            x2, y2, v2 = kpts[p2]
            if v1 > 0 and v2 > 0:
                edge_color = get_kpt_color(p2) if p1 == 0 else get_kpt_color(p1)
                draw.line([(x1, y1), (x2, y2)], fill=edge_color, width=2)

        # Draw Keypoints & Indices
        r = 4
        for idx, (kx, ky, kv) in enumerate(kpts):
            if kv == 0:
                continue  # Not visible / unannotated

            pt_color = PALETTE["occluded"] if kv == 1 else get_kpt_color(idx)
            draw.ellipse([kx - r, ky - r, kx + r, ky + r], fill=pt_color, outline=(0, 0, 0))
            # Text index
            draw.text((kx + 5, ky - 5), str(idx), fill=(255, 255, 255))

    # Top overlay header
    draw.rectangle([0, 0, w, 28], fill=(17, 17, 27))
    draw.text((8, 6), f"KINEX SKELETON QA: {image_path.name}", fill=(137, 180, 250))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(output_path, quality=92)
    return True


def batch_visualize(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path = PROJECT_ROOT / "reports" / "annotation_previews",
    max_per_folder: int = 15,
):
    valid_exts = {".jpg", ".jpeg", ".png"}
    img_files = [p for p in images_dir.rglob("*") if p.suffix.lower() in valid_exts]

    if not img_files:
        print(f"[INFO] No images found to visualize under {images_dir}")
        return

    count = 0
    for img_p in img_files[:max_per_folder]:
        lbl_p = labels_dir / f"{img_p.stem}.txt"
        out_p = output_dir / f"qa_{img_p.name}"
        if visualize_annotation(img_p, lbl_p, out_p):
            count += 1

    print(f"[VISUALIZER] Rendered {count} annotation previews in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize KINEX Hand Pose Annotations")
    parser.add_argument("--images-dir", default="datasets/hand_pose/images", help="Path to images directory")
    parser.add_argument("--labels-dir", default="datasets/hand_pose/labels", help="Path to labels directory")
    parser.add_argument("--output-dir", default="reports/annotation_previews", help="Output directory for previews")
    args = parser.parse_args()

    batch_visualize(
        images_dir=Path(args.images_dir),
        labels_dir=Path(args.labels_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()

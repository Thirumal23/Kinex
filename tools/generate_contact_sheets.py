#!/usr/bin/env python3
"""
KINEX Visual Contact Sheet Generator
Creates visual inspection contact sheets from collected raw dataset images for manual quality review.
Saves grids to reports/contact_sheets/{gesture}.jpg
"""

import os
import sys
import math
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def create_contact_sheet(
    image_paths: list,
    output_path: Path,
    title: str = "KINEX Inspection Sheet",
    cols: int = 5,
    thumb_w: int = 240,
    thumb_h: int = 180,
    max_images: int = 25,
):
    """Generates a high-quality visual contact sheet grid from a list of image paths."""
    selected_paths = image_paths[:max_images]
    if not selected_paths:
        return False

    n_images = len(selected_paths)
    rows = math.ceil(n_images / cols)

    header_h = 50
    margin = 10
    total_w = cols * thumb_w + (cols + 1) * margin
    total_h = rows * thumb_h + (rows + 1) * margin + header_h

    sheet = Image.new("RGB", (total_w, total_h), color=(24, 24, 37))
    draw = ImageDraw.Draw(sheet)

    # Header title
    draw.text((margin, 15), f"{title} (Showing {n_images} frames)", fill=(137, 180, 250))

    for idx, img_p in enumerate(selected_paths):
        c = idx % cols
        r = idx // cols

        x = margin + c * (thumb_w + margin)
        y = header_h + margin + r * (thumb_h + margin)

        try:
            with Image.open(img_p) as im:
                thumb = im.copy()
                thumb.thumbnail((thumb_w, thumb_h))
                # Center within thumbnail box
                paste_x = x + (thumb_w - thumb.width) // 2
                paste_y = y + (thumb_h - thumb.height) // 2
                sheet.paste(thumb, (paste_x, paste_y))
                # Subtitle text
                draw.text((x + 2, y + thumb_h - 18), img_p.stem[:28], fill=(205, 214, 244))
        except Exception as e:
            draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=(243, 139, 168))
            draw.text((x + 5, y + 5), f"Error: {e}", fill=(243, 139, 168))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(str(output_path), quality=90)
    print(f"[CONTACT SHEET] Generated: {output_path} ({n_images} images)")
    return True


def generate_all_contact_sheets(raw_root: str = "datasets/hand_pose/raw", output_root: str = "reports/contact_sheets"):
    raw_dir = Path(raw_root)
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group all .jpg by gesture
    all_images = list(raw_dir.rglob("*.jpg"))
    gesture_images = {}

    for p in all_images:
        # Path is raw/{subject}/{session}/{gesture}/{file}
        parts = p.parts
        if len(parts) >= 4:
            gesture = p.parent.name
            gesture_images.setdefault(gesture, []).append(p)

    if not gesture_images:
        print(f"[WARN] No raw images found under {raw_dir}")
        return

    for gesture, paths in gesture_images.items():
        out_file = out_dir / f"{gesture}.jpg"
        create_contact_sheet(
            paths,
            out_file,
            title=f"KINEX CONTACT SHEET — GESTURE: {gesture}",
            cols=5,
            max_images=25,
        )


def main():
    parser = argparse.ArgumentParser(description="KINEX Contact Sheet Generator")
    parser.add_argument("--raw-dir", default="datasets/hand_pose/raw", help="Path to raw dataset folder")
    parser.add_argument("--output-dir", default="reports/contact_sheets", help="Output contact sheet directory")
    args = parser.parse_args()

    generate_all_contact_sheets(raw_root=args.raw_dir, output_root=args.output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
KINEX Project Sharing & Distribution Packager
Creates a clean, production-ready ZIP archive of the project for sharing.
Excludes build artifacts, cache files, temporary logs, and virtual environments.
"""

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".github",
    ".venv",
    "venv",
    "env",
    "ENV",
    ".vscode",
    ".idea",
    ".pytest_cache",
    ".mypy_cache",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".zip",
    ".tar",
    ".gz",
    ".rar",
    ".7z",
    ".log",
    ".tmp",
    ".bak",
}

EXCLUDE_FILES = {
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    ".env",
    ".env.local",
}


def should_exclude(rel_path: Path) -> bool:
    """Determine whether a file or directory should be excluded from the zip."""
    # Check directory components
    for part in rel_path.parts:
        if part in EXCLUDE_DIRS:
            return True

    # Check filename
    if rel_path.name in EXCLUDE_FILES:
        return True

    # Check extension
    if rel_path.suffix.lower() in EXCLUDE_EXTENSIONS:
        return True

    return False


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def package_project(output_path: Path, prefix: str = "") -> Path:
    """Package the project into a zip file."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[PACKAGE] Scanning project directory: {PROJECT_ROOT}")
    files_to_zip = []
    dirs_to_zip = set()

    for root_dir, dirs, files in os.walk(PROJECT_ROOT):
        # Modify dirs in-place to avoid recursing into excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        rel_dir = Path(root_dir).relative_to(PROJECT_ROOT)

        # Track directory structure (to preserve empty directories)
        if rel_dir != Path("."):
            if not should_exclude(rel_dir):
                dirs_to_zip.add(rel_dir)

        for file_name in sorted(files):
            file_path = Path(root_dir) / file_name
            rel_file = file_path.relative_to(PROJECT_ROOT)

            # Skip the output zip itself if inside project root
            if file_path.resolve() == output_path:
                continue

            if not should_exclude(rel_file):
                files_to_zip.append((file_path, rel_file))

    print(f"[PACKAGE] Found {len(files_to_zip)} files across {len(dirs_to_zip)} directories.")
    print(f"[PACKAGE] Compressing into: {output_path}")

    # Write zip with DEFLATED compression (level 9 for optimal size)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        # Write directories to preserve empty folders
        for d in sorted(dirs_to_zip):
            arc_dir = f"{prefix}/{d.as_posix()}/" if prefix else f"{d.as_posix()}/"
            zip_info = zipfile.ZipInfo(arc_dir)
            zf.writestr(zip_info, "")

        # Write files
        for file_path, rel_file in files_to_zip:
            arc_path = f"{prefix}/{rel_file.as_posix()}" if prefix else rel_file.as_posix()
            zf.write(file_path, arc_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    sha256_hash = compute_sha256(output_path)

    print(f"[PACKAGE] Successfully created: {output_path.name}")
    print(f"          Size:       {size_mb:.2f} MB ({output_path.stat().st_size:,} bytes)")
    print(f"          Files:      {len(files_to_zip)}")
    print(f"          SHA-256:    {sha256_hash}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Package KINEX project for sharing")
    parser.add_argument(
        "--output",
        "-o",
        default=str(PROJECT_ROOT / "kinex_adv_project.zip"),
        help="Target ZIP archive path (default: kinex_adv_project.zip in project root)",
    )
    parser.add_argument(
        "--prefix",
        "-p",
        default="",
        help="Optional root folder prefix inside the archive (e.g., 'kinex_adv')",
    )
    args = parser.parse_args()

    target_zip = Path(args.output)
    package_project(target_zip, prefix=args.prefix)


if __name__ == "__main__":
    main()

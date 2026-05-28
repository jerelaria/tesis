#!/usr/bin/env python3
"""
cleanup_masks.py

Cleans up the XRayNicoSent GT mask dataset, normalizing all patients to a
consistent output format regardless of the source naming convention.

The dataset contains patients with two different lung mask formats:
  - lung_*.png  (fragmented: lung_1.png, lung_2.png, lung_3.png, ...)
  - left_lung.png / right_lung.png  (already split, but potentially
    using a different left/right convention than the rest)

Both formats are handled the same way:
  1. All lung masks for a patient are union-merged into one binary mask.
  2. Connected components are extracted.
  3. Each component is assigned to left or right based on which side of the
     image its centroid falls on (radiological convention: left side of image
     = patient's right lung).
  4. All components on the same side are unioned → right_lung.png / left_lung.png.

This guarantees a consistent convention across the entire dataset.

Additional steps:
  - Clavicle files (left_clavicle.png, right_clavicle.png) are skipped.
  - All other files (heart.png, image.png, etc.) are copied as-is.
  - The source directory is never modified.

Usage:
    python cleanup_masks.py --masks_dir <src> --output_dir <dst>
    python cleanup_masks.py --masks_dir <src> --output_dir <dst> --execute
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


# ─── Constants ────────────────────────────────────────────────────────────────

CLAVICLE_FILES = {"left_clavicle.png", "right_clavicle.png"}

# All file patterns that contain lung masks, regardless of source convention.
# These are collected, merged, and re-assigned by image position.
LUNG_PATTERNS = ("lung_*.png", "left_lung.png", "right_lung.png")

# Output names follow radiological convention:
#   patient's RIGHT lung → centroid on the LEFT  side of the image (smaller x)
#   patient's LEFT  lung → centroid on the RIGHT side of the image (larger  x)
RIGHT_LUNG_OUT = "right_lung.png"
LEFT_LUNG_OUT  = "left_lung.png"

# Connected components smaller than this pixel area are treated as noise.
MIN_COMPONENT_AREA = 50


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_binary_mask(path: Path) -> np.ndarray:
    """Load a grayscale PNG and return a boolean binary mask."""
    img = Image.open(path).convert("L")
    return np.array(img) > 127


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    """Save a boolean array as a binary PNG (0 / 255)."""
    Image.fromarray((mask * 255).astype(np.uint8)).save(path)


def collect_lung_files(patient_dir: Path) -> list[Path]:
    """
    Collect all lung mask files for a patient, regardless of naming convention.
    Matches lung_*.png, left_lung.png, and right_lung.png.
    """
    found = set()
    for pattern in LUNG_PATTERNS:
        found.update(patient_dir.glob(pattern))
    return sorted(found)


def merge_masks(paths: list[Path]) -> np.ndarray:
    """Union of all masks in the list into a single boolean array."""
    merged = None
    for p in paths:
        m = load_binary_mask(p)
        merged = m if merged is None else (merged | m)
    return merged if merged is not None else np.zeros((1, 1), dtype=bool)


def split_lungs_by_side(
    merged: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Assign each connected component to left or right lung based on which side
    of the image its centroid falls on, then union all components per side.

    This handles fragmented lungs correctly: a small detached piece whose
    centroid is on the left half of the image gets merged into right_lung
    rather than being discarded or misassigned.

    Radiological convention (applied uniformly to all patients):
        centroid_x <  image_width / 2  →  patient's RIGHT lung  (right_lung.png)
        centroid_x >= image_width / 2  →  patient's LEFT  lung  (left_lung.png)

    Returns:
        (right_lung_mask, left_lung_mask) — either can be None if no
        components were found on that side.
    """
    h, w = merged.shape
    mid_x = w / 2

    labeled, n_components = ndimage.label(merged)

    if n_components == 0:
        return None, None

    right_side = np.zeros(merged.shape, dtype=bool)
    left_side  = np.zeros(merged.shape, dtype=bool)

    for label_id in range(1, n_components + 1):
        component_mask = labeled == label_id
        area = int(component_mask.sum())

        if area < MIN_COMPONENT_AREA:
            # Discard tiny noise fragments
            continue

        centroid_x = float(np.where(component_mask)[1].mean())

        if centroid_x < mid_x:
            right_side |= component_mask  # left side of image → patient's right lung
        else:
            left_side  |= component_mask  # right side of image → patient's left lung

    right_lung = right_side if right_side.any() else None
    left_lung  = left_side  if left_side.any()  else None

    if right_lung is None:
        print("    [WARN] No components on image-left side (patient's right lung missing).")
    if left_lung is None:
        print("    [WARN] No components on image-right side (patient's left lung missing).")

    return right_lung, left_lung


# ─── Per-patient processing ────────────────────────────────────────────────────

def process_patient(
    patient_dir: Path,
    out_patient_dir: Path,
    dry_run: bool,
) -> dict:
    """
    Process a single patient directory and write normalized masks to out_patient_dir.
    The source directory is never modified.

    Steps:
      1. Collect all lung mask files (lung_*.png, left_lung.png, right_lung.png).
      2. Union-merge them and re-assign by image position → right_lung.png / left_lung.png.
      3. Skip clavicle files entirely.
      4. Copy all remaining files as-is (heart.png, image.png, etc.).
    """
    summary = {
        "patient":           patient_dir.name,
        "clavicles_skipped": 0,
        "lungs_merged":      False,
        "warnings":          [],
    }

    if not dry_run:
        out_patient_dir.mkdir(parents=True, exist_ok=True)

    lung_files     = collect_lung_files(patient_dir)
    lung_filenames = {lf.name for lf in lung_files}

    for src_file in sorted(patient_dir.iterdir()):
        if not src_file.is_file():
            continue

        # ── Clavicles: skip entirely ──────────────────────────────────────────
        if src_file.name in CLAVICLE_FILES:
            summary["clavicles_skipped"] += 1
            print(f"    [skip]  {src_file.name}  (clavicle)")
            continue

        # ── Lung files: collected and processed below ─────────────────────────
        if src_file.name in lung_filenames:
            continue

        # ── Everything else: copy as-is ───────────────────────────────────────
        if not dry_run:
            shutil.copy2(src_file, out_patient_dir / src_file.name)
        print(f"    [copy]  {src_file.name}")

    # ── Process lungs ─────────────────────────────────────────────────────────
    if not lung_files:
        msg = "No lung mask files found (lung_*.png / left_lung.png / right_lung.png)"
        summary["warnings"].append(msg)
        print(f"    [WARN] {msg}")
        return summary

    merged = merge_masks(lung_files)

    if not merged.any():
        msg = "Merged lung mask is empty"
        summary["warnings"].append(msg)
        print(f"    [WARN] {msg}")
        return summary

    right_mask, left_mask = split_lungs_by_side(merged)

    n_right   = int(right_mask.sum()) if right_mask is not None else 0
    n_left    = int(left_mask.sum())  if left_mask  is not None else 0
    src_names = [lf.name for lf in lung_files]

    print(
        f"    [lungs] {src_names} → "
        f"{RIGHT_LUNG_OUT} ({n_right}px), {LEFT_LUNG_OUT} ({n_left}px)"
    )

    if not dry_run:
        if right_mask is not None:
            save_binary_mask(right_mask, out_patient_dir / RIGHT_LUNG_OUT)
        if left_mask is not None:
            save_binary_mask(left_mask, out_patient_dir / LEFT_LUNG_OUT)

    summary["lungs_merged"] = True
    return summary


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize XRayNicoSent masks to a consistent format: "
            "skip clavicles, merge all lung masks and re-assign by image position. "
            "Source directory is never modified."
        )
    )
    parser.add_argument(
        "--masks_dir",
        type=Path,
        required=True,
        help="Source root directory with one sub-folder per patient.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Destination root directory for cleaned masks (created if needed).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually write files. Without this flag, runs in dry-run mode.",
    )
    args = parser.parse_args()

    if not args.masks_dir.exists():
        print(f"[ERROR] masks_dir not found: {args.masks_dir}")
        return

    if args.output_dir.resolve() == args.masks_dir.resolve():
        print("[ERROR] --output_dir must differ from --masks_dir.")
        return

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"=== Mask normalization — {mode} ===")
    print(f"  source : {args.masks_dir}")
    print(f"  output : {args.output_dir}")
    if not args.execute:
        print("  (pass --execute to write files)\n")

    patient_dirs = sorted(p for p in args.masks_dir.iterdir() if p.is_dir())
    print(f"Found {len(patient_dirs)} patient directories.\n")

    total_clavicles = 0
    total_lungs     = 0
    all_warnings    = []

    for patient_dir in patient_dirs:
        out_patient_dir = args.output_dir / patient_dir.name
        print(f"[{patient_dir.name}]")
        summary = process_patient(patient_dir, out_patient_dir, dry_run=not args.execute)
        total_clavicles += summary["clavicles_skipped"]
        total_lungs     += int(summary["lungs_merged"])
        for w in summary["warnings"]:
            all_warnings.append(f"  {patient_dir.name}: {w}")
            print(f"    [WARN] {w}")

    print("\n=== Summary ===")
    print(f"  Clavicle files skipped    : {total_clavicles}")
    print(f"  Patients lungs processed  : {total_lungs}")
    if all_warnings:
        print(f"\n  Warnings ({len(all_warnings)}):")
        for w in all_warnings:
            print(w)


if __name__ == "__main__":
    main()
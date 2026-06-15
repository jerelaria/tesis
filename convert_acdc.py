#!/usr/bin/env python3
"""
convert_acdc.py
---------------
Convert the Kaggle ACDC preprocessed dataset (HDF5 slices) into the 2D
per-slice PNG layout used by the project, matching the SunnybrookNicoSent
structure under data/processed/.

Input layout (Kaggle dataset 'anhoangvo/acdc-dataset'):
    ACDC_preprocessed/ACDC_training_slices/
        patientXXX_frameYY_slice_Z.h5
        patientXXX_frameYY_slice_Z(1).h5   <- duplicate, skipped

Each .h5 file contains:
    image   : (H, W) float32 in [0, 1]   -- already normalized
    label   : (H, W) uint8               -- 0=bg, 1=rv, 2=myo, 3=lv
    scribble: ignored

Output layout (identical to data/processed/SunnybrookNicoSent):
    {out_root}/images/{stem}.png              -- grayscale saved as RGB
    {out_root}/masks/{stem}/rv_cavity.png
    {out_root}/masks/{stem}/myocardium.png
    {out_root}/masks/{stem}/lv_cavity.png

Stem format: patientXXX_frameYY_slice_Z  (no parenthesized suffixes)

Only slices with at least one annotated organ are written. Slices where
label is all-zero (no structure present) are skipped entirely, since they
carry no useful GT for the prototype-transfer evaluation.

Dependencies: pip install h5py numpy pillow

Usage:
    python convert_acdc.py \
        --h5-root  ACDC_preprocessed/ACDC_training_slices \
        --out-root data/processed/ACDC
"""

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


LABEL_TO_ORGAN = {
    1: "rv_cavity",
    2: "myocardium",
    3: "lv_cavity",
}

# Files with a parenthesized suffix like _slice_2(1).h5 are exact duplicates.
_DUPLICATE_RE = re.compile(r"\(\d+\)$")


def is_duplicate(path: Path) -> bool:
    """Return True if this file is a parenthesized duplicate (e.g. slice_2(1).h5)."""
    return bool(_DUPLICATE_RE.search(path.stem))


def convert_slice(h5_path: Path, images_out: Path, masks_out: Path) -> bool:
    """Convert one HDF5 slice file. Returns True if the slice was written."""
    with h5py.File(h5_path, "r") as f:
        image = f["image"][()].astype(np.float32)   # (H, W) in [0, 1]
        label = f["label"][()].astype(np.int64)      # (H, W)

    # Skip slices with no annotated structure (all-background).
    if label.max() == 0:
        return False

    stem = h5_path.stem  # e.g. patient001_frame01_slice_2

    # Image: scale [0,1] -> uint8, save as RGB (matches project convention).
    img_uint8 = (image * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(img_uint8, mode="L").convert("RGB").save(
        images_out / f"{stem}.png"
    )

    # Masks: one binary PNG per organ present in this slice.
    slice_mask_dir = masks_out / stem
    slice_mask_dir.mkdir(parents=True, exist_ok=True)

    for label_value, organ in LABEL_TO_ORGAN.items():
        binary = label == label_value
        if not binary.any():
            # Organ absent in this slice; skip the file (project convention).
            continue
        mask_img = binary.astype(np.uint8) * 255
        Image.fromarray(mask_img, mode="L").save(slice_mask_dir / f"{organ}.png")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5-root", required=True,
        help="Directory containing patientXXX_frameYY_slice_Z.h5 files.",
    )
    parser.add_argument(
        "--out-root", required=True,
        help="Output dataset root, e.g. data/processed/ACDC.",
    )
    args = parser.parse_args()

    h5_root = Path(args.h5_root)
    if not h5_root.is_dir():
        raise FileNotFoundError(f"HDF5 root not found: {h5_root}")

    h5_files = sorted(h5_root.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found under {h5_root}.")

    out_root = Path(args.out_root)
    images_out = out_root / "images"
    masks_out = out_root / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    total = skipped_dup = skipped_empty = written = 0
    for h5_path in h5_files:
        total += 1
        if is_duplicate(h5_path):
            skipped_dup += 1
            continue
        if convert_slice(h5_path, images_out, masks_out):
            written += 1
        else:
            skipped_empty += 1

    print(f"\nDone.")
    print(f"  Total .h5 files found : {total}")
    print(f"  Skipped (duplicates)  : {skipped_dup}")
    print(f"  Skipped (empty label) : {skipped_empty}")
    print(f"  Slices written        : {written}")
    print(f"  Output: {out_root}")


if __name__ == "__main__":
    main()
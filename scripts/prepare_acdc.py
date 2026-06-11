#!/usr/bin/env python3
"""
prepare_acdc.py
---------------
Convert ACDC NIfTI volumes to 2D short-axis PNG slices.

For each patient directory and each requested cardiac phase (ED and/or ES):
  1. Parse Info.cfg to find the ED and ES frame numbers.
  2. Load the 3D image NIfTI and its GT label NIfTI for that frame.
  3. Normalize intensity per volume: clip to [1st, 99th percentile], scale to [0, 255].
  4. For each short-axis slice (Z index):
       - Skip slices where the GT is entirely background (no annotated structure).
         This matches the SunnybrookNicoSent convention, which only includes slices
         that have at least one organ annotation.
       - Write the 2D image to:  {output_dir}/images/{stem}.png  (uint8 grayscale)
       - For each ACDC organ label present in that slice, write a binary mask to:
         {output_dir}/masks/{stem}/{organ}.png  (uint8, values 0 or 255)

ACDC label -> organ name mapping:
  1  ->  rv_cavity
  2  ->  myocardium
  3  ->  lv_cavity

Stem convention: {patient_id}_{phase}_s{slice:02d}
  e.g.  patient001_ED_s00,  patient001_ES_s05

Expected ACDC raw layout (auto-detected: patients at root or under training/):
  <acdc_root>/
    patient001/
      patient001_frame01.nii.gz        <- ED image (frame number from Info.cfg)
      patient001_frame01_gt.nii.gz     <- ED GT
      patient001_frame12.nii.gz        <- ES image
      patient001_frame12_gt.nii.gz     <- ES GT
      Info.cfg                         <- ED:, ES: frame numbers
    patient002/
      ...

Output layout mirrors SunnybrookNicoSent (flat images, nested mask dirs):
  {output_dir}/
    images/
      patient001_ED_s00.png
      patient001_ED_s01.png
      ...
    masks/
      patient001_ED_s00/
        lv_cavity.png
        rv_cavity.png
        myocardium.png
      patient001_ED_s01/
        ...

Usage:
  python scripts/prepare_acdc.py \\
      --acdc-root /path/to/ACDC \\
      --output-dir data/processed/ACDC

  python scripts/prepare_acdc.py \\
      --acdc-root /path/to/ACDC \\
      --output-dir data/processed/ACDC \\
      --phases ed       # ED only
"""

import argparse
import configparser
import re
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


LABEL_TO_ORGAN: dict[int, str] = {
    1: "rv_cavity",
    2: "myocardium",
    3: "lv_cavity",
}


# ---------------------------------------------------------------------------
# Info.cfg parsing
# ---------------------------------------------------------------------------

def parse_info_cfg(info_path: Path) -> dict[str, int]:
    """Return {'ED': frame_number, 'ES': frame_number} from Info.cfg.

    Info.cfg uses bare key: value lines without a section header, so we
    inject a dummy section to satisfy configparser.
    """
    text = info_path.read_text()
    cfg = configparser.ConfigParser()
    cfg.read_string("[root]\n" + text)
    section = cfg["root"]

    result: dict[str, int] = {}
    for key in ("ED", "ES"):
        raw = section.get(key)
        if raw is None:
            raise ValueError(f"Key '{key}' missing in {info_path}")
        result[key] = int(raw.strip())
    return result


# ---------------------------------------------------------------------------
# Patient directory discovery
# ---------------------------------------------------------------------------

def find_patient_dirs(acdc_root: Path) -> list[Path]:
    """Return all patient* directories under acdc_root.

    Handles two common layouts:
      - Patients at root: acdc_root/patient001/
      - Patients under a split: acdc_root/training/patient001/
    """
    pattern = re.compile(r"^patient\d+$")
    dirs: list[Path] = []

    for candidate in sorted(acdc_root.iterdir()):
        if candidate.is_dir() and pattern.match(candidate.name):
            dirs.append(candidate)

    if not dirs:
        for sub in sorted(acdc_root.iterdir()):
            if not sub.is_dir():
                continue
            for candidate in sorted(sub.iterdir()):
                if candidate.is_dir() and pattern.match(candidate.name):
                    dirs.append(candidate)

    if not dirs:
        raise FileNotFoundError(
            f"No patient* directories found under {acdc_root}. "
            "Expected layout: patient001/, patient002/, ... (directly or under a split dir)."
        )
    return dirs


# ---------------------------------------------------------------------------
# Intensity normalization
# ---------------------------------------------------------------------------

def normalize_volume(arr: np.ndarray) -> np.ndarray:
    """Clip to [1st, 99th percentile] and scale to uint8 [0, 255].

    Operates on the full 3D array so relative intensities are preserved
    across slices of the same volume. Returns uint8 array of the same shape.
    """
    p1 = float(np.percentile(arr, 1))
    p99 = float(np.percentile(arr, 99))

    if p99 <= p1:
        raise ValueError(
            f"Degenerate intensity range: p1={p1}, p99={p99}. "
            "Volume may be empty or constant."
        )

    clipped = np.clip(arr, p1, p99)
    scaled = (clipped - p1) / (p99 - p1) * 255.0
    return scaled.astype(np.uint8)


# ---------------------------------------------------------------------------
# Single-phase extraction
# ---------------------------------------------------------------------------

def extract_phase(
    patient_dir: Path,
    patient_id: str,
    phase: str,
    frame_num: int,
    images_dir: Path,
    masks_dir: Path,
) -> dict:
    """Extract all annotated short-axis slices for one patient + phase.

    Returns a stats dict with keys: n_slices_total, n_slices_written,
    organ_counts (dict).
    """
    frame_str = f"{frame_num:02d}"
    img_path = patient_dir / f"{patient_id}_frame{frame_str}.nii.gz"
    gt_path  = patient_dir / f"{patient_id}_frame{frame_str}_gt.nii.gz"

    if not img_path.exists():
        raise FileNotFoundError(f"Image NIfTI not found: {img_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT NIfTI not found: {gt_path}")

    img_arr = nib.load(img_path).get_fdata(dtype=np.float32)   # (H, W, Z)
    gt_arr  = nib.load(gt_path).get_fdata(dtype=np.float32)    # (H, W, Z)
    gt_arr  = np.round(gt_arr).astype(np.int32)

    if img_arr.shape != gt_arr.shape:
        raise ValueError(
            f"{patient_id} {phase}: image shape {img_arr.shape} != "
            f"GT shape {gt_arr.shape}"
        )

    img_norm = normalize_volume(img_arr)                        # uint8, (H, W, Z)

    n_slices = img_arr.shape[2]
    n_written = 0
    organ_counts: dict[str, int] = defaultdict(int)

    for z in range(n_slices):
        gt_slice = gt_arr[:, :, z]

        # Skip unannotated slices — matches SunnybrookNicoSent convention
        labels_present = set(np.unique(gt_slice)) - {0}
        if not labels_present:
            continue

        stem = f"{patient_id}_{phase}_s{z:02d}"
        img_slice = img_norm[:, :, z]                           # (H, W) uint8

        # Write image
        Image.fromarray(img_slice, mode="L").save(images_dir / f"{stem}.png")

        # Write one binary mask per organ present in this slice
        mask_stem_dir = masks_dir / stem
        mask_stem_dir.mkdir(parents=True, exist_ok=True)

        for label, organ in LABEL_TO_ORGAN.items():
            if label not in labels_present:
                continue
            binary = (gt_slice == label).astype(np.uint8) * 255
            Image.fromarray(binary, mode="L").save(mask_stem_dir / f"{organ}.png")
            organ_counts[organ] += 1

        n_written += 1

    return {
        "n_slices_total": n_slices,
        "n_slices_written": n_written,
        "organ_counts": dict(organ_counts),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ACDC NIfTI volumes to 2D short-axis PNG slices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--acdc-root", required=True, type=Path,
        help="Root directory of the ACDC dataset (contains patient* dirs).",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Destination directory. Images -> images/, masks -> masks/.",
    )
    parser.add_argument(
        "--phases", nargs="+", choices=["ed", "es"], default=["ed", "es"],
        help="Cardiac phases to process (case-insensitive).",
    )
    args = parser.parse_args()

    acdc_root:  Path = args.acdc_root.resolve()
    output_dir: Path = args.output_dir.resolve()
    phases = [p.upper() for p in args.phases]

    if not acdc_root.is_dir():
        raise FileNotFoundError(f"ACDC root not found: {acdc_root}")

    images_dir = output_dir / "images"
    masks_dir  = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    patient_dirs = find_patient_dirs(acdc_root)
    print(f"Found {len(patient_dirs)} patient directories under {acdc_root}")
    print(f"Phases: {phases}")
    print(f"Output: {output_dir}")
    print()

    total_volumes  = 0
    total_slices   = 0
    organ_totals: dict[str, int] = defaultdict(int)
    skipped_phases: list[str] = []

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name

        info_path = patient_dir / "Info.cfg"
        if not info_path.exists():
            raise FileNotFoundError(
                f"Info.cfg missing for {patient_id}: {info_path}"
            )
        frame_nums = parse_info_cfg(info_path)

        for phase in phases:
            frame_num = frame_nums[phase]
            try:
                stats = extract_phase(
                    patient_dir, patient_id, phase, frame_num,
                    images_dir, masks_dir,
                )
            except FileNotFoundError as exc:
                print(f"  SKIP {patient_id} {phase}: {exc}", file=sys.stderr)
                skipped_phases.append(f"{patient_id}_{phase}")
                continue

            total_volumes += 1
            total_slices  += stats["n_slices_written"]
            for organ, count in stats["organ_counts"].items():
                organ_totals[organ] += count

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Volumes written : {total_volumes}")
    print(f"  Slices written  : {total_slices}")
    print()
    print("  Slices per organ:")
    for organ in sorted(organ_totals):
        print(f"    {organ:<14} {organ_totals[organ]}")
    if skipped_phases:
        print()
        print(f"  Skipped phases ({len(skipped_phases)}):")
        for s in skipped_phases:
            print(f"    {s}")
    print()
    print(f"  images/ -> {images_dir}")
    print(f"  masks/  -> {masks_dir}")


if __name__ == "__main__":
    main()

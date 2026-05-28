"""
relabel_baseline_masks.py
-------------------------
Re-label the masks produced by the unsupervised baseline (`obj_000.png`,
`obj_001.png`, ...) into anatomical names by matching each predicted mask
against the ground truth via IoU.

Rationale
---------
The unsupervised baseline (unsup_baseline) writes positional filenames
because, by definition, no labeling information is available. To use those
masks as training labels for HybridGNet (or any downstream model that
needs anatomical correspondence), we anchor the labels on the GT post-hoc.

This is NOT a fair use of the GT during pipeline evaluation — evaluate.py
already does this matching internally for metrics computation. The script
here is exclusively for *producing usable training masks from the
baseline output*, and that fact must be explicitly stated in any paper
or thesis where these masks are used.

Strategy
--------
For each image with both a prediction directory and a GT directory:
  1. Load all predicted masks (obj_*.png) and all GT masks (*.png).
  2. Build a per-pair IoU matrix.
  3. Run Hungarian matching to obtain an optimal assignment.
  4. Keep pairs whose IoU exceeds --min-iou (default 0.3).
  5. Rename the predicted file to the matched GT organ name.
  6. Predicted masks that did not match any GT (or matched below
     threshold) are by default DELETED. Use --keep-unmatched to keep
     them with a `unmatched_*.png` prefix instead.

The script operates in-place on the predictions directory unless
--output is given, in which case the entire predictions tree is copied
first and renaming happens on the copy.

Usage
-----
    # In-place relabel of v0_baseline_training_xray output
    python relabel_baseline_masks.py \\
        --pred results/v0_baseline_training_xray/XRayNicoSent/\\
unsup_baseline/masks \\
        --gt data/processed/XRayNicoSent/masks

    # Safe mode: produce a relabeled copy alongside the original
    python relabel_baseline_masks.py \\
        --pred results/v0_baseline_training_xray/XRayNicoSent/\\
unsup_baseline/masks \\
        --gt data/processed/XRayNicoSent/masks \\
        --output results/v0_baseline_training_xray/XRayNicoSent/\\
unsup_baseline/masks_relabeled
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Mask I/O
# ---------------------------------------------------------------------------

def _load_mask(path: Path) -> np.ndarray:
    """Load a mask PNG as a boolean numpy array."""
    img = np.array(Image.open(path))
    if img.ndim == 3:
        # Some masks may be RGB/RGBA; collapse to a single channel
        img = img[..., 0]
    return img > 0


def _iter_mask_files(directory: Path) -> list[Path]:
    """List all .png files in a directory, sorted."""
    return sorted(directory.glob("*.png"))


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two boolean masks. Returns 0 on shape mismatch."""
    if mask_a.shape != mask_b.shape:
        return 0.0
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def _hungarian_match(
    pred_masks: list[tuple[str, np.ndarray]],
    gt_masks: list[tuple[str, np.ndarray]],
    min_iou: float,
) -> list[tuple[str, str, float]]:
    """
    Hungarian-match predicted masks to GT masks by IoU.

    Returns
    -------
    list[tuple[pred_name, gt_name, iou]]
        Only pairs with iou >= min_iou are returned.
    """
    if not pred_masks or not gt_masks:
        return []

    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)

    for i, (_, pred) in enumerate(pred_masks):
        for j, (_, gt) in enumerate(gt_masks):
            iou_matrix[i, j] = _iou(pred, gt)

    # linear_sum_assignment minimizes cost; we want max IoU, so negate.
    pred_idx, gt_idx = linear_sum_assignment(-iou_matrix)

    matches = []
    for p, g in zip(pred_idx, gt_idx):
        iou_val = float(iou_matrix[p, g])
        if iou_val >= min_iou:
            pred_name = pred_masks[p][0]
            gt_name = gt_masks[g][0]
            matches.append((pred_name, gt_name, iou_val))
    return matches


# ---------------------------------------------------------------------------
# Per-image relabeling
# ---------------------------------------------------------------------------

def _relabel_image(
    pred_image_dir: Path,
    gt_image_dir: Path,
    min_iou: float,
    keep_unmatched: bool,
    verbose: bool = False,
) -> dict:
    """
    Relabel one image directory in-place.

    Returns a per-image report dict for the global summary.
    """
    # Discover masks. Ignore non-mask files like scores.json, *.json, etc.
    pred_files = [
        p for p in _iter_mask_files(pred_image_dir)
        if p.suffix == ".png"
    ]
    gt_files = [
        p for p in _iter_mask_files(gt_image_dir)
        if p.suffix == ".png"
    ]

    if not pred_files:
        return {
            "n_pred": 0, "n_gt": len(gt_files),
            "matches": [], "unmatched_pred": [], "unmatched_gt": [],
        }
    if not gt_files:
        return {
            "n_pred": len(pred_files), "n_gt": 0,
            "matches": [], "unmatched_pred": [p.name for p in pred_files],
            "unmatched_gt": [],
        }

    pred_masks = [(p.name, _load_mask(p)) for p in pred_files]
    gt_masks = [(p.name, _load_mask(p)) for p in gt_files]

    matches = _hungarian_match(pred_masks, gt_masks, min_iou)

    # Build sets for tracking unmatched files
    matched_pred = {m[0] for m in matches}
    matched_gt = {m[1] for m in matches}
    unmatched_pred = [
        p.name for p in pred_files if p.name not in matched_pred
    ]
    unmatched_gt = [
        p.name for p in gt_files if p.name not in matched_gt
    ]

    # Apply renames. Two-phase to avoid name collisions:
    #   1) rename matched pred -> tmp prefix
    #   2) rename tmp prefix -> final gt name
    # Then handle unmatched (delete or rename to unmatched_ prefix).

    for pred_name, gt_name, _iou in matches:
        src = pred_image_dir / pred_name
        tmp = pred_image_dir / f".tmp_{pred_name}"
        src.rename(tmp)

    for pred_name, gt_name, _iou in matches:
        tmp = pred_image_dir / f".tmp_{pred_name}"
        dst = pred_image_dir / gt_name
        if dst.exists():
            # Edge case: a GT file with that exact name already existed
            # among the predictions (e.g. obj_001.png matched a GT also
            # called obj_001.png). Append a numeric suffix to avoid loss.
            stem = dst.stem
            suffix = dst.suffix
            n = 1
            while (pred_image_dir / f"{stem}_{n}{suffix}").exists():
                n += 1
            dst = pred_image_dir / f"{stem}_{n}{suffix}"
        tmp.rename(dst)

    # Handle unmatched predicted files
    for pred_name in unmatched_pred:
        path = pred_image_dir / pred_name
        if keep_unmatched:
            path.rename(pred_image_dir / f"unmatched_{pred_name}")
        else:
            path.unlink()

    if verbose:
        print(f"  {pred_image_dir.name}: "
              f"{len(matches)} matched / "
              f"{len(unmatched_pred)} unmatched_pred / "
              f"{len(unmatched_gt)} unmatched_gt")

    return {
        "n_pred": len(pred_files),
        "n_gt": len(gt_files),
        "matches": [
            {"pred": p, "gt": g, "iou": round(i, 4)}
            for p, g, i in matches
        ],
        "unmatched_pred": unmatched_pred,
        "unmatched_gt": unmatched_gt,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def relabel_directory(
    pred_root: Path,
    gt_root: Path,
    output_root: Path | None,
    min_iou: float,
    keep_unmatched: bool,
    verbose: bool,
) -> None:
    """
    Relabel all per-image subdirectories under pred_root, anchoring on
    the GT masks at gt_root.

    If output_root is given, the entire pred_root tree is copied there
    first and renaming happens on the copy. Otherwise renaming is
    in-place on pred_root.
    """
    if output_root is not None:
        if output_root.exists():
            raise SystemExit(
                f"Refusing to overwrite existing output: {output_root}"
            )
        print(f"Copying {pred_root} -> {output_root} (safe mode)...")
        shutil.copytree(pred_root, output_root)
        working_root = output_root
    else:
        print(f"Operating in-place on {pred_root}")
        working_root = pred_root

    image_dirs = sorted(
        d for d in working_root.iterdir() if d.is_dir()
    )
    print(f"\nFound {len(image_dirs)} image directories to process")
    print(f"GT root: {gt_root}")
    print(f"min_iou: {min_iou}")
    print(f"keep_unmatched: {keep_unmatched}\n")

    report = {}
    n_no_gt = 0
    n_processed = 0

    for image_dir in image_dirs:
        gt_image_dir = gt_root / image_dir.name
        if not gt_image_dir.is_dir():
            if verbose:
                print(f"  [skip] {image_dir.name}: no GT directory")
            n_no_gt += 1
            continue

        report[image_dir.name] = _relabel_image(
            image_dir, gt_image_dir, min_iou, keep_unmatched, verbose,
        )
        n_processed += 1

    # Aggregate stats
    total_matches = sum(len(r["matches"]) for r in report.values())
    total_pred = sum(r["n_pred"] for r in report.values())
    total_gt = sum(r["n_gt"] for r in report.values())

    print("\n" + "=" * 60)
    print("Relabeling complete")
    print("=" * 60)
    print(f"  Processed:       {n_processed} images")
    print(f"  Skipped (no GT): {n_no_gt} images")
    print(f"  Total predicted: {total_pred}")
    print(f"  Total GT:        {total_gt}")
    print(f"  Total matched:   {total_matches} "
          f"({(total_matches / total_pred * 100) if total_pred else 0:.1f}% "
          f"of predicted, "
          f"{(total_matches / total_gt * 100) if total_gt else 0:.1f}% "
          f"of GT)")

    # Save report for inspection
    report_path = working_root / "_relabel_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "config": {
                "pred_root": str(pred_root),
                "gt_root": str(gt_root),
                "min_iou": min_iou,
                "keep_unmatched": keep_unmatched,
            },
            "totals": {
                "n_processed": n_processed,
                "n_no_gt": n_no_gt,
                "total_pred": total_pred,
                "total_gt": total_gt,
                "total_matched": total_matches,
            },
            "per_image": report,
        }, f, indent=2)
    print(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Relabel baseline `obj_NNN.png` masks to anatomical organ "
            "names by GT-anchored Hungarian matching on IoU."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pred", required=True,
        help=(
            "Path to the predictions root (the `masks/` directory under "
            "the experiment output; contains per-image subdirectories)."
        ),
    )
    parser.add_argument(
        "--gt", required=True,
        help="Path to the GT root (same structure as --pred).",
    )
    parser.add_argument(
        "--output", default=None,
        help=(
            "If given, copy the predictions tree to this path first and "
            "relabel on the copy. If omitted, relabeling is in-place."
        ),
    )
    parser.add_argument(
        "--min-iou", type=float, default=0.3,
        help="Minimum IoU for a match to be accepted.",
    )
    parser.add_argument(
        "--keep-unmatched", action="store_true",
        help=(
            "Keep predicted masks that did not match any GT, prefixed "
            "with `unmatched_`. Default: delete them."
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print one line per image.",
    )
    args = parser.parse_args()

    pred_root = Path(args.pred).resolve()
    gt_root = Path(args.gt).resolve()
    output_root = Path(args.output).resolve() if args.output else None

    if not pred_root.is_dir():
        raise SystemExit(f"--pred not found: {pred_root}")
    if not gt_root.is_dir():
        raise SystemExit(f"--gt not found: {gt_root}")

    relabel_directory(
        pred_root=pred_root,
        gt_root=gt_root,
        output_root=output_root,
        min_iou=args.min_iou,
        keep_unmatched=args.keep_unmatched,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
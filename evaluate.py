"""
evaluate.py
-----------
Evaluate predicted masks against ground truth.

Supports two matching strategies:
- Semantic: match by organ name (few-shot / text-guided modes).
- Hungarian: match by best IoU (unsupervised mode, where names are cluster_N).

Usage:
    python evaluate.py \
        --gt data/processed/XRayNicoSent/masks/ \
        --pred results/unsup/masks/ \
        --output results/unsup/

    python evaluate.py \
        --gt data/processed/XRayNicoSent/masks/ \
        --pred results/fs_indep_4ref/masks/ \
        --output results/fs_indep_4ref/ \
        --matching semantic

Output:
    <output>/metrics.csv      — per-image per-organ metrics
    <output>/summary.json     — aggregated metrics per organ and global
"""

import argparse
import csv
import json
import re
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.spatial.distance import directed_hausdorff
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient between two binary masks."""
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0  # both empty = perfect match
    return float(2 * intersection / total)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection over Union between two binary masks."""
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def hausdorff_95(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    95th percentile Hausdorff distance between mask boundaries.
    Returns 0.0 if both masks are empty, inf if only one is empty.
    """
    pred_points = np.argwhere(pred)
    gt_points = np.argwhere(gt)

    if len(pred_points) == 0 and len(gt_points) == 0:
        return 0.0
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float("inf")

    # Compute all pairwise distances from pred to gt and gt to pred
    from scipy.ndimage import distance_transform_edt

    dt_gt = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)

    distances_pred_to_gt = dt_gt[pred]
    distances_gt_to_pred = dt_pred[gt]

    all_distances = np.concatenate([distances_pred_to_gt, distances_gt_to_pred])
    return float(np.percentile(all_distances, 95))


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute all metrics for a single pred-gt mask pair."""
    return {
        "dice": dice_score(pred, gt),
        "iou": iou_score(pred, gt),
        "hausdorff_95": hausdorff_95(pred, gt),
    }


# ---------------------------------------------------------------------------
# Mask loading
# ---------------------------------------------------------------------------

def load_masks_from_dir(mask_dir: Path) -> dict[str, np.ndarray]:
    """
    Load all binary masks from a directory.

    Returns dict mapping organ name (e.g. 'lung_1', 'heart_1') to boolean mask.
    For predicted masks with cluster names (e.g. 'cluster_0_1.png'),
    the key is 'cluster_0_1'.
    """
    masks = {}
    for f in sorted(mask_dir.iterdir()):
        if f.suffix.lower() != ".png":
            continue
        if f.name == "image.png":
            continue

        arr = np.array(Image.open(f).convert("L"))
        masks[f.stem] = arr > 127

    return masks


def parse_organ_name(filename_stem: str) -> str:
    """
    Extract the base organ name (without the instance number) from a filename stem.

    'lung_1' -> 'lung'
    'lung_2' -> 'lung'
    'heart_1' -> 'heart'
    'cluster_0_1' -> 'cluster_0'
    """
    # Match trailing _N where N is a digit
    match = re.match(r"^(.+)_(\d+)$", filename_stem)
    if match:
        return match.group(1)
    return filename_stem


# ---------------------------------------------------------------------------
# Matching strategies
# ---------------------------------------------------------------------------

def match_semantic(
    pred_masks: dict[str, np.ndarray],
    gt_masks: dict[str, np.ndarray],
) -> list[dict]:
    """
    Match predictions to GT by organ name.

    Groups masks by base organ name (e.g. 'lung'), then within each organ
    uses Hungarian matching on IoU to pair instances (lung_1 <-> lung_1).

    GT organs with no prediction get dice=0, iou=0, hd95=inf.
    """
    # Group by base organ name
    gt_by_organ: dict[str, list[tuple[str, np.ndarray]]] = {}
    for name, mask in gt_masks.items():
        organ = parse_organ_name(name)
        gt_by_organ.setdefault(organ, []).append((name, mask))

    pred_by_organ: dict[str, list[tuple[str, np.ndarray]]] = {}
    for name, mask in pred_masks.items():
        organ = parse_organ_name(name)
        pred_by_organ.setdefault(organ, []).append((name, mask))

    results = []

    for organ, gt_list in gt_by_organ.items():
        pred_list = pred_by_organ.get(organ, [])

        if not pred_list:
            # No predictions for this organ — all GT get zero scores
            for gt_name, gt_mask in gt_list:
                h, w = gt_mask.shape
                results.append({
                    "gt_name": gt_name,
                    "pred_name": None,
                    "organ": organ,
                    **compute_metrics(np.zeros((h, w), dtype=bool), gt_mask),
                })
            continue

        # Hungarian matching within this organ group
        cost_matrix = np.zeros((len(gt_list), len(pred_list)))
        for i, (_, gt_mask) in enumerate(gt_list):
            for j, (_, pred_mask) in enumerate(pred_list):
                cost_matrix[i, j] = -iou_score(pred_mask, gt_mask)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_gt = set()
        for i, j in zip(row_ind, col_ind):
            gt_name, gt_mask = gt_list[i]
            pred_name, pred_mask = pred_list[j]
            results.append({
                "gt_name": gt_name,
                "pred_name": pred_name,
                "organ": organ,
                **compute_metrics(pred_mask, gt_mask),
            })
            matched_gt.add(i)

        # Unmatched GT entries
        for i, (gt_name, gt_mask) in enumerate(gt_list):
            if i not in matched_gt:
                h, w = gt_mask.shape
                results.append({
                    "gt_name": gt_name,
                    "pred_name": None,
                    "organ": organ,
                    **compute_metrics(np.zeros((h, w), dtype=bool), gt_mask),
                })

    return results


def match_hungarian(
    pred_masks: dict[str, np.ndarray],
    gt_masks: dict[str, np.ndarray],
) -> list[dict]:
    """
    Match predictions to GT using global Hungarian matching on IoU.

    Used for unsupervised mode where prediction names (cluster_N) have
    no semantic meaning. Finds the best global assignment.

    GT organs with no match get dice=0, iou=0, hd95=inf.
    """
    gt_list = list(gt_masks.items())
    pred_list = list(pred_masks.items())

    if not pred_list:
        results = []
        for gt_name, gt_mask in gt_list:
            h, w = gt_mask.shape
            results.append({
                "gt_name": gt_name,
                "pred_name": None,
                "organ": parse_organ_name(gt_name),
                **compute_metrics(np.zeros((h, w), dtype=bool), gt_mask),
            })
        return results

    # Build cost matrix (negative IoU for minimization)
    cost_matrix = np.zeros((len(gt_list), len(pred_list)))
    for i, (_, gt_mask) in enumerate(gt_list):
        for j, (_, pred_mask) in enumerate(pred_list):
            cost_matrix[i, j] = -iou_score(pred_mask, gt_mask)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    results = []
    matched_gt = set()

    for i, j in zip(row_ind, col_ind):
        gt_name, gt_mask = gt_list[i]
        pred_name, pred_mask = pred_list[j]

        # Only count as matched if IoU > 0
        if -cost_matrix[i, j] > 0:
            results.append({
                "gt_name": gt_name,
                "pred_name": pred_name,
                "organ": parse_organ_name(gt_name),
                **compute_metrics(pred_mask, gt_mask),
            })
            matched_gt.add(i)
        else:
            h, w = gt_mask.shape
            results.append({
                "gt_name": gt_name,
                "pred_name": None,
                "organ": parse_organ_name(gt_name),
                **compute_metrics(np.zeros((h, w), dtype=bool), gt_mask),
            })
            matched_gt.add(i)

    # Unmatched GT entries
    for i, (gt_name, gt_mask) in enumerate(gt_list):
        if i not in matched_gt:
            h, w = gt_mask.shape
            results.append({
                "gt_name": gt_name,
                "pred_name": None,
                "organ": parse_organ_name(gt_name),
                **compute_metrics(np.zeros((h, w), dtype=bool), gt_mask),
            })

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics(all_results: list[dict]) -> dict:
    """
    Compute summary statistics from per-image per-organ results.

    Returns per-organ and global mean ± std for each metric.
    """
    metrics_keys = ["dice", "iou", "hausdorff_95"]

    # Per organ
    by_organ: dict[str, list[dict]] = {}
    for r in all_results:
        by_organ.setdefault(r["organ"], []).append(r)

    summary = {"per_organ": {}, "global": {}, "n_images": 0, "n_entries": len(all_results)}

    images = {r["image"] for r in all_results if "image" in r}
    summary["n_images"] = len(images)

    for organ, entries in sorted(by_organ.items()):
        organ_summary = {"count": len(entries)}
        for key in metrics_keys:
            values = [e[key] for e in entries if not (key == "hausdorff_95" and e[key] == float("inf"))]
            if values:
                organ_summary[f"{key}_mean"] = float(np.mean(values))
                organ_summary[f"{key}_std"] = float(np.std(values))
            else:
                organ_summary[f"{key}_mean"] = None
                organ_summary[f"{key}_std"] = None
        # Track how many had no prediction
        organ_summary["missing"] = sum(1 for e in entries if e.get("pred_name") is None)
        summary["per_organ"][organ] = organ_summary

    # Global
    for key in metrics_keys:
        values = [r[key] for r in all_results if not (key == "hausdorff_95" and r[key] == float("inf"))]
        if values:
            summary["global"][f"{key}_mean"] = float(np.mean(values))
            summary["global"][f"{key}_std"] = float(np.std(values))
        else:
            summary["global"][f"{key}_mean"] = None
            summary["global"][f"{key}_std"] = None

    return summary


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def evaluate(gt_dir: Path, pred_dir: Path, matching: str) -> tuple[list[dict], dict]:
    """
    Run evaluation across all images.

    Parameters
    ----------
    gt_dir : Path
        Directory with GT masks (one subfolder per image stem).
    pred_dir : Path
        Directory with predicted masks (same structure).
    matching : str
        'semantic' or 'hungarian'.

    Returns
    -------
    all_results : list[dict]
        Per-image per-organ metric rows.
    summary : dict
        Aggregated statistics.
    """
    match_fn = match_semantic if matching == "semantic" else match_hungarian

    gt_stems = {d.name for d in gt_dir.iterdir() if d.is_dir()}
    pred_stems = {d.name for d in pred_dir.iterdir() if d.is_dir()}

    common = sorted(gt_stems & pred_stems)
    gt_only = sorted(gt_stems - pred_stems)
    pred_only = sorted(pred_stems - gt_stems)

    if gt_only:
        print(f"  Warning: {len(gt_only)} images in GT but not in predictions")
    if pred_only:
        print(f"  Warning: {len(pred_only)} images in predictions but not in GT")

    print(f"  Evaluating {len(common)} images (matching={matching})")

    all_results = []

    for stem in common:
        gt_masks = load_masks_from_dir(gt_dir / stem)
        pred_masks = load_masks_from_dir(pred_dir / stem)

        if not gt_masks:
            continue

        image_results = match_fn(pred_masks, gt_masks)

        for r in image_results:
            r["image"] = stem

        all_results.extend(image_results)

    summary = aggregate_metrics(all_results)
    return all_results, summary


def save_results(
    all_results: list[dict], summary: dict, output_dir: Path,
) -> None:
    """Save metrics.csv and summary.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # metrics.csv
    csv_path = output_dir / "metrics.csv"
    fieldnames = ["image", "organ", "gt_name", "pred_name", "dice", "iou", "hausdorff_95"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(all_results, key=lambda r: (r["image"], r["organ"])):
            writer.writerow(row)
    print(f"  Saved: {csv_path}")

    # summary.json
    json_path = output_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")


def print_summary(summary: dict) -> None:
    """Print a human-readable summary to stdout."""
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    print(f"\n  Images evaluated: {summary['n_images']}")
    print(f"  Total entries: {summary['n_entries']}")

    print(f"\n  {'Organ':<15} {'Dice':>10} {'IoU':>10} {'HD95':>10} {'Missing':>10}")
    print("  " + "-" * 55)

    for organ, stats in sorted(summary["per_organ"].items()):
        dice = f"{stats['dice_mean']:.3f}±{stats['dice_std']:.3f}" if stats["dice_mean"] is not None else "N/A"
        iou = f"{stats['iou_mean']:.3f}±{stats['iou_std']:.3f}" if stats["iou_mean"] is not None else "N/A"
        hd95 = f"{stats['hausdorff_95_mean']:.1f}±{stats['hausdorff_95_std']:.1f}" if stats["hausdorff_95_mean"] is not None else "N/A"
        missing = f"{stats['missing']}/{stats['count']}"
        print(f"  {organ:<15} {dice:>10} {iou:>10} {hd95:>10} {missing:>10}")

    g = summary["global"]
    print("  " + "-" * 55)
    dice = f"{g['dice_mean']:.3f}±{g['dice_std']:.3f}" if g["dice_mean"] is not None else "N/A"
    iou = f"{g['iou_mean']:.3f}±{g['iou_std']:.3f}" if g["iou_mean"] is not None else "N/A"
    hd95 = f"{g['hausdorff_95_mean']:.1f}±{g['hausdorff_95_std']:.1f}" if g["hausdorff_95_mean"] is not None else "N/A"
    print(f"  {'GLOBAL':<15} {dice:>10} {iou:>10} {hd95:>10}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predicted masks against ground truth."
    )
    parser.add_argument("--gt", required=True,
                        help="Path to GT masks directory (with per-image subfolders)")
    parser.add_argument("--pred", required=True,
                        help="Path to predicted masks directory (with per-image subfolders)")
    parser.add_argument("--output", required=True,
                        help="Directory to save metrics.csv and summary.json")
    parser.add_argument("--matching", default="hungarian",
                        choices=["semantic", "hungarian"],
                        help="Matching strategy: 'semantic' (by name) or 'hungarian' (by IoU). Default: hungarian")
    args = parser.parse_args()

    gt_dir = Path(args.gt)
    pred_dir = Path(args.pred)
    output_dir = Path(args.output)

    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Predictions directory not found: {pred_dir}")

    all_results, summary = evaluate(gt_dir, pred_dir, args.matching)
    save_results(all_results, summary, output_dir)
    print_summary(summary)


if __name__ == "__main__":
    main()
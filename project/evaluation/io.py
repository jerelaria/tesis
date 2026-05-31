"""I/O helpers: mask loading, score loading, result saving, summary printing."""

import csv
import json
import numpy as np
from pathlib import Path
from PIL import Image


def load_masks_from_dir(mask_dir: Path) -> dict[str, np.ndarray]:
    """
    Load all binary masks from a directory.

    Returns dict mapping filename stem (e.g., 'lung_1', 'heart_1', 'obj_002')
    to a boolean mask. The 'image.png' file (if present) is skipped.
    """
    masks: dict[str, np.ndarray] = {}
    for f in sorted(mask_dir.iterdir()):
        if f.suffix.lower() != ".png" or f.name == "image.png":
            continue
        masks[f.stem] = np.array(Image.open(f).convert("L")) > 127
    return masks


def load_scores_from_dir(mask_dir: Path) -> dict[str, float]:
    """
    Load per-prediction scores from a scores.json file in the mask dir.

    Returns a dict mapping mask stem (e.g., 'organ_a', 'lung_1') to the
    `combined` score. Returns an empty dict if scores.json is missing,
    which triggers the fallback score=1.0 for all predictions in
    evaluate(). This keeps the evaluator backward-compatible with runs
    that pre-date the scoring feature.
    """
    scores_path = mask_dir / "scores.json"
    if not scores_path.exists():
        return {}
    with open(scores_path) as f:
        raw = json.load(f)
    return {
        Path(filename).stem: float(entry.get("combined", 1.0))
        for filename, entry in raw.items()
    }


def save_results(
    all_results: list[dict],
    summary: dict,
    output_dir: Path,
) -> None:
    """Save metrics.csv (per-pair quality) and summary.json (aggregated)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "metrics.csv"
    fieldnames = [
        "image", "organ", "gt_name", "pred_name",
        "dice", "iou", "hausdorff", "hausdorff_95", "assd",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(all_results, key=lambda r: (r["image"], r["organ"])):
            writer.writerow(row)
    print(f"  Saved: {csv_path}")

    json_path = output_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")


def _fmt_ratio(mean: float | None, std: float | None) -> str:
    """Format a bounded metric (dice, iou) as 'mean±std' or 'N/A'."""
    if mean is None:
        return "N/A"
    return f"{mean:.3f}±{std:.3f}"


def _fmt_dist(mean: float | None, std: float | None) -> str:
    """Format a distance metric (hausdorff etc.) as 'mean±std' or 'N/A'."""
    if mean is None:
        return "N/A"
    return f"{mean:.1f}±{std:.1f}"


def print_summary(summary: dict) -> None:
    """Print a human-readable summary to stdout."""
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)

    print(f"\n  Images evaluated: {summary['n_images']}")
    print(f"  Total quality entries: {summary['n_entries']}")
    print(f"  Matching: {summary.get('matching', 'unknown')}")

    # ----- Per-organ quality -----
    # Two sub-rows per organ: (all) = missing penalised; (det.) = detected only
    print(f"\n  Per-organ quality  (all = missing penalised; det. = detected only):")
    print(f"  {'Organ':<14} {'':7} {'Dice':>14} {'IoU':>14} {'HD95':>14} {'Missing':>10}")
    print("  " + "-" * 74)
    for organ, stats in sorted(summary["per_organ"].items()):
        dice_all = _fmt_ratio(stats.get("dice_mean_with_missing"),
                              stats.get("dice_std_with_missing"))
        dice_det = _fmt_ratio(stats.get("dice_mean_detected_only"),
                              stats.get("dice_std_detected_only"))
        iou_all  = _fmt_ratio(stats.get("iou_mean_with_missing"),
                              stats.get("iou_std_with_missing"))
        iou_det  = _fmt_ratio(stats.get("iou_mean_detected_only"),
                              stats.get("iou_std_detected_only"))
        hd95_all = _fmt_dist(stats.get("hausdorff_95_mean_with_missing"),
                             stats.get("hausdorff_95_std_with_missing"))
        hd95_det = _fmt_dist(stats.get("hausdorff_95_mean_detected_only"),
                             stats.get("hausdorff_95_std_detected_only"))
        missing  = f"{stats.get('missing', 0)}/{stats.get('count', 0)}"
        print(f"  {organ:<14} {'(all)':7} {dice_all:>14} {iou_all:>14} {hd95_all:>14} {missing:>10}")
        print(f"  {'':14} {'(det.)':7} {dice_det:>14} {iou_det:>14} {hd95_det:>14}")

    # ----- Global quality -----
    g = summary["global"]
    print("  " + "-" * 74)
    dice_all = _fmt_ratio(g.get("dice_mean_with_missing"),
                          g.get("dice_std_with_missing"))
    dice_det = _fmt_ratio(g.get("dice_mean_detected_only"),
                          g.get("dice_std_detected_only"))
    iou_all  = _fmt_ratio(g.get("iou_mean_with_missing"),
                          g.get("iou_std_with_missing"))
    iou_det  = _fmt_ratio(g.get("iou_mean_detected_only"),
                          g.get("iou_std_detected_only"))
    hd95_all = _fmt_dist(g.get("hausdorff_95_mean_with_missing"),
                         g.get("hausdorff_95_std_with_missing"))
    hd95_det = _fmt_dist(g.get("hausdorff_95_mean_detected_only"),
                         g.get("hausdorff_95_std_detected_only"))
    print(f"  {'GLOBAL':<14} {'(all)':7} {dice_all:>14} {iou_all:>14} {hd95_all:>14}")
    print(f"  {'':14} {'(det.)':7} {dice_det:>14} {iou_det:>14} {hd95_det:>14}")

    # ----- Coverage / cleanliness at each threshold -----
    iou_thresholds = summary.get("iou_thresholds", [])
    if iou_thresholds:
        n_gt_total = g.get("n_gt_total", 0)
        n_pred_total = g.get("n_pred_total", 0)

        print(f"\n  Coverage / Cleanliness (n_gt={n_gt_total}, n_pred={n_pred_total}):")
        print(f"  {'Threshold':<12} {'Recall':>10} {'Precision':>11} {'F1':>10}"
              f"  {'GT cov.':>12} {'Pred rel.':>14}")
        print("  " + "-" * 70)
        for thr in iou_thresholds:
            recall = g.get(f"recall@{thr}", 0.0)
            precision = g.get(f"precision@{thr}", 0.0)
            f1 = g.get(f"f1@{thr}", 0.0)
            n_gt_cov = g.get(f"n_gt_covered@{thr}", 0)
            n_pred_rel = g.get(f"n_pred_relevant@{thr}", 0)
            print(f"  IoU >= {thr:<5} {recall:>10.3f} {precision:>11.3f}"
                  f" {f1:>10.3f}  {n_gt_cov:>4}/{n_gt_total:<6}"
                  f" {n_pred_rel:>5}/{n_pred_total:<7}")

        primary_thr = iou_thresholds[0]
        print(f"\n  Per-organ recall @ IoU >= {primary_thr}:")
        for organ, stats in sorted(summary["per_organ"].items()):
            recall = stats.get(f"recall@{primary_thr}", 0.0)
            n_cov = stats.get(f"n_covered@{primary_thr}", 0)
            n_tot = stats.get("n_total", 0)
            print(f"    {organ:<14} {recall:>6.3f}  ({n_cov}/{n_tot})")

"""
runner.py
---------
Main evaluation orchestrator and CLI entry point.

Coordinates mask loading, matching, coverage/cleanliness counting, mAP
computation, aggregation, and result saving across all image directories.
"""

import argparse
from pathlib import Path

from project.evaluation.aggregation import aggregate_quality, add_panoptic_quality
from project.evaluation.conflict_resolution import resolve_overlaps
from project.evaluation.coverage import compute_pr_counts, aggregate_pr
from project.evaluation.io import (
    load_masks_from_dir,
    load_scores_from_dir,
    save_results,
    print_summary,
)
from project.evaluation.map import compute_map
from project.evaluation.matching import match_semantic, match_greedy

# 1.0 is excluded from the fine IoU grid: at threshold=1.0 only pixel-perfect
# predictions are TP, which is degenerate for real-world masks.
_DEFAULT_IOU_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]


def evaluate(
    gt_dir: Path,
    pred_dir: Path,
    matching: str,
    iou_thresholds: list[float],
    compute_map_metric: bool = True,
    match_threshold: float = 0.5,
) -> tuple[list[dict], dict]:
    """
    Run evaluation across all images.

    Quality metrics use the chosen matching strategy with match_threshold as
    the IoU gate: paired predictions with IoU < match_threshold are demoted to
    missing (pred_name=None, quality against an empty mask).
    Coverage / cleanliness metrics are matching-agnostic.
    mAP@[0.5:0.95] is matching-agnostic and pools predictions across images.

    Parameters
    ----------
    match_threshold : float
        Minimum IoU for a matched pair to count as a detection in quality
        metrics. Below this threshold the GT is treated as not detected.
        Default 0.5.
    """
    match_fn = match_semantic if matching == "semantic" else match_greedy

    gt_stems = {d.name for d in gt_dir.iterdir() if d.is_dir()}
    pred_stems = {d.name for d in pred_dir.iterdir() if d.is_dir()}

    common = sorted(gt_stems & pred_stems)
    gt_only = sorted(gt_stems - pred_stems)
    pred_only = sorted(pred_stems - gt_stems)

    if gt_only:
        print(f"  Warning: {len(gt_only)} images in GT but not in predictions")
    if pred_only:
        print(f"  Warning: {len(pred_only)} images in predictions but not in GT")

    print(f"  Evaluating {len(common)} images "
          f"(matching={matching}, match_threshold={match_threshold})")
    print(f"  IoU thresholds for P/R/F1: {iou_thresholds}")

    all_results: list[dict] = []
    pr_counts_by_thr: dict[float, list[dict]] = {thr: [] for thr in iou_thresholds}

    pred_masks_all: list[dict] = []
    gt_masks_all: list[dict] = []
    pred_scores_all: list[dict] = []

    for stem in common:
        gt_masks = load_masks_from_dir(gt_dir / stem)
        pred_masks = load_masks_from_dir(pred_dir / stem)
        pred_scores = load_scores_from_dir(pred_dir / stem)

        if not gt_masks:
            continue

        # Resolve pixel-level overlaps between predictions before any metric
        # computation.  Only applied when scores.json is present (pred_scores
        # non-empty); older results without scores.json are evaluated as-is
        # for backward compatibility.  If scores.json exists but a name is
        # missing, resolve_overlaps raises rather than assuming a default score.
        if pred_scores:
            pred_masks = resolve_overlaps(pred_masks, pred_scores)

        image_results = match_fn(pred_masks, gt_masks, match_threshold)
        for r in image_results:
            r["image"] = stem
        all_results.extend(image_results)

        for thr in iou_thresholds:
            pr_counts_by_thr[thr].append(
                compute_pr_counts(pred_masks, gt_masks, thr)
            )

        pred_masks_all.append(pred_masks)
        gt_masks_all.append(gt_masks)
        pred_scores_all.append({
            name: pred_scores.get(name, 1.0) for name in pred_masks
        })

    # Compute and report the fraction of predictions with real (non-fallback) scores
    n_pred = sum(len(s) for s in pred_scores_all)
    n_with_score = sum(
        1 for image_scores in pred_scores_all
        for v in image_scores.values()
        if v != 1.0
    )
    if n_pred > 0:
        pct = 100 * n_with_score / n_pred
        print(f"  mAP scoring: {n_with_score}/{n_pred} predictions ({pct:.0f}%) "
              f"have real scores; rest use fallback=1.0")
    else:
        pct = 0.0

    quality_summary = aggregate_quality(all_results)
    pr_per_organ, pr_global = aggregate_pr(pr_counts_by_thr, iou_thresholds)

    n_images = len({r["image"] for r in all_results if "image" in r})
    summary: dict = {
        "per_organ": {},
        "global": {**quality_summary["global"], **pr_global},
        "n_images": n_images,
        "n_entries": len(all_results),
        "iou_thresholds": iou_thresholds,
        "matching": matching,
        "match_threshold": match_threshold,
    }

    # Persist scoring coverage so plots can warn when curves are degenerate
    summary["global"]["pct_real_scores"] = pct

    if compute_map_metric:
        map_result = compute_map(pred_masks_all, gt_masks_all, pred_scores_all)
        summary["global"]["map"] = map_result["map"]
        summary["global"]["map_50"] = map_result["map_50"]
        summary["global"]["map_75"] = map_result["map_75"]
        summary["map_per_threshold"] = map_result["ap_per_threshold"]
        summary["pr_curve_per_threshold"] = map_result["pr_curve_per_threshold"]

        # Compute P/R/F1 at each mAP grid threshold so plotting scripts can
        # draw full precision-recall curves (not just the @0.5 / @0.7 points).
        map_thrs = sorted(map_result["ap_per_threshold"].keys())
        det_pr_by_thr: dict[float, list[dict]] = {thr: [] for thr in map_thrs}
        for pm, gm in zip(pred_masks_all, gt_masks_all):
            for thr in map_thrs:
                det_pr_by_thr[thr].append(compute_pr_counts(pm, gm, thr))
        _, det_global = aggregate_pr(det_pr_by_thr, map_thrs)
        summary["detection_per_threshold"] = {
            str(thr): {
                "recall":          det_global.get(f"recall@{thr}"),
                "precision":       det_global.get(f"precision@{thr}"),
                "f1":              det_global.get(f"f1@{thr}"),
                "n_gt_covered":    det_global.get(f"n_gt_covered@{thr}"),
                "n_pred_relevant": det_global.get(f"n_pred_relevant@{thr}"),
            }
            for thr in map_thrs
        }

    organs = set(quality_summary["per_organ"].keys()) | set(pr_per_organ.keys())
    for organ in organs:
        merged: dict = {}
        merged.update(quality_summary["per_organ"].get(organ, {}))
        merged.update(pr_per_organ.get(organ, {}))
        summary["per_organ"][organ] = merged

    add_panoptic_quality(summary)
    return all_results, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted masks against ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gt", required=True,
                        help="GT masks directory (with per-image subfolders)")
    parser.add_argument("--pred", required=True,
                        help="Predicted masks directory (same structure)")
    parser.add_argument("--output", required=True,
                        help="Directory to save metrics.csv and summary.json")
    parser.add_argument("--matching", default="greedy",
                        choices=["semantic", "greedy"],
                        help="Matching strategy for quality metrics")
    parser.add_argument(
        "--iou-thresholds", nargs="+", type=float,
        default=_DEFAULT_IOU_THRESHOLDS,
        help="IoU thresholds at which to report P/R/F1 (coverage metrics). "
             "1.0 is excluded by default: at threshold=1.0 only pixel-perfect "
             "predictions are TP, which is degenerate for real-world masks.",
    )
    parser.add_argument(
        "--match-threshold", type=float, default=0.5,
        help="IoU gate for quality matching: pairs with IoU below this are "
             "demoted to missing (pred_name=None, Dice=0). Does not affect "
             "coverage/mAP metrics which are threshold-scanned independently.",
    )
    args = parser.parse_args()

    gt_dir = Path(args.gt)
    pred_dir = Path(args.pred)
    output_dir = Path(args.output)

    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Predictions directory not found: {pred_dir}")

    all_results, summary = evaluate(
        gt_dir, pred_dir, args.matching, args.iou_thresholds,
        match_threshold=args.match_threshold,
    )
    save_results(all_results, summary, output_dir)
    print_summary(summary)

"""
evaluate.py
-----------
Evaluate predicted masks against ground truth.

Reports three families of metrics:

1. Quality (Dice, IoU, HD95)
   Computed via a matching strategy (semantic or hungarian).
   Averaged over ALL GT entries — including missed ones, which contribute
   dice=0, iou=0, hd95=inf. Captures "given each GT, how good is the best
   matching prediction, including the case of no prediction at all".

2. Coverage (Recall @ IoU threshold)
   Per GT, max IoU vs ANY predicted mask >= threshold.
   Independent of matching. Captures "how much of the GT does the method
   actually find?".

3. Cleanliness (Precision @ IoU threshold)
   Per prediction, max IoU vs ANY GT mask >= threshold.
   Independent of matching. Captures "how many of our predictions actually
   correspond to anatomical structures?".

   F1 is the harmonic mean of precision and recall.

Why three families?

   Quality (Dice) mixes "how good is the match" with "how much do I miss".
   Coverage isolates the recall side. Cleanliness isolates the precision
   side. With all three, the matched-vs-missed-vs-junk story is fully
   visible — Dice alone hides false positives (Hungarian discards extra
   predictions) and dilutes recall gains across many already-matched GT
   entries.

P/R are reported at multiple IoU thresholds (default: 0.5 and 0.7) so the
gains can be checked under both a lenient "anatomical hit" criterion and
a stricter "tight boundary" criterion.

Two matching strategies for the quality metrics:
- semantic:  match by organ name (few-shot / text-guided modes).
- hungarian: match by best IoU (unsupervised mode, where names are obj_N).

Usage:
    python evaluate.py \\
        --gt data/processed/XRayNicoSent/masks/ \\
        --pred results/.../masks/ \\
        --output results/.../

    # Custom IoU thresholds for P/R/F1:
    python evaluate.py ... --iou-thresholds 0.5 0.75

Output:
    <output>/metrics.csv      — per-image per-organ quality metrics
    <output>/summary.json     — aggregated quality, coverage, cleanliness
"""

import argparse
import csv
import json
import re
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Per-pair quality metrics
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

    from scipy.ndimage import distance_transform_edt
    dt_gt = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)

    distances_pred_to_gt = dt_gt[pred]
    distances_gt_to_pred = dt_pred[gt]

    all_distances = np.concatenate([distances_pred_to_gt, distances_gt_to_pred])
    return float(np.percentile(all_distances, 95))


def compute_quality_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute all quality metrics for a single (pred, gt) pair."""
    return {
        "dice": dice_score(pred, gt),
        "iou": iou_score(pred, gt),
        "hausdorff": hausdorff_full(pred, gt),
        "hausdorff_95": hausdorff_95(pred, gt),
        "assd": assd(pred, gt),
    }

def hausdorff_full(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Symmetric Hausdorff distance (maximum, not 95th percentile).
    
    Defined as max over both directions of the maximum surface-to-surface
    distance. Captures the worst-case boundary error, complementing HD95
    (typical case) and ASSD (average case).
    
    Returns 0.0 if both masks are empty, inf if only one is empty.
    """
    pred_points = np.argwhere(pred)
    gt_points = np.argwhere(gt)
    
    if len(pred_points) == 0 and len(gt_points) == 0:
        return 0.0
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float("inf")
    
    from scipy.ndimage import distance_transform_edt
    dt_gt = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)
    
    # Directed max distances
    max_p_to_g = float(dt_gt[pred].max())
    max_g_to_p = float(dt_pred[gt].max())
    
    return max(max_p_to_g, max_g_to_p)


def assd(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Average Symmetric Surface Distance.
    
    Mean distance from each point on one boundary to the closest point on
    the other, computed symmetrically. Complements HD95 by capturing the
    average boundary error rather than the worst percentile.
    
    Returns 0.0 if both masks are empty, inf if only one is empty.
    """
    pred_points = np.argwhere(pred)
    gt_points = np.argwhere(gt)
    
    if len(pred_points) == 0 and len(gt_points) == 0:
        return 0.0
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float("inf")
    
    from scipy.ndimage import distance_transform_edt
    dt_gt = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)
    
    # Surface points only (boundary), via XOR with eroded mask
    from scipy.ndimage import binary_erosion
    pred_surface = pred & ~binary_erosion(pred)
    gt_surface = gt & ~binary_erosion(gt)
    
    if not pred_surface.any() or not gt_surface.any():
        # Degenerate masks (single pixel etc): fall back to all points
        distances = np.concatenate([dt_gt[pred], dt_pred[gt]])
    else:
        distances = np.concatenate([dt_gt[pred_surface], dt_pred[gt_surface]])
    
    return float(distances.mean())

def compute_average_precision(
    matched_ious: list[float],
    matched_scores: list[float],
    n_gt: int,
    iou_threshold: float,
) -> float:
    """
    Compute average precision at one IoU threshold (COCO-style).
    
    Given a list of (iou_with_matched_gt, prediction_score) pairs and the
    total number of GT instances, compute AP as the area under the
    precision-recall curve traced by varying the score threshold.
    
    Parameters
    ----------
    matched_ious : list[float]
        IoU of each prediction with its best-matching GT.
        For unmatched predictions, use 0.0.
    matched_scores : list[float]
        Confidence score of each prediction (e.g., combined SAM + labeling).
        Used to order predictions for the PR curve.
    n_gt : int
        Total number of GT instances in the dataset. Sets the recall
        denominator.
    iou_threshold : float
        Minimum IoU for a prediction to count as a true positive.
    
    Returns
    -------
    float
        AP in [0, 1]. Returns 0.0 if there are no predictions or no GTs.
    """
    if n_gt == 0 or len(matched_ious) == 0:
        return 0.0
    
    # Sort predictions by descending score
    order = np.argsort(-np.array(matched_scores))
    sorted_ious = np.array(matched_ious)[order]
    
    # Cumulative TP / FP along the sorted list
    tp = (sorted_ious >= iou_threshold).astype(np.float64)
    fp = 1.0 - tp
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    
    recall = cum_tp / n_gt
    precision = cum_tp / (cum_tp + cum_fp + 1e-12)
    
    # Pascal VOC 2010+ style: integrate using monotonic precision envelope
    # (precision never decreases as recall decreases)
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    
    # Integrate over recall axis
    recall_extended = np.concatenate([[0.0], recall, [1.0]])
    precision_extended = np.concatenate([[precision_envelope[0]],
                                          precision_envelope, [0.0]])
    
    # Sum rectangles (recall_i+1 - recall_i) * precision_i+1
    return float(np.sum(
        (recall_extended[1:] - recall_extended[:-1]) * precision_extended[1:]
    ))


def compute_map(
    pred_masks_per_image: list[dict[str, np.ndarray]],
    gt_masks_per_image: list[dict[str, np.ndarray]],
    pred_scores_per_image: list[dict[str, float]],
    iou_thresholds: list[float] | None = None,
) -> dict:
    """
    Compute mean Average Precision over a range of IoU thresholds (COCO-style).
    
    For each image, each prediction is greedy-matched to its best-IoU GT;
    each GT can be matched at most once (highest-scoring prediction wins).
    Predictions and GTs across all images are then pooled and AP is computed
    at each threshold. mAP is the mean of those APs.
    
    Parameters
    ----------
    pred_masks_per_image : list[dict[str, np.ndarray]]
        Per image, dict mapping prediction name to binary mask.
    gt_masks_per_image : list[dict[str, np.ndarray]]
        Per image, dict mapping GT name to binary mask. Same length as
        pred_masks_per_image and same image order.
    pred_scores_per_image : list[dict[str, float]]
        Per image, dict mapping prediction name to its confidence score
        (e.g., SAM score, or combined SAM + labeling confidence).
        Same length and image order.
    iou_thresholds : list[float] or None
        Thresholds at which to compute AP. Defaults to [0.5, 0.55, ..., 0.95].
    
    Returns
    -------
    dict with:
        ap_per_threshold: {thr: ap_value}
        map: mean over thresholds
        map_50: AP at IoU=0.5 (Pascal VOC criterion)
        map_75: AP at IoU=0.75 (stricter)
    """
    if iou_thresholds is None:
        iou_thresholds = list(np.arange(0.5, 1.0, 0.05))
    
    # Greedy match per image: for each prediction, its best GT IoU.
    # Each GT can be claimed only once (highest-scoring pred wins).
    pooled_ious: list[float] = []
    pooled_scores: list[float] = []
    total_gt = 0
    
    for preds, gts, scores in zip(
        pred_masks_per_image, gt_masks_per_image, pred_scores_per_image,
    ):
        total_gt += len(gts)
        if not preds:
            continue
        
        gt_items = list(gts.items())
        pred_items = list(preds.items())
        
        # Sort predictions by descending score (greedy match)
        pred_items_sorted = sorted(
            pred_items,
            key=lambda x: -scores.get(x[0], 0.0),
        )
        
        claimed_gt: set[int] = set()
        for pred_name, pred_mask in pred_items_sorted:
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, (_, gt_mask) in enumerate(gt_items):
                if gt_idx in claimed_gt:
                    continue
                iou = iou_score(pred_mask, gt_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            pooled_ious.append(best_iou)
            pooled_scores.append(scores.get(pred_name, 0.0))
            if best_gt_idx >= 0:
                claimed_gt.add(best_gt_idx)
    
    # AP at each threshold
    ap_per_threshold = {}
    for thr in iou_thresholds:
        ap = compute_average_precision(
            pooled_ious, pooled_scores, total_gt, thr,
        )
        ap_per_threshold[float(thr)] = float(ap)
    
    map_value = float(np.mean(list(ap_per_threshold.values())))
    
    return {
        "ap_per_threshold": ap_per_threshold,
        "map": map_value,
        "map_50": ap_per_threshold.get(0.5, 0.0),
        "map_75": ap_per_threshold.get(0.75, 0.0),
    }

# ---------------------------------------------------------------------------
# Mask loading
# ---------------------------------------------------------------------------

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
    """Load per-prediction scores from a scores.json file in the mask dir.

    Returns a dict mapping mask stem (e.g., 'organ_a', 'lung_1') to the
    `combined` score. Returns an empty dict if scores.json is missing,
    which triggers the fallback `score=1.0` for all predictions in
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

def parse_organ_name(stem: str) -> str:
    """
    Strip a trailing _N (instance number) from a filename stem.

    'lung_1'      -> 'lung'
    'heart_1'     -> 'heart'
    'obj_002'     -> 'obj'
    'cluster_0_1' -> 'cluster_0'
    """
    match = re.match(r"^(.+)_(\d+)$", stem)
    return match.group(1) if match else stem


# ---------------------------------------------------------------------------
# Matching strategies (for quality metrics)
# ---------------------------------------------------------------------------

def match_semantic(
    pred_masks: dict[str, np.ndarray],
    gt_masks: dict[str, np.ndarray],
) -> list[dict]:
    """
    Match predictions to GT by organ name.

    Groups masks by base organ name (e.g., 'lung'), then within each organ
    uses Hungarian matching on IoU to pair instances (lung_1 <-> lung_1).

    GT organs with no prediction get dice=0, iou=0, hd95=inf.
    """
    gt_by_organ: dict[str, list[tuple[str, np.ndarray]]] = {}
    for name, mask in gt_masks.items():
        gt_by_organ.setdefault(parse_organ_name(name), []).append((name, mask))

    pred_by_organ: dict[str, list[tuple[str, np.ndarray]]] = {}
    for name, mask in pred_masks.items():
        pred_by_organ.setdefault(parse_organ_name(name), []).append((name, mask))

    results: list[dict] = []

    for organ, gt_list in gt_by_organ.items():
        pred_list = pred_by_organ.get(organ, [])

        if not pred_list:
            for gt_name, gt_mask in gt_list:
                h, w = gt_mask.shape
                results.append({
                    "gt_name": gt_name, "pred_name": None, "organ": organ,
                    **compute_quality_metrics(np.zeros((h, w), dtype=bool), gt_mask),
                })
            continue

        cost_matrix = np.zeros((len(gt_list), len(pred_list)))
        for i, (_, gt_mask) in enumerate(gt_list):
            for j, (_, pred_mask) in enumerate(pred_list):
                cost_matrix[i, j] = -iou_score(pred_mask, gt_mask)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_gt: set[int] = set()

        for i, j in zip(row_ind, col_ind):
            gt_name, gt_mask = gt_list[i]
            pred_name, pred_mask = pred_list[j]
            results.append({
                "gt_name": gt_name, "pred_name": pred_name, "organ": organ,
                **compute_quality_metrics(pred_mask, gt_mask),
            })
            matched_gt.add(i)

        for i, (gt_name, gt_mask) in enumerate(gt_list):
            if i not in matched_gt:
                h, w = gt_mask.shape
                results.append({
                    "gt_name": gt_name, "pred_name": None, "organ": organ,
                    **compute_quality_metrics(np.zeros((h, w), dtype=bool), gt_mask),
                })

    return results


def match_hungarian(
    pred_masks: dict[str, np.ndarray],
    gt_masks: dict[str, np.ndarray],
) -> list[dict]:
    """
    Match predictions to GT using global Hungarian assignment on IoU.

    Used for unsupervised mode where prediction names (e.g., obj_NNN) carry
    no semantic meaning. Finds the best global GT-to-pred assignment.

    GT organs with no positive-IoU match (or no prediction at all) get
    dice=0, iou=0, hd95=inf.
    """
    gt_list = list(gt_masks.items())
    pred_list = list(pred_masks.items())

    if not pred_list:
        results: list[dict] = []
        for gt_name, gt_mask in gt_list:
            h, w = gt_mask.shape
            results.append({
                "gt_name": gt_name, "pred_name": None,
                "organ": parse_organ_name(gt_name),
                **compute_quality_metrics(np.zeros((h, w), dtype=bool), gt_mask),
            })
        return results

    cost_matrix = np.zeros((len(gt_list), len(pred_list)))
    for i, (_, gt_mask) in enumerate(gt_list):
        for j, (_, pred_mask) in enumerate(pred_list):
            cost_matrix[i, j] = -iou_score(pred_mask, gt_mask)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    results: list[dict] = []
    matched_gt: set[int] = set()

    for i, j in zip(row_ind, col_ind):
        gt_name, gt_mask = gt_list[i]
        pred_name, pred_mask = pred_list[j]

        if -cost_matrix[i, j] > 0:
            results.append({
                "gt_name": gt_name, "pred_name": pred_name,
                "organ": parse_organ_name(gt_name),
                **compute_quality_metrics(pred_mask, gt_mask),
            })
        else:
            h, w = gt_mask.shape
            results.append({
                "gt_name": gt_name, "pred_name": None,
                "organ": parse_organ_name(gt_name),
                **compute_quality_metrics(np.zeros((h, w), dtype=bool), gt_mask),
            })
        matched_gt.add(i)

    for i, (gt_name, gt_mask) in enumerate(gt_list):
        if i not in matched_gt:
            h, w = gt_mask.shape
            results.append({
                "gt_name": gt_name, "pred_name": None,
                "organ": parse_organ_name(gt_name),
                **compute_quality_metrics(np.zeros((h, w), dtype=bool), gt_mask),
            })

    return results


# ---------------------------------------------------------------------------
# Coverage / cleanliness counts (for P/R/F1 — independent of matching)
# ---------------------------------------------------------------------------

def compute_pr_counts(
    pred_masks: dict[str, np.ndarray],
    gt_masks: dict[str, np.ndarray],
    iou_threshold: float,
) -> dict:
    """
    Per-image P/R raw counts at one IoU threshold.

    A GT is "covered" if its max IoU vs any prediction is >= threshold.
    A prediction is "relevant" if its max IoU vs any GT is >= threshold.

    Predictions and GT are matched independently here (a single prediction
    can cover multiple GTs and vice-versa). This is intentional: coverage
    and cleanliness are name-agnostic metrics that complement the matched
    quality metrics.

    Returns
    -------
    dict with:
        n_gt:                total GT instances
        n_gt_covered:        GTs with max IoU >= threshold
        n_pred:              total predictions
        n_pred_relevant:     preds with max IoU >= threshold
        per_organ_covered:   {organ: covered_count}
        per_organ_total:     {organ: total_count}
    """
    gt_list = list(gt_masks.items())
    pred_list = list(pred_masks.items())
    pred_arrays = [m for _, m in pred_list]
    gt_arrays = [m for _, m in gt_list]

    # GT-side: coverage
    n_gt_covered = 0
    per_organ_covered: dict[str, int] = {}
    per_organ_total: dict[str, int] = {}
    for gt_name, gt_mask in gt_list:
        organ = parse_organ_name(gt_name)
        per_organ_total[organ] = per_organ_total.get(organ, 0) + 1

        max_iou = (
            max(iou_score(p, gt_mask) for p in pred_arrays) if pred_arrays else 0.0
        )
        if max_iou >= iou_threshold:
            n_gt_covered += 1
            per_organ_covered[organ] = per_organ_covered.get(organ, 0) + 1

    # Pred-side: relevance
    n_pred_relevant = 0
    if gt_arrays:
        for _, pred_mask in pred_list:
            max_iou = max(iou_score(pred_mask, g) for g in gt_arrays)
            if max_iou >= iou_threshold:
                n_pred_relevant += 1

    return {
        "n_gt": len(gt_list),
        "n_gt_covered": n_gt_covered,
        "n_pred": len(pred_list),
        "n_pred_relevant": n_pred_relevant,
        "per_organ_covered": per_organ_covered,
        "per_organ_total": per_organ_total,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_quality(all_results: list[dict]) -> dict:
    """
    Aggregate per-pair quality metrics per organ and globally.
    
    HD, HD95, and ASSD entries equal to inf are excluded from mean/std
    but counted toward the 'missing' field.
    """
    metrics_keys = ["dice", "iou", "hausdorff", "hausdorff_95", "assd"]
    inf_metrics = {"hausdorff", "hausdorff_95", "assd"}
    by_organ: dict[str, list[dict]] = {}
    for r in all_results:
        by_organ.setdefault(r["organ"], []).append(r)
    
    summary: dict = {"per_organ": {}, "global": {}}
    
    for organ, entries in sorted(by_organ.items()):
        organ_summary = {"count": len(entries)}
        for key in metrics_keys:
            values = [
                e[key] for e in entries
                if not (key in inf_metrics and e[key] == float("inf"))
            ]
            if values:
                organ_summary[f"{key}_mean"] = float(np.mean(values))
                organ_summary[f"{key}_std"] = float(np.std(values))
            else:
                organ_summary[f"{key}_mean"] = None
                organ_summary[f"{key}_std"] = None
        organ_summary["missing"] = sum(
            1 for e in entries if e.get("pred_name") is None
        )
        summary["per_organ"][organ] = organ_summary
    
    for key in metrics_keys:
        values = [
            r[key] for r in all_results
            if not (key in inf_metrics and r[key] == float("inf"))
        ]
        if values:
            summary["global"][f"{key}_mean"] = float(np.mean(values))
            summary["global"][f"{key}_std"] = float(np.std(values))
        else:
            summary["global"][f"{key}_mean"] = None
            summary["global"][f"{key}_std"] = None
    
    return summary


def aggregate_pr(
    pr_counts_by_thr: dict[float, list[dict]],
    iou_thresholds: list[float],
) -> tuple[dict, dict]:
    """
    Aggregate per-image P/R counts into global and per-organ metrics
    at each IoU threshold.

    Recall is computed both globally and per organ. Precision is computed
    only globally — per-organ precision would require knowing which
    prediction is "supposed to be" which organ, which is not well-defined
    when names are synthetic (hungarian mode).

    Returns
    -------
    (per_organ_metrics, global_metrics)
    """
    per_organ_metrics: dict[str, dict] = {}
    global_metrics: dict = {}

    for thr in iou_thresholds:
        thr_data = pr_counts_by_thr[thr]

        # Global recall
        total_gt = sum(d["n_gt"] for d in thr_data)
        total_gt_covered = sum(d["n_gt_covered"] for d in thr_data)
        recall = total_gt_covered / total_gt if total_gt > 0 else 0.0

        # Global precision
        total_pred = sum(d["n_pred"] for d in thr_data)
        total_pred_relevant = sum(d["n_pred_relevant"] for d in thr_data)
        precision = (
            total_pred_relevant / total_pred if total_pred > 0 else 0.0
        )

        # F1
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        global_metrics[f"recall@{thr}"] = recall
        global_metrics[f"precision@{thr}"] = precision
        global_metrics[f"f1@{thr}"] = f1
        global_metrics[f"n_gt_covered@{thr}"] = total_gt_covered
        global_metrics[f"n_pred_relevant@{thr}"] = total_pred_relevant

        # Per-organ recall
        organ_total: dict[str, int] = {}
        organ_covered: dict[str, int] = {}
        for d in thr_data:
            for organ, count in d["per_organ_total"].items():
                organ_total[organ] = organ_total.get(organ, 0) + count
            for organ, count in d["per_organ_covered"].items():
                organ_covered[organ] = organ_covered.get(organ, 0) + count

        for organ, total in organ_total.items():
            covered = organ_covered.get(organ, 0)
            entry = per_organ_metrics.setdefault(organ, {})
            entry[f"recall@{thr}"] = covered / total if total > 0 else 0.0
            entry[f"n_covered@{thr}"] = covered

    # n_total per organ is the same across thresholds — store it once
    if iou_thresholds:
        first_thr_data = pr_counts_by_thr[iou_thresholds[0]]
        organ_total: dict[str, int] = {}
        for d in first_thr_data:
            for organ, count in d["per_organ_total"].items():
                organ_total[organ] = organ_total.get(organ, 0) + count
        for organ, total in organ_total.items():
            per_organ_metrics.setdefault(organ, {})["n_total"] = total

    # Global totals (same across thresholds — store once)
    if iou_thresholds:
        first_thr_data = pr_counts_by_thr[iou_thresholds[0]]
        global_metrics["n_gt_total"] = sum(d["n_gt"] for d in first_thr_data)
        global_metrics["n_pred_total"] = sum(d["n_pred"] for d in first_thr_data)

    return per_organ_metrics, global_metrics


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def evaluate(
    gt_dir: Path,
    pred_dir: Path,
    matching: str,
    iou_thresholds: list[float],
    compute_map_metric: bool = True,
) -> tuple[list[dict], dict]:
    """
    Run evaluation across all images.

    Quality metrics use the chosen matching strategy.
    Coverage / cleanliness metrics are matching-agnostic.
    mAP@[0.5:0.95] is matching-agnostic and pools predictions across images.
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
    print(f"  IoU thresholds for P/R/F1: {iou_thresholds}")

    all_results: list[dict] = []
    pr_counts_by_thr: dict[float, list[dict]] = {thr: [] for thr in iou_thresholds}

    # mAP accumulators (initialized here so the loop can append into them)
    pred_masks_all: list[dict] = []
    gt_masks_all: list[dict] = []
    pred_scores_all: list[dict] = []

    for stem in common:
        gt_masks = load_masks_from_dir(gt_dir / stem)
        pred_masks = load_masks_from_dir(pred_dir / stem)
        pred_scores = load_scores_from_dir(pred_dir / stem)

        if not gt_masks:
            continue

        image_results = match_fn(pred_masks, gt_masks)
        for r in image_results:
            r["image"] = stem
        all_results.extend(image_results)

        for thr in iou_thresholds:
            pr_counts_by_thr[thr].append(
                compute_pr_counts(pred_masks, gt_masks, thr)
            )

        # Accumulate for mAP: real scores if available, fallback to 1.0
        pred_masks_all.append(pred_masks)
        gt_masks_all.append(gt_masks)
        pred_scores_all.append({
            name: pred_scores.get(name, 1.0) for name in pred_masks
        })

    # Report whether real scores or fallback were used (helps interpret mAP)
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

    # Aggregate
    quality_summary = aggregate_quality(all_results)
    pr_per_organ, pr_global = aggregate_pr(pr_counts_by_thr, iou_thresholds)

    # Merge into a single summary
    n_images = len({r["image"] for r in all_results if "image" in r})
    summary: dict = {
        "per_organ": {},
        "global": {**quality_summary["global"], **pr_global},
        "n_images": n_images,
        "n_entries": len(all_results),
        "iou_thresholds": iou_thresholds,
        "matching": matching,
    }

    # mAP (matching-agnostic, COCO-style)
    if compute_map_metric:
        map_result = compute_map(
            pred_masks_all, gt_masks_all, pred_scores_all,
        )
        summary["global"]["map"] = map_result["map"]
        summary["global"]["map_50"] = map_result["map_50"]
        summary["global"]["map_75"] = map_result["map_75"]
        summary["map_per_threshold"] = map_result["ap_per_threshold"]

    organs = set(quality_summary["per_organ"].keys()) | set(pr_per_organ.keys())
    for organ in organs:
        merged: dict = {}
        merged.update(quality_summary["per_organ"].get(organ, {}))
        merged.update(pr_per_organ.get(organ, {}))
        summary["per_organ"][organ] = merged

    return all_results, summary


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


def print_summary(summary: dict) -> None:
    """Print a human-readable summary to stdout."""
    print("\n" + "=" * 75)
    print("Evaluation Summary")
    print("=" * 75)

    print(f"\n  Images evaluated: {summary['n_images']}")
    print(f"  Total quality entries: {summary['n_entries']}")
    print(f"  Matching: {summary.get('matching', 'unknown')}")

    # ----- Per-organ quality -----
    print(f"\n  Per-organ quality:")
    print(f"  {'Organ':<14} {'Dice':>14} {'IoU':>14} {'HD95':>14} {'Missing':>10}")
    print("  " + "-" * 70)
    for organ, stats in sorted(summary["per_organ"].items()):
        dice = (
            f"{stats['dice_mean']:.3f}±{stats['dice_std']:.3f}"
            if stats.get("dice_mean") is not None else "N/A"
        )
        iou = (
            f"{stats['iou_mean']:.3f}±{stats['iou_std']:.3f}"
            if stats.get("iou_mean") is not None else "N/A"
        )
        hd95 = (
            f"{stats['hausdorff_95_mean']:.1f}±{stats['hausdorff_95_std']:.1f}"
            if stats.get("hausdorff_95_mean") is not None else "N/A"
        )
        missing = f"{stats.get('missing', 0)}/{stats.get('count', 0)}"
        print(f"  {organ:<14} {dice:>14} {iou:>14} {hd95:>14} {missing:>10}")

    # ----- Global quality -----
    g = summary["global"]
    print("  " + "-" * 70)
    dice = (
        f"{g['dice_mean']:.3f}±{g['dice_std']:.3f}"
        if g.get("dice_mean") is not None else "N/A"
    )
    iou = (
        f"{g['iou_mean']:.3f}±{g['iou_std']:.3f}"
        if g.get("iou_mean") is not None else "N/A"
    )
    hd95 = (
        f"{g['hausdorff_95_mean']:.1f}±{g['hausdorff_95_std']:.1f}"
        if g.get("hausdorff_95_mean") is not None else "N/A"
    )
    print(f"  {'GLOBAL':<14} {dice:>14} {iou:>14} {hd95:>14}")

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

        # Per-organ recall at the primary (first) threshold
        primary_thr = iou_thresholds[0]
        print(f"\n  Per-organ recall @ IoU >= {primary_thr}:")
        for organ, stats in sorted(summary["per_organ"].items()):
            recall = stats.get(f"recall@{primary_thr}", 0.0)
            n_cov = stats.get(f"n_covered@{primary_thr}", 0)
            n_tot = stats.get("n_total", 0)
            print(f"    {organ:<14} {recall:>6.3f}  ({n_cov}/{n_tot})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
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
    parser.add_argument("--matching", default="hungarian",
                        choices=["semantic", "hungarian"],
                        help="Matching strategy for quality metrics")
    parser.add_argument("--iou-thresholds", nargs="+", type=float,
                        default=[0.5, 0.7],
                        help="IoU thresholds at which to report P/R/F1")
    args = parser.parse_args()

    gt_dir = Path(args.gt)
    pred_dir = Path(args.pred)
    output_dir = Path(args.output)

    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Predictions directory not found: {pred_dir}")

    all_results, summary = evaluate(
        gt_dir, pred_dir, args.matching, args.iou_thresholds
    )
    save_results(all_results, summary, output_dir)
    print_summary(summary)


if __name__ == "__main__":
    main()
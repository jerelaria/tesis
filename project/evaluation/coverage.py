"""Coverage and cleanliness metrics: P/R counts and their aggregation."""

from project.evaluation.matching import parse_organ_name
from project.evaluation.metrics_paired import iou_score


def compute_pr_counts(
    pred_masks: dict,
    gt_masks: dict,
    iou_threshold: float,
) -> dict:
    """
    Per-image P/R raw counts at one IoU threshold.

    A GT is "covered" if its max IoU vs any prediction is >= threshold.
    A prediction is "relevant" if its max IoU vs any GT is >= threshold.

    Predictions and GT are matched independently (a single prediction can
    cover multiple GTs and vice-versa). This is intentional: coverage and
    cleanliness are name-agnostic metrics that complement the matched
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
    when names are synthetic (greedy mode).

    Returns
    -------
    (per_organ_metrics, global_metrics)
    """
    per_organ_metrics: dict[str, dict] = {}
    global_metrics: dict = {}

    for thr in iou_thresholds:
        thr_data = pr_counts_by_thr[thr]

        total_gt = sum(d["n_gt"] for d in thr_data)
        total_gt_covered = sum(d["n_gt_covered"] for d in thr_data)
        recall = total_gt_covered / total_gt if total_gt > 0 else 0.0

        total_pred = sum(d["n_pred"] for d in thr_data)
        total_pred_relevant = sum(d["n_pred_relevant"] for d in thr_data)
        precision = total_pred_relevant / total_pred if total_pred > 0 else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        global_metrics[f"recall@{thr}"] = recall
        global_metrics[f"precision@{thr}"] = precision
        global_metrics[f"f1@{thr}"] = f1
        global_metrics[f"n_gt_covered@{thr}"] = total_gt_covered
        global_metrics[f"n_pred_relevant@{thr}"] = total_pred_relevant

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

    # n_total per organ and global totals are threshold-independent — store once
    if iou_thresholds:
        first_thr_data = pr_counts_by_thr[iou_thresholds[0]]
        organ_total_first: dict[str, int] = {}
        for d in first_thr_data:
            for organ, count in d["per_organ_total"].items():
                organ_total_first[organ] = organ_total_first.get(organ, 0) + count
        for organ, total in organ_total_first.items():
            per_organ_metrics.setdefault(organ, {})["n_total"] = total

        global_metrics["n_gt_total"] = sum(d["n_gt"] for d in first_thr_data)
        global_metrics["n_pred_total"] = sum(d["n_pred"] for d in first_thr_data)

    return per_organ_metrics, global_metrics

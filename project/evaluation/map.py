"""mAP (mean Average Precision) computation, COCO-style."""

import numpy as np

from project.evaluation.metrics_paired import iou_score


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

    order = np.argsort(-np.array(matched_scores))
    sorted_ious = np.array(matched_ious)[order]

    tp = (sorted_ious >= iou_threshold).astype(np.float64)
    fp = 1.0 - tp
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recall = cum_tp / n_gt
    precision = cum_tp / (cum_tp + cum_fp + 1e-12)

    # Pascal VOC 2010+ style: integrate using monotonic precision envelope
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]

    recall_extended = np.concatenate([[0.0], recall, [1.0]])
    precision_extended = np.concatenate(
        [[precision_envelope[0]], precision_envelope, [0.0]]
    )

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
        Per image, dict mapping GT name to binary mask.
    pred_scores_per_image : list[dict[str, float]]
        Per image, dict mapping prediction name to its confidence score.
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
        pred_items_sorted = sorted(
            preds.items(),
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

    ap_per_threshold = {
        float(thr): float(
            compute_average_precision(pooled_ious, pooled_scores, total_gt, thr)
        )
        for thr in iou_thresholds
    }

    map_value = float(np.mean(list(ap_per_threshold.values())))

    return {
        "ap_per_threshold": ap_per_threshold,
        "map": map_value,
        "map_50": ap_per_threshold.get(0.5, 0.0),
        "map_75": ap_per_threshold.get(0.75, 0.0),
    }

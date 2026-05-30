"""Per-pair quality metrics: Dice, IoU, HD95, HD full, ASSD."""

import numpy as np


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
    if not np.any(pred) and not np.any(gt):
        return 0.0
    if not np.any(pred) or not np.any(gt):
        return float("inf")

    from scipy.ndimage import distance_transform_edt
    dt_gt = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)

    all_distances = np.concatenate([dt_gt[pred], dt_pred[gt]])
    return float(np.percentile(all_distances, 95))


def hausdorff_full(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Symmetric Hausdorff distance (maximum, not 95th percentile).

    Captures the worst-case boundary error, complementing HD95 (typical
    case) and ASSD (average case).
    Returns 0.0 if both masks are empty, inf if only one is empty.
    """
    if not np.any(pred) and not np.any(gt):
        return 0.0
    if not np.any(pred) or not np.any(gt):
        return float("inf")

    from scipy.ndimage import distance_transform_edt
    dt_gt = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)

    max_p_to_g = float(dt_gt[pred].max())
    max_g_to_p = float(dt_pred[gt].max())
    return max(max_p_to_g, max_g_to_p)


def assd(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Average Symmetric Surface Distance.

    Mean surface-to-surface distance computed symmetrically. Complements
    HD95 by capturing the average boundary error rather than the worst
    percentile.
    Returns 0.0 if both masks are empty, inf if only one is empty.
    """
    if not np.any(pred) and not np.any(gt):
        return 0.0
    if not np.any(pred) or not np.any(gt):
        return float("inf")

    from scipy.ndimage import distance_transform_edt, binary_erosion
    dt_gt = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)

    pred_surface = pred & ~binary_erosion(pred)
    gt_surface = gt & ~binary_erosion(gt)

    if not pred_surface.any() or not gt_surface.any():
        # Degenerate masks (single pixel etc.): fall back to all points
        distances = np.concatenate([dt_gt[pred], dt_pred[gt]])
    else:
        distances = np.concatenate([dt_gt[pred_surface], dt_pred[gt_surface]])

    return float(distances.mean())


def compute_quality_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute all quality metrics for a single (pred, gt) pair."""
    return {
        "dice": dice_score(pred, gt),
        "iou": iou_score(pred, gt),
        "hausdorff": hausdorff_full(pred, gt),
        "hausdorff_95": hausdorff_95(pred, gt),
        "assd": assd(pred, gt),
    }

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
    95th percentile Hausdorff distance between mask surfaces (medpy-real).

    Delegates to ``medpy.metric.binary.hd95``, which restricts distances to
    surface/border voxels of each mask. Computing this over ALL foreground
    pixels (border + interior) instead -- as an earlier version of this
    function did -- massively underestimates the true value: interior
    pixels of a well-overlapped mask contribute near-zero distances and
    dilute the percentile, since a compact blob has far more interior area
    than border perimeter (verified empirically: up to ~20x underestimate,
    occasionally collapsing a true HD95 of ~18px down to 0.0).
    Returns 0.0 if both masks are empty, inf if only one is empty.
    """
    if not np.any(pred) and not np.any(gt):
        return 0.0
    if not np.any(pred) or not np.any(gt):
        return float("inf")

    from medpy.metric.binary import hd95 as medpy_hd95
    return float(medpy_hd95(pred, gt, voxelspacing=(1, 1)))


def hausdorff_full(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Symmetric Hausdorff distance (maximum, not 95th percentile), medpy-real.

    Captures the worst-case boundary error, complementing HD95 (typical
    case) and ASSD (average case).
    Returns 0.0 if both masks are empty, inf if only one is empty.
    """
    if not np.any(pred) and not np.any(gt):
        return 0.0
    if not np.any(pred) or not np.any(gt):
        return float("inf")

    from medpy.metric.binary import hd as medpy_hd
    return float(medpy_hd(pred, gt, voxelspacing=(1, 1)))


def assd(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Average Symmetric Surface Distance (medpy-real).

    Mean surface-to-surface distance computed symmetrically. Complements
    HD95 by capturing the average boundary error rather than the worst
    percentile.
    Returns 0.0 if both masks are empty, inf if only one is empty.
    """
    if not np.any(pred) and not np.any(gt):
        return 0.0
    if not np.any(pred) or not np.any(gt):
        return float("inf")

    from medpy.metric.binary import assd as medpy_assd
    return float(medpy_assd(pred, gt, voxelspacing=(1, 1)))


def compute_quality_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Compute all quality metrics for a single (pred, gt) pair.

    The returned dict includes ``image_diagonal``, the Euclidean length of the
    image diagonal in pixels. Aggregation uses this as a finite worst-case
    distance to replace ``inf`` for missing organs, so that missing organs are
    penalised rather than excluded from the mean/std.
    """
    h, w = gt.shape
    image_diagonal = float(np.sqrt(h**2 + w**2))
    return {
        "dice": dice_score(pred, gt),
        "iou": iou_score(pred, gt),
        "hausdorff": hausdorff_full(pred, gt),
        "hausdorff_95": hausdorff_95(pred, gt),
        "assd": assd(pred, gt),
        "image_diagonal": image_diagonal,
    }

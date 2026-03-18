from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Image format conversions
# ---------------------------------------------------------------------------

def to_uint8(volume: np.ndarray) -> np.ndarray:
    """
    Converts a float32 array (H, W, 3) to uint8 RGB.
    Accepts values in [0, 1] or [0, 255].
    """
    arr = volume.copy()
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two boolean masks (H, W)."""
    intersection = np.logical_and(a, b).sum()
    union        = np.logical_or(a, b).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Bounding box of a boolean mask (H, W).
    Returns (x_min, y_min, x_max, y_max) or None if the mask is empty.
    """
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return (int(x_min), int(y_min), int(x_max), int(y_max))

# Por si quiero filtrar por área mínima en el futuro
def mask_area(mask: np.ndarray) -> int:
    """Number of True pixels in a boolean mask."""
    return int(mask.sum())


# ---------------------------------------------------------------------------
# Non-Maximum Suppression
# ---------------------------------------------------------------------------

def nms(
    results: List[Tuple[np.ndarray, float]],
    iou_threshold: float = 0.50,
) -> List[Tuple[np.ndarray, float]]:
    """
    Non-Maximum Suppression on a list of (mask, score).

    Sorts by descending score and discards any mask whose IoU with
    an already accepted mask exceeds `iou_threshold`.

    Parameters
    ----------
    results : list of (mask bool HxW, score float)
    iou_threshold : overlap threshold for suppression

    Returns
    -------
    Filtered list, sorted by descending score.
    """

    if not results:
        return []

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    kept: List[Tuple[np.ndarray, float]] = []

    for mask, score in sorted_results:
        if not any(mask_iou(mask, k_mask) > iou_threshold for k_mask, _ in kept):
            kept.append((mask, score))

    return kept


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

def make_point_grid(h: int, w: int, grid_side: int = 6) -> np.ndarray:
    """
    Uniform grid of points (col, row) with 10% margin on each edge.

    Returns array of shape (grid_side², 2), dtype float32.
    """

    xs = np.linspace(w * 0.02, w * 0.98, grid_side).astype(int)
    ys = np.linspace(h * 0.02, h * 0.98, grid_side).astype(int)
    return np.array([(x,y) for y in ys for x in xs])
"""
conflict_resolution.py
----------------------
Resolve pixel-level ownership conflicts in overlapping predicted masks.

Used by the evaluation runner before matching and coverage counting so that
each pixel is attributed to exactly one prediction.  Masks on disk are never
touched; the function works on in-memory arrays and returns copies.
"""

import numpy as np


def resolve_overlaps(
    pred_masks: dict[str, np.ndarray],
    pred_scores: dict[str, float],
) -> dict[str, np.ndarray]:
    """
    Assign each contested pixel to the single highest-priority prediction.

    For every pixel claimed by more than one mask, ownership is granted to
    the mask whose score is highest.  Ties are resolved deterministically:

      1. Largest area (number of True pixels in the *original* mask).
      2. Alphabetically smallest name (e.g. "alpha" beats "beta").

    Pixels claimed by only one mask are left untouched.  Masks in the input
    are not mutated; the returned dict contains independent copies.

    Parameters
    ----------
    pred_masks :
        Mapping from prediction name to boolean mask array.  All arrays must
        have the same shape.
    pred_scores :
        Score for every name that appears in pred_masks.  Raises ``KeyError``
        if any name is absent — callers must not fall back to a default score
        here, because an arbitrary default would silently corrupt the ranking.

    Returns
    -------
    dict[str, np.ndarray]
        Same keys as pred_masks; each value is a boolean array where every
        True pixel belongs to exactly one prediction.

    Raises
    ------
    KeyError
        If pred_masks contains a name with no corresponding entry in
        pred_scores.
    ValueError
        If the masks do not all share the same shape.
    """
    if not pred_masks:
        return {}

    missing = sorted(name for name in pred_masks if name not in pred_scores)
    if missing:
        raise KeyError(
            f"resolve_overlaps: no score found for predictions {missing}. "
            "Every name in pred_masks must have a corresponding entry in "
            "pred_scores; do not assume a default score here."
        )

    shapes = {name: pred_masks[name].shape for name in pred_masks}
    unique_shapes = set(shapes.values())
    if len(unique_shapes) > 1:
        raise ValueError(
            f"resolve_overlaps: masks have inconsistent shapes: {shapes}"
        )
    canvas_shape = next(iter(unique_shapes))

    areas = {name: int(pred_masks[name].sum()) for name in pred_masks}

    # Determine write order: ascending priority so the highest-priority mask
    # is written last and therefore wins on contested pixels.
    #
    # Winner priority (descending):  score DESC, area DESC, name ASC.
    # Write order   (ascending):     score ASC,  area ASC,  name DESC.
    #
    # Two-pass stable sort: first by name DESC so that within equal (score,
    # area) ties, the alphabetically earlier name ends up written last (wins).
    by_name_desc = sorted(pred_masks, reverse=True)
    write_order = sorted(by_name_desc, key=lambda n: (pred_scores[n], areas[n]))

    # canvas[r, c] holds the write_order index of the mask that owns that
    # pixel.  -1 means no mask claims this pixel.
    canvas = np.full(canvas_shape, -1, dtype=np.intp)
    for idx, name in enumerate(write_order):
        canvas[pred_masks[name]] = idx

    name_to_idx = {name: idx for idx, name in enumerate(write_order)}
    return {name: (canvas == name_to_idx[name]) for name in pred_masks}

"""Tests for project.evaluation.conflict_resolution.resolve_overlaps."""

import numpy as np
import pytest

from project.evaluation.conflict_resolution import resolve_overlaps


def _mask(h: int, w: int, rows: slice, cols: slice) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[rows, cols] = True
    return m


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_overlapping_higher_score_wins():
    """Contested pixels go to the mask with the higher score."""
    h, w = 4, 8
    # "lo" covers cols 0-4; "hi" covers cols 2-7; overlap is cols 2-4.
    lo = _mask(h, w, slice(None), slice(0, 5))   # score 0.4
    hi = _mask(h, w, slice(None), slice(2, 8))   # score 0.8

    resolved = resolve_overlaps({"lo": lo, "hi": hi}, {"lo": 0.4, "hi": 0.8})

    expected_lo = _mask(h, w, slice(None), slice(0, 2))   # keeps cols 0-1 only
    expected_hi = _mask(h, w, slice(None), slice(2, 8))   # keeps cols 2-7 (all of hi)
    np.testing.assert_array_equal(resolved["lo"], expected_lo)
    np.testing.assert_array_equal(resolved["hi"], expected_hi)


def test_disjoint_masks_pass_through():
    """Disjoint masks are returned as copies with identical pixel patterns."""
    h, w = 4, 8
    a = _mask(h, w, slice(None), slice(0, 4))
    b = _mask(h, w, slice(None), slice(4, 8))

    masks = {"a": a.copy(), "b": b.copy()}
    resolved = resolve_overlaps(masks, {"a": 0.5, "b": 0.9})

    np.testing.assert_array_equal(resolved["a"], a)
    np.testing.assert_array_equal(resolved["b"], b)


def test_disjoint_masks_are_copies():
    """resolve_overlaps returns new arrays, not views of the input."""
    h, w = 4, 4
    a = _mask(h, w, slice(None), slice(0, 2))
    b = _mask(h, w, slice(None), slice(2, 4))

    masks = {"a": a, "b": b}
    resolved = resolve_overlaps(masks, {"a": 0.3, "b": 0.7})

    assert resolved["a"] is not masks["a"]
    assert resolved["b"] is not masks["b"]


def test_input_masks_not_mutated():
    """Input masks must not be modified by resolve_overlaps."""
    h, w = 4, 4
    a = _mask(h, w, slice(None), slice(0, 3))   # overlaps cols 1-2
    b = _mask(h, w, slice(None), slice(1, 4))

    a_original = a.copy()
    b_original = b.copy()

    resolve_overlaps({"a": a, "b": b}, {"a": 0.2, "b": 0.9})

    np.testing.assert_array_equal(a, a_original)
    np.testing.assert_array_equal(b, b_original)


# ---------------------------------------------------------------------------
# Tie-breaking: area
# ---------------------------------------------------------------------------

def test_score_tie_larger_area_wins():
    """Equal scores: the mask with more True pixels wins contested pixels."""
    h, w = 4, 6
    # "big" covers cols 0-4 (20 px), "small" covers cols 2-3 (8 px).
    # Contest: cols 2-3.  big has more area → wins.
    big   = _mask(h, w, slice(None), slice(0, 5))   # 20 True pixels
    small = _mask(h, w, slice(None), slice(2, 4))   #  8 True pixels

    resolved = resolve_overlaps(
        {"big": big, "small": small},
        {"big": 0.6, "small": 0.6},
    )

    # big keeps all 20 pixels (including the contested 8)
    np.testing.assert_array_equal(resolved["big"], big)
    # small loses all its pixels to big
    np.testing.assert_array_equal(resolved["small"], np.zeros((h, w), dtype=bool))


# ---------------------------------------------------------------------------
# Tie-breaking: alphabetical name
# ---------------------------------------------------------------------------

def test_score_and_area_tie_alphabetical_name_wins():
    """Equal score and area: alphabetically earlier name wins."""
    h, w = 4, 4
    # Both masks are identical: same score, same area, same pixels.
    m = _mask(h, w, slice(0, 2), slice(0, 2))

    resolved = resolve_overlaps(
        {"beta": m.copy(), "alpha": m.copy()},
        {"beta": 0.5, "alpha": 0.5},
    )

    # "alpha" < "beta" → alpha wins all pixels
    np.testing.assert_array_equal(resolved["alpha"], m)
    np.testing.assert_array_equal(resolved["beta"], np.zeros((h, w), dtype=bool))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_mask_unchanged():
    """A single mask has no conflicts; it should be returned as-is (copy)."""
    h, w = 4, 4
    m = _mask(h, w, slice(1, 3), slice(1, 3))

    resolved = resolve_overlaps({"only": m}, {"only": 0.7})

    np.testing.assert_array_equal(resolved["only"], m)
    assert resolved["only"] is not m


def test_empty_pred_masks_returns_empty():
    """Empty input returns an empty dict without error."""
    assert resolve_overlaps({}, {}) == {}


def test_empty_mask_produces_empty_output():
    """A mask with no True pixels produces an all-False output."""
    h, w = 4, 4
    empty = np.zeros((h, w), dtype=bool)
    full  = np.ones((h, w), dtype=bool)

    resolved = resolve_overlaps(
        {"empty": empty, "full": full},
        {"empty": 0.9, "full": 0.1},
    )

    # "empty" has no pixels to win regardless of score
    np.testing.assert_array_equal(resolved["empty"], np.zeros((h, w), dtype=bool))
    # "full" keeps everything (empty had nothing to contest)
    np.testing.assert_array_equal(resolved["full"], full)


# ---------------------------------------------------------------------------
# Error conditions
# ---------------------------------------------------------------------------

def test_missing_score_raises():
    """KeyError is raised when pred_masks has a name absent from pred_scores."""
    h, w = 4, 4
    a = _mask(h, w, slice(None), slice(0, 2))

    with pytest.raises(KeyError, match="no score found for predictions"):
        resolve_overlaps({"a": a}, {})


def test_partial_missing_score_raises():
    """KeyError is raised even when only some names are missing scores."""
    h, w = 4, 4
    a = _mask(h, w, slice(None), slice(0, 2))
    b = _mask(h, w, slice(None), slice(1, 3))

    with pytest.raises(KeyError, match="no score found for predictions"):
        resolve_overlaps({"a": a, "b": b}, {"a": 0.5})  # "b" missing


def test_inconsistent_shapes_raises():
    """ValueError is raised when masks have different shapes."""
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((5, 5), dtype=bool)

    with pytest.raises(ValueError, match="inconsistent shapes"):
        resolve_overlaps({"a": a, "b": b}, {"a": 0.5, "b": 0.7})

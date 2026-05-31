"""Tests for aggregate_quality: two-variant (with_missing / detected_only) aggregation."""
import math
import pytest
from project.evaluation.aggregation import aggregate_quality


def _entry(
    organ: str,
    pred_name,
    dice: float,
    iou: float,
    hd95: float,
    diagonal: float | None = None,
) -> dict:
    """Build a minimal result entry suitable for aggregate_quality."""
    e = {
        "organ": organ,
        "gt_name": f"{organ}_1",
        "pred_name": pred_name,
        "dice": dice,
        "iou": iou,
        "hausdorff": hd95,
        "hausdorff_95": hd95,
        "assd": hd95,
    }
    if diagonal is not None:
        e["image_diagonal"] = diagonal
    return e


# ---------------------------------------------------------------------------
# 1. Missing organ uses diagonal, not excluded (with_missing variant)
# ---------------------------------------------------------------------------

def test_missing_organ_uses_diagonal_not_excluded():
    """
    Two entries for one organ: one normal (hd95=10) and one missing (hd95=inf).
    with_missing mean must be (10 + 362) / 2 = 186, NOT 10 (old exclude behaviour).
    """
    entries = [
        _entry("lung", "lung_1", dice=0.9, iou=0.8, hd95=10.0, diagonal=362.0),
        _entry("lung", None,     dice=0.0, iou=0.0, hd95=float("inf"), diagonal=362.0),
    ]
    summary = aggregate_quality(entries)
    lung = summary["per_organ"]["lung"]
    assert lung["hausdorff_95_mean_with_missing"] == pytest.approx(186.0), (
        "Expected (10 + 362) / 2 = 186; old code would give 10 by excluding inf"
    )


# ---------------------------------------------------------------------------
# 2. 'missing' count is independent of metric values
# ---------------------------------------------------------------------------

def test_missing_count_reflects_pred_name_not_metric():
    """
    The 'missing' field must equal the number of entries where pred_name is
    None, regardless of what hd95 value those entries carry.
    """
    entries = [
        _entry("heart", "heart_1", dice=0.8, iou=0.7, hd95=5.0,         diagonal=200.0),
        _entry("heart", None,      dice=0.0, iou=0.0, hd95=float("inf"), diagonal=200.0),
        _entry("heart", None,      dice=0.0, iou=0.0, hd95=float("inf"), diagonal=200.0),
    ]
    summary = aggregate_quality(entries)
    assert summary["per_organ"]["heart"]["missing"] == 2


# ---------------------------------------------------------------------------
# 3. Dice / IoU with_missing aggregation unaffected (missing contributes 0.0)
# ---------------------------------------------------------------------------

def test_dice_iou_unaffected_by_inf_substitution():
    """
    Dice and IoU for missing organs are already 0.0, not inf, so the
    with_missing variant leaves them unchanged and the mean is correct.
    """
    entries = [
        _entry("spleen", "spleen_1", dice=1.0, iou=1.0, hd95=0.0,         diagonal=300.0),
        _entry("spleen", None,       dice=0.0, iou=0.0, hd95=float("inf"), diagonal=300.0),
    ]
    summary = aggregate_quality(entries)
    spleen = summary["per_organ"]["spleen"]
    assert spleen["dice_mean_with_missing"] == pytest.approx(0.5)
    assert spleen["iou_mean_with_missing"]  == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 4. Fallback: inf with no image_diagonal key degrades gracefully
# ---------------------------------------------------------------------------

def test_fallback_no_image_diagonal_returns_inf():
    """
    An entry with hd95=inf but no image_diagonal key (older CSV row) must
    not crash. The helper falls back to inf, so the mean includes inf and
    resolves to inf — a detectable, non-crashing outcome.
    """
    entries = [
        _entry("liver", None, dice=0.0, iou=0.0, hd95=float("inf"), diagonal=None),
    ]
    summary = aggregate_quality(entries)
    liver = summary["per_organ"]["liver"]
    assert math.isinf(liver["hausdorff_95_mean_with_missing"]), (
        "Without image_diagonal the helper falls back to inf; mean should be inf"
    )


# ---------------------------------------------------------------------------
# NEW 1. with_missing == penalise missing (same arithmetic as existing test)
# ---------------------------------------------------------------------------

def test_with_missing_penalises_with_diagonal():
    """
    with_missing mean for hd95: (10 + 362) / 2 = 186.
    This is a named assertion of the expected with_missing arithmetic.
    """
    entries = [
        _entry("lung", "lung_1", dice=0.9, iou=0.8, hd95=10.0,          diagonal=362.0),
        _entry("lung", None,     dice=0.0, iou=0.0, hd95=float("inf"),  diagonal=362.0),
    ]
    summary = aggregate_quality(entries)
    lung = summary["per_organ"]["lung"]
    assert lung["hausdorff_95_mean_with_missing"] == pytest.approx(186.0)


# ---------------------------------------------------------------------------
# NEW 2. detected_only drops missing entry entirely
# ---------------------------------------------------------------------------

def test_detected_only_excludes_missing_entry():
    """
    detected_only mean for hd95: only the detected entry (hd95=10) contributes.
    The missing entry (hd95=inf) is dropped, so mean = 10.0, not 186.
    """
    entries = [
        _entry("lung", "lung_1", dice=0.9, iou=0.8, hd95=10.0,          diagonal=362.0),
        _entry("lung", None,     dice=0.0, iou=0.0, hd95=float("inf"),  diagonal=362.0),
    ]
    summary = aggregate_quality(entries)
    lung = summary["per_organ"]["lung"]
    assert lung["hausdorff_95_mean_detected_only"] == pytest.approx(10.0)
    assert lung["hausdorff_95_std_detected_only"]  == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# NEW 3. Fully-missing organ: detected_only is None, with_missing uses diagonal
# ---------------------------------------------------------------------------

def test_fully_missing_organ_detected_only_none():
    """
    When all entries for an organ are missing (pred_name=None), detected_only
    returns None/None (no data to average). with_missing still has a value
    (the diagonal for every entry).
    """
    entries = [
        _entry("heart", None, dice=0.0, iou=0.0, hd95=float("inf"), diagonal=362.0),
        _entry("heart", None, dice=0.0, iou=0.0, hd95=float("inf"), diagonal=362.0),
    ]
    summary = aggregate_quality(entries)
    heart = summary["per_organ"]["heart"]

    assert heart["hausdorff_95_mean_detected_only"] is None
    assert heart["hausdorff_95_std_detected_only"]  is None

    assert heart["hausdorff_95_mean_with_missing"] == pytest.approx(362.0)


# ---------------------------------------------------------------------------
# NEW 4. Dice: both variants computed correctly
# ---------------------------------------------------------------------------

def test_dice_two_variants():
    """
    One detected entry (dice=0.9) + one missing (dice=0.0, pred_name=None).
    with_missing: (0.9 + 0.0) / 2 = 0.45
    detected_only: only the 0.9 entry -> mean = 0.9
    """
    entries = [
        _entry("spleen", "spleen_1", dice=0.9, iou=0.85, hd95=5.0,          diagonal=300.0),
        _entry("spleen", None,       dice=0.0, iou=0.00, hd95=float("inf"), diagonal=300.0),
    ]
    summary = aggregate_quality(entries)
    spleen = summary["per_organ"]["spleen"]

    assert spleen["dice_mean_with_missing"]  == pytest.approx(0.45)
    assert spleen["dice_mean_detected_only"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# NEW 5. 'missing' count and 'count' are identical regardless of variant
# ---------------------------------------------------------------------------

def test_counts_unaffected_by_variant():
    """
    'count' is always the total number of entries and 'missing' is always
    pred_name-is-None, independent of any metric variant.
    """
    entries = [
        _entry("liver", "liver_1", dice=0.7, iou=0.6, hd95=8.0,          diagonal=400.0),
        _entry("liver", None,      dice=0.0, iou=0.0, hd95=float("inf"), diagonal=400.0),
        _entry("liver", None,      dice=0.0, iou=0.0, hd95=float("inf"), diagonal=400.0),
    ]
    summary = aggregate_quality(entries)
    liver = summary["per_organ"]["liver"]

    assert liver["count"]   == 3
    assert liver["missing"] == 2

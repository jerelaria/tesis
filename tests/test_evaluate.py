"""Tests for evaluate.py metric and matching functions."""
import numpy as np
import pytest
from evaluate import dice_score, iou_score, match_semantic, match_hungarian


# ---------------------------------------------------------------------------
# dice_score
# ---------------------------------------------------------------------------

def test_dice_identical():
    mask = np.array([[True, True], [False, False]])
    assert dice_score(mask, mask) == 1.0


def test_dice_disjoint():
    pred = np.array([[True, False], [False, False]])
    gt   = np.array([[False, True], [False, False]])
    assert dice_score(pred, gt) == 0.0


def test_dice_one_empty():
    pred = np.zeros((4, 4), dtype=bool)
    gt   = np.ones((4, 4), dtype=bool)
    assert dice_score(pred, gt) == 0.0


def test_dice_both_empty():
    pred = np.zeros((4, 4), dtype=bool)
    gt   = np.zeros((4, 4), dtype=bool)
    # Both empty → defined as perfect match
    assert dice_score(pred, gt) == 1.0


# ---------------------------------------------------------------------------
# iou_score
# ---------------------------------------------------------------------------

def test_iou_identical():
    mask = np.ones((5, 5), dtype=bool)
    assert iou_score(mask, mask) == 1.0


def test_iou_disjoint():
    pred = np.array([[True, False]])
    gt   = np.array([[False, True]])
    assert iou_score(pred, gt) == 0.0


def test_iou_one_empty():
    assert iou_score(np.zeros((3, 3), dtype=bool), np.ones((3, 3), dtype=bool)) == 0.0


# ---------------------------------------------------------------------------
# match_semantic
# ---------------------------------------------------------------------------

def test_match_semantic_by_name():
    """lung_1 + lung_2 in GT and pred should match within the 'lung' group."""
    lung_1 = np.zeros((10, 10), dtype=bool)
    lung_1[:5, :5] = True

    lung_2 = np.zeros((10, 10), dtype=bool)
    lung_2[5:, 5:] = True

    gt_masks   = {"lung_1": lung_1, "lung_2": lung_2}
    pred_masks = {"lung_1": lung_1, "lung_2": lung_2}

    results = match_semantic(pred_masks, gt_masks)
    for r in results:
        assert r["dice"] == pytest.approx(1.0), f"Expected perfect match for {r['gt_name']}"


def test_match_semantic_missing_organ():
    """A GT organ with no matching pred should get dice=0."""
    gt_masks   = {"heart": np.ones((5, 5), dtype=bool)}
    pred_masks = {}

    results = match_semantic(pred_masks, gt_masks)
    assert len(results) == 1
    assert results[0]["dice"] == 0.0
    assert results[0]["pred_name"] is None


# ---------------------------------------------------------------------------
# match_hungarian
# ---------------------------------------------------------------------------

def test_match_hungarian_prefers_high_iou():
    """
    2 GT, 2 pred with cross-IoU matrix:
        pred_A  pred_B
    gt_1  0.3    0.7
    gt_2  0.6    0.2

    Optimal: gt_1->pred_B (0.7), gt_2->pred_A (0.6)  (sum=1.3)
    Greedy (by column max): gt_1->pred_A? no — hungarian should find global optimum.
    """
    gt_1 = np.zeros((10, 10), dtype=bool)
    gt_1[0:7, 0:7] = True   # 49 px

    gt_2 = np.zeros((10, 10), dtype=bool)
    gt_2[0:6, 0:6] = True   # 36 px

    # pred_A overlaps gt_2 more than gt_1
    pred_A = np.zeros((10, 10), dtype=bool)
    pred_A[0:6, 0:6] = True  # perfect for gt_2, partial for gt_1

    # pred_B overlaps gt_1 more
    pred_B = np.zeros((10, 10), dtype=bool)
    pred_B[0:7, 0:7] = True  # same as gt_1

    gt_masks   = {"gt_1": gt_1, "gt_2": gt_2}
    pred_masks = {"pred_A": pred_A, "pred_B": pred_B}

    results = match_hungarian(pred_masks, gt_masks)

    matching = {r["gt_name"]: r["pred_name"] for r in results}
    assert matching.get("gt_1") == "pred_B", f"gt_1 should match pred_B, got {matching}"
    assert matching.get("gt_2") == "pred_A", f"gt_2 should match pred_A, got {matching}"

"""Tests for add_panoptic_quality."""

import pytest

from project.evaluation.aggregation import add_panoptic_quality


def _make_summary(global_sq, global_rq, organ_sq):
    return {
        "global": {
            "iou_mean_detected_only": global_sq,
            "f1@0.5": global_rq,
        },
        "per_organ": {
            "heart": {"iou_mean_detected_only": organ_sq},
        },
    }


def test_pq_equals_sq_times_rq():
    summary = _make_summary(0.8, 0.75, 0.7)
    add_panoptic_quality(summary)
    assert summary["global"]["sq"] == pytest.approx(0.8)
    assert summary["global"]["rq"] == pytest.approx(0.75)
    assert summary["global"]["pq"] == pytest.approx(0.6)


def test_pq_none_when_sq_is_none():
    summary = _make_summary(None, 0.75, None)
    add_panoptic_quality(summary)
    assert summary["global"]["sq"] is None
    assert summary["global"]["pq"] is None


def test_pq_none_when_rq_is_none():
    summary = _make_summary(0.8, None, 0.7)
    add_panoptic_quality(summary)
    assert summary["global"]["rq"] is None
    assert summary["global"]["pq"] is None


def test_per_organ_sq_set_rq_and_pq_none():
    summary = _make_summary(0.8, 0.75, 0.7)
    add_panoptic_quality(summary)
    heart = summary["per_organ"]["heart"]
    assert heart["sq"] == pytest.approx(0.7)
    assert heart["rq"] is None
    assert heart["pq"] is None

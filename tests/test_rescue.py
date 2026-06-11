"""Smoke tests for scripts/rescue_report.py core logic."""

import json
import numpy as np
import pytest
from pathlib import Path

from scripts.rescue_report import (
    _baseline_valid,
    _final_valid,
    _load_sam_scores,
    compute_rescue_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_mask(path: Path, mask: np.ndarray) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(path)


def _write_scores(path: Path, scores: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(scores, f)


def _solid_mask(shape=(64, 64), region=None) -> np.ndarray:
    """Binary mask with a solid rectangle; region=(r0,r1,c0,c1) or full."""
    m = np.zeros(shape, dtype=bool)
    if region is None:
        m[:] = True
    else:
        r0, r1, c0, c1 = region
        m[r0:r1, c0:c1] = True
    return m


# ---------------------------------------------------------------------------
# Unit tests for _baseline_valid and _final_valid
# ---------------------------------------------------------------------------

def test_baseline_valid_requires_both_score_and_iou():
    gt = _solid_mask(region=(0, 32, 0, 32))
    overlapping = _solid_mask(region=(0, 32, 0, 32))   # IoU = 1.0
    non_overlap = _solid_mask(region=(32, 64, 32, 64))  # IoU = 0.0

    # High score + good IoU -> valid
    assert _baseline_valid(gt, {"a": overlapping}, {"a": 0.9}, sam_threshold=0.5)

    # High score but bad IoU -> invalid
    assert not _baseline_valid(gt, {"a": non_overlap}, {"a": 0.9}, sam_threshold=0.5)

    # Good IoU but score at threshold (not >) -> invalid
    assert not _baseline_valid(gt, {"a": overlapping}, {"a": 0.5}, sam_threshold=0.5)

    # Good IoU, score just above threshold -> valid
    assert _baseline_valid(gt, {"a": overlapping}, {"a": 0.51}, sam_threshold=0.5)


def test_final_valid_iou_gate():
    gt = _solid_mask(region=(0, 32, 0, 32))
    good = _solid_mask(region=(0, 32, 0, 32))   # IoU = 1.0
    bad  = _solid_mask(region=(32, 64, 32, 64))  # IoU = 0.0

    assert _final_valid(gt, {"a": good})
    assert not _final_valid(gt, {"a": bad})
    assert not _final_valid(gt, {})


# ---------------------------------------------------------------------------
# Integration: compute_rescue_stats on a tiny synthetic dataset
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_dirs(tmp_path):
    """
    Two images, one organ (lv_cavity) each.

    Image A: baseline misses (low sam score), final hits -> rescued
    Image B: baseline hits,  final misses               -> lost
    """
    gt_dir       = tmp_path / "gt"
    baseline_dir = tmp_path / "baseline"
    final_dir    = tmp_path / "final"

    organ_mask = _solid_mask(region=(10, 54, 10, 54))
    empty_mask = _solid_mask(region=(0, 0, 0, 0))   # all zeros

    # --- Image A: rescued ---
    _write_mask(gt_dir / "imgA" / "lv_cavity.png", organ_mask)
    # baseline: present but sam score too low to be valid
    _write_mask(baseline_dir / "imgA" / "obj_000.png", organ_mask)
    _write_scores(
        baseline_dir / "imgA" / "scores.json",
        {"obj_000.png": {"sam": 0.3, "combined": 0.3, "labeling": 0.3}},
    )
    # final: good prediction
    _write_mask(final_dir / "imgA" / "cluster_0.png", organ_mask)

    # --- Image B: lost ---
    _write_mask(gt_dir / "imgB" / "lv_cavity.png", organ_mask)
    # baseline: valid (high score + good IoU)
    _write_mask(baseline_dir / "imgB" / "obj_000.png", organ_mask)
    _write_scores(
        baseline_dir / "imgB" / "scores.json",
        {"obj_000.png": {"sam": 0.9, "combined": 0.9, "labeling": 0.9}},
    )
    # final: empty (misses entirely)
    _write_mask(final_dir / "imgB" / "cluster_0.png", empty_mask)

    return gt_dir, baseline_dir, final_dir


def test_compute_rescue_stats(synthetic_dirs):
    gt_dir, baseline_dir, final_dir = synthetic_dirs
    per_organ, rescued_examples = compute_rescue_stats(
        baseline_dir, final_dir, gt_dir, sam_threshold=0.5
    )

    assert "lv_cavity" in per_organ
    stats = per_organ["lv_cavity"]
    assert stats["rescued"] == 1
    assert stats["lost"] == 1

    assert len(rescued_examples) == 1
    ex = rescued_examples[0]
    assert ex["image"] == "imgA"
    assert ex["organ"] == "lv_cavity"
    assert ex["gt_name"] == "lv_cavity"


def test_missing_scores_json_raises(synthetic_dirs, tmp_path):
    gt_dir, baseline_dir, final_dir = synthetic_dirs
    # Remove scores.json from one image
    (baseline_dir / "imgA" / "scores.json").unlink()
    with pytest.raises(FileNotFoundError, match="scores.json not found"):
        compute_rescue_stats(baseline_dir, final_dir, gt_dir, sam_threshold=0.5)

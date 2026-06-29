"""Tests for project.evaluation.cluster_map."""
import json

import numpy as np
import pytest
from PIL import Image

from project.evaluation.cluster_map import (
    DISCARD_LABEL,
    load_cluster_map,
    parse_cluster_map,
    relabel_predictions,
)
from project.evaluation.runner import evaluate


# ---------------------------------------------------------------------------
# parse_cluster_map
# ---------------------------------------------------------------------------

def test_parse_cluster_map_valid():
    assert parse_cluster_map("0=heart 1=left_lung 2=right_lung") == {
        0: "heart", 1: "left_lung", 2: "right_lung",
    }


def test_parse_cluster_map_discard_token():
    assert parse_cluster_map("3=__discard__") == {3: DISCARD_LABEL}


@pytest.mark.parametrize("raw", ["heart", "x=heart", "0=", ""])
def test_parse_cluster_map_malformed(raw):
    with pytest.raises(ValueError):
        parse_cluster_map(raw)


# ---------------------------------------------------------------------------
# load_cluster_map
# ---------------------------------------------------------------------------

def test_load_cluster_map_valid(tmp_path):
    path = tmp_path / "cluster_map.json"
    path.write_text(json.dumps({"0": "heart", "1": "left_lung", "3": "__discard__"}))
    assert load_cluster_map(path) == {0: "heart", 1: "left_lung", 3: DISCARD_LABEL}


def test_load_cluster_map_non_integer_key(tmp_path):
    path = tmp_path / "cluster_map.json"
    path.write_text(json.dumps({"heart": "heart"}))
    with pytest.raises(ValueError):
        load_cluster_map(path)


def test_load_cluster_map_empty_value(tmp_path):
    path = tmp_path / "cluster_map.json"
    path.write_text(json.dumps({"0": "  "}))
    with pytest.raises(ValueError):
        load_cluster_map(path)


# ---------------------------------------------------------------------------
# relabel_predictions
# ---------------------------------------------------------------------------

def _masks(*stems):
    return {s: np.ones((4, 4), dtype=bool) for s in stems}


def _scores(*stems):
    return {s: 1.0 for s in stems}


def test_relabel_renames_and_preserves_instance():
    masks = _masks("cluster_0", "cluster_3_2")
    scores = _scores("cluster_0", "cluster_3_2")
    cmap = {0: "heart", 3: "left_lung"}

    out_masks, out_scores = relabel_predictions(masks, scores, cmap)

    assert set(out_masks) == {"heart", "left_lung_2"}
    assert set(out_scores) == set(out_masks)


def test_relabel_discards():
    masks = _masks("cluster_0", "cluster_1")
    scores = _scores("cluster_0", "cluster_1")
    cmap = {0: "heart", 1: DISCARD_LABEL}

    out_masks, out_scores = relabel_predictions(masks, scores, cmap)

    assert set(out_masks) == {"heart"}
    assert "cluster_1" not in out_masks
    assert set(out_scores) == {"heart"}


def test_relabel_keeps_unmapped_cluster():
    masks = _masks("cluster_5")
    scores = _scores("cluster_5")
    cmap = {0: "heart"}

    out_masks, out_scores = relabel_predictions(masks, scores, cmap)

    assert set(out_masks) == {"cluster_5"}
    assert set(out_scores) == {"cluster_5"}


def test_relabel_reindexes_on_collision():
    # Two clusters mapped to the same organ must reindex without collision.
    masks = _masks("cluster_0", "cluster_1")
    scores = _scores("cluster_0", "cluster_1")
    cmap = {0: "heart", 1: "heart"}

    out_masks, out_scores = relabel_predictions(masks, scores, cmap)

    assert set(out_masks) == {"heart_1", "heart_2"}
    assert set(out_scores) == set(out_masks)
    assert len(out_masks) == 2  # nothing overwritten


def test_relabel_masks_and_scores_keys_identical():
    masks = _masks("cluster_0", "cluster_1_1", "cluster_1_2")
    scores = _scores("cluster_0", "cluster_1_1", "cluster_1_2")
    cmap = {0: "heart", 1: "left_lung"}

    out_masks, out_scores = relabel_predictions(masks, scores, cmap)

    assert set(out_masks) == set(out_scores)


def test_relabel_non_cluster_stem_kept():
    masks = _masks("heart_1")
    scores = _scores("heart_1")
    cmap = {0: "heart"}

    out_masks, out_scores = relabel_predictions(masks, scores, cmap)

    assert set(out_masks) == {"heart_1"}


def test_relabel_desync_score_without_mask_raises():
    masks = _masks("cluster_0")
    scores = _scores("cluster_0", "cluster_9")
    with pytest.raises(ValueError):
        relabel_predictions(masks, scores, {0: "heart"})


# ---------------------------------------------------------------------------
# evaluate() smoke test with a cluster-map
# ---------------------------------------------------------------------------

def _save_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(path)


def test_evaluate_with_cluster_map_forces_semantic(tmp_path):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"

    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True

    # GT names the organ "heart"; prediction is an anonymous cluster_0.
    _save_mask(gt_dir / "img1" / "heart.png", mask)
    _save_mask(pred_dir / "img1" / "cluster_0.png", mask)
    _save_mask(pred_dir / "img1" / "cluster_9.png", mask)  # discarded
    (pred_dir / "img1" / "scores.json").write_text(json.dumps({
        "cluster_0.png": {"combined": 0.9},
        "cluster_9.png": {"combined": 0.9},
    }))

    cluster_map = {0: "heart", 9: DISCARD_LABEL}
    _, summary = evaluate(
        gt_dir=gt_dir,
        pred_dir=pred_dir,
        matching="greedy",          # must be overridden to semantic
        iou_thresholds=[0.5],
        cluster_map=cluster_map,
    )

    assert summary["matching"] == "semantic"
    assert summary["cluster_map"] == {"0": "heart", "9": DISCARD_LABEL}
    assert summary["cluster_map_discarded"] == ["9"]
    # heart was detected with a perfect mask -> recall 1.0 at IoU 0.5.
    assert summary["global"]["recall@0.5"] == pytest.approx(1.0)

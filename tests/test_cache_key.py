"""Tests for segmentation fingerprinting utilities."""

import pytest

from project.pipeline.cache_key import (
    build_fingerprint_payload,
    cache_dir_name,
    segmentation_fingerprint,
)

_BASE_PAYLOAD = {
    "dataset_short": "XRayNicoSent",
    "image_stems": ["img001", "img002", "img003"],
    "mode": "unsupervised",
    "propagation_mode": "-",
    "segmenter": {
        "grid_side": 6,
        "score_threshold": 0.50,
        "iou_threshold": 0.50,
    },
    "reference_stems": [],
    "model_id": "medsam2",
}


def _mutate(payload: dict, **overrides) -> dict:
    """Return a shallow copy of payload with top-level keys overridden."""
    return {**payload, **overrides}


def _mutate_seg(payload: dict, **seg_overrides) -> dict:
    return _mutate(payload, segmenter={**payload["segmenter"], **seg_overrides})


# ── Determinism ────────────────────────────────────────────────────────────

def test_fingerprint_deterministic():
    h1 = segmentation_fingerprint(_BASE_PAYLOAD)
    h2 = segmentation_fingerprint(_BASE_PAYLOAD)
    assert h1 == h2


def test_fingerprint_length():
    assert len(segmentation_fingerprint(_BASE_PAYLOAD)) == 8


def test_cache_dir_name_deterministic():
    assert cache_dir_name(_BASE_PAYLOAD) == cache_dir_name(_BASE_PAYLOAD)


# ── Sensitivity ────────────────────────────────────────────────────────────

def test_hash_changes_on_score_threshold():
    p2 = _mutate_seg(_BASE_PAYLOAD, score_threshold=0.7)
    assert segmentation_fingerprint(_BASE_PAYLOAD) != segmentation_fingerprint(p2)


def test_hash_changes_on_iou_threshold():
    p2 = _mutate_seg(_BASE_PAYLOAD, iou_threshold=0.3)
    assert segmentation_fingerprint(_BASE_PAYLOAD) != segmentation_fingerprint(p2)


def test_hash_changes_on_grid_side():
    p2 = _mutate_seg(_BASE_PAYLOAD, grid_side=8)
    assert segmentation_fingerprint(_BASE_PAYLOAD) != segmentation_fingerprint(p2)


def test_hash_changes_on_image_order():
    p2 = _mutate(_BASE_PAYLOAD, image_stems=["img002", "img001", "img003"])
    assert segmentation_fingerprint(_BASE_PAYLOAD) != segmentation_fingerprint(p2)


def test_hash_changes_on_image_set():
    p2 = _mutate(_BASE_PAYLOAD, image_stems=["img001", "img002", "img004"])
    assert segmentation_fingerprint(_BASE_PAYLOAD) != segmentation_fingerprint(p2)


def test_hash_changes_on_ref_stems():
    p2 = _mutate(_BASE_PAYLOAD, reference_stems=["ref001"])
    assert segmentation_fingerprint(_BASE_PAYLOAD) != segmentation_fingerprint(p2)


def test_hash_changes_on_mode():
    p2 = _mutate(_BASE_PAYLOAD, mode="few_shot")
    assert segmentation_fingerprint(_BASE_PAYLOAD) != segmentation_fingerprint(p2)


def test_hash_changes_on_dataset():
    p2 = _mutate(_BASE_PAYLOAD, dataset_short="OtherDS")
    assert segmentation_fingerprint(_BASE_PAYLOAD) != segmentation_fingerprint(p2)


# ── build_fingerprint_payload ──────────────────────────────────────────────

def test_build_fingerprint_payload_excludes_device():
    from pathlib import Path

    paths = [Path("/data/img001.png"), Path("/data/img002.png")]
    seg_with_device = {
        "grid_side": 6,
        "score_threshold": 0.5,
        "iou_threshold": 0.5,
        "device": "cuda",
    }
    seg_no_device = {
        "grid_side": 6,
        "score_threshold": 0.5,
        "iou_threshold": 0.5,
        "device": "cpu",
    }
    p1 = build_fingerprint_payload("DS", paths, "unsupervised", None, seg_with_device, [])
    p2 = build_fingerprint_payload("DS", paths, "unsupervised", None, seg_no_device, [])
    assert segmentation_fingerprint(p1) == segmentation_fingerprint(p2)


def test_build_fingerprint_payload_propagation_none_becomes_dash():
    from pathlib import Path

    paths = [Path("/data/img001.png")]
    p = build_fingerprint_payload("DS", paths, "unsupervised", None, {"grid_side": 6, "score_threshold": 0.5, "iou_threshold": 0.5}, [])
    assert p["propagation_mode"] == "-"


# ── cache_dir_name format ──────────────────────────────────────────────────

def test_cache_dir_name_unsupervised_format():
    name = cache_dir_name(_BASE_PAYLOAD)
    h8 = segmentation_fingerprint(_BASE_PAYLOAD)
    assert name.startswith("unsupervised_grid6_st0.50_iou0.50_n3__")
    assert name.endswith(h8)
    assert name == f"unsupervised_grid6_st0.50_iou0.50_n3__{h8}"


def test_cache_dir_name_few_shot_format():
    payload = {
        **_BASE_PAYLOAD,
        "mode": "few_shot",
        "propagation_mode": "iterative",
        "reference_stems": ["ref001", "ref002", "ref003", "ref004"],
    }
    name = cache_dir_name(payload)
    h8 = segmentation_fingerprint(payload)
    assert name.startswith("few_shot_iterative_K4_grid6_st0.50_iou0.50_n3__")
    assert name.endswith(h8)


def test_cache_dir_name_few_shot_independent():
    payload = {
        **_BASE_PAYLOAD,
        "mode": "few_shot",
        "propagation_mode": "independent",
        "reference_stems": ["ref001", "ref002"],
    }
    name = cache_dir_name(payload)
    assert "few_shot_independent_K2" in name

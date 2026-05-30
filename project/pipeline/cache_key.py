"""Segmentation cache fingerprinting: payload, hash, and directory naming."""

import hashlib
import json
from pathlib import Path


def build_fingerprint_payload(
    dataset_short: str,
    image_paths: list[Path],
    mode: str,
    propagation_mode: str | None,
    segmenter_cfg: dict,
    reference_stems: list[str] | None,
) -> dict:
    """Construct the canonical fingerprint payload for the segmentation cache.

    Only segmenter fields that affect Phase 1 output are included; device is
    excluded intentionally so CPU/GPU runs share the same cache.
    """
    seg = {
        "grid_side": int(segmenter_cfg["grid_side"]),
        "score_threshold": float(segmenter_cfg["score_threshold"]),
        "iou_threshold": float(segmenter_cfg["iou_threshold"]),
    }
    return {
        "dataset_short": dataset_short,
        "image_stems": [p.stem for p in image_paths],
        "mode": mode,
        "propagation_mode": propagation_mode or "-",
        "segmenter": seg,
        "reference_stems": reference_stems or [],
        # TODO: derive from checkpoint/config names when they become configurable
        "model_id": "medsam2",
    }


def segmentation_fingerprint(payload: dict) -> str:
    """Return an 8-character SHA-1 hash of the canonicalized payload."""
    serialized = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(serialized).hexdigest()[:8]


def cache_dir_name(payload: dict) -> str:
    """Return a human-readable directory name with an embedded hash suffix.

    Unsupervised:
        unsupervised_grid{G}_st{S:.2f}_iou{I:.2f}_n{N}__{hash8}
    Few-shot:
        few_shot_{prop}_K{k}_grid{G}_st{S:.2f}_iou{I:.2f}_n{N}__{hash8}
    """
    mode = payload["mode"]
    seg = payload["segmenter"]
    n = len(payload["image_stems"])
    grid = seg["grid_side"]
    st = seg["score_threshold"]
    iou = seg["iou_threshold"]
    h8 = segmentation_fingerprint(payload)

    if mode == "unsupervised":
        prefix = f"unsupervised_grid{grid}_st{st:.2f}_iou{iou:.2f}_n{n}"
    else:
        prop = payload.get("propagation_mode", "independent")
        k = len(payload.get("reference_stems", []))
        prefix = f"few_shot_{prop}_K{k}_grid{grid}_st{st:.2f}_iou{iou:.2f}_n{n}"

    return f"{prefix}__{h8}"

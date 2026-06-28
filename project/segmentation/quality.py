"""
Cluster quality scoring and prototype selection for batch propagation.

This module provides:
1. ClusterQualityConfig / identify_good_clusters — three-threshold criterion
   to decide which clusters represent real organs.
2. MaskSelectionConfig / select_prototypes — select the top-K prototype masks
   per good cluster, ranked by combined score.
3. ClusterNMSConfig / suppress_overlapping_clusters — cluster-level non-maximum
   suppression that drops duplicate clusters of the same organ before
   propagation.

Combined score formula:
    combined_score = alpha * sam_score + (1 - alpha) * labeling_confidence
where alpha = MaskSelectionConfig.sam_score_weight.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from project.core.data_types import LabeledObject
from project.segmentation.utils import mask_iou


def _resize_mask_to(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize of a boolean mask to ``shape`` = (height, width).

    Mirrors project.segmentation.medsam2.video._resize_mask_to; kept local to
    avoid importing the medsam2 stack just for a mask reshape.
    """
    if mask.shape == shape:
        return mask
    from PIL import Image as PILImage

    height, width = shape
    resized = PILImage.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
        (width, height), resample=PILImage.NEAREST
    )
    return np.array(resized) > 127


@dataclass
class MaskSelectionConfig:
    """Configuration for prototype selection and reference mask scoring."""
    sam_score_weight: float = 0.5     # alpha: weight for SAM confidence
    min_combined_score: float = 0.75  # minimum score to qualify as a prototype


@dataclass
class ClusterNMSConfig:
    """Configuration for cluster-level non-maximum suppression over prototypes."""
    enabled: bool = True
    iou_threshold: float = 0.5   # IoU above which two cluster prototypes are
                                 # treated as the same anatomical structure


@dataclass
class ClusterQualityConfig:
    """
    Thresholds for determining whether a cluster represents a real organ.

    A cluster passes if ALL three conditions hold (globally, across all images):
      - it appears in at least min_image_frequency fraction of images
      - its mean labeling confidence is >= min_avg_labeling_confidence
      - its mean SAM segmentation score is >= min_avg_sam_confidence
        (objects where SAM returned no score are excluded from the mean;
         if no object has a score the cluster is not penalised for it)
    """
    min_image_frequency: float = 0.3
    min_avg_labeling_confidence: float = 0.5
    min_avg_sam_confidence: float = 0.60


def compute_cluster_quality(
    labeled_by_image: dict[Path, list[LabeledObject]],
    config: ClusterQualityConfig,
) -> dict[int, dict]:
    """
    Return per-cluster quality metrics and pass/fail status.

    Parameters
    ----------
    labeled_by_image : dict[Path, list[LabeledObject]]
        All labeled objects grouped by image path.
    config : ClusterQualityConfig
        The three quality thresholds.

    Returns
    -------
    dict[int, dict]
        Maps cluster_id -> {
            "image_frequency":          float,
            "avg_labeling_confidence":  float,
            "avg_sam_confidence":       float | None,
            "n_objects":                int,
            "good":                     bool,
            "failed":                   list[str],  # empty when good=True
        }
    """
    total_images = len(labeled_by_image)
    if total_images == 0:
        return {}

    all_valid = [
        obj
        for objs in labeled_by_image.values()
        for obj in objs
        if not obj.is_noise and obj.organ_id != -1
    ]

    report: dict[int, dict] = {}
    for cid in sorted({obj.organ_id for obj in all_valid}):
        members = [obj for obj in all_valid if obj.organ_id == cid]

        unique_images = {obj.segmented_object.source_image.source_path for obj in members}
        image_freq = len(unique_images) / total_images

        avg_lconf = sum(obj.labeling_confidence for obj in members) / len(members)

        sam_scores = [
            obj.segmented_object.confidence
            for obj in members
            if obj.segmented_object.confidence is not None
        ]
        avg_sam = (sum(sam_scores) / len(sam_scores)) if sam_scores else None

        failed = []
        if image_freq < config.min_image_frequency:
            failed.append(
                f"image_frequency={image_freq:.3f} < {config.min_image_frequency}"
            )
        if avg_lconf < config.min_avg_labeling_confidence:
            failed.append(
                f"avg_labeling_confidence={avg_lconf:.3f} < {config.min_avg_labeling_confidence}"
            )
        if avg_sam is not None and avg_sam < config.min_avg_sam_confidence:
            failed.append(
                f"avg_sam_confidence={avg_sam:.3f} < {config.min_avg_sam_confidence}"
            )

        report[cid] = {
            "image_frequency": round(image_freq, 4),
            "avg_labeling_confidence": round(avg_lconf, 4),
            "avg_sam_confidence": round(avg_sam, 4) if avg_sam is not None else None,
            "n_objects": len(members),
            "good": len(failed) == 0,
            "failed": failed,
        }

    return report


def identify_good_clusters(
    labeled_by_image: dict[Path, list[LabeledObject]],
    config: ClusterQualityConfig,
) -> set[int]:
    """
    Return cluster IDs that pass all three quality thresholds.

    Delegates to compute_cluster_quality; kept as a thin wrapper so call
    sites that only need the set don't change.
    """
    return {
        cid
        for cid, info in compute_cluster_quality(labeled_by_image, config).items()
        if info["good"]
    }


def select_prototypes(
    labeled_by_image: dict[Path, list[LabeledObject]],
    good_clusters: set[int],
    k: int,
    config: MaskSelectionConfig,
) -> dict[int, list[LabeledObject]]:
    """
    Select the top-k prototype masks for each good cluster.

    Only non-noise objects whose combined score meets config.min_combined_score
    are considered.  Each cluster's candidates are sorted by descending combined
    score and truncated to k.  Clusters with no qualifying objects appear in
    the result with an empty list.

    Parameters
    ----------
    labeled_by_image : dict[Path, list[LabeledObject]]
        All labeled objects grouped by image path.
    good_clusters : set[int]
        Cluster IDs to build prototypes for (from identify_good_clusters).
    k : int
        Maximum number of prototype masks per cluster.
    config : MaskSelectionConfig
        Provides sam_score_weight (alpha) and min_combined_score.

    Returns
    -------
    dict[int, list[LabeledObject]]
        Maps cluster_id -> list of up to k objects, best-scored first.
    """
    alpha = config.sam_score_weight
    candidates: dict[int, list[tuple[float, LabeledObject]]] = {
        cid: [] for cid in good_clusters
    }

    for objs in labeled_by_image.values():
        for obj in objs:
            if obj.is_noise or obj.organ_id not in good_clusters:
                continue
            sam_score = obj.segmented_object.confidence or 0.0
            score = alpha * sam_score + (1.0 - alpha) * obj.labeling_confidence
            if score >= config.min_combined_score:
                candidates[obj.organ_id].append((score, obj))

    result: dict[int, list[LabeledObject]] = {}
    for cid, scored in candidates.items():
        scored.sort(key=lambda x: -x[0])
        result[cid] = [obj for _, obj in scored[:k]]

    return result


def _prototype_strength(obj: LabeledObject, alpha: float) -> float:
    """Combined prototype score: alpha * sam_score + (1 - alpha) * lconf.

    Identical to the score select_prototypes ranks candidates by, so the NMS
    representative strength is consistent with prototype selection.
    """
    sam_score = obj.segmented_object.confidence or 0.0
    return alpha * sam_score + (1.0 - alpha) * obj.labeling_confidence


def suppress_overlapping_clusters(
    prototypes: dict[int, list[LabeledObject]],
    cluster_sizes: dict[int, int],
    selection_config: MaskSelectionConfig,
    nms_config: ClusterNMSConfig,
) -> tuple[dict[int, list[LabeledObject]], list[dict]]:
    """Greedy cluster-level NMS over prototype masks, run before propagation.

    Over-segmentation produces several clusters that describe the same
    anatomical structure.  This collapses them so the cluster->organ mapping is
    consistent across every image: each cluster is represented by its top
    prototype (index 0); the strongest representative survives and weaker
    representatives whose mask overlaps it (IoU >= nms_config.iou_threshold) are
    suppressed and never propagated.

    Representative strength is the same combined score select_prototypes ranks
    by, with alpha = selection_config.sam_score_weight:
        strength = alpha * (sam_score or 0.0) + (1 - alpha) * labeling_confidence

    Survivor criterion: clusters are visited in deterministic order — strength
    descending, then cluster size descending, then cluster_id ascending — so the
    strongest (and, on ties, largest then lowest-id) cluster of an overlapping
    group is the one kept.  Clusters with an empty prototype list are not
    candidates and pass through unchanged (they cannot duplicate anything).

    Parameters
    ----------
    prototypes
        Mapping cluster_id -> list of prototype objects (best-scored first), as
        returned by select_prototypes.
    cluster_sizes
        Mapping cluster_id -> number of members, used as the tie-breaker.
    selection_config
        Provides sam_score_weight (alpha) for the strength score.
    nms_config
        Enable flag and IoU threshold.

    Returns
    -------
    survivors : dict[int, list[LabeledObject]]
        The input mapping with the suppressed clusters removed.
    report : list[dict]
        One entry per suppressed cluster, in suppression order:
        {"cluster": int, "suppressed_by": int, "iou": float}.

    Representatives from differently-sized source images (variable-resolution
    datasets such as ACDC) are reconciled with a nearest-neighbor resize before
    the overlap is measured, so the IoU is computed in a common frame.

    Raises
    ------
    ValueError
        If a representative mask is None.
    """
    if not nms_config.enabled:
        return prototypes, []

    alpha = selection_config.sam_score_weight

    # Build a representative (top prototype) for every non-empty cluster.
    reps: dict[int, tuple[float, int, np.ndarray]] = {}
    for cid, proto_list in prototypes.items():
        if not proto_list:
            continue
        mask = proto_list[0].segmented_object.mask
        if mask is None:
            raise ValueError(
                f"cluster {cid}: representative prototype has no mask (None)."
            )
        strength = _prototype_strength(proto_list[0], alpha)
        reps[cid] = (strength, cluster_sizes.get(cid, 0), np.asarray(mask, dtype=bool))

    # Deterministic order: strength desc, size desc, cluster_id asc.
    order = sorted(reps, key=lambda cid: (-reps[cid][0], -reps[cid][1], cid))

    kept: list[int] = []
    report: list[dict] = []
    for cid in order:
        mask = reps[cid][2]
        for kept_id in kept:
            kept_mask = reps[kept_id][2]
            # Cluster representatives come from different source images, which on
            # variable-resolution datasets (e.g. ACDC) may differ in shape. Bring
            # the candidate into the kept mask's frame with a nearest-neighbor
            # resize before measuring overlap — same reconciliation the pipeline
            # already uses for cross-resolution masks (medsam2 _resize_mask_to).
            cand_mask = (
                mask if mask.shape == kept_mask.shape
                else _resize_mask_to(mask, kept_mask.shape)
            )
            iou = mask_iou(cand_mask, kept_mask)
            if iou >= nms_config.iou_threshold:
                report.append(
                    {"cluster": cid, "suppressed_by": kept_id, "iou": float(iou)}
                )
                break
        else:
            kept.append(cid)

    suppressed_ids = {entry["cluster"] for entry in report}
    survivors = {
        cid: proto_list
        for cid, proto_list in prototypes.items()
        if cid not in suppressed_ids
    }
    return survivors, report

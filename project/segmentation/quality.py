"""
Cluster quality scoring and prototype selection for batch propagation.

This module provides:
1. ClusterQualityConfig / identify_good_clusters — three-threshold criterion
   to decide which clusters represent real organs.
2. MaskSelectionConfig / select_prototypes — select the top-K prototype masks
   per good cluster, ranked by combined score.

Combined score formula:
    combined_score = alpha * sam_score + (1 - alpha) * labeling_confidence
where alpha = MaskSelectionConfig.sam_score_weight.
"""

from dataclasses import dataclass
from pathlib import Path

from project.core.data_types import LabeledObject


@dataclass
class MaskSelectionConfig:
    """Configuration for prototype selection and reference mask scoring."""
    sam_score_weight: float = 0.5     # alpha: weight for SAM confidence
    min_combined_score: float = 0.75  # minimum score to qualify as a prototype
    num_reference_frames: int = 5     # retained for API compatibility; not used
                                      # by select_prototypes (governed by k)


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
        num_reference_frames is NOT used here; k governs the limit.

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

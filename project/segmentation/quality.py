"""
Quality scoring and reference mask selection for retroactive refinement.

This module provides functions to:
1. Determine which clusters are absent from an image (candidates for refinement).
2. Select the best reference masks from other images to use as context frames
   for the SAM2 video predictor.

A cluster C is considered absent from image I if no object in I was assigned
to C with labeling_confidence >= min_cluster_confidence. This uses clustering
confidence (certainty of cluster membership), not SAM score, because a high
SAM score does not guarantee correct cluster assignment.

Reference masks are selected using a combined score:
    combined_score = alpha * sam_score + (1 - alpha) * cluster_confidence
Only candidates above min_combined_score are kept, truncated to num_reference_frames.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass

from project.core.data_types import LabeledObject


@dataclass
class MaskSelectionConfig:
    """Configuration for reference mask selection in retroactive refinement."""
    sam_score_weight: float = 0.5           # alpha: weight for SAM confidence
    min_combined_score: float = 0.75        # minimum combined score to be a candidate
    num_reference_frames: int = 5           # max number of reference frames for the video


def find_absent_clusters(
    labeled_objects: list[LabeledObject],
    all_cluster_ids: set[int],
    min_cluster_confidence: float,
) -> set[int]:
    """
    Find clusters that are absent from a single image.

    A cluster is absent if no object in the image was assigned to it with
    labeling_confidence >= min_cluster_confidence.

    Parameters
    ----------
    labeled_objects : list[LabeledObject]
        Labeled objects from a single image.
    all_cluster_ids : set[int]
        All known cluster IDs from the global clustering.
    min_cluster_confidence : float
        Threshold for considering a cluster "present" in an image.

    Returns
    -------
    set[int]
        Cluster IDs that are absent from this image.
    """
    present_clusters = {
        obj.organ_id
        for obj in labeled_objects
        if not obj.is_noise and obj.labeling_confidence >= min_cluster_confidence
    }
    return all_cluster_ids - present_clusters


def select_reference_masks(
    cluster_id: int,
    labeled_by_image: dict[Path, list[LabeledObject]],
    target_path: Path,
    config: MaskSelectionConfig,
) -> list[LabeledObject]:
    """
    Select the best reference masks for a given cluster to use as context
    frames in the video predictor.

    Candidates come from all images except the target. They are ranked by
    a combined score of SAM confidence and cluster membership confidence.

    Parameters
    ----------
    cluster_id : int
        The cluster whose masks we want to propagate.
    labeled_by_image : dict[Path, list[LabeledObject]]
        All labeled objects grouped by image path.
    target_path : Path
        Path of the target image (excluded from candidates).
    config : MaskSelectionConfig
        Selection hyperparameters.

    Returns
    -------
    list[LabeledObject]
        Top-K reference objects, sorted by descending combined score.
        K = config.num_reference_frames.
    """
    alpha = config.sam_score_weight

    candidates = []
    for path, objects in labeled_by_image.items():
        if path == target_path:
            continue

        for obj in objects:
            if obj.organ_id != cluster_id or obj.is_noise:
                continue

            sam_score = obj.segmented_object.confidence or 0.0
            cluster_conf = obj.labeling_confidence

            combined = alpha * sam_score + (1.0 - alpha) * cluster_conf

            if combined >= config.min_combined_score:
                candidates.append((obj, combined))

    # Sort by descending combined score
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Truncate to num_reference_frames
    selected = [obj for obj, _ in candidates[:config.num_reference_frames]]

    return selected


def get_good_cluster_ids(
    labeled_by_image: dict[Path, list[LabeledObject]],
    min_image_frequency: float = 0.3,
) -> set[int]:
    """
    Return cluster IDs that appear in at least min_image_frequency fraction
    of all images. Only these clusters are worth attempting to refine.

    Parameters
    ----------
    labeled_by_image : dict[Path, list[LabeledObject]]
        All labeled objects grouped by image path.
    min_image_frequency : float
        Minimum fraction of images a cluster must appear in.

    Returns
    -------
    set[int]
        Cluster IDs considered "good" for refinement.
    """
    total_images = len(labeled_by_image)
    if total_images == 0:
        return set()

    cluster_image_count: dict[int, int] = {}
    for path, objects in labeled_by_image.items():
        present_ids = {obj.organ_id for obj in objects if not obj.is_noise}
        for cid in present_ids:
            cluster_image_count[cid] = cluster_image_count.get(cid, 0) + 1

    return {
        cid
        for cid, count in cluster_image_count.items()
        if count / total_images >= min_image_frequency
    }
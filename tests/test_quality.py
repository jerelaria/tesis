"""Tests for quality scoring and reference selection functions."""
import numpy as np
import pytest
from pathlib import Path
from project.core.data_types import MedicalImage, SegmentedObject, LabeledObject
from project.segmentation.quality import (
    find_absent_clusters,
    select_reference_masks,
    build_reference_index,
    select_from_index,
    MaskSelectionConfig,
)


def _make_labeled(
    organ_id: int,
    labeling_confidence: float,
    sam_confidence: float = 0.9,
    is_noise: bool = False,
    source_path: str = "/tmp/img.png",
) -> LabeledObject:
    image = MedicalImage(
        volume=np.zeros((10, 10, 3), dtype=np.float32),
        modality="synthetic",
        source_path=source_path,
    )
    seg = SegmentedObject(
        mask=np.ones((10, 10), dtype=bool),
        source_image=image,
        confidence=sam_confidence,
    )
    return LabeledObject(
        segmented_object=seg,
        organ_id=organ_id,
        organ_name=f"cluster_{organ_id}",
        labeling_confidence=labeling_confidence,
        method_used="test",
        is_noise=is_noise,
    )


def test_find_absent_clusters_detects_low_confidence():
    """A cluster present with confidence below threshold should count as absent."""
    obj = _make_labeled(organ_id=1, labeling_confidence=0.4)
    all_cluster_ids = {1, 2, 3}
    absent = find_absent_clusters([obj], all_cluster_ids, min_cluster_confidence=0.5)
    assert 1 in absent, "cluster 1 (conf=0.4 < 0.5) should be absent"
    assert 2 in absent
    assert 3 in absent


def test_find_absent_clusters_present_cluster():
    """A cluster with confidence above threshold should not be absent."""
    obj = _make_labeled(organ_id=1, labeling_confidence=0.8)
    absent = find_absent_clusters([obj], {1, 2}, min_cluster_confidence=0.5)
    assert 1 not in absent
    assert 2 in absent


def test_select_reference_masks_returns_top_k():
    """select_reference_masks should return ≤ num_reference_frames candidates above threshold."""
    target_path = Path("/tmp/target.png")
    labeled_by_image = {}
    for i in range(10):
        path = Path(f"/tmp/img_{i}.png")
        labeled_by_image[path] = [
            _make_labeled(
                organ_id=0,
                labeling_confidence=0.8,
                sam_confidence=0.9,
                source_path=str(path),
            )
        ]

    cfg = MaskSelectionConfig(
        sam_score_weight=0.5,
        min_combined_score=0.5,
        num_reference_frames=3,
    )
    refs = select_reference_masks(0, labeled_by_image, target_path, cfg)
    assert len(refs) <= 3


def test_build_and_select_from_index_matches_select_reference():
    """build_reference_index + select_from_index must match select_reference_masks output."""
    target_path = Path("/tmp/target.png")
    labeled_by_image = {}
    for i in range(10):
        path = Path(f"/tmp/img_{i}.png")
        labeled_by_image[path] = [
            _make_labeled(
                organ_id=0,
                labeling_confidence=0.6 + i * 0.01,
                sam_confidence=0.9,
                source_path=str(path),
            )
        ]

    cfg = MaskSelectionConfig(
        sam_score_weight=0.5,
        min_combined_score=0.5,
        num_reference_frames=3,
    )
    direct = select_reference_masks(0, labeled_by_image, target_path, cfg)
    index = build_reference_index(labeled_by_image, cfg)
    via_index = select_from_index(0, target_path, index, cfg)

    assert len(direct) == len(via_index)
    direct_ids = [id(o) for o in direct]
    index_ids = [id(o) for o in via_index]
    assert direct_ids == index_ids, "Index and direct selection returned different objects"

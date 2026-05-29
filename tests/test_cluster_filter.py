"""Tests for ClusterFilter."""
import numpy as np
import pytest
from pathlib import Path
from project.core.data_types import MedicalImage, SegmentedObject, LabeledObject
from project.labeling.clustering_filter import ClusterFilter, ClusterFilterConfig


def _make_labeled(
    organ_id: int,
    labeling_confidence: float = 0.8,
    sam_confidence: float = 0.9,
    source_path: str = "/tmp/img.png",
    is_noise: bool = False,
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


def test_low_frequency_cluster_marked_as_noise():
    """Cluster appearing in 2/10 images should be noise when min_image_frequency=0.5."""
    total_images = 10
    labeled_by_image = {}

    # cluster 0: appears in 8/10 images (good)
    for i in range(8):
        path = Path(f"/tmp/img_{i}.png")
        labeled_by_image[path] = [
            _make_labeled(organ_id=0, source_path=str(path))
        ]

    # cluster 1: appears only in 2/10 images (bad)
    for i in range(2):
        path = Path(f"/tmp/img_{i}.png")
        labeled_by_image[path].append(
            _make_labeled(organ_id=1, source_path=str(path))
        )

    # Fill remaining images with only cluster 0
    for i in range(8, total_images):
        path = Path(f"/tmp/img_{i}.png")
        labeled_by_image[path] = [
            _make_labeled(organ_id=0, source_path=str(path))
        ]

    cfg = ClusterFilterConfig(
        min_image_frequency=0.5,
        min_avg_labeling_confidence=0.0,
        min_avg_sam_confidence=0.0,
    )
    cf = ClusterFilter(cfg)
    cf.filter(labeled_by_image)

    # cluster 1 objects should be noise, cluster 0 should not
    for path, objs in labeled_by_image.items():
        for obj in objs:
            if obj.organ_id == 1:
                assert obj.is_noise, f"cluster_1 object at {path} should be noise"
            elif obj.organ_id == 0:
                assert not obj.is_noise, f"cluster_0 object at {path} should not be noise"

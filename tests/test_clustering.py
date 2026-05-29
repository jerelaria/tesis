"""Tests for ClusteringLabeler."""
import numpy as np
import pytest
from pathlib import Path
from project.core.data_types import MedicalImage, SegmentedObject
from project.labeling.clustering import (
    ClusteringLabeler,
    ClusteringConfig,
    KMeansConfig,
    HDBSCANConfig,
    ClusteringAlgorithm,
)
from project.feature_extraction.feature_names import FEATURE_NAMES


def _make_obj_with_features(features: np.ndarray) -> SegmentedObject:
    image = MedicalImage(
        volume=np.zeros((10, 10, 3), dtype=np.float32),
        modality="synthetic",
    )
    obj = SegmentedObject(
        mask=np.ones((10, 10), dtype=bool),
        source_image=image,
    )
    obj.features = features
    return obj


def _make_group(center: np.ndarray, n: int = 10, noise: float = 0.01) -> list:
    rng = np.random.default_rng(42)
    objects = []
    for _ in range(n):
        f = center + rng.normal(scale=noise, size=center.shape)
        objects.append(_make_obj_with_features(f.astype(np.float32)))
    return objects


def test_kmeans_three_groups():
    """KMeans K=3 on clearly separated groups should assign correctly (mod permutation)."""
    n_features = len(FEATURE_NAMES)
    centers = [
        np.zeros(n_features),
        np.ones(n_features) * 5.0,
        np.ones(n_features) * 10.0,
    ]
    groups = [_make_group(c) for c in centers]
    all_objects = [obj for g in groups for obj in g]

    cfg = ClusteringConfig(
        algorithm=ClusteringAlgorithm.KMEANS,
        kmeans=KMeansConfig(n_clusters=3, random_state=0),
    )
    labeler = ClusteringLabeler(cfg)
    labeler.fit(all_objects)

    # Label each group and collect the assigned cluster ids
    assigned_ids = []
    for group in groups:
        labeled = labeler.label(group)
        ids = {obj.organ_id for obj in labeled}
        assert len(ids) == 1, f"Group split across clusters: {ids}"
        assigned_ids.append(ids.pop())

    # All three groups must map to distinct clusters
    assert len(set(assigned_ids)) == 3, f"Not all groups got distinct clusters: {assigned_ids}"


@pytest.mark.parametrize("num_images", [10, 50, 100])
def test_hdbscan_adaptive_resolution(num_images):
    """resolve_adaptive_params() should compute absolute values from fractions."""
    cfg = ClusteringConfig(
        algorithm=ClusteringAlgorithm.HDBSCAN,
        hdbscan=HDBSCANConfig(
            min_cluster_size_fraction=0.1,
            min_samples_fraction=0.05,
        ),
    )
    labeler = ClusteringLabeler(cfg)
    labeler.resolve_adaptive_params(num_images)

    resolved = labeler.config.hdbscan
    expected_mcs = max(2, int(0.1 * num_images))
    expected_ms = max(1, int(0.05 * num_images))
    assert resolved.min_cluster_size == expected_mcs
    assert resolved.min_samples == expected_ms


def test_embeddings_only_without_enabled_raises():
    """features=None without embedding.enabled=True must raise ValueError."""
    with pytest.raises(ValueError, match="embedding"):
        ClusteringConfig(features=None)

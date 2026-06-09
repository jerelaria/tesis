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
    DBSCANConfig,
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


# ------------------------------------------------------------------
# cluster_selection_method tests
# ------------------------------------------------------------------

def test_hdbscan_default_cluster_selection_method():
    """Default cluster_selection_method must be 'eom' (backward-compatible with sklearn)."""
    cfg = HDBSCANConfig(min_cluster_size=2, min_samples=1)
    assert cfg.cluster_selection_method == "eom"

    labeler = ClusteringLabeler(
        ClusteringConfig(algorithm=ClusteringAlgorithm.HDBSCAN, hdbscan=cfg)
    )
    assert labeler._model.cluster_selection_method == "eom"


def test_hdbscan_leaf_propagates_to_model():
    """cluster_selection_method='leaf' must reach the sklearn HDBSCAN instance."""
    cfg = HDBSCANConfig(min_cluster_size=2, min_samples=1, cluster_selection_method="leaf")
    labeler = ClusteringLabeler(
        ClusteringConfig(algorithm=ClusteringAlgorithm.HDBSCAN, hdbscan=cfg)
    )
    assert labeler._model.cluster_selection_method == "leaf"


def test_hdbscan_invalid_cluster_selection_method_raises():
    """Any value other than 'eom' or 'leaf' must raise ValueError at config build time."""
    with pytest.raises(ValueError, match="cluster_selection_method"):
        HDBSCANConfig(cluster_selection_method="invalid")


def test_hdbscan_default_matches_explicit_eom():
    """Sanity: default config produces the same result as an explicit 'eom'."""
    n_features = len(FEATURE_NAMES)
    centers = [np.zeros(n_features), np.ones(n_features) * 5.0]
    all_objects = [obj for c in centers for obj in _make_group(c, n=15)]

    cfg_default = ClusteringConfig(
        algorithm=ClusteringAlgorithm.HDBSCAN,
        hdbscan=HDBSCANConfig(min_cluster_size=5, min_samples=1),
    )
    cfg_explicit_eom = ClusteringConfig(
        algorithm=ClusteringAlgorithm.HDBSCAN,
        hdbscan=HDBSCANConfig(min_cluster_size=5, min_samples=1, cluster_selection_method="eom"),
    )

    labeler_default = ClusteringLabeler(cfg_default)
    labeler_eom = ClusteringLabeler(cfg_explicit_eom)
    labeler_default.fit(all_objects)
    labeler_eom.fit(all_objects)

    ids_default = [labeler_default._id_to_cluster[o.id] for o in all_objects]
    ids_eom = [labeler_eom._id_to_cluster[o.id] for o in all_objects]
    assert ids_default == ids_eom, "Default and explicit 'eom' must produce identical assignments"


def test_hdbscan_resolve_preserves_cluster_selection_method():
    """resolve() must carry cluster_selection_method to the returned config."""
    cfg = HDBSCANConfig(
        min_cluster_size_fraction=0.1,
        cluster_selection_method="leaf",
    )
    resolved = cfg.resolve(num_images=50)
    assert resolved.cluster_selection_method == "leaf"


def test_hdbscan_confidence_from_probabilities():
    """HDBSCAN labeling_confidence must come from the model's `probabilities_`,
    not a binary clustered/noise flag."""
    n_features = len(FEATURE_NAMES)
    centers = [np.zeros(n_features), np.ones(n_features) * 5.0]
    all_objects = [obj for c in centers for obj in _make_group(c, n=15)]

    cfg = ClusteringConfig(
        algorithm=ClusteringAlgorithm.HDBSCAN,
        hdbscan=HDBSCANConfig(min_cluster_size=5, min_samples=1),
    )
    labeler = ClusteringLabeler(cfg)
    labeler.fit(all_objects)
    labeled = labeler.label(all_objects)

    proba = labeler._model.probabilities_
    for lab, p in zip(labeled, proba):
        if lab.organ_id == -1:
            assert lab.labeling_confidence == 0.0
        else:
            assert lab.labeling_confidence == pytest.approx(float(p))


def test_dbscan_confidence_is_binary():
    """DBSCAN has no soft assignment: non-noise -> 1.0, noise -> 0.0."""
    n_features = len(FEATURE_NAMES)
    centers = [np.zeros(n_features), np.ones(n_features) * 5.0]
    all_objects = [obj for c in centers for obj in _make_group(c, n=15)]

    cfg = ClusteringConfig(
        algorithm=ClusteringAlgorithm.DBSCAN,
        dbscan=DBSCANConfig(eps=0.5, min_samples=2),
    )
    labeler = ClusteringLabeler(cfg)
    labeler.fit(all_objects)
    labeled = labeler.label(all_objects)
    for lab in labeled:
        assert lab.labeling_confidence in (0.0, 1.0)

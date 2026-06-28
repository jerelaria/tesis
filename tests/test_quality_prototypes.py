"""Tests for identify_good_clusters and select_prototypes in quality.py."""
import pytest
from pathlib import Path
import numpy as np

from project.core.data_types import MedicalImage, SegmentedObject, LabeledObject
from project.segmentation.quality import (
    ClusterNMSConfig,
    ClusterQualityConfig,
    MaskSelectionConfig,
    identify_good_clusters,
    select_prototypes,
    suppress_overlapping_clusters,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_labeled(
    organ_id: int,
    labeling_confidence: float = 0.8,
    sam_confidence: float | None = 0.9,
    source_path: str = "/tmp/img.png",
    is_noise: bool = False,
) -> LabeledObject:
    image = MedicalImage(
        volume=np.zeros((4, 4, 3), dtype=np.float32),
        modality="synthetic",
        source_path=source_path,
    )
    seg = SegmentedObject(
        mask=np.ones((4, 4), dtype=bool),
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


def _build_dataset(
    n_images: int,
    cluster_ids: list[int],
    labeling_confidence: float = 0.8,
    sam_confidence: float = 0.9,
) -> dict[Path, list[LabeledObject]]:
    """All n_images carry every cluster_id once."""
    dataset: dict[Path, list[LabeledObject]] = {}
    for i in range(n_images):
        path = Path(f"/tmp/img_{i}.png")
        dataset[path] = [
            _make_labeled(
                organ_id=cid,
                labeling_confidence=labeling_confidence,
                sam_confidence=sam_confidence,
                source_path=str(path),
            )
            for cid in cluster_ids
        ]
    return dataset


# ---------------------------------------------------------------------------
# identify_good_clusters — individual threshold tests
# ---------------------------------------------------------------------------

def test_identify_good_clusters_empty_returns_empty():
    assert identify_good_clusters({}, ClusterQualityConfig()) == set()


def test_identify_all_pass():
    dataset = _build_dataset(n_images=10, cluster_ids=[0, 1])
    cfg = ClusterQualityConfig(
        min_image_frequency=0.5,
        min_avg_labeling_confidence=0.5,
        min_avg_sam_confidence=0.5,
    )
    assert identify_good_clusters(dataset, cfg) == {0, 1}


def test_identify_fails_image_frequency():
    """Cluster 1 appears in only 2/10 images → below 0.5 threshold."""
    dataset = _build_dataset(n_images=10, cluster_ids=[0])
    # Add cluster 1 to only 2 images
    for i in range(2):
        path = Path(f"/tmp/img_{i}.png")
        dataset[path].append(
            _make_labeled(organ_id=1, source_path=str(path))
        )

    cfg = ClusterQualityConfig(
        min_image_frequency=0.5,
        min_avg_labeling_confidence=0.0,
        min_avg_sam_confidence=0.0,
    )
    result = identify_good_clusters(dataset, cfg)
    assert 0 in result
    assert 1 not in result


def test_identify_fails_avg_labeling_confidence():
    """Cluster with low labeling confidence is excluded."""
    dataset: dict[Path, list[LabeledObject]] = {}
    for i in range(10):
        path = Path(f"/tmp/img_{i}.png")
        dataset[path] = [
            _make_labeled(organ_id=0, labeling_confidence=0.8, source_path=str(path)),
            _make_labeled(organ_id=1, labeling_confidence=0.1, source_path=str(path)),
        ]

    cfg = ClusterQualityConfig(
        min_image_frequency=0.0,
        min_avg_labeling_confidence=0.5,
        min_avg_sam_confidence=0.0,
    )
    result = identify_good_clusters(dataset, cfg)
    assert 0 in result
    assert 1 not in result


def test_identify_fails_avg_sam_confidence():
    """Cluster with low SAM score is excluded."""
    dataset: dict[Path, list[LabeledObject]] = {}
    for i in range(10):
        path = Path(f"/tmp/img_{i}.png")
        dataset[path] = [
            _make_labeled(organ_id=0, sam_confidence=0.9, source_path=str(path)),
            _make_labeled(organ_id=1, sam_confidence=0.2, source_path=str(path)),
        ]

    cfg = ClusterQualityConfig(
        min_image_frequency=0.0,
        min_avg_labeling_confidence=0.0,
        min_avg_sam_confidence=0.6,
    )
    result = identify_good_clusters(dataset, cfg)
    assert 0 in result
    assert 1 not in result


def test_identify_noise_objects_ignored():
    """Noise objects must not contribute to image frequency or mean metrics."""
    # Cluster 0: 10 non-noise objects across 10 images → should pass.
    # Cluster 1: 10 noise objects across 10 images → no valid members → excluded.
    dataset: dict[Path, list[LabeledObject]] = {}
    for i in range(10):
        path = Path(f"/tmp/img_{i}.png")
        dataset[path] = [
            _make_labeled(organ_id=0, is_noise=False, source_path=str(path)),
            _make_labeled(organ_id=1, is_noise=True, source_path=str(path)),
        ]

    cfg = ClusterQualityConfig(
        min_image_frequency=0.1,
        min_avg_labeling_confidence=0.0,
        min_avg_sam_confidence=0.0,
    )
    result = identify_good_clusters(dataset, cfg)
    assert 0 in result
    assert 1 not in result


def test_identify_sam_none_not_penalised():
    """Cluster with sam_confidence=None on all objects should not fail the SAM threshold."""
    dataset: dict[Path, list[LabeledObject]] = {}
    for i in range(10):
        path = Path(f"/tmp/img_{i}.png")
        dataset[path] = [
            _make_labeled(organ_id=0, sam_confidence=None, source_path=str(path))
        ]

    cfg = ClusterQualityConfig(
        min_image_frequency=0.0,
        min_avg_labeling_confidence=0.0,
        min_avg_sam_confidence=0.9,
    )
    assert 0 in identify_good_clusters(dataset, cfg)


def test_identify_hdbscan_noise_id_excluded():
    """organ_id == -1 (HDBSCAN noise label) must be excluded even if is_noise=False."""
    dataset: dict[Path, list[LabeledObject]] = {}
    for i in range(10):
        path = Path(f"/tmp/img_{i}.png")
        dataset[path] = [
            _make_labeled(organ_id=-1, is_noise=False, source_path=str(path))
        ]

    cfg = ClusterQualityConfig(min_image_frequency=0.0, min_avg_labeling_confidence=0.0,
                               min_avg_sam_confidence=0.0)
    assert -1 not in identify_good_clusters(dataset, cfg)


# ---------------------------------------------------------------------------
# select_prototypes — core behaviour
# ---------------------------------------------------------------------------

def test_select_prototypes_returns_at_most_k():
    dataset = _build_dataset(n_images=10, cluster_ids=[0])
    good = {0}
    cfg = MaskSelectionConfig(sam_score_weight=0.5, min_combined_score=0.0)
    result = select_prototypes(dataset, good, k=3, config=cfg)
    assert len(result[0]) <= 3


def test_select_prototypes_sorted_descending():
    """Objects returned for each cluster must be sorted best-score first."""
    dataset: dict[Path, list[LabeledObject]] = {}
    scores = [0.9, 0.7, 0.5, 0.3]
    for i, lc in enumerate(scores):
        path = Path(f"/tmp/img_{i}.png")
        dataset[path] = [
            _make_labeled(organ_id=0, labeling_confidence=lc, sam_confidence=lc,
                          source_path=str(path))
        ]

    cfg = MaskSelectionConfig(sam_score_weight=0.5, min_combined_score=0.0)
    result = select_prototypes(dataset, good_clusters={0}, k=4, config=cfg)
    confs = [obj.labeling_confidence for obj in result[0]]
    assert confs == sorted(confs, reverse=True)


def test_select_prototypes_noise_excluded():
    """Noise objects must never appear in prototypes."""
    dataset: dict[Path, list[LabeledObject]] = {}
    for i in range(5):
        path = Path(f"/tmp/img_{i}.png")
        dataset[path] = [
            _make_labeled(organ_id=0, is_noise=True, source_path=str(path))
        ]

    cfg = MaskSelectionConfig(sam_score_weight=0.5, min_combined_score=0.0)
    result = select_prototypes(dataset, good_clusters={0}, k=10, config=cfg)
    assert result[0] == []


def test_select_prototypes_clusters_not_in_good_ignored():
    """Objects whose cluster is not in good_clusters must not appear in the result."""
    dataset = _build_dataset(n_images=5, cluster_ids=[0, 1, 2])
    cfg = MaskSelectionConfig(sam_score_weight=0.5, min_combined_score=0.0)
    result = select_prototypes(dataset, good_clusters={0}, k=10, config=cfg)
    assert set(result.keys()) == {0}
    assert all(obj.organ_id == 0 for obj in result[0])


def test_select_prototypes_min_combined_score_gate():
    """Objects below min_combined_score must be excluded."""
    dataset: dict[Path, list[LabeledObject]] = {}
    for i in range(5):
        path = Path(f"/tmp/img_{i}.png")
        dataset[path] = [
            # score = 0.5 * 0.4 + 0.5 * 0.4 = 0.4  → below 0.6 gate
            _make_labeled(organ_id=0, labeling_confidence=0.4, sam_confidence=0.4,
                          source_path=str(path))
        ]

    cfg = MaskSelectionConfig(sam_score_weight=0.5, min_combined_score=0.6)
    result = select_prototypes(dataset, good_clusters={0}, k=10, config=cfg)
    assert result[0] == []


def test_select_prototypes_all_clusters_present_as_keys():
    """Every cluster in good_clusters must appear as a key even with zero prototypes."""
    dataset = _build_dataset(n_images=3, cluster_ids=[0])
    cfg = MaskSelectionConfig(sam_score_weight=0.5, min_combined_score=0.99)
    result = select_prototypes(dataset, good_clusters={0, 1}, k=5, config=cfg)
    assert 0 in result
    assert 1 in result


def test_select_prototypes_k_governs_limit():
    """k must govern the number of prototypes returned per cluster."""
    dataset = _build_dataset(n_images=20, cluster_ids=[0])
    cfg = MaskSelectionConfig(sam_score_weight=0.5, min_combined_score=0.0)
    result = select_prototypes(dataset, good_clusters={0}, k=7, config=cfg)
    assert len(result[0]) == 7


# ---------------------------------------------------------------------------
# Integration: identify then select
# ---------------------------------------------------------------------------

def test_identify_then_select_only_good_clusters():
    """Full pipeline: bad clusters identified by identify_good_clusters are not
    included in select_prototypes."""
    dataset: dict[Path, list[LabeledObject]] = {}
    for i in range(10):
        path = Path(f"/tmp/img_{i}.png")
        objects = [
            # cluster 0: high quality, all images
            _make_labeled(organ_id=0, labeling_confidence=0.9, sam_confidence=0.9,
                          source_path=str(path)),
        ]
        if i < 2:
            # cluster 1: appears in only 2/10 images → bad frequency
            objects.append(
                _make_labeled(organ_id=1, labeling_confidence=0.9, sam_confidence=0.9,
                              source_path=str(path))
            )
        dataset[path] = objects

    quality_cfg = ClusterQualityConfig(
        min_image_frequency=0.5,
        min_avg_labeling_confidence=0.0,
        min_avg_sam_confidence=0.0,
    )
    sel_cfg = MaskSelectionConfig(sam_score_weight=0.5, min_combined_score=0.0)

    good = identify_good_clusters(dataset, quality_cfg)
    assert good == {0}

    prototypes = select_prototypes(dataset, good, k=3, config=sel_cfg)
    assert set(prototypes.keys()) == {0}
    assert len(prototypes[0]) == 3


# ---------------------------------------------------------------------------
# suppress_overlapping_clusters — cluster-level NMS
# ---------------------------------------------------------------------------

# With sam_score_weight=0.5 and sam==lconf==strength, the combined score equals
# `strength`, so these helpers give full control over each representative.
_NMS_SEL = MaskSelectionConfig(sam_score_weight=0.5, min_combined_score=0.0)
_NMS_CFG = ClusterNMSConfig(enabled=True, iou_threshold=0.5)


def _proto(mask, strength: float) -> list[LabeledObject]:
    """One-element prototype list with the given mask and combined strength."""
    image = MedicalImage(
        volume=np.zeros((2, 2, 3), dtype=np.float32),
        modality="synthetic",
        source_path="/tmp/x.png",
    )
    seg = SegmentedObject(mask=mask, source_image=image, confidence=strength)
    obj = LabeledObject(
        segmented_object=seg,
        organ_id=0,
        organ_name="c",
        labeling_confidence=strength,
        method_used="test",
    )
    return [obj]


def _full(h: int = 4, w: int = 4) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def test_two_overlapping_clusters_weaker_suppressed():
    """Overlapping prototypes: the weaker cluster is suppressed by the stronger."""
    prototypes = {0: _proto(_full(), 0.9), 1: _proto(_full(), 0.5)}
    survivors, report = suppress_overlapping_clusters(
        prototypes, {0: 5, 1: 5}, _NMS_SEL, _NMS_CFG
    )
    assert set(survivors) == {0}
    assert report == [{"cluster": 1, "suppressed_by": 0, "iou": 1.0}]


def test_disjoint_clusters_both_kept():
    """Non-overlapping prototypes both survive; report is empty."""
    a = np.zeros((4, 4), dtype=bool); a[:2, :2] = True
    b = np.zeros((4, 4), dtype=bool); b[2:, 2:] = True
    prototypes = {0: _proto(a, 0.9), 1: _proto(b, 0.8)}
    survivors, report = suppress_overlapping_clusters(
        prototypes, {0: 1, 1: 1}, _NMS_SEL, _NMS_CFG
    )
    assert set(survivors) == {0, 1}
    assert report == []


def test_tie_is_deterministic():
    """Equal strength → larger size wins; equal size → lower id wins."""
    for _ in range(5):
        prototypes = {0: _proto(_full(), 0.7), 1: _proto(_full(), 0.7)}
        survivors, report = suppress_overlapping_clusters(
            prototypes, {0: 3, 1: 9}, _NMS_SEL, _NMS_CFG
        )
        assert set(survivors) == {1}
        assert report == [{"cluster": 0, "suppressed_by": 1, "iou": 1.0}]

    for _ in range(5):
        prototypes = {0: _proto(_full(), 0.7), 1: _proto(_full(), 0.7)}
        survivors, report = suppress_overlapping_clusters(
            prototypes, {0: 5, 1: 5}, _NMS_SEL, _NMS_CFG
        )
        assert set(survivors) == {0}
        assert report == [{"cluster": 1, "suppressed_by": 0, "iou": 1.0}]


def test_disabled_is_noop():
    """enabled=False returns the prototypes unchanged and an empty report."""
    prototypes = {0: _proto(_full(), 0.9), 1: _proto(_full(), 0.5)}
    survivors, report = suppress_overlapping_clusters(
        prototypes, {0: 5, 1: 5}, _NMS_SEL,
        ClusterNMSConfig(enabled=False, iou_threshold=0.5),
    )
    assert survivors is prototypes
    assert report == []


def test_malformed_mask_none_raises():
    """A None representative mask raises ValueError, not silent skip."""
    prototypes = {0: _proto(None, 0.9)}
    with pytest.raises(ValueError):
        suppress_overlapping_clusters(prototypes, {0: 1}, _NMS_SEL, _NMS_CFG)


def test_mismatched_shapes_reconciled_by_resize():
    """Representatives of different shapes (variable-resolution datasets like
    ACDC) are reconciled with a nearest-neighbor resize before the IoU instead
    of raising: two full masks of different sizes overlap fully → weaker dropped."""
    prototypes = {
        0: _proto(np.ones((4, 4), dtype=bool), 0.9),
        1: _proto(np.ones((3, 3), dtype=bool), 0.8),
    }
    survivors, report = suppress_overlapping_clusters(
        prototypes, {0: 1, 1: 1}, _NMS_SEL, _NMS_CFG
    )
    assert set(survivors) == {0}
    assert report == [{"cluster": 1, "suppressed_by": 0, "iou": 1.0}]


def test_mismatched_shapes_disjoint_both_kept():
    """Different-shape representatives that map to disjoint regions both survive."""
    a = np.zeros((4, 4), dtype=bool); a[:2, :] = True          # top half
    b = np.zeros((8, 8), dtype=bool); b[4:, :] = True          # bottom half
    prototypes = {0: _proto(a, 0.9), 1: _proto(b, 0.8)}
    survivors, report = suppress_overlapping_clusters(
        prototypes, {0: 1, 1: 1}, _NMS_SEL, _NMS_CFG
    )
    assert set(survivors) == {0, 1}
    assert report == []

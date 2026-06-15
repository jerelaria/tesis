"""Tests for build_unsupervised_multiorgan_references and its pure helpers.

GPU is NOT used.  The merge/selection helpers are exercised directly on hand-
built anchor_to_masks dicts; the end-to-end builder reuses StubVideoSegmenter
and StubReader from test_propagator.
"""
import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace

from project.core.data_types import MedicalImage, SegmentedObject, LabeledObject
from project.data_io.few_shot_reader import FewShotReference
from project.pipeline.reference_builder import (
    build_unsupervised_multiorgan_references,
    _merge_clusters_by_iou,
    _select_complete_topk_anchors,
)
from project.segmentation.quality import ClusterQualityConfig, MaskSelectionConfig
from project.pipeline.propagator import PropagationConfig

from tests.test_propagator import StubVideoSegmenter, StubReader, H, W


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _disc(cy: float, cx: float, r: float = 3.0) -> np.ndarray:
    yy, xx = np.mgrid[:H, :W]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2


# Distinct, well-separated centers so clusters never merge by IoU.
_CENTERS = {0: (3, 3), 1: (3, 16), 2: (16, 9)}


def _labeled(cid: int, path: Path, sam: float = 0.9) -> LabeledObject:
    cy, cx = _CENTERS[cid]
    image = MedicalImage(
        volume=np.zeros((H, W, 3), dtype=np.float32),
        modality="synthetic",
        source_path=str(path),
    )
    seg = SegmentedObject(
        mask=_disc(cy, cx), source_image=image, confidence=sam
    )
    return LabeledObject(
        segmented_object=seg,
        organ_id=cid,
        organ_name=f"cluster_{cid}",
        labeling_confidence=0.8,
        method_used="test",
    )


def _multiorgan_dataset(
    image_clusters: dict[str, list[int]],
) -> dict[Path, list[LabeledObject]]:
    """Build a labeled-by-image dict with distinct disc masks per cluster."""
    dataset: dict[Path, list[LabeledObject]] = {}
    for name, cids in image_clusters.items():
        path = Path(name)
        dataset[path] = [_labeled(cid, path) for cid in cids]
    return dataset


def _stub_with_config(score_threshold=0.0, iou_threshold=0.5):
    stub = StubVideoSegmenter()
    stub.config = SimpleNamespace(
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )
    return stub


def _multiorgan_config(k=2):
    return PropagationConfig(
        references_per_cluster=k,
        quality=ClusterQualityConfig(
            min_image_frequency=0.0,
            min_avg_labeling_confidence=0.0,
            min_avg_sam_confidence=0.0,
        ),
        mask_selection=MaskSelectionConfig(
            sam_score_weight=0.5,
            min_combined_score=0.0,
        ),
    )


# ---------------------------------------------------------------------------
# _merge_clusters_by_iou
# ---------------------------------------------------------------------------

class TestMergeClustersByIou:

    def test_near_identical_labels_collapse(self):
        """Two labels with near-equal masks across anchors collapse to one."""
        mask = _disc(10, 10)
        anchor_to_masks = {
            Path(f"/tmp/a{i}.png"): {
                "cluster_0": (mask, 0.7),
                "cluster_1": (mask.copy(), 0.9),
            }
            for i in range(3)
        }
        merged, report = _merge_clusters_by_iou(anchor_to_masks, iou_threshold=0.5)

        organs = {label for masks in merged.values() for label in masks}
        assert organs == {"cluster_1"}, "higher mean-SAM label must survive"
        assert len(report) == 1
        assert report[0]["canonical"] == "cluster_1"
        assert report[0]["discarded"] == ["cluster_0"]
        # one mask per anchor after collapse
        for masks in merged.values():
            assert set(masks.keys()) == {"cluster_1"}

    def test_disjoint_labels_stay_separate(self):
        """Spatially disjoint labels are not merged."""
        left, right = _disc(5, 5), _disc(15, 15)
        anchor_to_masks = {
            Path(f"/tmp/a{i}.png"): {
                "cluster_0": (left, 0.8),
                "cluster_1": (right, 0.8),
            }
            for i in range(3)
        }
        merged, report = _merge_clusters_by_iou(anchor_to_masks, iou_threshold=0.5)

        organs = {label for masks in merged.values() for label in masks}
        assert organs == {"cluster_0", "cluster_1"}
        assert report == []

    def test_canonical_is_highest_mean_sam(self):
        """When collapsing, the label with the highest mean SAM is canonical."""
        mask = _disc(10, 10)
        anchor_to_masks = {
            Path("/tmp/a0.png"): {
                "cluster_0": (mask, 0.95),
                "cluster_5": (mask.copy(), 0.60),
            },
            Path("/tmp/a1.png"): {
                "cluster_0": (mask.copy(), 0.95),
                "cluster_5": (mask.copy(), 0.60),
            },
        }
        merged, report = _merge_clusters_by_iou(anchor_to_masks, iou_threshold=0.5)
        assert report[0]["canonical"] == "cluster_0"
        for masks in merged.values():
            assert set(masks.keys()) == {"cluster_0"}


# ---------------------------------------------------------------------------
# _select_complete_topk_anchors
# ---------------------------------------------------------------------------

class TestSelectCompleteTopK:

    def test_discards_incomplete_anchors(self):
        organs = {"cluster_0", "cluster_1"}
        m = np.ones((H, W), dtype=bool)
        anchor_to_masks = {
            Path("/tmp/full.png"): {"cluster_0": (m, 0.8), "cluster_1": (m, 0.7)},
            Path("/tmp/partial.png"): {"cluster_0": (m, 0.9)},
        }
        chosen = _select_complete_topk_anchors(anchor_to_masks, organs, k=5)
        assert [a for a, _ in chosen] == [Path("/tmp/full.png")]

    def test_respects_k_and_min_sam_order(self):
        organs = {"cluster_0", "cluster_1"}
        m = np.ones((H, W), dtype=bool)
        anchor_to_masks = {
            Path("/tmp/low.png"): {"cluster_0": (m, 0.5), "cluster_1": (m, 0.9)},
            Path("/tmp/high.png"): {"cluster_0": (m, 0.8), "cluster_1": (m, 0.85)},
            Path("/tmp/mid.png"): {"cluster_0": (m, 0.7), "cluster_1": (m, 0.95)},
        }
        chosen = _select_complete_topk_anchors(anchor_to_masks, organs, k=2)
        # min-SAM per anchor: low=0.5, high=0.8, mid=0.7 -> top2 = high, mid
        assert [a for a, _ in chosen] == [Path("/tmp/high.png"), Path("/tmp/mid.png")]
        assert chosen[0][1] == pytest.approx(0.8)
        assert chosen[1][1] == pytest.approx(0.7)

    def test_raises_when_no_complete_anchor(self):
        organs = {"cluster_0", "cluster_1"}
        m = np.ones((H, W), dtype=bool)
        anchor_to_masks = {
            Path("/tmp/a.png"): {"cluster_0": (m, 0.8)},
            Path("/tmp/b.png"): {"cluster_1": (m, 0.8)},
        }
        with pytest.raises(ValueError):
            _select_complete_topk_anchors(anchor_to_masks, organs, k=2)


# ---------------------------------------------------------------------------
# End-to-end integration
# ---------------------------------------------------------------------------

class TestBuildMultiorgan:

    def test_end_to_end_multiorgan_references(self):
        """All-native-complete images yield multi-organ FewShotReferences."""
        labeled_by_image = _multiorgan_dataset({
            f"/tmp/img_{i}.png": [0, 1, 2] for i in range(6)
        })
        stub = _stub_with_config(score_threshold=0.0, iou_threshold=0.5)
        config = _multiorgan_config(k=2)

        refs, scores = build_unsupervised_multiorgan_references(
            labeled_by_image, config, StubReader(), stub
        )

        assert len(refs) == len(scores)
        assert len(refs) == config.references_per_cluster
        for ref in refs:
            assert isinstance(ref, FewShotReference)
            assert set(ref.masks.keys()) == {"cluster_0", "cluster_1", "cluster_2"}
            for mask in ref.masks.values():
                assert mask.shape == (H, W)
        for s in scores:
            assert isinstance(s, float)

    def test_propagation_fills_missing_organ(self):
        """An organ absent from an anchor is filled by propagation (gate open)."""
        labeled_by_image = _multiorgan_dataset({
            "/tmp/both.png": [0, 1],
            "/tmp/only0.png": [0],
        })
        stub = _stub_with_config(score_threshold=0.0, iou_threshold=0.5)
        config = _multiorgan_config(k=2)

        refs, _ = build_unsupervised_multiorgan_references(
            labeled_by_image, config, StubReader(), stub
        )
        # both anchors complete: only0.png gets cluster_1 from propagation
        assert len(refs) == 2
        for ref in refs:
            assert set(ref.masks.keys()) == {"cluster_0", "cluster_1"}

    def test_gate_drops_low_score_propagations(self):
        """A gate above the propagated SAM (0.8) drops the propagation-only anchor."""
        labeled_by_image = _multiorgan_dataset({
            "/tmp/both.png": [0, 1],
            "/tmp/only0.png": [0],
        })
        stub = _stub_with_config(score_threshold=0.9, iou_threshold=0.5)
        config = _multiorgan_config(k=2)

        refs, _ = build_unsupervised_multiorgan_references(
            labeled_by_image, config, StubReader(), stub
        )
        # only0.png becomes incomplete (cluster_1 propagation gated out)
        assert len(refs) == 1
        assert set(refs[0].masks.keys()) == {"cluster_0", "cluster_1"}
        assert Path(refs[0].source_path) == Path("/tmp/both.png")

    def test_no_good_clusters_returns_empty(self):
        labeled_by_image = _multiorgan_dataset({"/tmp/a.png": [0], "/tmp/b.png": [0]})
        # drive SAM below the quality threshold
        for objs in labeled_by_image.values():
            for o in objs:
                o.segmented_object.confidence = 0.3
        stub = _stub_with_config()
        config = PropagationConfig(
            references_per_cluster=1,
            quality=ClusterQualityConfig(
                min_image_frequency=0.0,
                min_avg_labeling_confidence=0.0,
                min_avg_sam_confidence=0.9,
            ),
            mask_selection=MaskSelectionConfig(min_combined_score=0.0),
        )
        refs, scores = build_unsupervised_multiorgan_references(
            labeled_by_image, config, StubReader(), stub
        )
        assert refs == [] and scores == []

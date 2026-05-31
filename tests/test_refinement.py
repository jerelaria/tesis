"""Tests for hardened _improve_existing_masks and select_closest_component."""
import numpy as np
import pytest
from pathlib import Path

from project.core.data_types import MedicalImage, SegmentedObject, LabeledObject
from project.segmentation.utils import select_closest_component
from project.segmentation.refinement import RetroactiveRefiner, RefinementConfig
from project.segmentation.quality import MaskSelectionConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

H, W = 100, 100


def _image():
    return MedicalImage(
        volume=np.zeros((H, W, 3), dtype=np.float32),
        modality="synthetic",
        source_path="/tmp/fake.png",
    )


def _seg(mask: np.ndarray, sam: float = 0.9, source_image=None):
    si = source_image if source_image is not None else _image()
    return SegmentedObject(mask=mask, source_image=si, confidence=sam)


def _labeled(seg: SegmentedObject, organ_id: int = 0, conf: float = 0.9):
    return LabeledObject(
        segmented_object=seg,
        organ_id=organ_id,
        organ_name=f"cluster_{organ_id}",
        labeling_confidence=conf,
        method_used="test",
    )


def _large_mask(r0=20, r1=80, c0=20, c1=80):
    """60x60 block; area=3600, centroid≈(49.5, 49.5)."""
    m = np.zeros((H, W), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def _tiny_mask(row=47, col=47, side=5):
    """Small patch near the center; area=side^2."""
    m = np.zeros((H, W), dtype=bool)
    m[row:row + side, col:col + side] = True
    return m


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class StubSegmenter:
    """Returns a fixed SegmentedObject from segment_with_multi_reference."""

    def __init__(self):
        self.result: SegmentedObject | None = None

    def segment_with_multi_reference(self, target_image, reference_entries, organ_name):
        return self.result

    def encode_image(self, image):
        return None


class StubExtractor:
    def extract(self, obj):
        obj.features = np.zeros(16, dtype=np.float32)
        return obj.features


class StubReader:
    def load(self, path: str):
        return _image()


# ---------------------------------------------------------------------------
# Refiner factory
# ---------------------------------------------------------------------------

def _make_refiner(config: RefinementConfig) -> RetroactiveRefiner:
    refiner = RetroactiveRefiner(
        segmenter=StubSegmenter(),
        extractor=StubExtractor(),
        config=config,
        labeler=None,
        extract_embeddings=False,
        debug_dir=None,
    )
    refiner._refinement_log = None
    return refiner


# ---------------------------------------------------------------------------
# select_closest_component — max_centroid_distance gate
# ---------------------------------------------------------------------------

class TestSelectClosestComponentDistanceGate:

    def test_far_single_component_rejected(self):
        """Single component far from ref centroid is rejected when gate is on."""
        mask = _tiny_mask(row=5, col=5, side=5)   # top-left corner
        ref_centroid = (80.0, 80.0)               # bottom-right corner
        result = select_closest_component(mask, ref_centroid, max_centroid_distance=10.0)
        assert result is None

    def test_near_single_component_accepted(self):
        """Single component near ref centroid passes even with a tight gate."""
        mask = _tiny_mask(row=47, col=47, side=5)
        ref_centroid = (49.0, 49.0)
        result = select_closest_component(mask, ref_centroid, max_centroid_distance=10.0)
        assert result is not None
        assert result.any()

    def test_multi_component_near_accepted(self):
        """Among two large components, the one near the ref is accepted."""
        mask = np.zeros((H, W), dtype=bool)
        mask[5:15, 5:15] = True    # centroid ≈ (9.5, 9.5), area=100
        mask[70:80, 70:80] = True  # centroid ≈ (74.5, 74.5), area=100
        ref_centroid = (9.5, 9.5)

        # Closest component is the top-left block; its distance ≈ 0 < gate=20
        result = select_closest_component(mask, ref_centroid, max_centroid_distance=20.0)
        assert result is not None
        # Verify it's the top-left block (pixel at row=10, col=10 should be True)
        assert result[10, 10]

    def test_multi_component_both_too_far_rejected(self):
        """When the closest component still exceeds the gate, return None."""
        mask = np.zeros((H, W), dtype=bool)
        mask[5:15, 5:15] = True    # centroid ≈ (9.5, 9.5)
        mask[70:80, 70:80] = True  # centroid ≈ (74.5, 74.5)
        # ref sits at (50, 50); closest block is (9.5, 9.5), dist ≈ 56.9
        ref_centroid = (50.0, 50.0)
        result = select_closest_component(mask, ref_centroid, max_centroid_distance=10.0)
        assert result is None

    def test_no_gate_backward_compatible(self):
        """max_centroid_distance=None preserves original behavior."""
        mask = _tiny_mask(row=5, col=5, side=5)
        ref_centroid = (80.0, 80.0)
        result = select_closest_component(mask, ref_centroid, max_centroid_distance=None)
        # Without gate, the only component is returned regardless of distance.
        assert result is not None


# ---------------------------------------------------------------------------
# _improve_existing_masks — structural guards (Fix 1)
# ---------------------------------------------------------------------------

CLUSTER_ID = 0


def _setup_improve_scenario(old_mask: np.ndarray, new_mask: np.ndarray, config: RefinementConfig):
    """
    Build the minimal RefinerActively-with-stubs scenario and run _improve.
    Returns (refiner, labeled_by_image, was_replaced: bool).
    """
    ref_img = _image()
    ref_img.source_path = "/tmp/ref.png"
    ref_img.volume = np.zeros((H, W, 3), dtype=np.float32)

    ref_mask = _large_mask()
    ref_seg = _seg(ref_mask, sam=0.9, source_image=ref_img)
    ref_labeled = _labeled(ref_seg, organ_id=CLUSTER_ID, conf=0.9)

    tgt_img = _image()
    tgt_img.source_path = "/tmp/target.png"

    tgt_seg = _seg(old_mask, sam=0.5, source_image=tgt_img)
    tgt_labeled = _labeled(tgt_seg, organ_id=CLUSTER_ID, conf=0.6)

    ref_path = Path("/tmp/ref.png")
    tgt_path = Path("/tmp/target.png")

    labeled_by_image = {
        ref_path: [ref_labeled],
        tgt_path: [tgt_labeled],
    }
    objects_by_image = {
        ref_path: [ref_seg],
        tgt_path: [tgt_seg],
    }

    refiner = _make_refiner(config)
    refiner.segmenter.result = _seg(new_mask, sam=0.9)

    good_clusters = {CLUSTER_ID}
    refiner._improve_existing_masks(
        objects_by_image, labeled_by_image, good_clusters, StubReader()
    )

    was_replaced = (tgt_labeled.method_used == "refinement_improved")
    return refiner, tgt_labeled, was_replaced


class TestImproveExistingMasksGuards:

    def _config(self, **kwargs):
        defaults = dict(
            enabled=True,
            improve_existing=True,
            improve_min_combined_score=1.0,
            improve_sam_score_weight=0.2,
            improve_min_area_ratio=0.5,
            improve_max_area_ratio=None,
            improve_min_iou_with_original=0.5,
            max_centroid_distance_factor=None,
            mask_selection=MaskSelectionConfig(
                sam_score_weight=0.5,
                min_combined_score=0.75,
                num_reference_frames=1,
            ),
        )
        defaults.update(kwargs)
        return RefinementConfig(**defaults)

    def test_tiny_mask_not_replaced(self):
        """Degenerate tiny mask (high SAM but tiny area) must NOT replace a large original."""
        old_mask = _large_mask()   # 3600 px
        new_mask = _tiny_mask()    # 25 px → area_ratio ≈ 0.007 < 0.5

        _, tgt, replaced = _setup_improve_scenario(old_mask, new_mask, self._config())
        assert not replaced, "Tiny mask should be rejected by area_ratio guard"
        assert tgt.method_used != "refinement_improved"

    def test_teleported_mask_not_replaced(self):
        """Mask that relocated to a non-overlapping region must NOT replace original."""
        old_mask = _large_mask(r0=10, r1=50, c0=10, c1=50)  # top-left quadrant
        new_mask = _large_mask(r0=60, r1=99, c0=60, c1=99)  # bottom-right quadrant; IoU=0

        _, tgt, replaced = _setup_improve_scenario(old_mask, new_mask, self._config())
        assert not replaced, "Teleported mask (IoU=0) should be rejected by iou guard"

    def test_valid_adjusted_mask_is_replaced(self):
        """A mask that slightly shrinks original (high IoU, similar area, better SAM) IS replaced."""
        old_mask = _large_mask(r0=20, r1=80, c0=20, c1=80)  # 3600 px
        # Slightly smaller: 3481 px, IoU = 3481/3600 ≈ 0.967
        new_mask = _large_mask(r0=20, r1=79, c0=20, c1=79)

        _, tgt, replaced = _setup_improve_scenario(old_mask, new_mask, self._config())
        # old_score = 0.2*0.5 + 0.8*0.6 = 0.58
        # new_score = 0.2*0.9 + 0.8*0.6 = 0.66 > 0.58
        assert replaced, "Valid high-IoU mask with better score should be accepted"
        assert tgt.method_used == "refinement_improved"

    def test_worse_score_not_replaced(self):
        """Even a structurally valid mask is skipped when its score is lower."""
        old_mask = _large_mask(r0=20, r1=80, c0=20, c1=80)
        new_mask = _large_mask(r0=20, r1=79, c0=20, c1=79)

        # Force new_obj.confidence very low so new_score < original_score.
        cfg = self._config()
        refiner = _make_refiner(cfg)
        # new SAM = 0.1 → new_score = 0.2*0.1 + 0.8*0.6 = 0.50 < 0.58
        refiner.segmenter.result = _seg(new_mask, sam=0.1)

        ref_path = Path("/tmp/ref2.png")
        tgt_path = Path("/tmp/target2.png")
        ref_img = _image()
        ref_img.source_path = str(ref_path)
        ref_img.volume = np.zeros((H, W, 3), dtype=np.float32)

        tgt_seg = _seg(old_mask, sam=0.5, source_image=_image())
        tgt_labeled = _labeled(tgt_seg, organ_id=CLUSTER_ID, conf=0.6)
        ref_seg = _seg(_large_mask(), sam=0.9, source_image=ref_img)
        ref_labeled = _labeled(ref_seg, organ_id=CLUSTER_ID, conf=0.9)

        labeled_by_image = {ref_path: [ref_labeled], tgt_path: [tgt_labeled]}
        objects_by_image = {ref_path: [ref_seg], tgt_path: [tgt_seg]}

        refiner._improve_existing_masks(
            objects_by_image, labeled_by_image, {CLUSTER_ID}, StubReader()
        )
        assert tgt_labeled.method_used != "refinement_improved"

    def test_ablation_zero_thresholds_allows_tiny_replacement(self):
        """With all guards disabled (ablation), a tiny high-SAM mask replaces the original."""
        old_mask = _large_mask()
        new_mask = _tiny_mask()  # 25 px — would normally fail area_ratio

        cfg = self._config(
            improve_min_area_ratio=0.0,
            improve_min_iou_with_original=0.0,
            max_centroid_distance_factor=None,
        )
        _, tgt, replaced = _setup_improve_scenario(old_mask, new_mask, cfg)
        # new_score = 0.2*0.9 + 0.8*0.6 = 0.66 > 0.58 → accepted with no guards
        assert replaced, "Ablation mode (zero thresholds) should reproduce the old bug"

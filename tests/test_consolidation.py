"""Tests for the offline, GPU-free consolidation tool.

Covers project.consolidation.core.consolidate directly (built from hand-made
MaskRecords, following the _disc helper style used in
tests/test_reference_builder_multiorgan.py) plus a disk round-trip through
project.consolidation.io.
"""

import json

import numpy as np
import pytest
from PIL import Image

from project.consolidation.core import MaskRecord, consolidate
from project.consolidation.io import read_records_from_results, write_consolidated_results

H, W = 40, 40


def _disc(cy: float, cx: float, r: float = 10.0) -> np.ndarray:
    yy, xx = np.mgrid[:H, :W]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2


def _mr(label: str, mask: np.ndarray, sam: float) -> MaskRecord:
    return MaskRecord(
        label=label,
        mask=mask,
        sam=sam,
        scores={"sam": sam, "labeling": 0.8, "combined": sam},
    )


# Well-separated centers so unrelated organs never accidentally overlap.
_A = (10, 10)
_B = (10, 30)


# ---------------------------------------------------------------------------
# consolidate()
# ---------------------------------------------------------------------------

class TestConsolidate:

    def test_near_identical_labels_collapse_to_higher_sam_canonical(self):
        mask = _disc(*_A)
        records_by_image = {
            f"img{i}": [
                _mr("cluster_0", mask.copy(), sam=0.7),
                _mr("cluster_1", mask.copy(), sam=0.9),
            ]
            for i in range(3)
        }
        consolidated, report = consolidate(records_by_image, iou_threshold=0.5)

        labels = {r.label for recs in consolidated.values() for r in recs}
        assert labels == {"cluster_1"}
        assert report["n_labels_before"] == 2
        assert report["n_organs_after"] == 1
        assert len(report["groups"]) == 1
        group = report["groups"][0]
        assert group["canonical"] == "cluster_1"
        assert group["discarded"] == ["cluster_0"]
        assert group["median_iou"]["cluster_0|cluster_1"] == pytest.approx(1.0)

    def test_disjoint_labels_stay_separate(self):
        left, right = _disc(*_A), _disc(*_B)
        records_by_image = {
            f"img{i}": [
                _mr("cluster_0", left.copy(), sam=0.8),
                _mr("cluster_1", right.copy(), sam=0.8),
            ]
            for i in range(3)
        }
        consolidated, report = consolidate(records_by_image, iou_threshold=0.5)

        labels = {r.label for recs in consolidated.values() for r in recs}
        assert labels == {"cluster_0", "cluster_1"}
        assert report["groups"] == []
        assert report["n_organs_after"] == 2

    def test_hanging_cluster_survives_under_canonical_label(self):
        mask = _disc(*_A)
        records_by_image = {
            "img0": [
                _mr("cluster_1", mask.copy(), sam=0.9),
                _mr("cluster_3", mask.copy(), sam=0.6),
            ],
            "img1": [
                _mr("cluster_1", mask.copy(), sam=0.9),
                _mr("cluster_3", mask.copy(), sam=0.6),
            ],
            "img2": [
                _mr("cluster_3", mask.copy(), sam=0.6),
            ],
        }
        consolidated, report = consolidate(records_by_image, iou_threshold=0.5)

        assert report["label_to_canonical"]["cluster_3"] == "cluster_1"
        img2_labels = {r.label for r in consolidated["img2"]}
        assert img2_labels == {"cluster_1"}
        all_labels = {r.label for recs in consolidated.values() for r in recs}
        assert "cluster_3" not in all_labels

    def test_per_image_dedup_keeps_higher_sam(self):
        mask = _disc(*_A)
        records_by_image = {
            "img0": [
                _mr("cluster_1", mask.copy(), sam=0.9),
                _mr("cluster_3", mask.copy(), sam=0.4),
            ],
            "img1": [
                _mr("cluster_1", mask.copy(), sam=0.9),
                _mr("cluster_3", mask.copy(), sam=0.4),
            ],
        }
        consolidated, report = consolidate(records_by_image, iou_threshold=0.5)

        for stem in ("img0", "img1"):
            recs = consolidated[stem]
            assert len(recs) == 1
            assert recs[0].label == "cluster_1"
            assert recs[0].sam == pytest.approx(0.9)
        assert report["n_masks_dropped_dedup"] == 2

    def test_transitive_chain_forms_one_component(self):
        m1 = _disc(20, 16)
        m3 = _disc(20, 20)
        m7 = _disc(20, 24)
        records_by_image = {
            f"img{i}": [
                _mr("cluster_1", m1.copy(), sam=0.9),
                _mr("cluster_3", m3.copy(), sam=0.8),
                _mr("cluster_7", m7.copy(), sam=0.7),
            ]
            for i in range(3)
        }
        consolidated, report = consolidate(records_by_image, iou_threshold=0.5)

        assert (
            report["label_to_canonical"]["cluster_1"]
            == report["label_to_canonical"]["cluster_3"]
            == report["label_to_canonical"]["cluster_7"]
        )
        assert report["n_organs_after"] == 1
        group = report["groups"][0]
        assert set(group["members"]) == {"cluster_1", "cluster_3", "cluster_7"}
        assert "cluster_1|cluster_7" not in group["median_iou"]

    def test_insufficient_overlap_not_merged(self):
        mask = _disc(*_A)
        records_by_image = {
            "img0": [
                _mr("cluster_0", mask.copy(), sam=0.8),
                _mr("cluster_1", mask.copy(), sam=0.8),
            ],
            "img1": [
                _mr("cluster_0", mask.copy(), sam=0.8),
                _mr("cluster_1", mask.copy(), sam=0.8),
            ],
            # A third image with only cluster_0, so total co-occurrence stays at 2.
            "img2": [
                _mr("cluster_0", mask.copy(), sam=0.8),
            ],
        }
        consolidated, report = consolidate(
            records_by_image, iou_threshold=0.5, min_overlap_images=3,
        )

        labels = {r.label for recs in consolidated.values() for r in recs}
        assert labels == {"cluster_0", "cluster_1"}
        assert report["groups"] == []
        assert ["cluster_0", "cluster_1", 2] in report["pairs_with_insufficient_overlap"]

    def test_organ_id_map_contiguous_and_1_based(self):
        # Centers spaced far enough apart (distance > 2r) that no pair overlaps.
        left, mid, right = _disc(8, 8), _disc(8, 32), _disc(32, 20)
        records_by_image = {
            "img0": [
                _mr("cluster_b", left.copy(), sam=0.8),
                _mr("cluster_a", mid.copy(), sam=0.8),
                _mr("cluster_c", right.copy(), sam=0.8),
            ],
        }
        _, report = consolidate(records_by_image, iou_threshold=0.5)

        ids = sorted(report["organ_id_map"].values())
        assert ids == list(range(1, len(report["organ_id_map"]) + 1))
        canonicals_sorted = sorted(report["organ_id_map"].keys())
        for idx, canonical in enumerate(canonicals_sorted, start=1):
            assert report["organ_id_map"][canonical] == idx

    def test_idempotent_on_own_output(self):
        mask = _disc(*_A)
        records_by_image = {
            f"img{i}": [
                _mr("cluster_0", mask.copy(), sam=0.7),
                _mr("cluster_1", mask.copy(), sam=0.9),
            ]
            for i in range(3)
        }
        consolidated_once, _ = consolidate(records_by_image, iou_threshold=0.5)
        consolidated_twice, report_twice = consolidate(consolidated_once, iou_threshold=0.5)

        assert report_twice["groups"] == []
        assert report_twice["n_masks_dropped_dedup"] == 0
        assert report_twice["n_labels_before"] == report_twice["n_organs_after"]
        for stem in consolidated_once:
            labels_once = {r.label for r in consolidated_once[stem]}
            labels_twice = {r.label for r in consolidated_twice[stem]}
            assert labels_once == labels_twice

    def test_all_empty_masks_returns_empty_without_raising(self):
        empty = np.zeros((H, W), dtype=bool)
        records_by_image = {"img0": [_mr("cluster_0", empty, sam=0.8)]}
        consolidated, report = consolidate(records_by_image, iou_threshold=0.5)
        assert consolidated == {}
        assert report["n_labels_before"] == 0
        assert report["n_organs_after"] == 0

    def test_invalid_iou_threshold_raises(self):
        with pytest.raises(ValueError):
            consolidate({"img0": [_mr("cluster_0", _disc(*_A), sam=0.8)]}, iou_threshold=1.5)

    def test_invalid_min_overlap_images_raises(self):
        with pytest.raises(ValueError):
            consolidate(
                {"img0": [_mr("cluster_0", _disc(*_A), sam=0.8)]},
                min_overlap_images=0,
            )


# ---------------------------------------------------------------------------
# Disk round-trip (io.py)
# ---------------------------------------------------------------------------

class TestDiskRoundTrip:

    def _write_fake_results(self, results_dir):
        mask = _disc(*_A)
        masks_dir = results_dir / "masks"
        for stem in ("img0", "img1"):
            image_dir = masks_dir / stem
            image_dir.mkdir(parents=True)
            for label, sam in (("cluster_1", 0.9), ("cluster_3", 0.6)):
                mask_uint8 = (mask.astype(np.uint8)) * 255
                Image.fromarray(mask_uint8, mode="L").save(image_dir / f"{label}.png")
            scores = {
                "cluster_1.png": {"sam": 0.9, "labeling": 0.8, "combined": 0.85},
                "cluster_3.png": {"sam": 0.6, "labeling": 0.8, "combined": 0.7},
            }
            with open(image_dir / "scores.json", "w") as f:
                json.dump(scores, f)

    def test_read_consolidate_write_round_trip(self, tmp_path):
        results_dir = tmp_path / "run"
        self._write_fake_results(results_dir)

        records_by_image = read_records_from_results(results_dir)
        consolidated, report = consolidate(records_by_image, iou_threshold=0.5)
        out_dir = results_dir / "masks_consolidated"
        n_written = write_consolidated_results(consolidated, out_dir)

        assert n_written == 2  # one canonical mask per image
        for stem in ("img0", "img1"):
            png_files = sorted(p.name for p in (out_dir / stem).glob("*.png"))
            assert png_files == ["cluster_1.png"]
            scores_path = out_dir / stem / "scores.json"
            assert scores_path.exists()
            with open(scores_path) as f:
                scores = json.load(f)
            assert set(scores.keys()) == {"cluster_1.png"}

    def test_empty_image_dir_without_scores_json_is_not_an_error(self, tmp_path):
        # save_predicted_masks leaves an image dir with zero pngs and no
        # scores.json when every candidate mask for that image was noise.
        results_dir = tmp_path / "run"
        self._write_fake_results(results_dir)
        (results_dir / "masks" / "img_empty").mkdir(parents=True)

        records_by_image = read_records_from_results(results_dir)

        assert records_by_image["img_empty"] == []
        consolidated, _ = consolidate(records_by_image, iou_threshold=0.5)
        assert "img_empty" not in consolidated

    def test_missing_score_entry_raises(self, tmp_path):
        results_dir = tmp_path / "run"
        masks_dir = results_dir / "masks"
        image_dir = masks_dir / "img0"
        image_dir.mkdir(parents=True)
        mask_uint8 = (_disc(*_A).astype(np.uint8)) * 255
        Image.fromarray(mask_uint8, mode="L").save(image_dir / "cluster_1.png")
        with open(image_dir / "scores.json", "w") as f:
            json.dump({}, f)  # no entry for cluster_1.png

        with pytest.raises(ValueError):
            read_records_from_results(results_dir)

    def test_missing_masks_dir_raises(self, tmp_path):
        with pytest.raises(ValueError):
            read_records_from_results(tmp_path / "no_such_run")

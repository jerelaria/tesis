"""Tests for T7: match_threshold gate, AP curve consistency, PR persistence, legacy fallback."""

import json
import numpy as np
import pytest
from PIL import Image
from pathlib import Path

from project.evaluation.matching import match_semantic, match_greedy
from project.evaluation.map import compute_average_precision, compute_map
from project.evaluation.runner import evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mask(h: int, w: int, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    """Return a bool mask of shape (h, w) with a filled rectangle."""
    m = np.zeros((h, w), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# T7.1 — match_threshold gate (both matching strategies)
# ---------------------------------------------------------------------------

class TestMatchThresholdGate:
    """
    GT: rows 0-9, cols 0-9 (100 px)
    Pred: rows 3-12, cols 3-12 (100 px)
    Overlap: rows 3-9, cols 3-9 = 7x7 = 49 px
    IoU = 49 / (100+100-49) = 49/151 ≈ 0.324
    """

    GT   = _make_mask(20, 20, 0, 10, 0, 10)
    PRED = _make_mask(20, 20, 3, 13, 3, 13)

    def test_iou_in_expected_range(self):
        iou = _iou(self.GT, self.PRED)
        assert 0.30 < iou < 0.35

    # --- semantic ---

    def test_semantic_below_threshold_is_missing(self):
        results = match_semantic(
            {"organ_1": self.PRED}, {"organ_1": self.GT},
            match_threshold=0.5,
        )
        assert len(results) == 1
        assert results[0]["pred_name"] is None
        assert results[0]["dice"] == pytest.approx(0.0)

    def test_semantic_above_threshold_is_detected(self):
        results = match_semantic(
            {"organ_1": self.PRED}, {"organ_1": self.GT},
            match_threshold=0.3,
        )
        assert len(results) == 1
        assert results[0]["pred_name"] is not None
        assert results[0]["dice"] > 0

    def test_semantic_missing_is_penalised_with_zeros(self):
        results = match_semantic(
            {"organ_1": self.PRED}, {"organ_1": self.GT},
            match_threshold=0.5,
        )
        assert results[0]["iou"] == pytest.approx(0.0)

    # --- greedy ---

    def test_greedy_below_threshold_is_missing(self):
        results = match_greedy(
            {"obj_001": self.PRED}, {"organ_1": self.GT},
            match_threshold=0.5,
        )
        assert len(results) == 1
        assert results[0]["pred_name"] is None
        assert results[0]["dice"] == pytest.approx(0.0)

    def test_greedy_above_threshold_is_detected(self):
        results = match_greedy(
            {"obj_001": self.PRED}, {"organ_1": self.GT},
            match_threshold=0.3,
        )
        assert len(results) == 1
        assert results[0]["pred_name"] is not None
        assert results[0]["dice"] > 0

    def test_default_threshold_is_0_5(self):
        # At default threshold=0.5, IoU≈0.324 is below threshold → missing
        results_default = match_greedy(
            {"obj_001": self.PRED}, {"organ_1": self.GT},
        )
        assert results_default[0]["pred_name"] is None

    def test_exact_threshold_boundary(self):
        # threshold equals IoU → should be detected (>= not >)
        iou = _iou(self.GT, self.PRED)
        results = match_greedy(
            {"obj_001": self.PRED}, {"organ_1": self.GT},
            match_threshold=float(iou),
        )
        assert results[0]["pred_name"] is not None


# ---------------------------------------------------------------------------
# T7.2 — AP curve consistency
# ---------------------------------------------------------------------------

class TestApCurveConsistency:

    def test_trivial_perfect_case(self):
        """1 GT, 1 pred, IoU=0.8 > threshold=0.5, score=0.9 → AP=1.0."""
        ap, curve = compute_average_precision(
            matched_ious=[0.8],
            matched_scores=[0.9],
            n_gt=1,
            iou_threshold=0.5,
            return_curve=True,
        )
        assert ap == pytest.approx(1.0)
        assert len(curve["recall"]) == 1
        assert curve["recall"][-1] == pytest.approx(1.0)

    def test_curve_recall_monotone(self):
        """Recall sequence must be non-decreasing."""
        ap, curve = compute_average_precision(
            matched_ious=[0.7, 0.3, 0.8, 0.6],
            matched_scores=[0.9, 0.8, 0.7, 0.6],
            n_gt=3,
            iou_threshold=0.5,
            return_curve=True,
        )
        r = curve["recall"]
        assert all(r[i] <= r[i + 1] for i in range(len(r) - 1))

    def test_ap_recomputed_from_curve_matches_scalar(self):
        """
        Recomputing AP from the raw curve using the Pascal VOC envelope must
        reproduce the scalar AP returned by the function.
        """
        ious   = [0.8, 0.4, 0.9, 0.2, 0.6]
        scores = [0.9, 0.85, 0.7, 0.65, 0.5]
        n_gt = 4

        ap, curve = compute_average_precision(
            matched_ious=ious,
            matched_scores=scores,
            n_gt=n_gt,
            iou_threshold=0.5,
            return_curve=True,
        )

        r = np.array(curve["recall"])
        p = np.array(curve["precision"])
        p_env = np.maximum.accumulate(p[::-1])[::-1]
        r_ext = np.concatenate([[0.0], r, [1.0]])
        p_ext = np.concatenate([[p_env[0]], p_env, [0.0]])
        ap_recomputed = float(np.sum((r_ext[1:] - r_ext[:-1]) * p_ext[1:]))

        assert ap == pytest.approx(ap_recomputed, abs=1e-9)

    def test_empty_predictions_returns_empty_curve(self):
        ap, curve = compute_average_precision(
            matched_ious=[], matched_scores=[], n_gt=3,
            iou_threshold=0.5, return_curve=True,
        )
        assert ap == pytest.approx(0.0)
        assert curve["recall"] == []
        assert curve["precision"] == []

    def test_zero_gt_returns_empty_curve(self):
        ap, curve = compute_average_precision(
            matched_ious=[0.8], matched_scores=[0.9], n_gt=0,
            iou_threshold=0.5, return_curve=True,
        )
        assert ap == pytest.approx(0.0)
        assert curve["recall"] == []

    def test_return_curve_false_gives_scalar(self):
        result = compute_average_precision(
            matched_ious=[0.8], matched_scores=[0.9],
            n_gt=1, iou_threshold=0.5,
        )
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# T7.3 — PR curve persisted in summary after evaluate()
# ---------------------------------------------------------------------------

class TestPrCurvePersisted:

    @pytest.fixture()
    def simple_eval_dirs(self, tmp_path: Path):
        """
        Single image 'img1' with one GT (heart_1) and one perfect prediction.
        Returns (gt_dir, pred_dir).
        """
        gt_dir   = tmp_path / "gt"
        pred_dir = tmp_path / "pred"
        img_gt   = gt_dir   / "img1"
        img_pred = pred_dir / "img1"
        img_gt.mkdir(parents=True)
        img_pred.mkdir(parents=True)

        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        img = Image.fromarray(mask)
        img.save(img_gt   / "heart_1.png")
        img.save(img_pred / "heart_1.png")

        return gt_dir, pred_dir

    def test_pr_curve_keys_present(self, simple_eval_dirs):
        gt_dir, pred_dir = simple_eval_dirs
        _, summary = evaluate(
            gt_dir, pred_dir, "semantic",
            iou_thresholds=[0.5],
            compute_map_metric=True,
        )
        assert "pr_curve_per_threshold" in summary
        curves = summary["pr_curve_per_threshold"]
        assert "0.50" in curves
        assert "0.75" in curves

    def test_pr_curve_has_nonempty_lists(self, simple_eval_dirs):
        gt_dir, pred_dir = simple_eval_dirs
        _, summary = evaluate(
            gt_dir, pred_dir, "semantic",
            iou_thresholds=[0.5],
            compute_map_metric=True,
        )
        c50 = summary["pr_curve_per_threshold"]["0.50"]
        assert isinstance(c50["recall"], list)
        assert len(c50["recall"]) > 0
        assert len(c50["precision"]) == len(c50["recall"])

    def test_match_threshold_in_summary(self, simple_eval_dirs):
        gt_dir, pred_dir = simple_eval_dirs
        _, summary = evaluate(
            gt_dir, pred_dir, "semantic",
            iou_thresholds=[0.5],
            match_threshold=0.75,
        )
        assert summary.get("match_threshold") == pytest.approx(0.75)

    def test_pct_real_scores_in_summary(self, simple_eval_dirs):
        gt_dir, pred_dir = simple_eval_dirs
        _, summary = evaluate(
            gt_dir, pred_dir, "semantic",
            iou_thresholds=[0.5],
        )
        assert "pct_real_scores" in summary.get("global", {})

    def test_existing_keys_not_removed(self, simple_eval_dirs):
        gt_dir, pred_dir = simple_eval_dirs
        _, summary = evaluate(
            gt_dir, pred_dir, "semantic",
            iou_thresholds=[0.5],
        )
        for key in ("n_images", "n_entries", "matching", "iou_thresholds",
                    "per_organ", "global"):
            assert key in summary, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# T7.4 — Legacy summary fallback smoke test for plot_pr_curve_grid
# ---------------------------------------------------------------------------

class TestLegacySummaryFallback:

    def test_no_crash_without_pr_curve(self, tmp_path: Path):
        """
        plot_pr_curve_grid must not raise when pr_curve_per_threshold is absent.
        It should fall back to the single-point mode and save a figure.
        """
        from project.evaluation.plots_extra import plot_pr_curve_grid

        legacy_summary = {
            "global": {
                "recall@0.5":    0.8,
                "precision@0.5": 0.7,
                "recall@0.75":   0.5,
                "precision@0.75": 0.6,
                "map_50":  0.65,
                "map_75":  0.45,
                "map":     0.55,
                "pct_real_scores": 0.0,
            },
            # pr_curve_per_threshold intentionally absent
        }
        data = {"v1": {"JSRT": {"unsup_hdbscan": legacy_summary}}}

        plot_pr_curve_grid(
            data,
            output_dir=tmp_path,
            configs=["unsup_hdbscan"],
            feature_versions=[("v1", "Baseline")],
            dataset_order=["JSRT"],
            dataset_labels={"JSRT": "JSRT"},
        )

        out_file = tmp_path / "pr_grid_unsup-hdbscan.png"
        assert out_file.exists(), "Expected pr_grid_unsup-hdbscan.png to be saved"

    def test_no_crash_with_none_pr_curve(self, tmp_path: Path):
        """pr_curve_per_threshold=None is treated as absent (not a key error)."""
        from project.evaluation.plots_extra import plot_pr_curve_grid

        summary_with_none = {
            "global": {
                "recall@0.5": 0.7, "precision@0.5": 0.6,
                "map_50": 0.5, "map_75": 0.3, "map": 0.4,
                "pct_real_scores": 2.0,
            },
            "pr_curve_per_threshold": None,
        }
        data = {"v1": {"JSRT": {"unsup_hdbscan": summary_with_none}}}

        plot_pr_curve_grid(
            data,
            output_dir=tmp_path,
            configs=["unsup_hdbscan"],
            feature_versions=[("v1", "Baseline")],
            dataset_order=["JSRT"],
            dataset_labels={"JSRT": "JSRT"},
        )

        assert (tmp_path / "pr_grid_unsup-hdbscan.png").exists()

    def test_no_data_cell_does_not_crash(self, tmp_path: Path):
        """A (version, dataset, config) combination absent from data → 'no data' cell."""
        from project.evaluation.plots_extra import plot_pr_curve_grid

        data = {"v1": {"JSRT": {}}}  # config not present

        plot_pr_curve_grid(
            data,
            output_dir=tmp_path,
            configs=["unsup_hdbscan"],
            feature_versions=[("v1", "Baseline")],
            dataset_order=["JSRT"],
            dataset_labels={"JSRT": "JSRT"},
        )

        assert (tmp_path / "pr_grid_unsup-hdbscan.png").exists()

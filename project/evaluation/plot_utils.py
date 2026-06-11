"""
Shared utilities for comparison and extra plotting scripts.

Extracted from compare_versions.py so that both compare_versions.py and
plots_extra.py can import from a single source of truth without either
script importing from the other.
"""

import numpy as np

from project.evaluation.display_names import EXPERIMENT_ORDER, get_display_name  # noqa: F401


# ---------------------------------------------------------------------------
# v0_baseline name equivalences
# ---------------------------------------------------------------------------

V0_FS_NAME_EQUIVALENCES: dict[str, str | list[str]] = {
    "fs_indep_baseline_1ref": [
        "fs_indep_1ref", "fs_indep_refine_1ref",
        "fs_iter_1ref",  "fs_iter_refine_1ref",
    ],
    "fs_indep_baseline_3ref": [
        "fs_indep_3ref", "fs_indep_refine_3ref",
        "fs_iter_3ref",  "fs_iter_refine_3ref",
    ],
    "fs_indep_baseline_5ref": [
        "fs_indep_5ref", "fs_indep_refine_5ref",
        "fs_iter_5ref",  "fs_iter_refine_5ref",
    ],
    "fs_indep_baseline_10ref": [
        "fs_indep_10ref", "fs_indep_refine_10ref",
        "fs_iter_10ref",  "fs_iter_refine_10ref",
    ],
}

V0_UNSUP_NAME_EQUIVALENCES: dict[str, str | list[str]] = {
    "unsup_baseline": [
        "unsup_kmeans", "unsup_hdbscan",
        "unsup_kmeans_refine", "unsup_hdbscan_refine",
    ],
}

ALL_V0_EQUIVALENCES: dict[str, str | list[str]] = {
    **V0_FS_NAME_EQUIVALENCES,
    **V0_UNSUP_NAME_EQUIVALENCES,
}


# ---------------------------------------------------------------------------
# Experiment group classification
# ---------------------------------------------------------------------------

_GROUP_PREFIXES = [
    ("unsup_",               "Unsupervised"),
    ("fs_indep_baseline_1",  "1-ref"),
    ("fs_indep_1",           "1-ref"),
    ("fs_indep_refine_1",    "1-ref"),
    ("fs_iter_1",            "1-ref"),
    ("fs_iter_refine_1",     "1-ref"),
    ("fs_indep_baseline_3",  "3-ref"),
    ("fs_indep_3",           "3-ref"),
    ("fs_indep_refine_3",    "3-ref"),
    ("fs_iter_3",            "3-ref"),
    ("fs_iter_refine_3",     "3-ref"),
    ("fs_indep_baseline_5",  "5-ref"),
    ("fs_indep_5",           "5-ref"),
    ("fs_indep_refine_5",    "5-ref"),
    ("fs_iter_5",            "5-ref"),
    ("fs_iter_refine_5",     "5-ref"),
    ("fs_indep_baseline_10", "10-ref"),
    ("fs_indep_10",          "10-ref"),
    ("fs_indep_refine_10",   "10-ref"),
    ("fs_iter_10",           "10-ref"),
    ("fs_iter_refine_10",    "10-ref"),
]


def _get_experiment_group(exp_name: str) -> str:
    for prefix, group in _GROUP_PREFIXES:
        if exp_name.startswith(prefix):
            return group
    return "Other"


def _sort_experiments(experiments: list[str]) -> list[str]:
    order_map = {name: i for i, name in enumerate(EXPERIMENT_ORDER)}
    return sorted(experiments, key=lambda e: (order_map.get(e, 999), e))


# ---------------------------------------------------------------------------
# Metric metadata
# ---------------------------------------------------------------------------

# Each entry: (display label, colormap, value range or None for data-driven).
METRIC_META: dict[str, tuple[str, str, tuple | None]] = {
    "dice_mean_with_missing":          ("Dice (incl. missing)",  "RdYlGn", (0.0, 1.0)),
    "dice_mean_detected_only":         ("Dice (detected)",       "RdYlGn", (0.0, 1.0)),
    "iou_mean_with_missing":           ("IoU (incl. missing)",   "RdYlGn", (0.0, 1.0)),
    "iou_mean_detected_only":          ("IoU (detected)",        "RdYlGn", (0.0, 1.0)),
    "hausdorff_95_mean_with_missing":  ("HD95 (incl. missing)",  "RdYlGn_r", None),
    "hausdorff_95_mean_detected_only": ("HD95 (detected)",       "RdYlGn_r", None),
    "assd_mean_with_missing":          ("ASSD (incl. missing)",  "RdYlGn_r", None),
    "assd_mean_detected_only":         ("ASSD (detected)",       "RdYlGn_r", None),
    "hausdorff_mean_with_missing":     ("Hausdorff (incl. miss)","RdYlGn_r", None),
    "hausdorff_mean_detected_only":    ("Hausdorff (detected)",  "RdYlGn_r", None),
    "recall@0.5":    ("Recall@0.5",    "RdYlGn", (0.0, 1.0)),
    "precision@0.5": ("Precision@0.5", "RdYlGn", (0.0, 1.0)),
    "f1@0.5":        ("F1@0.5",        "RdYlGn", (0.0, 1.0)),
    "recall@0.7":    ("Recall@0.7",    "RdYlGn", (0.0, 1.0)),
    "precision@0.7": ("Precision@0.7", "RdYlGn", (0.0, 1.0)),
    "f1@0.7":        ("F1@0.7",        "RdYlGn", (0.0, 1.0)),
    "recall@0.75":    ("Recall@0.75",    "RdYlGn", (0.0, 1.0)),
    "precision@0.75": ("Precision@0.75", "RdYlGn", (0.0, 1.0)),
    "f1@0.75":        ("F1@0.75",        "RdYlGn", (0.0, 1.0)),
    "map":           ("mAP@[.5:.95]",  "RdYlGn", (0.0, 1.0)),
    "map_50":        ("mAP@0.5",       "RdYlGn", (0.0, 1.0)),
    "map_75":        ("mAP@0.75",      "RdYlGn", (0.0, 1.0)),
}

DEFAULT_METRICS = [
    "dice_mean_with_missing",
    "dice_mean_detected_only",
    "recall@0.5",
    "f1@0.5",
]

_GLOBAL_ONLY_METRICS = {"map", "map_50", "map_75"}


def _is_global_only(metric: str) -> bool:
    return (
        metric in _GLOBAL_ONLY_METRICS
        or metric.startswith("precision@")
        or metric.startswith("f1@")
    )


def _is_distance_metric(metric: str) -> bool:
    return any(metric.startswith(p) for p in ("hausdorff", "assd"))


def _metric_label(metric: str) -> str:
    return METRIC_META.get(metric, (metric.replace("_", " "), None, None))[0]


def _metric_cmap(metric: str) -> str:
    return METRIC_META.get(metric, (None, "RdYlGn", None))[1]


# ---------------------------------------------------------------------------
# Backward-compatibility helpers for old vs new summary keys
# ---------------------------------------------------------------------------

_COMPAT_FALLBACK: dict[str, str] = {
    "dice_mean_with_missing":          "dice_mean",
    "dice_std_with_missing":           "dice_std",
    "iou_mean_with_missing":           "iou_mean",
    "iou_std_with_missing":            "iou_std",
    "hausdorff_95_mean_with_missing":  "hausdorff_95_mean",
    "hausdorff_95_std_with_missing":   "hausdorff_95_std",
    "assd_mean_with_missing":          "assd_mean",
    "assd_std_with_missing":           "assd_std",
    "hausdorff_mean_with_missing":     "hausdorff_mean",
    "hausdorff_std_with_missing":      "hausdorff_std",
}


def _compat_get_global(summary: dict, metric: str):
    """Fetch a global metric, falling back to legacy flat key if needed."""
    val = summary.get("global", {}).get(metric)
    if val is None:
        old = _COMPAT_FALLBACK.get(metric)
        if old:
            val = summary.get("global", {}).get(old)
    return val


def _compat_get_per_organ(summary: dict, organ: str, metric: str):
    """Fetch a per-organ metric, falling back to legacy flat key if needed."""
    stats = summary.get("per_organ", {}).get(organ, {})
    val = stats.get(metric)
    if val is None:
        old = _COMPAT_FALLBACK.get(metric)
        if old:
            val = stats.get(old)
    return val


# ---------------------------------------------------------------------------
# Heatmap helpers
# ---------------------------------------------------------------------------

def _heatmap_vrange(metric: str, matrix: np.ndarray) -> tuple[float, float]:
    """
    Return (vmin, vmax) for a heatmap.

    For [0,1]-bounded metrics: fixed (0, 1).
    For distance *_detected_only: data range with small margin.
    For distance *_with_missing: clip upper end at p95 so the worst-case
    diagonal does not flatten the colormap.
    """
    meta = METRIC_META.get(metric)
    if meta and meta[2] is not None:
        return meta[2]

    valid = matrix[~np.isnan(matrix)]
    if len(valid) == 0:
        return (0.0, 1.0)
    lo = float(valid.min())
    if metric.endswith("_with_missing") and _is_distance_metric(metric):
        hi = float(np.percentile(valid, 95))
    else:
        hi = float(valid.max())
    margin = max((hi - lo) * 0.05, 1e-6)
    return (max(0.0, lo - margin), hi + margin)


def _annotate_heatmap_cell(
    ax, j: int, i: int, value: float, vmin: float, vmax: float
) -> None:
    """Write a value annotation in a heatmap cell with auto contrast."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return
    span = vmax - vmin if vmax > vmin else 1.0
    norm = (value - vmin) / span
    text_color = "white" if norm < 0.20 or norm > 0.85 else "black"
    ax.text(
        j, i, f"{value:.3f}", ha="center", va="center",
        fontsize=9, color=text_color, fontweight="bold",
    )


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _parse_threshold_dict(d: dict) -> list[tuple[float, dict]]:
    """
    Parse a dict with string float keys (e.g. '0.6000000000000001') into a
    sorted list of (float_threshold, value_dict) pairs.
    """
    items = []
    for k, v in d.items():
        try:
            items.append((float(k), v))
        except (ValueError, TypeError):
            pass
    return sorted(items, key=lambda t: t[0])

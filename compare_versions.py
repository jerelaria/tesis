"""
compare_versions.py
-------------------
Generate cross-version, cross-dataset comparison plots.

Scans results/{version}/{dataset}/{experiment}/summary.json and produces
a set of figures designed for the paper.

Generated figures
-----------------
1.  metric_heatmap_{metric}_{dataset}.png
        Global heatmap experiment x version for each (metric, dataset).

2.  cross_dataset_heatmap_{metric}_{version}.png
        Global heatmap experiment x dataset for each (metric, version).

3.  per_organ_{metric}_{dataset}.png  [replaces old per-organ separate files]
        One figure per (metric, dataset) with 1 × (n_organs + 1) subplots.
        Shared colorbar. Experiment y-labels only on the leftmost panel.

3N. dice_gap_{dataset}.png  [new]
        Dice gap: solid bar = with_missing; translucent extension to
        detected_only.  Reveals the cost of non-detection per organ.

4N. recall_per_organ_{dataset}.png  [new]
        Grouped bars: x = experiments, groups = organs.  Direct view of
        how well each method detects each organ class.

4.  metric_story_{dataset}_{version}.png  [optional, --skip-metric-story]
        Grouped bars per experiment, multiple metrics, fixed version.

5.  version_story_{dataset}_{experiment}.png
        Grouped bars per version for one experiment group.

5N. refinement_impact_{dataset}.png  [new]
        Cross-version refinement impact (with vs without _refine suffix).

6.  delta_vs_{ref}_pipeline_{metric}_{dataset}.png
        Delta heatmaps vs reference version.

7.  organ_recovery_{dataset}_{version}.png
        Baseline-miss → pipeline-hit recovery counts.

8.  box_plot_dice_{dataset}_{version}_{slug}.png  [replaces per-organ files]
        One figure per (dataset, version, baseline-slug) with subplots = organs.

9N. pr_vs_threshold_{dataset}.png  [new]
        Precision / Recall / F1 curves vs IoU threshold.
        Primary source: detection_per_threshold.  Fallback: @0.5/@0.7 points.

10N. method_profile_{dataset}_{exp}.png  [new]
        Multi-panel profile for the best (or --highlight-experiment) method:
        mAP curve, Dice per organ (with_missing vs detected_only), HD95/ASSD.

11. all_versions_summary.csv
        Flat CSV with every (version, dataset, experiment, metric).

Usage
-----
    python compare_versions.py --results_dir results/ --output results/comparison/

    python compare_versions.py \\
        --versions v0_baseline v1_baseline v5_ext_emb_red \\
        --experiments unsup_hdbscan_refine fs_iter_refine_1ref \\
        --output results/comparison_focused/

    python compare_versions.py --skip-metric-story --skip-pr-curves

    python compare_versions.py --highlight-experiment fs_iter_refine_1ref
"""

import argparse
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict

from project.evaluation.display_names import (
    DISPLAY_NAMES, EXPERIMENT_ORDER, get_display_name,
)

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


# ---------------------------------------------------------------------------
# v0_baseline name equivalences for delta heatmaps
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

_ALL_BASELINE_EXPERIMENTS: set[str] = set(ALL_V0_EQUIVALENCES.keys())

# ---------------------------------------------------------------------------
# Metric metadata
# ---------------------------------------------------------------------------

# Each entry: (display label, colormap, value range or None for data-driven).
# Distance metrics use an inverted colormap (lower = better = greener).
# _with_missing distance metrics get a data-driven range clipped at p95 to
# avoid the worst-case diagonal (≈1448 px for JSRT 1024²) dominating colors.
METRIC_META: dict[str, tuple[str, str, tuple | None]] = {
    # Overlap — new split keys
    "dice_mean_with_missing":          ("Dice (incl. missing)",  "RdYlGn", (0.0, 1.0)),
    "dice_mean_detected_only":         ("Dice (detected)",       "RdYlGn", (0.0, 1.0)),
    "iou_mean_with_missing":           ("IoU (incl. missing)",   "RdYlGn", (0.0, 1.0)),
    "iou_mean_detected_only":          ("IoU (detected)",        "RdYlGn", (0.0, 1.0)),
    # Distance — with_missing: data-driven range; detected_only: data-driven
    "hausdorff_95_mean_with_missing":  ("HD95 (incl. missing)",  "RdYlGn_r", None),
    "hausdorff_95_mean_detected_only": ("HD95 (detected)",       "RdYlGn_r", None),
    "assd_mean_with_missing":          ("ASSD (incl. missing)",  "RdYlGn_r", None),
    "assd_mean_detected_only":         ("ASSD (detected)",       "RdYlGn_r", None),
    "hausdorff_mean_with_missing":     ("Hausdorff (incl. miss)","RdYlGn_r", None),
    "hausdorff_mean_detected_only":    ("Hausdorff (detected)",  "RdYlGn_r", None),
    # Detection
    "recall@0.5":    ("Recall@0.5",    "RdYlGn", (0.0, 1.0)),
    "precision@0.5": ("Precision@0.5", "RdYlGn", (0.0, 1.0)),
    "f1@0.5":        ("F1@0.5",        "RdYlGn", (0.0, 1.0)),
    "recall@0.7":    ("Recall@0.7",    "RdYlGn", (0.0, 1.0)),
    "precision@0.7": ("Precision@0.7", "RdYlGn", (0.0, 1.0)),
    "f1@0.7":        ("F1@0.7",        "RdYlGn", (0.0, 1.0)),
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

# Metrics with no meaningful per-organ breakdown.
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
# Backward-compatibility helpers
#
# New summaries: dice_mean_with_missing / dice_mean_detected_only
# Old summaries: dice_mean (flat)
# These helpers try the new key first; fall back to old for legacy files.
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
# Small helpers
# ---------------------------------------------------------------------------

def _sort_experiments(experiments: list[str]) -> list[str]:
    order_map = {name: i for i, name in enumerate(EXPERIMENT_ORDER)}
    return sorted(experiments, key=lambda e: (order_map.get(e, 999), e))


def _annotate_heatmap_cell(
    ax, j: int, i: int, value: float, vmin: float, vmax: float
):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return
    span = vmax - vmin if vmax > vmin else 1.0
    norm = (value - vmin) / span
    text_color = "white" if norm < 0.20 or norm > 0.85 else "black"
    ax.text(
        j, i, f"{value:.3f}", ha="center", va="center",
        fontsize=9, color=text_color, fontweight="bold",
    )


def _draw_group_separators(ax, experiments: list[str], axis: str = "y"):
    prev_group = None
    for idx, exp in enumerate(experiments):
        group = _get_experiment_group(exp)
        if prev_group is not None and group != prev_group:
            pos = idx - 0.5
            if axis == "y":
                ax.axhline(y=pos, color="grey", linewidth=1.2,
                           linestyle="--", alpha=0.5)
            else:
                ax.axvline(x=pos, color="grey", linewidth=1.2,
                           linestyle="--", alpha=0.5)
        prev_group = group


def _common_prefix(strings: list[str]) -> str:
    if not strings:
        return ""
    s_min, s_max = min(strings), max(strings)
    for i, ch in enumerate(s_min):
        if ch != s_max[i]:
            return s_min[:i]
    return s_min


def _short_dataset_labels(datasets: list[str]) -> list[str]:
    if len(datasets) < 2:
        return list(datasets)
    prefix = _common_prefix(datasets)
    last_sep = max(prefix.rfind("_"), prefix.rfind("-"), prefix.rfind("/"))
    if last_sep < 0:
        return list(datasets)
    cut = last_sep + 1
    return [d[cut:] for d in datasets]


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


# ---------------------------------------------------------------------------
# Heatmap colorbar range
# ---------------------------------------------------------------------------

def _heatmap_vrange(metric: str, matrix: np.ndarray) -> tuple[float, float]:
    """
    Return (vmin, vmax) for a heatmap.

    For [0,1]-bounded metrics: fixed (0, 1).
    For distance *_detected_only: data range with small margin.
    For distance *_with_missing: clip upper end at p95 so the worst-case
    diagonal (≈1448 px in JSRT 1024²) does not flatten the colormap.
    """
    meta = METRIC_META.get(metric)
    if meta and meta[2] is not None:
        return meta[2]

    valid = matrix[~np.isnan(matrix)]
    if len(valid) == 0:
        return (0.0, 1.0)
    lo = float(valid.min())
    if metric.endswith("_with_missing") and _is_distance_metric(metric):
        # Clip upper bound at 95th percentile to avoid diagonal domination
        hi = float(np.percentile(valid, 95))
    else:
        hi = float(valid.max())
    margin = max((hi - lo) * 0.05, 1e-6)
    return (max(0.0, lo - margin), hi + margin)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _extend_filter_with_baselines(experiments_filter: list[str]) -> list[str]:
    filter_set = set(experiments_filter)
    extras: list[str] = []
    for ref_exp, targets in ALL_V0_EQUIVALENCES.items():
        if ref_exp in filter_set:
            continue
        targets_list = targets if isinstance(targets, list) else [targets]
        if any(t in filter_set for t in targets_list):
            extras.append(ref_exp)
    if extras:
        print(f"  [auto-filter] Adding baseline experiments: {extras}")
    return list(filter_set | set(extras))


def load_all_results(
    results_dir: Path,
    versions: list[str] | None = None,
    experiments_filter: list[str] | None = None,
) -> dict[str, dict[str, dict[str, dict]]]:
    """
    Recursively load summary.json files.

    Returns {version: {dataset: {experiment: summary_dict}}}
    """
    if experiments_filter:
        experiments_filter = _extend_filter_with_baselines(experiments_filter)

    data: dict = {}

    if versions:
        version_dirs = [results_dir / v for v in versions]
    else:
        version_dirs = sorted([
            d for d in results_dir.iterdir()
            if d.is_dir()
            and not d.name.startswith(".")
            and d.name != "comparison"
        ])

    for version_dir in version_dirs:
        if not version_dir.is_dir():
            print(f"  [SKIP] Not a directory: {version_dir}")
            continue

        version = version_dir.name
        data[version] = {}

        for dataset_dir in sorted(version_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            data[version][dataset] = {}

            for exp_dir in sorted(dataset_dir.iterdir()):
                if experiments_filter and exp_dir.name not in experiments_filter:
                    continue
                summary_path = exp_dir / "summary.json"
                if not summary_path.exists():
                    continue
                with open(summary_path) as f:
                    data[version][dataset][exp_dir.name] = json.load(f)

    for version in data:
        for dataset in data[version]:
            n = len(data[version][dataset])
            print(f"  {version}/{dataset}: {n} experiments")

    return data


def _filter_datasets(data: dict, datasets_filter: list[str]) -> dict:
    keep = list(datasets_filter)
    keep_set = set(keep)
    pruned: dict = {}
    for v, ds_map in data.items():
        pruned[v] = {d: ds_map[d] for d in keep if d in ds_map}
        missing = keep_set - set(ds_map.keys())
        if missing:
            print(f"  [WARN] {v}: datasets not found and skipped: "
                  f"{sorted(missing)}")
    return pruned


def _load_metrics_csv(csv_path: Path) -> list[dict]:
    rows = []
    if not csv_path.exists():
        return rows
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Plot 1: Per-dataset global metric heatmap (experiment x version)
# ---------------------------------------------------------------------------

def plot_metric_heatmap_per_dataset(data: dict, output_dir: Path, metric: str):
    """
    For each dataset, heatmap with experiments on y-axis and versions on x-axis
    showing the absolute global value of `metric`.
    """
    versions = list(data.keys())
    if len(versions) < 2:
        print(f"  [SKIP] {metric}: need at least 2 versions")
        return

    label = _metric_label(metric)
    cmap = _metric_cmap(metric)

    all_datasets: set[str] = set()
    for v in data.values():
        all_datasets.update(v.keys())

    for dataset in sorted(all_datasets):
        all_exps: set[str] = set()
        for v in versions:
            all_exps.update(data.get(v, {}).get(dataset, {}).keys())
        experiments = _sort_experiments(list(all_exps))
        if not experiments:
            continue

        matrix = np.full((len(experiments), len(versions)), np.nan)
        for j, v in enumerate(versions):
            for i, exp in enumerate(experiments):
                summary = data.get(v, {}).get(dataset, {}).get(exp)
                if summary:
                    val = _compat_get_global(summary, metric)
                    if val is not None:
                        matrix[i, j] = val

        vmin, vmax = _heatmap_vrange(metric, matrix)
        fig, ax = plt.subplots(
            figsize=(max(6, len(versions) * 2),
                     max(6, len(experiments) * 0.55)),
        )
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        for i in range(len(experiments)):
            for j in range(len(versions)):
                _annotate_heatmap_cell(ax, j, i, matrix[i, j], vmin, vmax)

        if "v0_baseline" in versions:
            ref_col = versions.index("v0_baseline")
            ax.axvline(x=ref_col + 0.5, color="white",
                       linewidth=2.5, linestyle="--")

        _draw_group_separators(ax, experiments, axis="y")
        ax.set_xticks(range(len(versions)))
        ax.set_xticklabels(versions, fontsize=10, rotation=30, ha="right")
        ax.set_yticks(range(len(experiments)))
        ax.set_yticklabels(
            [get_display_name(e) for e in experiments], fontsize=9,
        )
        ax.set_title(f"Global {label} — {dataset}",
                     fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8, label=label)
        fig.tight_layout()

        out_path = (
            output_dir
            / f"metric_heatmap_{metric.replace('@', '_at_')}_{dataset}.png"
        )
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Cross-dataset heatmap (experiment x dataset, per version)
# ---------------------------------------------------------------------------

def plot_cross_dataset_heatmap(
    data: dict,
    output_dir: Path,
    metric: str,
    dataset_order: list[str] | None = None,
):
    versions = list(data.keys())
    if not versions:
        return

    all_datasets: set[str] = set()
    for v in versions:
        all_datasets.update(data.get(v, {}).keys())

    if dataset_order:
        datasets = [d for d in dataset_order if d in all_datasets]
    else:
        datasets = sorted(all_datasets)

    if len(datasets) < 2:
        print(f"  [SKIP] {metric}: cross-dataset plot needs >= 2 datasets "
              f"(got {len(datasets)})")
        return

    label = _metric_label(metric)
    cmap = _metric_cmap(metric)
    short_labels = _short_dataset_labels(datasets)

    for version in versions:
        all_exps: set[str] = set()
        for d in datasets:
            all_exps.update(data.get(version, {}).get(d, {}).keys())
        experiments = _sort_experiments(list(all_exps))
        if not experiments:
            continue

        matrix = np.full((len(experiments), len(datasets)), np.nan)
        for j, d in enumerate(datasets):
            for i, exp in enumerate(experiments):
                summary = data.get(version, {}).get(d, {}).get(exp)
                if summary:
                    val = _compat_get_global(summary, metric)
                    if val is not None:
                        matrix[i, j] = val

        vmin, vmax = _heatmap_vrange(metric, matrix)
        fig, ax = plt.subplots(
            figsize=(max(6, len(datasets) * 2.2 + 2),
                     max(6, len(experiments) * 0.55)),
        )
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        for i in range(len(experiments)):
            for j in range(len(datasets)):
                _annotate_heatmap_cell(ax, j, i, matrix[i, j], vmin, vmax)

        _draw_group_separators(ax, experiments, axis="y")
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(short_labels, fontsize=10, rotation=20, ha="right")
        ax.set_yticks(range(len(experiments)))
        ax.set_yticklabels(
            [get_display_name(e) for e in experiments], fontsize=9,
        )
        ax.set_title(f"Cross-dataset Global {label} — {version}",
                     fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8, label=label)
        fig.tight_layout()

        out_path = (
            output_dir
            / f"cross_dataset_heatmap_{metric.replace('@', '_at_')}_{version}.png"
        )
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: Per-organ consolidated heatmap (one figure per metric+dataset)
# ---------------------------------------------------------------------------

def plot_per_organ_consolidated(data: dict, output_dir: Path, metric: str):
    """
    For each dataset, one figure per metric with 1 × (n_organs + 1) subplots.

    Each subplot is a heatmap experiment × version for one organ (or global).
    A single shared colorbar is drawn on the right.  Experiment y-axis labels
    appear only on the leftmost panel.  Group separators and the v0_baseline
    column marker are applied in every panel.

    Skips global-only metrics.
    """
    if _is_global_only(metric):
        print(f"  [SKIP] per-organ consolidated for {metric}: global-only")
        return

    versions = list(data.keys())
    if not versions:
        return

    label = _metric_label(metric)
    cmap = _metric_cmap(metric)

    all_datasets: set[str] = set()
    for v in data.values():
        all_datasets.update(v.keys())

    for dataset in sorted(all_datasets):
        all_exps: set[str] = set()
        all_organs: set[str] = set()
        for v in versions:
            for exp_name, summary in data.get(v, {}).get(dataset, {}).items():
                all_exps.add(exp_name)
                all_organs.update(summary.get("per_organ", {}).keys())

        experiments = _sort_experiments(list(all_exps))
        organs = sorted(all_organs)

        if not experiments or not organs:
            continue

        panels = organs + [None]  # last panel = global
        n_panels = len(panels)
        n_exp = len(experiments)
        n_ver = len(versions)

        # Collect all values to derive a shared color range
        all_vals: list[float] = []
        matrices = []
        for panel in panels:
            mat = np.full((n_exp, n_ver), np.nan)
            for j, v in enumerate(versions):
                for i, exp in enumerate(experiments):
                    summary = data.get(v, {}).get(dataset, {}).get(exp)
                    if summary is None:
                        continue
                    if panel is None:
                        val = _compat_get_global(summary, metric)
                    else:
                        val = _compat_get_per_organ(summary, panel, metric)
                    if val is not None:
                        mat[i, j] = val
                        all_vals.append(val)
            matrices.append(mat)

        if not all_vals:
            continue

        # Shared vmin/vmax from combined matrix
        combined = np.array(all_vals)
        dummy = np.full((1, 1), np.nan)
        # Use the data-driven clip logic via _heatmap_vrange
        dummy_mat = np.array([[v] for v in all_vals]).reshape(-1, 1)
        vmin, vmax = _heatmap_vrange(metric, dummy_mat)

        panel_w = max(1.5, n_ver * 1.4)
        label_w = 2.5  # extra width for y-axis labels on leftmost panel
        fig_w = label_w + panel_w * n_panels + 0.8  # +0.8 for colorbar
        fig_h = max(4, n_exp * 0.5 + 1.5)

        fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
        # GridSpec: n_panels columns + 1 narrow column for colorbar
        gs = gridspec.GridSpec(
            1, n_panels + 1,
            figure=fig,
            width_ratios=[panel_w] * n_panels + [0.35],
            wspace=0.05,
        )

        axes = [fig.add_subplot(gs[0, k]) for k in range(n_panels)]
        cax = fig.add_subplot(gs[0, n_panels])

        im_last = None
        for k, (panel, mat) in enumerate(zip(panels, matrices)):
            ax = axes[k]
            if np.all(np.isnan(mat)):
                ax.set_visible(False)
                continue

            im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
            im_last = im

            for i in range(n_exp):
                for j in range(n_ver):
                    _annotate_heatmap_cell(ax, j, i, mat[i, j], vmin, vmax)

            if "v0_baseline" in versions:
                ref_col = versions.index("v0_baseline")
                ax.axvline(x=ref_col + 0.5, color="white",
                           linewidth=1.8, linestyle="--")

            _draw_group_separators(ax, experiments, axis="y")

            ax.set_xticks(range(n_ver))
            ax.set_xticklabels(versions, fontsize=7, rotation=35, ha="right")
            ax.set_yticks(range(n_exp))

            panel_title = (
                panel.replace("_", " ").title() if panel else "Global"
            )
            ax.set_title(panel_title, fontsize=9, fontweight="bold")

            if k == 0:
                ax.set_yticklabels(
                    [get_display_name(e) for e in experiments], fontsize=8,
                )
            else:
                ax.set_yticklabels([])

        if im_last is not None:
            plt.colorbar(im_last, cax=cax, label=label)

        fig.suptitle(
            f"{label} per organ — {dataset}",
            fontsize=12, fontweight="bold",
        )

        out_path = (
            output_dir
            / f"per_organ_{metric.replace('@', '_at_')}_{dataset}.png"
        )
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3-NEW: Dice gap (with_missing vs detected_only)
# ---------------------------------------------------------------------------

def plot_dice_gap(
    data: dict,
    output_dir: Path,
):
    """
    For each dataset, one figure showing the Dice gap between with_missing and
    detected_only variants, faceted by organ + global panel.

    Solid bar = dice_mean_with_missing.
    Translucent extension from with_missing to detected_only = the gap caused
    by missing detections (detected_only >= with_missing always).
    If detected_only is None (organ never detected), only the solid bar at 0
    is drawn — communicating "never found".
    """
    all_datasets: set[str] = set()
    for v in data.values():
        all_datasets.update(v.keys())

    for dataset in sorted(all_datasets):
        # Collect experiments and organs across all versions
        all_exps: set[str] = set()
        all_organs: set[str] = set()
        for v in data:
            for exp_name, summary in data.get(v, {}).get(dataset, {}).items():
                all_exps.add(exp_name)
                all_organs.update(summary.get("per_organ", {}).keys())

        experiments = _sort_experiments(list(all_exps))
        organs = sorted(all_organs)

        if not experiments or not organs:
            continue

        # Collect values: (with_missing, detected_only) per (exp, panel)
        # We aggregate across versions by taking the mean of latest version
        # that has data for that experiment.  Use the last version found.
        versions = list(data.keys())

        def _get_wm_do(exp: str, organ: str | None) -> tuple[float | None, float | None]:
            wm, do = None, None
            for v in reversed(versions):
                summary = data.get(v, {}).get(dataset, {}).get(exp)
                if summary is None:
                    continue
                if organ is None:
                    wm = _compat_get_global(summary, "dice_mean_with_missing")
                    do = summary.get("global", {}).get("dice_mean_detected_only")
                else:
                    wm = _compat_get_per_organ(summary, organ, "dice_mean_with_missing")
                    do = summary.get("per_organ", {}).get(organ, {}).get(
                        "dice_mean_detected_only"
                    )
                if wm is not None:
                    break
            return wm, do

        panels = organs + [None]
        n_panels = len(panels)
        n_exp = len(experiments)

        ncols = min(n_panels, 4)
        nrows = (n_panels + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * max(3.5, n_exp * 0.55 + 1), nrows * 4),
            squeeze=False,
        )

        colors = plt.get_cmap("tab10").colors
        x = np.arange(n_exp)

        for panel_idx, panel in enumerate(panels):
            row = panel_idx // ncols
            col = panel_idx % ncols
            ax = axes[row][col]

            wm_vals, do_vals = [], []
            for exp in experiments:
                wm, do = _get_wm_do(exp, panel)
                wm_vals.append(wm if wm is not None else 0.0)
                do_vals.append(do)  # may be None

            panel_title = panel.replace("_", " ").title() if panel else "Global"
            bar_colors = [
                colors[_sort_experiments(list(all_exps)).index(e) % len(colors)]
                for e in experiments
            ]

            # Solid bars: with_missing
            bars = ax.bar(
                x, wm_vals, color=bar_colors, alpha=0.85,
                edgecolor="white", linewidth=0.5,
            )

            # Translucent gap extension to detected_only
            for xi, (wm_v, do_v) in enumerate(zip(wm_vals, do_vals)):
                if do_v is not None and do_v > wm_v:
                    ax.bar(
                        xi, do_v - wm_v, bottom=wm_v,
                        color=bar_colors[xi], alpha=0.28,
                        edgecolor="none",
                    )
                    ax.text(
                        xi, do_v + 0.02, f"{do_v:.2f}",
                        ha="center", va="bottom", fontsize=6, color="#555",
                    )

            # Value labels on with_missing bars
            for xi, v in enumerate(wm_vals):
                if v > 0.01:
                    ax.text(
                        xi, v / 2, f"{v:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="white", fontweight="bold",
                    )

            _draw_group_separators(ax, experiments, axis="x")

            ax.set_title(panel_title, fontsize=10, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [get_display_name(e) for e in experiments],
                fontsize=7, rotation=35, ha="right",
            )
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("Dice", fontsize=9)
            ax.grid(axis="y", alpha=0.3)

        # Hide unused subplots
        for panel_idx in range(n_panels, nrows * ncols):
            row = panel_idx // ncols
            col = panel_idx % ncols
            axes[row][col].set_visible(False)

        fig.suptitle(
            f"Dice: incl. missing (solid) vs detected-only (+ translucent gap)"
            f" — {dataset}",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()
        out_path = output_dir / f"dice_gap_{dataset}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 4-NEW: Recall@0.5 per organ consolidated
# ---------------------------------------------------------------------------

def plot_recall_per_organ(data: dict, output_dir: Path):
    """
    For each dataset, grouped bars: x = experiments, one bar group per organ.
    Uses per_organ[organ]["recall@0.5"] (or n_covered@0.5 / n_total).

    Direct expression of detection success per organ class.
    """
    all_datasets: set[str] = set()
    for v in data.values():
        all_datasets.update(v.keys())

    # Aggregate over versions: use the last version that has data per experiment
    versions = list(data.keys())

    for dataset in sorted(all_datasets):
        all_exps: set[str] = set()
        all_organs: set[str] = set()
        for v in versions:
            for exp_name, summary in data.get(v, {}).get(dataset, {}).items():
                all_exps.add(exp_name)
                all_organs.update(summary.get("per_organ", {}).keys())

        experiments = _sort_experiments(list(all_exps))
        organs = sorted(all_organs)

        if not experiments or not organs:
            continue

        n_exp = len(experiments)
        n_org = len(organs)
        x = np.arange(n_exp)
        width = 0.8 / max(n_org, 1)

        organ_colors = {
            "heart":      "#D45B5B",
            "left_lung":  "#5B8DB8",
            "right_lung": "#6BBF6B",
            "lv_cavity":  "#9C27B0",
            "myocardium": "#FF9800",
            "lung":       "#5BA8D4",
        }

        fig, ax = plt.subplots(figsize=(max(8, n_exp * 1.2), 5))

        for oi, organ in enumerate(organs):
            recalls = []
            for exp in experiments:
                val = None
                for v in reversed(versions):
                    summary = data.get(v, {}).get(dataset, {}).get(exp)
                    if summary is None:
                        continue
                    od = summary.get("per_organ", {}).get(organ, {})
                    # Try recall@0.5 first, then n_covered / n_total
                    val = od.get("recall@0.5")
                    if val is None:
                        n_tot = od.get("n_total", od.get("count", 0))
                        n_cov = od.get("n_covered@0.5", 0)
                        if n_tot and n_tot > 0:
                            val = n_cov / n_tot
                    if val is not None:
                        break
                recalls.append(val if val is not None else 0.0)

            color = organ_colors.get(organ, f"C{oi}")
            offset = (oi - n_org / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, recalls, width,
                label=organ.replace("_", " ").title(),
                color=color, alpha=0.82,
                edgecolor="white", linewidth=0.5,
            )
            for bar, v in zip(bars, recalls):
                if v < 1.0 and v > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        v + 0.02, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6,
                    )

        _draw_group_separators(ax, experiments, axis="x")
        ax.set_ylabel("Recall @ IoU ≥ 0.5", fontsize=11)
        ax.set_title(f"Recall@0.5 per Organ — {dataset}", fontsize=13,
                     fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [get_display_name(e) for e in experiments],
            fontsize=8, rotation=35, ha="right",
        )
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        out_path = output_dir / f"recall_per_organ_{dataset}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 4: Multi-metric story plot (optional)
# ---------------------------------------------------------------------------

def plot_metric_story(data: dict, output_dir: Path, metrics: list[str]):
    """
    For each (dataset, version), grouped bars per experiment showing several
    metrics side by side. Distance metrics on a secondary y-axis.
    """
    bounded_colors = {
        "dice_mean_with_missing":  "#5B8DB8",
        "dice_mean_detected_only": "#4A7DA8",
        "iou_mean_with_missing":   "#7FA7C9",
        "iou_mean_detected_only":  "#6897B9",
        "recall@0.5":    "#6BBF6B",
        "precision@0.5": "#E8A838",
        "f1@0.5":        "#D45B5B",
        "recall@0.7":    "#4FA84F",
        "precision@0.7": "#C68E2C",
        "f1@0.7":        "#A84444",
    }
    distance_colors = {
        "hausdorff_95_mean_with_missing":  "#9C27B0",
        "hausdorff_95_mean_detected_only": "#7B1FA2",
        "assd_mean_with_missing":          "#E91E63",
        "assd_mean_detected_only":         "#C2185B",
        "hausdorff_mean_with_missing":     "#673AB7",
        "hausdorff_mean_detected_only":    "#512DA8",
    }

    bounded_metrics  = [m for m in metrics if m not in distance_colors]
    distance_metrics = [m for m in metrics if m in distance_colors]

    for version, datasets_dict in data.items():
        for dataset, exps in datasets_dict.items():
            if not exps:
                continue

            experiments = _sort_experiments(list(exps.keys()))
            x = np.arange(len(experiments))
            n_groups = len(bounded_metrics) + len(distance_metrics)
            width = 0.8 / max(n_groups, 1)
            has_two_axes = bool(distance_metrics)

            fig, ax = plt.subplots(figsize=(max(8, len(experiments) * 1.0), 6))
            ax2 = ax.twinx() if has_two_axes else None
            any_data = False

            for k, metric in enumerate(bounded_metrics):
                values = []
                for exp in experiments:
                    val = _compat_get_global(exps.get(exp, {}), metric)
                    values.append(val if val is not None else np.nan)
                if all(np.isnan(v) for v in values):
                    continue
                any_data = True
                offset = (k - n_groups / 2 + 0.5) * width
                color = bounded_colors.get(metric, f"C{k}")
                bars = ax.bar(
                    x + offset, values, width,
                    label=_metric_label(metric),
                    color=color, alpha=0.85,
                    edgecolor="white", linewidth=0.5,
                )
                for bar, v in zip(bars, values):
                    if not np.isnan(v):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=7,
                        )

            for dk, metric in enumerate(distance_metrics):
                values = []
                for exp in experiments:
                    val = _compat_get_global(exps.get(exp, {}), metric)
                    values.append(val if val is not None else np.nan)
                if all(np.isnan(v) for v in values):
                    continue
                any_data = True
                k = len(bounded_metrics) + dk
                offset = (k - n_groups / 2 + 0.5) * width
                color = distance_colors.get(metric, f"C{k}")
                bars = ax2.bar(
                    x + offset, values, width,
                    label=f"{_metric_label(metric)} (right axis)",
                    color=color, alpha=0.60,
                    edgecolor="white", linewidth=0.5, hatch="//",
                )

            if not any_data:
                plt.close(fig)
                continue

            _draw_group_separators(ax, experiments, axis="x")
            ax.set_ylim(0, 1.10)
            ax.set_ylabel("Metric value", fontsize=11)
            if ax2:
                ax2.set_ylabel("Distance (pixels)", fontsize=11, color="#9C27B0")
            ax.set_title(f"Metrics overview — {dataset} / {version}",
                         fontsize=13, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [get_display_name(e) for e in experiments],
                fontsize=8, rotation=35, ha="right",
            )
            handles, labels_leg = ax.get_legend_handles_labels()
            if ax2:
                h2, l2 = ax2.get_legend_handles_labels()
                handles += h2
                labels_leg += l2
            ax.legend(handles, labels_leg, fontsize=8, loc="upper left",
                      ncol=min(n_groups, 4))
            ax.grid(axis="y", alpha=0.3)
            ax.axhline(y=0, color="black", linewidth=0.6)
            fig.tight_layout()

            out_path = output_dir / f"metric_story_{dataset}_{version}.png"
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 5: Delta heatmap vs reference version
# ---------------------------------------------------------------------------

def _expand_equivalences(
    ref_exps: dict,
    name_equivalences: dict,
    metric: str,
) -> list[tuple[str, str, float]]:
    rows = []
    for ref_exp, summary in ref_exps.items():
        ref_val = _compat_get_global(summary, metric)
        if ref_val is None:
            continue
        mapping = name_equivalences.get(ref_exp, ref_exp)
        if isinstance(mapping, list):
            for equiv in mapping:
                rows.append((ref_exp, equiv, ref_val))
        else:
            rows.append((ref_exp, mapping, ref_val))
    rows.sort(key=lambda t: (
        EXPERIMENT_ORDER.index(t[1]) if t[1] in EXPERIMENT_ORDER else 999,
        t[1],
    ))
    return rows


def plot_delta_vs_baseline_heatmap(
    data: dict,
    output_dir: Path,
    reference_version: str,
    metric: str,
    name_equivalences: dict | None = None,
    output_suffix: str = "",
):
    if name_equivalences is None:
        name_equivalences = ALL_V0_EQUIVALENCES

    versions = [v for v in data.keys() if v != reference_version]
    ref_data = data.get(reference_version, {})

    if not ref_data:
        print(f"  [SKIP] reference version '{reference_version}' not in data")
        return

    label = _metric_label(metric)
    is_distance = _is_distance_metric(metric)

    all_datasets: set[str] = set(ref_data.keys())
    for v in versions:
        all_datasets.update(data.get(v, {}).keys())

    suffix = f"_{output_suffix}" if output_suffix else ""

    for dataset in sorted(all_datasets):
        ref_exps = ref_data.get(dataset, {})
        expanded = _expand_equivalences(ref_exps, name_equivalences, metric)

        if not expanded or not versions:
            continue

        n_rows = len(expanded)
        delta_matrix = np.full((n_rows, len(versions)), np.nan)
        abs_matrix   = np.full((n_rows, len(versions)), np.nan)

        for j, v in enumerate(versions):
            for i, (_, equiv_name, ref_val) in enumerate(expanded):
                comp_val = _compat_get_global(
                    data.get(v, {}).get(dataset, {}).get(equiv_name, {}),
                    metric,
                )
                if comp_val is not None:
                    delta_matrix[i, j] = comp_val - ref_val
                    abs_matrix[i, j]   = comp_val

        fig, axes = plt.subplots(
            1, 2,
            figsize=(
                max(10, len(versions) * 2.5 + 4),
                max(4, n_rows * 0.9 + 2),
            ),
            gridspec_kw={"width_ratios": [1, max(len(versions), 1)]},
        )

        ax_ref = axes[0]
        ref_vals = np.array([[rv] for _, _, rv in expanded])
        vmin_ref, vmax_ref = _heatmap_vrange(metric, ref_vals)
        ax_ref.imshow(ref_vals, cmap=_metric_cmap(metric),
                      aspect="auto", vmin=vmin_ref, vmax=vmax_ref)

        for i, (_, _, rv) in enumerate(expanded):
            _annotate_heatmap_cell(ax_ref, 0, i, rv, vmin_ref, vmax_ref)

        ax_ref.set_xticks([0])
        ax_ref.set_xticklabels([reference_version], fontsize=9,
                               rotation=30, ha="right")
        ax_ref.set_yticks(range(n_rows))

        y_labels = []
        for ref_exp, equiv_name, _ in expanded:
            display = get_display_name(equiv_name)
            if equiv_name != ref_exp:
                y_labels.append(
                    f"{display}\n(ref: {get_display_name(ref_exp)})"
                )
            else:
                y_labels.append(display)
        ax_ref.set_yticklabels(y_labels, fontsize=8)
        ax_ref.set_title(f"Reference\n({label} absolute)", fontsize=10)

        prev_group = None
        for idx, (_, equiv_name, _) in enumerate(expanded):
            group = _get_experiment_group(equiv_name)
            if prev_group is not None and group != prev_group:
                ax_ref.axhline(y=idx - 0.5, color="grey",
                               linewidth=1.2, linestyle="--", alpha=0.5)
            prev_group = group

        ax_delta = axes[1]
        max_abs = (
            np.nanmax(np.abs(delta_matrix))
            if not np.all(np.isnan(delta_matrix)) else 0.1
        )
        clim = max(max_abs, 0.05)
        # Distance: negative delta = improvement → RdBu_r (blue = improvement)
        delta_cmap = "RdBu_r" if is_distance else "RdBu"

        im_delta = ax_delta.imshow(
            delta_matrix, cmap=delta_cmap, aspect="auto",
            vmin=-clim, vmax=clim,
        )

        for i in range(n_rows):
            for j in range(len(versions)):
                delta = delta_matrix[i, j]
                abs_val = abs_matrix[i, j]
                if np.isnan(delta):
                    continue
                sign = "+" if delta >= 0 else ""
                lbl = f"{sign}{delta:.3f}\n({abs_val:.3f})"
                text_color = "white" if abs(delta) > clim * 0.6 else "black"
                ax_delta.text(j, i, lbl, ha="center", va="center",
                              fontsize=7.5, color=text_color, fontweight="bold")

        ax_delta.set_xticks(range(len(versions)))
        ax_delta.set_xticklabels(versions, fontsize=9, rotation=30, ha="right")
        ax_delta.set_yticks(range(n_rows))
        ax_delta.set_yticklabels([])

        prev_group = None
        for idx, (_, equiv_name, _) in enumerate(expanded):
            group = _get_experiment_group(equiv_name)
            if prev_group is not None and group != prev_group:
                ax_delta.axhline(y=idx - 0.5, color="grey",
                                 linewidth=1.2, linestyle="--", alpha=0.5)
            prev_group = group

        cbar = fig.colorbar(im_delta, ax=ax_delta, shrink=0.8)
        cbar.set_label(f"Δ {label} vs {reference_version}", fontsize=9)
        improvement = "blue = lower = improvement" if is_distance else "blue = improvement"
        ax_delta.set_title(
            f"Δ {label}  ({improvement}, red = regression)", fontsize=10,
        )

        fig.suptitle(
            f"{label} vs {reference_version}"
            f" ({output_suffix or 'pipeline'}) — {dataset}",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()

        out_path = output_dir / (
            f"delta_vs_{reference_version}{suffix}_"
            f"{metric.replace('@', '_at_')}_{dataset}.png"
        )
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 5-NEW: Refinement impact cross-version
# ---------------------------------------------------------------------------

def plot_refinement_impact_cross_version(data: dict, output_dir: Path):
    """
    For each dataset, detect (_base, _base_refine) experiment pairs by suffix.
    For each pair, grouped bars per version showing Dice (incl. missing) and
    F1@0.5 with and without refinement, with delta annotation.

    Supports the paper claim that the Refiner is the isolated contributor.
    """
    metrics_to_plot = ["dice_mean_with_missing", "f1@0.5"]
    m_labels = [_metric_label(m) for m in metrics_to_plot]
    colors_base   = ["#5B8DB8", "#6BBF6B"]
    colors_refine = ["#E8A838", "#D45B5B"]

    versions = list(data.keys())
    all_datasets: set[str] = set()
    for v in data.values():
        all_datasets.update(v.keys())

    for dataset in sorted(all_datasets):
        # Detect refine pairs: experiments where both name and name_refine exist
        # in ANY version.
        all_exps_seen: set[str] = set()
        for v in versions:
            all_exps_seen.update(data.get(v, {}).get(dataset, {}).keys())

        pairs = [
            (base, base + "_refine")
            for base in sorted(all_exps_seen)
            if base + "_refine" in all_exps_seen
        ]

        if not pairs:
            continue

        n_pairs = len(pairs)
        n_metrics = len(metrics_to_plot)
        n_versions = len(versions)

        fig, axes = plt.subplots(
            n_pairs, n_metrics,
            figsize=(n_metrics * max(4, n_versions * 1.2), n_pairs * 3.5),
            squeeze=False,
        )

        for pi, (base, refine) in enumerate(pairs):
            for mi, metric in enumerate(metrics_to_plot):
                ax = axes[pi][mi]

                base_vals, refine_vals = [], []
                present_vers = []
                for v in versions:
                    b_sum = data.get(v, {}).get(dataset, {}).get(base)
                    r_sum = data.get(v, {}).get(dataset, {}).get(refine)
                    if b_sum is None and r_sum is None:
                        continue
                    present_vers.append(v)
                    bv = _compat_get_global(b_sum or {}, metric)
                    rv = _compat_get_global(r_sum or {}, metric)
                    base_vals.append(bv if bv is not None else np.nan)
                    refine_vals.append(rv if rv is not None else np.nan)

                if not present_vers:
                    ax.set_visible(False)
                    continue

                x = np.arange(len(present_vers))
                width = 0.35
                ax.bar(x - width / 2, base_vals, width,
                       label="Without refine",
                       color=colors_base[mi], alpha=0.88,
                       edgecolor="white", linewidth=0.5)
                ax.bar(x + width / 2, refine_vals, width,
                       label="With refine",
                       color=colors_refine[mi], alpha=0.88,
                       edgecolor="white", linewidth=0.5)

                for i, (bv, rv) in enumerate(zip(base_vals, refine_vals)):
                    if np.isnan(bv) or np.isnan(rv):
                        continue
                    delta = rv - bv
                    sign = "+" if delta >= 0 else ""
                    ax.text(
                        i, max(bv, rv) + 0.03,
                        f"{sign}{delta:.3f}",
                        ha="center", fontsize=8, fontweight="bold",
                        color="#2E7D32" if delta >= 0 else "#C62828",
                    )

                ax.set_ylim(0, 1.18)
                ax.set_ylabel(m_labels[mi], fontsize=9)
                ax.set_xticks(x)
                ax.set_xticklabels(present_vers, fontsize=8, rotation=25, ha="right")
                ax.legend(fontsize=7)
                ax.grid(axis="y", alpha=0.3)
                if pi == 0:
                    ax.set_title(m_labels[mi], fontsize=10, fontweight="bold")

            axes[pi][0].set_ylabel(
                f"{get_display_name(base)}\nvs refine",
                fontsize=8,
            )

        fig.suptitle(
            f"Refinement Impact across Versions — {dataset}",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()
        out_path = output_dir / f"refinement_impact_{dataset}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 6: Cross-version story per experiment group
# ---------------------------------------------------------------------------

CROSS_VERSION_EXPERIMENTS = [
    [
        "unsup_baseline", "unsup_hdbscan", "unsup_hdbscan_refine",
        "unsup_kmeans", "unsup_kmeans_refine",
    ],
    ["fs_indep_baseline_1ref", "fs_iter_1ref",        "fs_indep_1ref"],
    ["fs_indep_baseline_1ref", "fs_iter_refine_1ref", "fs_indep_refine_1ref"],
    ["fs_indep_baseline_3ref", "fs_iter_3ref",        "fs_indep_3ref"],
    ["fs_indep_baseline_3ref", "fs_iter_refine_3ref", "fs_indep_refine_3ref"],
    ["fs_indep_baseline_5ref", "fs_iter_5ref",        "fs_indep_5ref"],
    ["fs_indep_baseline_5ref", "fs_iter_refine_5ref", "fs_indep_refine_5ref"],
    ["fs_indep_baseline_10ref","fs_iter_10ref",        "fs_indep_10ref"],
    ["fs_indep_baseline_10ref","fs_iter_refine_10ref", "fs_indep_refine_10ref"],
]


def plot_experiment_across_versions(
    data: dict,
    output_dir: Path,
    metrics: list[str],
    target_experiments: list = CROSS_VERSION_EXPERIMENTS,
):
    """
    For each (dataset, target_experiment_group), grouped bars per version.
    Bounded metrics on primary axis; distance metrics on secondary axis.
    """
    bounded_colors = {
        "dice_mean_with_missing":  "#5B8DB8",
        "dice_mean_detected_only": "#4A7DA8",
        "iou_mean_with_missing":   "#7FA7C9",
        "recall@0.5":    "#6BBF6B",
        "precision@0.5": "#E8A838",
        "f1@0.5":        "#D45B5B",
        "recall@0.7":    "#4FA84F",
        "precision@0.7": "#C68E2C",
        "f1@0.7":        "#A84444",
    }
    distance_colors = {
        "hausdorff_95_mean_with_missing":  "#9C27B0",
        "hausdorff_95_mean_detected_only": "#7B1FA2",
        "assd_mean_with_missing":          "#E91E63",
        "assd_mean_detected_only":         "#C2185B",
        "hausdorff_mean_with_missing":     "#673AB7",
        "hausdorff_mean_detected_only":    "#512DA8",
    }

    bounded_metrics  = [m for m in metrics if m not in distance_colors]
    distance_metrics = [m for m in metrics if m in distance_colors]
    n_groups = len(bounded_metrics) + len(distance_metrics)

    versions = list(data.keys())
    if not versions:
        return

    all_datasets: set[str] = set()
    for v in data.values():
        all_datasets.update(v.keys())

    for dataset in sorted(all_datasets):
        for exp_entry in target_experiments:
            candidates = [exp_entry] if isinstance(exp_entry, str) else exp_entry
            display_name = get_display_name(candidates[0])
            file_slug = candidates[0]
            primary_pipeline = candidates[1] if len(candidates) > 1 else candidates[0]
            if "refine" in primary_pipeline and "refine" not in file_slug:
                file_slug = file_slug + "_with_refine"

            values_by_metric: dict[str, list] = {m: [] for m in metrics}
            present_versions = []
            matched_experiments = []

            for v in versions:
                summary = None
                matched = None
                for candidate in candidates:
                    summary = data.get(v, {}).get(dataset, {}).get(candidate)
                    if summary is not None:
                        matched = candidate
                        break
                if summary is None:
                    continue
                present_versions.append(v)
                matched_experiments.append(matched)
                for m in metrics:
                    val = _compat_get_global(summary, m)
                    values_by_metric[m].append(val if val is not None else np.nan)

            if not present_versions:
                continue

            x = np.arange(len(present_versions))
            width = 0.8 / max(n_groups, 1)
            has_two_axes = bool(distance_metrics)

            fig, ax = plt.subplots(
                figsize=(max(8, len(present_versions) * 1.2), 6)
            )
            ax2 = ax.twinx() if has_two_axes else None
            all_bounded_values: list[float] = []
            any_data = False

            for k, metric in enumerate(bounded_metrics):
                values = values_by_metric[metric]
                if all(np.isnan(v) for v in values):
                    continue
                any_data = True
                all_bounded_values.extend(v for v in values if not np.isnan(v))
                offset = (k - n_groups / 2 + 0.5) * width
                color = bounded_colors.get(metric, f"C{k}")
                bars = ax.bar(
                    x + offset, values, width,
                    label=_metric_label(metric),
                    color=color, alpha=0.85,
                    edgecolor="white", linewidth=0.5,
                )
                for bar, v in zip(bars, values):
                    if not np.isnan(v):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=7,
                        )

            for dk, metric in enumerate(distance_metrics):
                values = values_by_metric[metric]
                if all(np.isnan(v) for v in values):
                    continue
                any_data = True
                k = len(bounded_metrics) + dk
                offset = (k - n_groups / 2 + 0.5) * width
                color = distance_colors.get(metric, f"C{k}")
                ax2.bar(
                    x + offset, values, width,
                    label=f"{_metric_label(metric)} (right axis)",
                    color=color, alpha=0.60,
                    edgecolor="white", linewidth=0.5, hatch="//",
                )

            if not any_data:
                plt.close(fig)
                continue

            if all_bounded_values:
                y_min_data = min(all_bounded_values)
                y_max = max(all_bounded_values)
                y_floor = max(0, y_min_data - 0.05) if y_min_data > 0.15 else 0
                y_ceil = min(y_max * 1.20, 1.10) if y_max < 0.9 else 1.10
            else:
                y_floor, y_ceil = 0, 1.10

            ax.set_ylim(y_floor, y_ceil)
            if ax2:
                ax2.set_ylabel("Distance (pixels)", fontsize=11, color="#9C27B0")

            if "v0_baseline" in present_versions:
                ref_idx = present_versions.index("v0_baseline")
                ax.axvline(x=ref_idx + 0.5, color="grey",
                           linewidth=1.5, linestyle="--", alpha=0.6)

            x_labels = []
            for v, matched in zip(present_versions, matched_experiments):
                md = get_display_name(matched) if matched else ""
                if matched and md != display_name:
                    x_labels.append(f"{v}\n({md})")
                else:
                    x_labels.append(v)

            refine_label = (
                " + Refine"
                if "refine" in primary_pipeline and "refine" not in candidates[0]
                else ""
            )
            ax.set_ylabel("Metric value", fontsize=11)
            ax.set_title(
                f"{display_name}{refine_label} across versions — {dataset}",
                fontsize=13, fontweight="bold",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=9, rotation=30, ha="right")

            handles, labels_leg = ax.get_legend_handles_labels()
            if ax2:
                h2, l2 = ax2.get_legend_handles_labels()
                handles += h2
                labels_leg += l2
            ax.legend(handles, labels_leg, fontsize=8, loc="upper left",
                      ncol=min(n_groups, 4))
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()

            out_path = output_dir / f"version_story_{dataset}_{file_slug}.png"
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 7: Organ recovery analysis
# ---------------------------------------------------------------------------

RECOVERY_PAIRS_UNSUP = {
    "unsup_baseline": [
        "unsup_kmeans", "unsup_kmeans_refine",
        "unsup_hdbscan", "unsup_hdbscan_refine",
    ],
}

RECOVERY_PAIRS_FS = {
    "fs_indep_baseline_1ref": [
        "fs_indep_1ref", "fs_indep_refine_1ref",
        "fs_iter_1ref",  "fs_iter_refine_1ref",
    ],
    "fs_indep_baseline_3ref": [
        "fs_indep_3ref", "fs_indep_refine_3ref",
        "fs_iter_3ref",  "fs_iter_refine_3ref",
    ],
}


def _build_organ_status(rows: list[dict]) -> dict[tuple[str, str], dict]:
    status = {}
    for row in rows:
        key = (row["image"], row["organ"])
        pred_name = row.get("pred_name", "")
        is_missing = (
            pred_name == "" or pred_name == "None" or pred_name is None
        )
        status[key] = {
            "pred_name": None if is_missing else pred_name,
            "dice": float(row.get("dice", 0)),
        }
    return status


def _analyze_recovery(baseline_rows: list[dict], pipeline_rows: list[dict]) -> dict:
    b_status = _build_organ_status(baseline_rows)
    p_status = _build_organ_status(pipeline_rows)
    all_keys = set(b_status.keys()) | set(p_status.keys())

    recovered, lost, both_hit, both_miss = [], [], [], []

    for key in sorted(all_keys):
        image, organ = key
        b = b_status.get(key, {"pred_name": None, "dice": 0})
        p = p_status.get(key, {"pred_name": None, "dice": 0})
        entry = {
            "image": image, "organ": organ,
            "baseline_dice": b["dice"], "pipeline_dice": p["dice"],
        }
        b_present = b["pred_name"] is not None
        p_present = p["pred_name"] is not None

        if not b_present and p_present:
            recovered.append(entry)
        elif b_present and not p_present:
            lost.append(entry)
        elif b_present and p_present:
            both_hit.append(entry)
        else:
            both_miss.append(entry)

    recovery_by_organ: dict[str, list] = defaultdict(list)
    for r in recovered:
        recovery_by_organ[r["organ"]].append(r)

    return {
        "n_recovered": len(recovered),
        "n_lost": len(lost),
        "n_both_hit": len(both_hit),
        "n_both_miss": len(both_miss),
        "total": len(all_keys),
        "recovered": recovered,
        "lost": lost,
        "recovery_by_organ": dict(recovery_by_organ),
        "recovered_dice_mean": (
            float(np.mean([r["pipeline_dice"] for r in recovered]))
            if recovered else None
        ),
    }


def plot_organ_recovery(
    results_dir: Path,
    data: dict,
    output_dir: Path,
    recovery_pairs: dict | None = None,
):
    if recovery_pairs is None:
        recovery_pairs = {**RECOVERY_PAIRS_UNSUP, **RECOVERY_PAIRS_FS}

    for version in data:
        for dataset in data[version]:
            for baseline_exp, pipeline_exps in recovery_pairs.items():
                baseline_csv = (
                    results_dir / version / dataset
                    / baseline_exp / "metrics.csv"
                )
                if not baseline_csv.exists():
                    continue

                baseline_rows = _load_metrics_csv(baseline_csv)
                if not baseline_rows:
                    continue

                all_results: dict[str, dict] = {}

                for pipe_exp in pipeline_exps:
                    pipe_csv = (
                        results_dir / version / dataset
                        / pipe_exp / "metrics.csv"
                    )
                    if not pipe_csv.exists():
                        continue
                    pipe_rows = _load_metrics_csv(pipe_csv)
                    result = _analyze_recovery(baseline_rows, pipe_rows)
                    all_results[pipe_exp] = result
                    print(
                        f"  {version}/{dataset}: {pipe_exp} vs {baseline_exp}"
                        f" — recovered={result['n_recovered']}"
                        f", lost={result['n_lost']}"
                    )

                if not all_results:
                    continue

                labels = list(all_results.keys())
                n_rec  = [r["n_recovered"] for r in all_results.values()]
                n_lost = [-r["n_lost"] for r in all_results.values()]

                x = np.arange(len(labels))
                width = 0.35
                fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 6))

                bars_r = ax.bar(x - width / 2, n_rec, width,
                                label="Recovered", color="#4CAF50", alpha=0.85)
                bars_l = ax.bar(x + width / 2, n_lost, width,
                                label="Lost", color="#F44336", alpha=0.85)

                for bar, v in zip(bars_r, n_rec):
                    if v > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.5,
                            str(v), ha="center", va="bottom",
                            fontsize=10, fontweight="bold",
                        )
                for bar, v in zip(bars_l, n_lost):
                    if v < 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            v - 0.5, str(abs(v)),
                            ha="center", va="top",
                            fontsize=10, fontweight="bold", color="#B71C1C",
                        )

                ax.axhline(y=0, color="black", linewidth=0.8)
                ax.set_ylabel("Organ instances", fontsize=12)
                ax.set_title(
                    f"Organ Recovery vs {get_display_name(baseline_exp)}"
                    f" — {dataset} / {version}",
                    fontsize=13, fontweight="bold",
                )
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [get_display_name(l) for l in labels],
                    fontsize=9, rotation=20, ha="right",
                )
                ax.legend(fontsize=10)
                ax.grid(axis="y", alpha=0.3)
                fig.tight_layout()

                slug = baseline_exp.replace("fs_indep_baseline_", "fs_")
                out_path = (
                    output_dir
                    / f"organ_recovery_{dataset}_{version}_{slug}.png"
                )
                fig.savefig(out_path, dpi=200)
                plt.close(fig)
                print(f"  Saved: {out_path}")

                # Per-organ breakdown for best pipeline
                best_exp = max(
                    all_results, key=lambda k: all_results[k]["n_recovered"]
                )
                best = all_results[best_exp]

                if best["recovery_by_organ"]:
                    organs = sorted(best["recovery_by_organ"].keys())
                    counts = [len(best["recovery_by_organ"][o]) for o in organs]
                    mean_dices = [
                        np.mean([
                            e["pipeline_dice"]
                            for e in best["recovery_by_organ"][o]
                        ])
                        for o in organs
                    ]

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    ax1.barh(organs, counts, color="#4CAF50", alpha=0.85)
                    for i, v in enumerate(counts):
                        ax1.text(v + 0.3, i, str(v), va="center",
                                 fontweight="bold")
                    ax1.set_xlabel("Recovered instances")
                    ax1.set_title("Recovery count")

                    ax2.barh(organs, mean_dices, color="#2196F3", alpha=0.85)
                    for i, v in enumerate(mean_dices):
                        ax2.text(v + 0.01, i, f"{v:.3f}", va="center",
                                 fontweight="bold")
                    ax2.set_xlabel("Mean Dice of recovered organs")
                    ax2.set_xlim(0, 1)
                    ax2.set_title("Recovery quality")

                    fig.suptitle(
                        f"Recovery by Organ — {get_display_name(best_exp)}"
                        f" — {dataset} / {version}",
                        fontsize=13, fontweight="bold",
                    )
                    fig.tight_layout()
                    out_path = (
                        output_dir
                        / f"organ_recovery_detail_{dataset}_{version}_{slug}.png"
                    )
                    fig.savefig(out_path, dpi=200)
                    plt.close(fig)
                    print(f"  Saved: {out_path}")

                # CSV per-case details
                csv_rows = []
                for pipe_exp, result in all_results.items():
                    for r in result["recovered"]:
                        csv_rows.append({
                            "version": version, "dataset": dataset,
                            "baseline": baseline_exp, "pipeline": pipe_exp,
                            "status": "recovered", **r,
                        })
                    for r in result["lost"]:
                        csv_rows.append({
                            "version": version, "dataset": dataset,
                            "baseline": baseline_exp, "pipeline": pipe_exp,
                            "status": "lost", **r,
                        })

                if csv_rows:
                    csv_path = (
                        output_dir
                        / f"organ_recovery_{dataset}_{version}_{slug}.csv"
                    )
                    fieldnames = [
                        "version", "dataset", "baseline", "pipeline",
                        "status", "image", "organ",
                        "baseline_dice", "pipeline_dice",
                    ]
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(
                            f, fieldnames=fieldnames, extrasaction="ignore",
                        )
                        writer.writeheader()
                        writer.writerows(csv_rows)
                    print(f"  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Plot 8: Box plots consolidated (one figure per dataset/version/pair)
# ---------------------------------------------------------------------------

BOX_PLOT_PAIRS_UNSUP = {
    "unsup_baseline": [
        "unsup_hdbscan", "unsup_hdbscan_refine",
        "unsup_kmeans_refine",
    ],
}

BOX_PLOT_PAIRS_FS = {
    "fs_indep_baseline_1ref": ["fs_iter_1ref", "fs_iter_refine_1ref"],
    "fs_indep_baseline_3ref": ["fs_iter_3ref", "fs_iter_refine_3ref"],
}


def plot_box_plots_consolidated(
    results_dir: Path,
    data: dict,
    output_dir: Path,
    box_plot_pairs: dict | None = None,
):
    """
    For each (version, dataset, baseline_exp), one figure with subplots = organs.

    Each organ subplot shows side-by-side box plots comparing baseline vs all
    pipeline variants.  The mean is marked with a diamond.
    """
    if box_plot_pairs is None:
        box_plot_pairs = {**BOX_PLOT_PAIRS_UNSUP, **BOX_PLOT_PAIRS_FS}

    for version in data:
        for dataset in data[version]:
            for baseline_exp, pipeline_exps in box_plot_pairs.items():
                baseline_csv = (
                    results_dir / version / dataset
                    / baseline_exp / "metrics.csv"
                )
                if not baseline_csv.exists():
                    continue

                baseline_rows = _load_metrics_csv(baseline_csv)
                if not baseline_rows:
                    continue

                dice_by_exp: dict[str, dict[str, list[float]]] = {}
                dice_by_exp[baseline_exp] = defaultdict(list)
                for row in baseline_rows:
                    dice_by_exp[baseline_exp][row["organ"]].append(
                        float(row.get("dice", 0))
                    )

                present_exps = [baseline_exp]
                for pipe_exp in pipeline_exps:
                    pipe_csv = (
                        results_dir / version / dataset
                        / pipe_exp / "metrics.csv"
                    )
                    if not pipe_csv.exists():
                        continue
                    pipe_rows = _load_metrics_csv(pipe_csv)
                    dice_by_exp[pipe_exp] = defaultdict(list)
                    for row in pipe_rows:
                        dice_by_exp[pipe_exp][row["organ"]].append(
                            float(row.get("dice", 0))
                        )
                    present_exps.append(pipe_exp)

                if len(present_exps) < 2:
                    continue

                all_organs: set[str] = set()
                for exp_data in dice_by_exp.values():
                    all_organs.update(exp_data.keys())
                organs = sorted(all_organs)

                if not organs:
                    continue

                n_organs = len(organs)
                ncols = min(n_organs, 3)
                nrows = (n_organs + ncols - 1) // ncols
                fig, axes = plt.subplots(
                    nrows, ncols,
                    figsize=(ncols * max(4, len(present_exps) * 1.0),
                             nrows * 4),
                    squeeze=False,
                )
                colors = [
                    "#BDBDBD", "#4CAF50", "#2196F3", "#FF9800", "#9C27B0"
                ]

                for oi, organ in enumerate(organs):
                    row_idx = oi // ncols
                    col_idx = oi % ncols
                    ax = axes[row_idx][col_idx]

                    box_data, box_labels, box_colors = [], [], []
                    for idx, exp in enumerate(present_exps):
                        values = dice_by_exp.get(exp, {}).get(organ, [])
                        if values:
                            box_data.append(values)
                            box_labels.append(get_display_name(exp))
                            box_colors.append(colors[idx % len(colors)])

                    if len(box_data) < 1:
                        ax.set_visible(False)
                        continue

                    bp = ax.boxplot(
                        box_data,
                        labels=box_labels,
                        patch_artist=True,
                        widths=0.6,
                        showmeans=True,
                        meanprops=dict(
                            marker="D", markerfacecolor="white",
                            markeredgecolor="black", markersize=6,
                        ),
                    )
                    for patch, color in zip(bp["boxes"], box_colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)

                    for i, vals in enumerate(box_data):
                        mean_v = np.mean(vals)
                        ax.text(
                            i + 1, mean_v + 0.02,
                            f"μ={mean_v:.3f}",
                            ha="center", fontsize=7, color="#333",
                        )

                    ax.set_title(
                        organ.replace("_", " ").title(),
                        fontsize=10, fontweight="bold",
                    )
                    ax.set_ylabel("Dice", fontsize=9)
                    ax.set_ylim(-0.05, 1.10)
                    ax.grid(axis="y", alpha=0.3)
                    ax.tick_params(axis="x", rotation=20, labelsize=8)

                # Hide unused subplots
                for oi in range(n_organs, nrows * ncols):
                    axes[oi // ncols][oi % ncols].set_visible(False)

                slug = baseline_exp.replace("fs_indep_baseline_", "fs_")
                fig.suptitle(
                    f"Dice Distribution — {dataset} / {version}"
                    f" (vs {get_display_name(baseline_exp)})",
                    fontsize=12, fontweight="bold",
                )
                fig.tight_layout()
                out_path = (
                    output_dir
                    / f"box_plot_dice_{dataset}_{version}_{slug}.png"
                )
                fig.savefig(out_path, dpi=200)
                plt.close(fig)
                print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 9-NEW: Precision / Recall / F1 vs IoU threshold
# ---------------------------------------------------------------------------

def plot_pr_vs_threshold(
    data: dict,
    output_dir: Path,
    experiments_subset: list[str] | None = None,
):
    """
    For each dataset, one figure with subplots per experiment.

    Primary source: detection_per_threshold (full curves).
    Fallback: scatter points at @0.5 / @0.7 from global + overlay of
    map_per_threshold as a degradation reference curve.

    experiments_subset filters to a specific list to avoid saturation.
    """
    versions = list(data.keys())
    all_datasets: set[str] = set()
    for v in data.values():
        all_datasets.update(v.keys())

    for dataset in sorted(all_datasets):
        all_exps: set[str] = set()
        for v in versions:
            all_exps.update(data.get(v, {}).get(dataset, {}).keys())
        experiments = _sort_experiments(list(all_exps))
        if experiments_subset:
            experiments = [e for e in experiments if e in experiments_subset]

        if not experiments:
            continue

        # Use the last version that has data per experiment
        def _best_summary(exp: str) -> dict | None:
            for v in reversed(versions):
                s = data.get(v, {}).get(dataset, {}).get(exp)
                if s is not None:
                    return s
            return None

        n_exp = len(experiments)
        ncols = min(n_exp, 3)
        nrows = (n_exp + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 4, nrows * 3.5),
            squeeze=False,
        )

        for ei, exp in enumerate(experiments):
            row = ei // ncols
            col = ei % ncols
            ax = axes[row][col]
            summary = _best_summary(exp)

            if summary is None:
                ax.set_visible(False)
                continue

            det_thr = summary.get("detection_per_threshold")
            map_thr = summary.get("map_per_threshold")

            if det_thr:
                thr_items = _parse_threshold_dict(det_thr)
                thrs  = [t for t, _ in thr_items]
                precs = [d.get("precision") for _, d in thr_items]
                recs  = [d.get("recall") for _, d in thr_items]
                f1s   = [d.get("f1") for _, d in thr_items]

                def _safe(lst):
                    return [v if v is not None else np.nan for v in lst]

                ax.plot(thrs, _safe(precs), "b-o", markersize=3,
                        label="Precision", linewidth=1.5)
                ax.plot(thrs, _safe(recs),  "g-s", markersize=3,
                        label="Recall", linewidth=1.5)
                ax.plot(thrs, _safe(f1s),   "r-^", markersize=3,
                        label="F1", linewidth=1.5)
            else:
                # Fallback: scatter points at @0.5 and @0.7
                g = summary.get("global", {})
                fallback_thrs = [0.5, 0.7]
                for thr in fallback_thrs:
                    p = g.get(f"precision@{thr}")
                    r = g.get(f"recall@{thr}")
                    f = g.get(f"f1@{thr}")
                    if p is not None:
                        ax.scatter([thr], [p], color="blue", marker="o", zorder=5)
                    if r is not None:
                        ax.scatter([thr], [r], color="green", marker="s", zorder=5)
                    if f is not None:
                        ax.scatter([thr], [f], color="red", marker="^", zorder=5)

                # Overlay map_per_threshold as AP reference
                if map_thr:
                    map_items = _parse_threshold_dict(map_thr)
                    mt = [t for t, _ in map_items]
                    mv = [v for _, v in map_items]
                    ax.plot(mt, mv, "k--", linewidth=1, alpha=0.6,
                            label="AP (mAP ref)")

                ax.text(
                    0.5, 0.03, "No det_per_thr — points only",
                    ha="center", transform=ax.transAxes,
                    fontsize=7, color="#888",
                )

            ax.set_xlim(0.45, 1.0)
            ax.set_ylim(-0.05, 1.10)
            ax.set_xlabel("IoU threshold", fontsize=8)
            ax.set_ylabel("Score", fontsize=8)
            ax.set_title(get_display_name(exp), fontsize=9, fontweight="bold")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(alpha=0.3)

        for ei in range(n_exp, nrows * ncols):
            axes[ei // ncols][ei % ncols].set_visible(False)

        fig.suptitle(
            f"Precision / Recall / F1 vs IoU Threshold — {dataset}",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()
        out_path = output_dir / f"pr_vs_threshold_{dataset}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 10-NEW: Method profile for the best (or highlighted) experiment
# ---------------------------------------------------------------------------

def _find_best_experiment(
    data: dict,
    dataset: str,
    metric: str = "f1@0.5",
) -> str | None:
    """Return the experiment name with the highest average value of metric."""
    best_exp, best_val = None, -1.0
    for v in data:
        for exp, summary in data.get(v, {}).get(dataset, {}).items():
            val = _compat_get_global(summary, metric)
            if val is not None and val > best_val:
                best_val = val
                best_exp = exp
    return best_exp


def plot_method_profile(
    data: dict,
    output_dir: Path,
    highlight_experiment: str | None = None,
):
    """
    For the highlighted experiment (or best f1@0.5 if not specified), produce
    a multi-panel profile figure per dataset:
      (a) mAP per threshold curve
      (b) Dice per organ: with_missing vs detected_only bars
      (c) HD95 and ASSD detected_only per organ

    Skips dataset/experiment combinations where data is absent.
    """
    versions = list(data.keys())
    all_datasets: set[str] = set()
    for v in data.values():
        all_datasets.update(v.keys())

    for dataset in sorted(all_datasets):
        if highlight_experiment is not None:
            exp = highlight_experiment
            # Check if it exists anywhere for this dataset
            summary = None
            for v in reversed(versions):
                s = data.get(v, {}).get(dataset, {}).get(exp)
                if s is not None:
                    summary = s
                    break
            if summary is None:
                print(
                    f"  [WARN] method_profile: {exp} not found in "
                    f"{dataset}, skipping"
                )
                continue
        else:
            exp = _find_best_experiment(data, dataset)
            if exp is None:
                continue
            summary = None
            for v in reversed(versions):
                s = data.get(v, {}).get(dataset, {}).get(exp)
                if s is not None:
                    summary = s
                    break
            if summary is None:
                continue

        g = summary.get("global", {})
        organs = sorted(summary.get("per_organ", {}).keys())

        fig = plt.figure(figsize=(14, 9))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        ax_map  = fig.add_subplot(gs[0, 0])
        ax_dice = fig.add_subplot(gs[0, 1])
        ax_hd   = fig.add_subplot(gs[1, :])

        # (a) mAP per threshold
        map_thr = summary.get("map_per_threshold")
        det_thr = summary.get("detection_per_threshold")

        if map_thr:
            items = _parse_threshold_dict(map_thr)
            thrs = [t for t, _ in items]
            aps  = [v for _, v in items]
            ax_map.plot(thrs, aps, "k-o", markersize=4, linewidth=2)
            ax_map.fill_between(thrs, aps, alpha=0.15, color="black")
            map_val = g.get("map")
            if map_val is not None:
                ax_map.axhline(map_val, color="red", linestyle="--",
                               linewidth=1, label=f"mAP={map_val:.3f}")
                ax_map.legend(fontsize=8)
        else:
            ax_map.text(0.5, 0.5, "No map_per_threshold data",
                        ha="center", va="center", transform=ax_map.transAxes,
                        fontsize=9, color="#888")

        if det_thr:
            items_d = _parse_threshold_dict(det_thr)
            td = [t for t, _ in items_d]
            f1s = [d.get("f1") for _, d in items_d]
            ax_map.plot(td, [v if v is not None else np.nan for v in f1s],
                        "r--^", markersize=3, linewidth=1, alpha=0.7,
                        label="F1")
            ax_map.legend(fontsize=8)

        ax_map.set_xlim(0.45, 1.0)
        ax_map.set_ylim(-0.02, 1.05)
        ax_map.set_xlabel("IoU threshold", fontsize=9)
        ax_map.set_ylabel("AP / F1", fontsize=9)
        ax_map.set_title("mAP per threshold", fontsize=10, fontweight="bold")
        ax_map.grid(alpha=0.3)

        # (b) Dice per organ: with_missing vs detected_only
        if organs:
            x = np.arange(len(organs))
            width = 0.35
            wm_vals = [
                _compat_get_per_organ(summary, o, "dice_mean_with_missing") or 0.0
                for o in organs
            ]
            do_vals = [
                summary.get("per_organ", {}).get(o, {}).get(
                    "dice_mean_detected_only"
                )
                for o in organs
            ]

            ax_dice.bar(x - width / 2, wm_vals, width,
                        label="Dice (incl. missing)",
                        color="#5B8DB8", alpha=0.85,
                        edgecolor="white", linewidth=0.5)
            do_plot = [v if v is not None else 0.0 for v in do_vals]
            ax_dice.bar(x + width / 2, do_plot, width,
                        label="Dice (detected)",
                        color="#E8A838", alpha=0.60,
                        edgecolor="white", linewidth=0.5)

            ax_dice.set_xticks(x)
            ax_dice.set_xticklabels(
                [o.replace("_", " ").title() for o in organs],
                fontsize=9, rotation=20, ha="right",
            )
            ax_dice.set_ylim(0, 1.15)
            ax_dice.set_ylabel("Dice", fontsize=9)
            ax_dice.set_title("Dice per organ", fontsize=10, fontweight="bold")
            ax_dice.legend(fontsize=8)
            ax_dice.grid(axis="y", alpha=0.3)

        # (c) HD95 and ASSD detected_only per organ
        if organs:
            x = np.arange(len(organs))
            width = 0.35
            hd95_vals = [
                summary.get("per_organ", {}).get(o, {}).get(
                    "hausdorff_95_mean_detected_only"
                )
                for o in organs
            ]
            assd_vals = [
                summary.get("per_organ", {}).get(o, {}).get(
                    "assd_mean_detected_only"
                )
                for o in organs
            ]

            has_hd95 = any(v is not None for v in hd95_vals)
            has_assd = any(v is not None for v in assd_vals)

            if has_hd95:
                ax_hd.bar(
                    x - width / 2,
                    [v if v is not None else 0.0 for v in hd95_vals],
                    width, label="HD95 (detected)",
                    color="#9C27B0", alpha=0.80,
                    edgecolor="white", linewidth=0.5,
                )
            if has_assd:
                ax_hd.bar(
                    x + width / 2,
                    [v if v is not None else 0.0 for v in assd_vals],
                    width, label="ASSD (detected)",
                    color="#E91E63", alpha=0.80,
                    edgecolor="white", linewidth=0.5,
                )

            ax_hd.set_xticks(x)
            ax_hd.set_xticklabels(
                [o.replace("_", " ").title() for o in organs],
                fontsize=9,
            )
            ax_hd.set_ylabel("Distance (pixels)", fontsize=9)
            ax_hd.set_title(
                "HD95 & ASSD per organ (detected only)",
                fontsize=10, fontweight="bold",
            )
            if has_hd95 or has_assd:
                ax_hd.legend(fontsize=8)
            ax_hd.grid(axis="y", alpha=0.3)

        fig.suptitle(
            f"Method Profile: {get_display_name(exp)} — {dataset}",
            fontsize=13, fontweight="bold",
        )

        out_path = output_dir / f"method_profile_{dataset}_{exp}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Flat CSV export
# ---------------------------------------------------------------------------

def save_full_csv(data: dict, output_dir: Path):
    """
    Flatten all results into a single CSV.

    Includes global quality metrics (both split variants), P/R/F1 at each
    available IoU threshold, per-organ metrics, and P/R/F1 from
    detection_per_threshold if present.
    """
    rows: list[dict] = []

    thresholds_seen: set[float] = set()
    for v in data.values():
        for ds in v.values():
            for s in ds.values():
                for thr in s.get("iou_thresholds", []):
                    thresholds_seen.add(float(thr))
    thresholds = sorted(thresholds_seen)

    for version in data:
        for dataset in data[version]:
            for exp, summary in data[version][dataset].items():
                g = summary.get("global", {})
                row: dict = {
                    "version":    version,
                    "dataset":    dataset,
                    "experiment": exp,
                    "n_images":   summary.get("n_images", 0),
                    "matching":   summary.get("matching", ""),
                    # New split keys
                    "dice_mean_with_missing":   g.get("dice_mean_with_missing"),
                    "dice_std_with_missing":    g.get("dice_std_with_missing"),
                    "dice_mean_detected_only":  g.get("dice_mean_detected_only"),
                    "dice_std_detected_only":   g.get("dice_std_detected_only"),
                    "iou_mean_with_missing":    g.get("iou_mean_with_missing"),
                    "iou_std_with_missing":     g.get("iou_std_with_missing"),
                    "iou_mean_detected_only":   g.get("iou_mean_detected_only"),
                    "iou_std_detected_only":    g.get("iou_std_detected_only"),
                    # Legacy flat keys (may be None for new-format summaries)
                    "dice_mean":  g.get("dice_mean"),
                    "dice_std":   g.get("dice_std"),
                    "iou_mean":   g.get("iou_mean"),
                    "iou_std":    g.get("iou_std"),
                    # Distance (prefer _with_missing; fall back to flat)
                    "hd95_mean_with_missing":
                        g.get("hausdorff_95_mean_with_missing")
                        or g.get("hausdorff_95_mean"),
                    "hd95_std_with_missing":
                        g.get("hausdorff_95_std_with_missing")
                        or g.get("hausdorff_95_std"),
                    "hd95_mean_detected_only":
                        g.get("hausdorff_95_mean_detected_only"),
                    "assd_mean_with_missing":
                        g.get("assd_mean_with_missing") or g.get("assd_mean"),
                    "assd_mean_detected_only":
                        g.get("assd_mean_detected_only"),
                    # Counts
                    "n_gt_total":   g.get("n_gt_total"),
                    "n_pred_total": g.get("n_pred_total"),
                    # Detection (mAP)
                    "map":      g.get("map"),
                    "map_50":   g.get("map_50"),
                    "map_75":   g.get("map_75"),
                }

                # P/R/F1 at each threshold
                for thr in thresholds:
                    row[f"recall@{thr}"]    = g.get(f"recall@{thr}")
                    row[f"precision@{thr}"] = g.get(f"precision@{thr}")
                    row[f"f1@{thr}"]        = g.get(f"f1@{thr}")
                    row[f"n_gt_covered@{thr}"]    = g.get(f"n_gt_covered@{thr}")
                    row[f"n_pred_relevant@{thr}"] = g.get(f"n_pred_relevant@{thr}")

                # detection_per_threshold: P/R/F1 at mAP grid thresholds
                det_thr = summary.get("detection_per_threshold")
                if det_thr:
                    for thr_f, d in _parse_threshold_dict(det_thr):
                        t_str = f"{thr_f:.2f}"
                        row[f"det_recall@{t_str}"]    = d.get("recall")
                        row[f"det_precision@{t_str}"] = d.get("precision")
                        row[f"det_f1@{t_str}"]        = d.get("f1")

                # Per-organ metrics
                for organ, stats in summary.get("per_organ", {}).items():
                    row[f"{organ}_dice_with_missing"] = _compat_get_per_organ(
                        summary, organ, "dice_mean_with_missing"
                    )
                    row[f"{organ}_dice_detected_only"] = stats.get(
                        "dice_mean_detected_only"
                    )
                    row[f"{organ}_missing"] = stats.get("missing", 0)
                    for thr in thresholds:
                        row[f"{organ}_recall@{thr}"] = stats.get(f"recall@{thr}")

                rows.append(row)

    if not rows:
        return

    fieldnames = [
        "version", "dataset", "experiment", "n_images", "matching",
        "dice_mean_with_missing", "dice_std_with_missing",
        "dice_mean_detected_only", "dice_std_detected_only",
        "iou_mean_with_missing", "iou_std_with_missing",
        "iou_mean_detected_only", "iou_std_detected_only",
        "dice_mean", "dice_std", "iou_mean", "iou_std",
        "hd95_mean_with_missing", "hd95_std_with_missing",
        "hd95_mean_detected_only",
        "assd_mean_with_missing", "assd_mean_detected_only",
        "n_gt_total", "n_pred_total",
        "map", "map_50", "map_75",
    ]
    for thr in thresholds:
        fieldnames += [
            f"recall@{thr}", f"precision@{thr}", f"f1@{thr}",
            f"n_gt_covered@{thr}", f"n_pred_relevant@{thr}",
        ]
    extra = sorted({k for row in rows for k in row if k not in fieldnames})
    fieldnames.extend(extra)

    csv_path = output_dir / "all_versions_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare pipeline results across versions and datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir", default="results",
        help="Root results dir (version/dataset/experiment/summary.json)",
    )
    parser.add_argument(
        "--versions", nargs="*",
        help="Versions to compare (default: all subdirs of results_dir)",
    )
    parser.add_argument(
        "--datasets", nargs="*",
        help="Datasets to include (default: all). Order is preserved for "
             "cross-dataset heatmap columns.",
    )
    parser.add_argument(
        "--experiments", nargs="*",
        help="Filter to these experiment names only (default: all). "
             "Baseline experiments whose pipeline targets are in the filter "
             "are added automatically.",
    )
    parser.add_argument(
        "--reference", default="v0_baseline",
        help="Reference version for delta heatmaps.",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=DEFAULT_METRICS,
        help="Metrics to plot (keys in summary.json[global]).",
    )
    parser.add_argument(
        "--output", default="results/comparison",
        help="Output directory for plots and CSV.",
    )
    parser.add_argument(
        "--highlight-experiment", default=None,
        help="Experiment to highlight in method profile (default: best f1@0.5).",
    )
    # Skip flags
    parser.add_argument("--skip-metric-story", action="store_true",
                        help="Skip metric story plots.")
    parser.add_argument("--skip-recovery", action="store_true",
                        help="Skip organ recovery analysis (needs metrics.csv).")
    parser.add_argument("--skip-box-plots", action="store_true",
                        help="Skip per-organ box plot distributions.")
    parser.add_argument("--skip-dice-gap", action="store_true",
                        help="Skip Dice gap (with_missing vs detected_only).")
    parser.add_argument("--skip-pr-curves", action="store_true",
                        help="Skip P/R vs threshold curves.")
    parser.add_argument("--skip-method-profile", action="store_true",
                        help="Skip method profile multi-panel figure.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    data = load_all_results(
        results_dir,
        versions=args.versions,
        experiments_filter=args.experiments,
    )

    if args.datasets:
        data = _filter_datasets(data, args.datasets)
        print(f"\nFiltered to {len(args.datasets)} dataset(s): "
              f"{list(args.datasets)}")

    total = sum(len(exps) for v in data.values() for exps in v.values())
    if total == 0:
        print("No results found.")
        return

    print(f"\nVersions: {list(data.keys())}")
    print(f"Total experiment results: {total}")
    print(f"Metrics to plot: {args.metrics}")
    print(f"Reference version: {args.reference}")

    # ---------- Tier 1: core paper figures ----------

    # [1/14] Delta heatmaps
    print("\n[1/14] Delta heatmaps vs reference version:")
    for metric in args.metrics:
        plot_delta_vs_baseline_heatmap(
            data, output_dir,
            reference_version=args.reference,
            metric=metric,
            name_equivalences=ALL_V0_EQUIVALENCES,
            output_suffix="pipeline",
        )

    # [2/14] Organ recovery
    if args.skip_recovery:
        print("\n[2/14] Organ recovery: SKIPPED (--skip-recovery)")
    else:
        print("\n[2/14] Organ recovery analysis:")
        plot_organ_recovery(results_dir, data, output_dir)

    # [3/14] Dice gap
    if args.skip_dice_gap:
        print("\n[3/14] Dice gap: SKIPPED (--skip-dice-gap)")
    else:
        print("\n[3/14] Dice gap (with_missing vs detected_only):")
        plot_dice_gap(data, output_dir)

    # [4/14] Recall per organ
    print("\n[4/14] Recall@0.5 per organ:")
    plot_recall_per_organ(data, output_dir)

    # [5/14] Refinement impact cross-version
    print("\n[5/14] Refinement impact cross-version:")
    plot_refinement_impact_cross_version(data, output_dir)

    # ---------- Tier 2: detailed analysis figures ----------

    # [6/14] Per-dataset global metric heatmaps
    print("\n[6/14] Per-dataset global metric heatmaps:")
    for metric in args.metrics:
        plot_metric_heatmap_per_dataset(data, output_dir, metric=metric)

    # [7/14] Cross-dataset metric heatmaps
    print("\n[7/14] Cross-dataset metric heatmaps:")
    for metric in args.metrics:
        plot_cross_dataset_heatmap(
            data, output_dir,
            metric=metric,
            dataset_order=args.datasets,
        )

    # [8/14] Per-organ consolidated heatmaps
    print("\n[8/14] Per-organ consolidated heatmaps:")
    for metric in args.metrics:
        plot_per_organ_consolidated(data, output_dir, metric=metric)

    # [9/14] P/R vs threshold curves
    if args.skip_pr_curves:
        print("\n[9/14] P/R curves: SKIPPED (--skip-pr-curves)")
    else:
        print("\n[9/14] Precision/Recall vs threshold curves:")
        plot_pr_vs_threshold(data, output_dir)

    # [10/14] Method profile
    if args.skip_method_profile:
        print("\n[10/14] Method profile: SKIPPED (--skip-method-profile)")
    else:
        print("\n[10/14] Method profile:")
        plot_method_profile(
            data, output_dir,
            highlight_experiment=args.highlight_experiment,
        )

    # [11/14] Box plots consolidated
    if args.skip_box_plots:
        print("\n[11/14] Box plots: SKIPPED (--skip-box-plots)")
    else:
        print("\n[11/14] Per-organ box plots (consolidated):")
        plot_box_plots_consolidated(results_dir, data, output_dir)

    # [12/14] Cross-version story
    print("\n[12/14] Cross-version story plots:")
    plot_experiment_across_versions(data, output_dir, metrics=args.metrics)

    # [13/14] Metric story (optional)
    if args.skip_metric_story:
        print("\n[13/14] Metric story: SKIPPED (--skip-metric-story)")
    else:
        print("\n[13/14] Multi-metric story plots:")
        plot_metric_story(data, output_dir, metrics=args.metrics)

    # [14/14] CSV
    print("\n[14/14] Flat CSV export:")
    save_full_csv(data, output_dir)

    print(f"\nAll comparison artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()

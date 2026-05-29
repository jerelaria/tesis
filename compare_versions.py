"""
compare_versions.py
-------------------
Generate cross-version, cross-dataset comparison plots.

Scans results/{version}/{dataset}/{experiment}/summary.json and produces
a small, opinionated set of figures designed for the paper:

1. metric_heatmap_{metric}_{dataset}.png
       For each (metric, dataset), a heatmap experiment x version showing
       the absolute metric value. One figure per dataset and metric.

2. cross_dataset_heatmap_{metric}_{version}.png
       For each (metric, version), a heatmap experiment x dataset showing
       the absolute metric value. Direct stratum / dataset comparison.
       Column order follows the order passed to --datasets when given.

3. per_organ_{metric}_{dataset}_{organ}.png
       For each (metric, dataset, organ), a separate heatmap figure with
       experiment x version. Each organ gets its own plot for readability.

4. metric_story_{dataset}_{version}.png  [optional, --skip-metric-story]
       For one (dataset, version), grouped bars per experiment showing
       Dice, Recall@0.5, Precision@0.5 and F1@0.5 side by side.

5. version_story_{dataset}_{experiment}.png
       For one (dataset, experiment_group), grouped bars per version
       comparing metrics across pipeline evolution.

6. delta_vs_{ref}_pipeline_{metric}_{dataset}.png
       Delta heatmaps vs the reference version (v0_baseline by default).
       Covers both unsup and fs baselines in a single call since both
       live inside v0_baseline.

7. organ_recovery_{dataset}_{version}.png
       Bar charts showing how many GT organs were recovered (baseline miss
       -> pipeline hit) vs lost, per pipeline variant and per organ.

8. box_plot_{metric}_{dataset}_{organ}.png
       Per-organ Dice distribution (box plots) comparing baseline vs
       pipeline variants, revealing whether improvements are uniform or
       driven by outlier images.

9. all_versions_summary.csv
       Flat CSV with every (version, dataset, experiment, metric).

Usage:
    python compare_versions.py --results_dir results/ --output results/comparison/

    # Specify versions explicitly:
    python compare_versions.py \\
        --versions v0_baseline v1_baseline v2_standardized \\
        --results_dir results/ \\
        --output results/comparison/

    # Filter to specific experiments:
    python compare_versions.py \\
        --versions v0_baseline v3_extended v5_ext_emb_red \\
        --experiments unsup_baseline unsup_hdbscan_refine \\
            fs_iter_1ref fs_iter_refine_1ref \\
        --output results/comparison_focused/

    # Filter to specific datasets, in a chosen column order (e.g. strata):
    python compare_versions.py \\
        --versions v3_extended \\
        --datasets SunnybrookNicoSent_basal SunnybrookNicoSent_mid \\
            SunnybrookNicoSent_apex \\
        --output results/comparison_strata/

    # Override reference version:
    python compare_versions.py --reference v0_baseline

    # Skip the metric story plots:
    python compare_versions.py --skip-metric-story
"""

import argparse
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from project.evaluation.display_names import (
    DISPLAY_NAMES, EXPERIMENT_ORDER, get_display_name,
)

# Classify each experiment into a group for visual separators.
# Groups are detected by prefix; order matches EXPERIMENT_ORDER.
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
    """Return the group label for an experiment name."""
    for prefix, group in _GROUP_PREFIXES:
        if exp_name.startswith(prefix):
            return group
    return "Other"


# ---------------------------------------------------------------------------
# Name equivalences for delta heatmaps
#
# v0_baseline contains BOTH the unsup baseline and the fs baselines.
# There is no separate v0_baseline_fs version.
# ---------------------------------------------------------------------------

# Few-shot: baseline -> pipeline variants (1-to-many).
V0_FS_NAME_EQUIVALENCES: dict[str, str | list[str]] = {
    "fs_indep_baseline_1ref": [
        "fs_indep_1ref",        "fs_indep_refine_1ref",
        "fs_iter_1ref",         "fs_iter_refine_1ref",
    ],
    "fs_indep_baseline_3ref": [
        "fs_indep_3ref",        "fs_indep_refine_3ref",
        "fs_iter_3ref",         "fs_iter_refine_3ref",
    ],
    "fs_indep_baseline_5ref": [
        "fs_indep_5ref",        "fs_indep_refine_5ref",
        "fs_iter_5ref",         "fs_iter_refine_5ref",
    ],
    "fs_indep_baseline_10ref": [
        "fs_indep_10ref",       "fs_indep_refine_10ref",
        "fs_iter_10ref",        "fs_iter_refine_10ref",
    ],
}

# Unsupervised: baseline -> pipeline variants.
V0_UNSUP_NAME_EQUIVALENCES: dict[str, str | list[str]] = {
    "unsup_baseline": [
        "unsup_kmeans",
        "unsup_hdbscan",
        "unsup_kmeans_refine",
        "unsup_hdbscan_refine",
    ],
}

# Combined: both unsup and fs baselines live in v0_baseline, so a single
# reference version covers everything.
ALL_V0_EQUIVALENCES: dict[str, str | list[str]] = {
    **V0_FS_NAME_EQUIVALENCES,
    **V0_UNSUP_NAME_EQUIVALENCES,
}

# Flat set of all baseline experiment names (used to auto-extend filters).
_ALL_BASELINE_EXPERIMENTS: set[str] = set(ALL_V0_EQUIVALENCES.keys())


# ---------------------------------------------------------------------------
# Metric metadata
# ---------------------------------------------------------------------------

# Each metric: (display label, colormap, value range for shared colorbar).
# Distance metrics use an inverted colormap (lower = better = greener).
METRIC_META = {
    "dice_mean":         ("Dice",           "RdYlGn", (0.0, 1.0)),
    "iou_mean":          ("IoU",            "RdYlGn", (0.0, 1.0)),
    "recall@0.5":        ("Recall@0.5",     "RdYlGn", (0.0, 1.0)),
    "precision@0.5":     ("Precision@0.5",  "RdYlGn", (0.0, 1.0)),
    "f1@0.5":            ("F1@0.5",         "RdYlGn", (0.0, 1.0)),
    "recall@0.7":        ("Recall@0.7",     "RdYlGn", (0.0, 1.0)),
    "precision@0.7":     ("Precision@0.7",  "RdYlGn", (0.0, 1.0)),
    "f1@0.7":            ("F1@0.7",         "RdYlGn", (0.0, 1.0)),
    "map":               ("mAP@[.5:.95]",   "RdYlGn", (0.0, 1.0)),
    "map_50":            ("mAP@0.5",        "RdYlGn", (0.0, 1.0)),
    "map_75":            ("mAP@0.75",       "RdYlGn", (0.0, 1.0)),
    # Distance metrics: lower is better -> inverted colormap, no fixed range.
    "hausdorff_mean":    ("Hausdorff",      "RdYlGn_r", None),
    "hausdorff_95_mean": ("HD95",           "RdYlGn_r", None),
    "assd_mean":         ("ASSD",           "RdYlGn_r", None),
}

# Default metrics produced by evaluate.py, covering all tracked aspects:
#   - overlap quality  : Dice, IoU
#   - detection @ 0.5  : Recall, Precision, F1
#   - detection @ 0.7  : Recall, Precision, F1  (stricter threshold)
#   - distance quality : HD95, ASSD
DEFAULT_METRICS = [
    "dice_mean",
    "iou_mean",
    "recall@0.5", "precision@0.5", "f1@0.5",
    "recall@0.7", "precision@0.7", "f1@0.7",
    "hausdorff_95_mean", "assd_mean",
]

# Metrics explicitly known to be global-only (no per-organ breakdown).
# precision@T and f1@T are also global-only because they require counting
# across the full prediction set, which cannot be decomposed per organ.
_GLOBAL_ONLY_METRICS = {"map", "map_50", "map_75"}


def _is_global_only(metric: str) -> bool:
    """Return True if the metric has no meaningful per-organ breakdown."""
    return (
        metric in _GLOBAL_ONLY_METRICS
        or metric.startswith("precision@")
        or metric.startswith("f1@")
    )


def _metric_label(metric: str) -> str:
    return METRIC_META.get(metric, (metric.replace("_", " "), None, None))[0]


def _metric_cmap(metric: str) -> str:
    return METRIC_META.get(metric, (None, "RdYlGn", None))[1]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _sort_experiments(experiments: list[str]) -> list[str]:
    order_map = {name: i for i, name in enumerate(EXPERIMENT_ORDER)}
    return sorted(experiments, key=lambda e: (order_map.get(e, 999), e))


def _get_global(summary: dict, metric: str):
    """Fetch a global metric from a summary dict, returning None if absent."""
    return summary.get("global", {}).get(metric)


def _get_per_organ(summary: dict, organ: str, metric: str):
    """Fetch a per-organ metric from a summary dict, returning None if absent."""
    return summary.get("per_organ", {}).get(organ, {}).get(metric)


def _annotate_heatmap_cell(ax, j: int, i: int, value: float, vmin: float, vmax: float):
    """Place a numeric label inside a heatmap cell with auto contrast."""
    if value is None or np.isnan(value):
        return
    span = vmax - vmin if vmax > vmin else 1.0
    norm = (value - vmin) / span
    text_color = "white" if norm < 0.20 or norm > 0.85 else "black"
    ax.text(j, i, f"{value:.3f}", ha="center", va="center",
            fontsize=9, color=text_color, fontweight="bold")


def _draw_group_separators(ax, experiments: list[str], axis: str = "y"):
    """
    Draw dashed lines between experiment groups on the given axis.

    axis='y' means experiments are on the y-axis (heatmap rows).
    axis='x' means experiments are on the x-axis (bar charts).
    """
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
    """Return the longest common prefix of a list of strings."""
    if not strings:
        return ""
    s_min, s_max = min(strings), max(strings)
    for i, ch in enumerate(s_min):
        if ch != s_max[i]:
            return s_min[:i]
    return s_min


def _short_dataset_labels(datasets: list[str]) -> list[str]:
    """
    Strip the longest shared prefix from dataset names for plot labels.

    Only strips when the prefix ends at a natural separator ('_', '-', '/'),
    so a list like ['SunnybrookNicoSent_basal', 'SunnybrookNicoSent_mid',
    'SunnybrookNicoSent_apex'] becomes ['basal', 'mid', 'apex']. If no clean
    separator is found, the original names are returned unchanged.
    """
    if len(datasets) < 2:
        return list(datasets)
    prefix = _common_prefix(datasets)
    last_sep = max(prefix.rfind("_"), prefix.rfind("-"), prefix.rfind("/"))
    if last_sep < 0:
        return list(datasets)
    cut = last_sep + 1
    return [d[cut:] for d in datasets]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _extend_filter_with_baselines(experiments_filter: list[str]) -> list[str]:
    """
    Given a user-supplied experiment filter, automatically add any baseline
    experiment whose pipeline targets appear in the filter.

    This ensures that, e.g., passing --experiments fs_iter_1ref also loads
    fs_indep_baseline_1ref from v0_baseline so that delta heatmaps and
    cross-version story plots have reference data for v0.

    Returns the extended list (original list is not mutated).
    """
    filter_set = set(experiments_filter)
    extras: list[str] = []
    for ref_exp, targets in ALL_V0_EQUIVALENCES.items():
        if ref_exp in filter_set:
            continue  # already included
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

    Parameters
    ----------
    results_dir : Path
        Root results directory.
    versions : list[str] | None
        If provided, only load these version subdirectories.
    experiments_filter : list[str] | None
        If provided, only load experiments whose directory name is in this
        list. Baseline experiments whose pipeline targets are in the filter
        are automatically added (see _extend_filter_with_baselines).

    Returns: {version: {dataset: {experiment: summary_dict}}}
    """
    # Auto-extend filter so baselines are never accidentally excluded.
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


def _filter_datasets(
    data: dict,
    datasets_filter: list[str],
) -> dict:
    """
    Prune the loaded data dict to keep only the requested datasets,
    preserving the user-supplied order. Datasets not present in a given
    version are silently skipped for that version, but warned about.
    """
    keep = list(datasets_filter)
    keep_set = set(keep)
    pruned: dict = {}
    for v, ds_map in data.items():
        # Preserve user-supplied order
        pruned[v] = {d: ds_map[d] for d in keep if d in ds_map}
        missing = keep_set - set(ds_map.keys())
        if missing:
            print(f"  [WARN] {v}: datasets not found and skipped: "
                  f"{sorted(missing)}")
    return pruned


def _load_metrics_csv(csv_path: Path) -> list[dict]:
    """Load a metrics.csv file into a list of row dicts."""
    rows = []
    if not csv_path.exists():
        return rows
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _find_experiment_dir(
    results_dir: Path,
    version: str,
    dataset: str,
    experiment: str,
) -> Path:
    """Resolve the full path for a specific experiment directory."""
    return results_dir / version / dataset / experiment


# ---------------------------------------------------------------------------
# Heatmap colorbar helpers
# ---------------------------------------------------------------------------

def _heatmap_vrange(metric: str, matrix: np.ndarray) -> tuple[float, float]:
    """
    Return (vmin, vmax) for a heatmap.

    For [0, 1]-bounded metrics the range is fixed at (0, 1).
    For distance metrics (no fixed range in METRIC_META) the range is
    derived from the data so the colormap is always informative.
    """
    meta = METRIC_META.get(metric)
    if meta and meta[2] is not None:
        return meta[2]  # e.g. (0.0, 1.0)

    # Distance metrics: use data range with a small margin.
    valid = matrix[~np.isnan(matrix)]
    if len(valid) == 0:
        return (0.0, 1.0)
    lo, hi = float(valid.min()), float(valid.max())
    margin = max((hi - lo) * 0.05, 1e-6)
    return (max(0.0, lo - margin), hi + margin)


# ---------------------------------------------------------------------------
# Plot 1: Per-dataset heatmap of one global metric (experiment x version)
# ---------------------------------------------------------------------------

def plot_metric_heatmap_per_dataset(
    data: dict,
    output_dir: Path,
    metric: str,
):
    """
    For each dataset, produce a heatmap with experiments on y-axis and
    versions on x-axis, showing the absolute value of `metric` (global).

    Includes group separators between unsup / 1-ref / 3-ref / etc.
    Distance metrics (HD95, ASSD) use data-driven color range and an
    inverted colormap so that lower values appear greener.
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
                    val = _get_global(summary, metric)
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

        # Reference version separator
        if "v0_baseline" in versions:
            ref_col = versions.index("v0_baseline")
            ax.axvline(x=ref_col + 0.5, color="white",
                       linewidth=2.5, linestyle="--")

        # Group separators between unsup / 1-ref / 3-ref / ...
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

        out_path = output_dir / f"metric_heatmap_{metric.replace('@', '_at_')}_{dataset}.png"
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
    """
    For each (version, metric) produce a heatmap with experiments on the
    y-axis and datasets on the x-axis. Useful for stratum comparisons:
    one figure per version lets you read off how the same pipeline behaves
    across datasets (e.g. basal / mid / apex) at a glance.

    If `dataset_order` is provided, columns follow that order; otherwise
    alphabetical. Skipped when fewer than 2 datasets are loaded.

    Output filename: cross_dataset_heatmap_{metric}_{version}.png
    """
    versions = list(data.keys())
    if not versions:
        return

    # Collect all datasets across all loaded versions
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
        # Collect experiments present in any of the selected datasets
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
                    val = _get_global(summary, metric)
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

        # Group separators between unsup / 1-ref / 3-ref / ...
        _draw_group_separators(ax, experiments, axis="y")

        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(short_labels, fontsize=10, rotation=20, ha="right")
        ax.set_yticks(range(len(experiments)))
        ax.set_yticklabels(
            [get_display_name(e) for e in experiments], fontsize=9,
        )
        ax.set_title(
            f"Cross-dataset Global {label} — {version}",
            fontsize=14, fontweight="bold",
        )
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
# Plot 3: Per-organ heatmap — one separate figure per organ
# ---------------------------------------------------------------------------

def _plot_single_organ_heatmap(
    versions: list[str],
    experiments: list[str],
    data: dict,
    dataset: str,
    organ: str | None,
    metric: str,
    output_dir: Path,
):
    """
    Render a single heatmap (experiment x version) for one organ or for
    the global metric. Produces one standalone figure with full y-axis labels.

    organ=None means global metric.
    """
    label = _metric_label(metric)
    cmap = _metric_cmap(metric)
    panel_name = organ if organ else "global"
    panel_title = organ.replace("_", " ").title() if organ else "Global"

    matrix = np.full((len(experiments), len(versions)), np.nan)
    for j, v in enumerate(versions):
        for i, exp in enumerate(experiments):
            summary = data.get(v, {}).get(dataset, {}).get(exp)
            if summary:
                if organ:
                    val = _get_per_organ(summary, organ, metric)
                else:
                    val = _get_global(summary, metric)
                if val is not None:
                    matrix[i, j] = val

    # Skip entirely empty panels
    if np.all(np.isnan(matrix)):
        return

    vmin, vmax = _heatmap_vrange(metric, matrix)

    fig, ax = plt.subplots(
        figsize=(max(5, len(versions) * 1.8),
                 max(5, len(experiments) * 0.55)),
    )
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    for i in range(len(experiments)):
        for j in range(len(versions)):
            _annotate_heatmap_cell(ax, j, i, matrix[i, j], vmin, vmax)

    # Reference version separator
    if "v0_baseline" in versions:
        ref_col = versions.index("v0_baseline")
        ax.axvline(x=ref_col + 0.5, color="white",
                   linewidth=2.0, linestyle="--")

    # Group separators
    _draw_group_separators(ax, experiments, axis="y")

    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels(versions, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(len(experiments)))
    ax.set_yticklabels(
        [get_display_name(e) for e in experiments], fontsize=9,
    )
    ax.set_title(f"{label} — {panel_title} — {dataset}",
                 fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label=label)
    fig.tight_layout()

    out_path = (
        output_dir
        / f"per_organ_{metric.replace('@', '_at_')}_{dataset}_{panel_name}.png"
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_per_organ_heatmap(
    data: dict,
    output_dir: Path,
    metric: str,
):
    """
    For each dataset, produce one standalone heatmap figure per organ plus
    one for global. Each figure has full y-axis labels for readability.

    Metrics that are inherently global-only (precision@T, f1@T, mAP) are
    skipped here; per-organ panels that come back empty are also skipped
    silently inside _plot_single_organ_heatmap.
    """
    versions = list(data.keys())
    if len(versions) < 1:
        return

    if _is_global_only(metric):
        print(f"  [SKIP] per-organ heatmap for {metric}: global-only metric")
        return

    all_datasets: set[str] = set()
    for v in data.values():
        all_datasets.update(v.keys())

    for dataset in sorted(all_datasets):
        # Collect all experiments and organs across versions
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

        # One figure per organ
        for organ in organs:
            _plot_single_organ_heatmap(
                versions, experiments, data, dataset, organ, metric, output_dir,
            )

        # Global figure
        _plot_single_organ_heatmap(
            versions, experiments, data, dataset, None, metric, output_dir,
        )


# ---------------------------------------------------------------------------
# Plot 4: Multi-metric story plot (optional, fixed version, compare exps)
# ---------------------------------------------------------------------------

def plot_metric_story(
    data: dict,
    output_dir: Path,
    metrics: list[str],
):
    """
    For each (dataset, version), one figure with grouped bars per
    experiment showing several metrics side by side.

    This is complementary to version_story: it fixes a version and
    compares experiments. Can be skipped with --skip-metric-story.

    Distance metrics (HD95, ASSD) are plotted on a secondary y-axis so
    they do not visually compete with [0, 1]-bounded metrics.
    """
    # Colors for [0, 1]-bounded metrics
    bounded_colors = {
        "dice_mean":      "#5B8DB8",
        "iou_mean":       "#7FA7C9",
        "recall@0.5":     "#6BBF6B",
        "precision@0.5":  "#E8A838",
        "f1@0.5":         "#D45B5B",
        "recall@0.7":     "#4FA84F",
        "precision@0.7":  "#C68E2C",
        "f1@0.7":         "#A84444",
    }
    # Colors for distance metrics (plotted on secondary axis)
    distance_colors = {
        "hausdorff_95_mean": "#9C27B0",
        "assd_mean":         "#E91E63",
        "hausdorff_mean":    "#673AB7",
    }

    bounded_metrics  = [m for m in metrics if m not in distance_colors]
    distance_metrics = [m for m in metrics if m in distance_colors]

    for version, datasets_dict in data.items():
        for dataset, exps in datasets_dict.items():
            if not exps:
                continue

            experiments = _sort_experiments(list(exps.keys()))
            x = np.arange(len(experiments))

            # Total bar groups: bounded + distance (on secondary axis)
            n_groups = len(bounded_metrics) + len(distance_metrics)
            width = 0.8 / max(n_groups, 1)

            has_two_axes = bool(distance_metrics)
            fig, ax = plt.subplots(figsize=(max(8, len(experiments) * 1.0), 6))
            ax2 = ax.twinx() if has_two_axes else None

            any_data = False

            # --- Bounded metrics on primary axis ---
            for k, metric in enumerate(bounded_metrics):
                values = []
                for exp in experiments:
                    val = _get_global(exps.get(exp, {}), metric)
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
                    if np.isnan(v):
                        continue
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7,
                    )

            # --- Distance metrics on secondary axis ---
            for dk, metric in enumerate(distance_metrics):
                values = []
                for exp in experiments:
                    val = _get_global(exps.get(exp, {}), metric)
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
                    edgecolor="white", linewidth=0.5,
                    hatch="//",
                )
                for bar, v in zip(bars, values):
                    if np.isnan(v):
                        continue
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{v:.1f}",
                        ha="center", va="bottom", fontsize=6,
                    )

            if not any_data:
                plt.close(fig)
                continue

            # Group separators
            _draw_group_separators(ax, experiments, axis="x")

            ax.set_ylim(0, 1.10)
            ax.set_ylabel("Metric value", fontsize=11)
            if ax2:
                ax2.set_ylabel("Distance (pixels)", fontsize=11, color="#9C27B0")

            ax.set_title(
                f"Metrics overview — {dataset} / {version}",
                fontsize=13, fontweight="bold",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                [get_display_name(e) for e in experiments],
                fontsize=8, rotation=35, ha="right",
            )

            # Merge legends from both axes
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
    name_equivalences: dict[str, str | list[str]],
    metric: str,
) -> list[tuple[str, str, float]]:
    """
    Build a flat list of (ref_exp_name, v1plus_exp_name, ref_metric_value).

    Supports both 1-to-1 (str value) and 1-to-many (list value) mappings.
    Only reference experiments with a non-None metric are included.
    """
    rows: list[tuple[str, str, float]] = []
    for ref_exp, summary in ref_exps.items():
        ref_val = _get_global(summary, metric)
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
    name_equivalences: dict[str, str | list[str]] | None = None,
    output_suffix: str = "",
):
    """
    Side-by-side figure per dataset:
      - Left panel:  absolute metric value for the reference version
      - Right panel: delta heatmap (version - reference) for the rest

    Uses ALL_V0_EQUIVALENCES by default, which covers both unsup and fs
    baselines since they both live in v0_baseline.

    Output filename: delta_vs_{reference}_{suffix}_{metric}_{dataset}.png
    """
    if name_equivalences is None:
        name_equivalences = ALL_V0_EQUIVALENCES

    versions = [v for v in data.keys() if v != reference_version]
    ref_data = data.get(reference_version, {})

    if not ref_data:
        print(f"  [SKIP] reference version '{reference_version}' not found in loaded data")
        return

    label = _metric_label(metric)

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
        abs_matrix = np.full((n_rows, len(versions)), np.nan)

        for j, v in enumerate(versions):
            for i, (_, equiv_name, ref_val) in enumerate(expanded):
                comp_val = _get_global(
                    data.get(v, {}).get(dataset, {}).get(equiv_name, {}),
                    metric,
                )
                if comp_val is not None:
                    delta_matrix[i, j] = comp_val - ref_val
                    abs_matrix[i, j] = comp_val

        fig, axes = plt.subplots(
            1, 2,
            figsize=(
                max(10, len(versions) * 2.5 + 4),
                max(4, n_rows * 0.9 + 2),
            ),
            gridspec_kw={"width_ratios": [1, max(len(versions), 1)]},
        )

        # Left panel: reference absolute values
        ax_ref = axes[0]
        ref_vals = np.array([[ref_val] for _, _, ref_val in expanded])
        vmin_ref, vmax_ref = _heatmap_vrange(metric, ref_vals)
        ax_ref.imshow(ref_vals, cmap=_metric_cmap(metric),
                      aspect="auto", vmin=vmin_ref, vmax=vmax_ref)

        for i, (_, _, ref_val) in enumerate(expanded):
            _annotate_heatmap_cell(ax_ref, 0, i, ref_val, vmin_ref, vmax_ref)

        ax_ref.set_xticks([0])
        ax_ref.set_xticklabels([reference_version],
                               fontsize=9, rotation=30, ha="right")
        ax_ref.set_yticks(range(n_rows))

        y_labels = []
        for ref_exp, equiv_name, _ in expanded:
            display = get_display_name(equiv_name)
            if equiv_name != ref_exp:
                ref_display = get_display_name(ref_exp)
                y_labels.append(f"{display}\n(ref: {ref_display})")
            else:
                y_labels.append(display)
        ax_ref.set_yticklabels(y_labels, fontsize=8)
        ax_ref.set_title(f"Reference\n({label} absolute)", fontsize=10)

        # Draw group separators on the reference panel rows
        prev_group = None
        for idx, (_, equiv_name, _) in enumerate(expanded):
            group = _get_experiment_group(equiv_name)
            if prev_group is not None and group != prev_group:
                ax_ref.axhline(y=idx - 0.5, color="grey",
                               linewidth=1.2, linestyle="--", alpha=0.5)
            prev_group = group

        # Right panel: delta heatmap (diverging colormap).
        # For distance metrics the sign of "better" is flipped: a negative
        # delta (lower distance) is an improvement, so we invert the colormap
        # so that blue = improvement stays consistent with overlap metrics.
        ax_delta = axes[1]
        max_abs = (
            np.nanmax(np.abs(delta_matrix))
            if not np.all(np.isnan(delta_matrix)) else 0.1
        )
        clim = max(max_abs, 0.05)

        is_distance = metric in {"hausdorff_mean", "hausdorff_95_mean", "assd_mean"}
        # For distance metrics: negative delta = improvement, so use RdBu_r
        # (red = positive = regression, blue = negative = improvement).
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
                              fontsize=7.5, color=text_color,
                              fontweight="bold")

        ax_delta.set_xticks(range(len(versions)))
        ax_delta.set_xticklabels(versions, fontsize=9, rotation=30, ha="right")
        ax_delta.set_yticks(range(n_rows))
        ax_delta.set_yticklabels([])

        # Group separators on delta panel
        prev_group = None
        for idx, (_, equiv_name, _) in enumerate(expanded):
            group = _get_experiment_group(equiv_name)
            if prev_group is not None and group != prev_group:
                ax_delta.axhline(y=idx - 0.5, color="grey",
                                 linewidth=1.2, linestyle="--", alpha=0.5)
            prev_group = group

        cbar = fig.colorbar(im_delta, ax=ax_delta, shrink=0.8)
        cbar.set_label(f"Δ {label} vs {reference_version}", fontsize=9)

        improvement_dir = "blue = lower = improvement" if is_distance else "blue = improvement"
        ax_delta.set_title(
            f"Δ {label}  ({improvement_dir}, red = regression)",
            fontsize=10,
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
# Plot 6: Multi-metric story per (dataset, experiment) across all versions
# ---------------------------------------------------------------------------

# Experiments to track across versions (one figure each).
#
# Each entry is a list of candidate experiment names tried in order:
# the first candidate that exists in a given version is used.
# Baseline names are listed first so that v0_baseline is correctly
# represented even when pipeline variants are absent in that version.
#
# For few-shot experiments we produce TWO separate figures per ref count:
#   - one tracking the iterative pipeline WITHOUT refinement
#   - one tracking the iterative pipeline WITH refinement
# Both use the corresponding fs_indep_baseline as the v0 anchor, so the
# plots are directly comparable.
CROSS_VERSION_EXPERIMENTS = [
    # ------------------------------------------------------------------ #
    # Unsupervised
    # ------------------------------------------------------------------ #
    [
        "unsup_baseline", "unsup_hdbscan", "unsup_hdbscan_refine",
        "unsup_kmeans", "unsup_kmeans_refine",
    ],

    # ------------------------------------------------------------------ #
    # 1-ref  —  without refinement
    # ------------------------------------------------------------------ #
    [
        "fs_indep_baseline_1ref",   # v0 anchor
        "fs_iter_1ref",             # v1+ pipeline (no refine)
        "fs_indep_1ref",            # fallback independent variant
    ],
    # 1-ref  —  with refinement
    [
        "fs_indep_baseline_1ref",   # v0 anchor (same baseline, different pipeline)
        "fs_iter_refine_1ref",      # v1+ pipeline (with refine)
        "fs_indep_refine_1ref",     # fallback independent+refine variant
    ],

    # ------------------------------------------------------------------ #
    # 3-ref  —  without refinement
    # ------------------------------------------------------------------ #
    [
        "fs_indep_baseline_3ref",
        "fs_iter_3ref",
        "fs_indep_3ref",
    ],
    # 3-ref  —  with refinement
    [
        "fs_indep_baseline_3ref",
        "fs_iter_refine_3ref",
        "fs_indep_refine_3ref",
    ],

    # ------------------------------------------------------------------ #
    # 5-ref  —  without refinement
    # ------------------------------------------------------------------ #
    [
        "fs_indep_baseline_5ref",
        "fs_iter_5ref",
        "fs_indep_5ref",
    ],
    # 5-ref  —  with refinement
    [
        "fs_indep_baseline_5ref",
        "fs_iter_refine_5ref",
        "fs_indep_refine_5ref",
    ],

    # ------------------------------------------------------------------ #
    # 10-ref  —  without refinement
    # ------------------------------------------------------------------ #
    [
        "fs_indep_baseline_10ref",
        "fs_iter_10ref",
        "fs_indep_10ref",
    ],
    # 10-ref  —  with refinement
    [
        "fs_indep_baseline_10ref",
        "fs_iter_refine_10ref",
        "fs_indep_refine_10ref",
    ],
]


def plot_experiment_across_versions(
    data: dict,
    output_dir: Path,
    metrics: list[str],
    target_experiments: list = CROSS_VERSION_EXPERIMENTS,
):
    """
    For each (dataset, target_experiment_group), one figure showing the chosen
    metrics as grouped bars where each group is a version.

    For each version, the first candidate experiment that exists is used.
    Because baseline names are listed first in each candidate group, v0_baseline
    is correctly represented even when pipeline variants are absent in that version.

    Bounded metrics [0, 1] use the primary y-axis; distance metrics use a
    secondary y-axis so they remain readable alongside detection metrics.

    Output: version_story_{dataset}_{experiment}.png
    """
    bounded_colors = {
        "dice_mean":      "#5B8DB8",
        "iou_mean":       "#7FA7C9",
        "recall@0.5":     "#6BBF6B",
        "precision@0.5":  "#E8A838",
        "f1@0.5":         "#D45B5B",
        "recall@0.7":     "#4FA84F",
        "precision@0.7":  "#C68E2C",
        "f1@0.7":         "#A84444",
    }
    distance_colors = {
        "hausdorff_95_mean": "#9C27B0",
        "assd_mean":         "#E91E63",
        "hausdorff_mean":    "#673AB7",
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
            # Resolve candidate names and display label
            candidates = [exp_entry] if isinstance(exp_entry, str) else exp_entry
            display_name = get_display_name(candidates[0])
            file_slug = candidates[0]

            # When two entries share the same baseline anchor (candidates[0])
            # but differ in whether refinement is used, append a suffix to
            # avoid overwriting the first file with the second.
            primary_pipeline = candidates[1] if len(candidates) > 1 else candidates[0]
            if "refine" in primary_pipeline and "refine" not in file_slug:
                file_slug = file_slug + "_with_refine"

            values_by_metric: dict[str, list[float]] = {m: [] for m in metrics}
            present_versions = []
            matched_experiments = []  # track which candidate was matched per version

            for v in versions:
                # Pick the first candidate experiment that exists
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
                    val = _get_global(summary, m)
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

            # --- Bounded metrics on primary axis ---
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
                    if np.isnan(v):
                        continue
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7,
                    )

            # --- Distance metrics on secondary axis ---
            for dk, metric in enumerate(distance_metrics):
                values = values_by_metric[metric]
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
                    edgecolor="white", linewidth=0.5,
                    hatch="//",
                )
                for bar, v in zip(bars, values):
                    if np.isnan(v):
                        continue
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{v:.1f}",
                        ha="center", va="bottom", fontsize=6,
                    )

            if not any_data:
                plt.close(fig)
                continue

            # Adaptive y-axis for bounded metrics
            if all_bounded_values:
                y_max = max(all_bounded_values)
                y_min_data = min(all_bounded_values)
                y_floor = max(0, y_min_data - 0.05) if y_min_data > 0.15 else 0
                y_ceil = min(y_max * 1.20, 1.10) if y_max < 0.9 else 1.10
            else:
                y_floor, y_ceil = 0, 1.10

            ax.set_ylim(y_floor, y_ceil)

            if ax2:
                ax2.set_ylabel("Distance (pixels)", fontsize=11, color="#9C27B0")

            # Mark reference version with a dashed vertical separator
            if "v0_baseline" in present_versions:
                ref_idx = present_versions.index("v0_baseline")
                ax.axvline(
                    x=ref_idx + 0.5,
                    color="grey", linewidth=1.5, linestyle="--", alpha=0.6,
                )

            # X-tick labels show version + matched experiment name when it
            # differs from the group's display name (e.g. baseline vs pipeline).
            x_labels = []
            for v, matched in zip(present_versions, matched_experiments):
                matched_display = get_display_name(matched) if matched else ""
                if matched and matched_display != display_name:
                    x_labels.append(f"{v}\n({matched_display})")
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

            # Merge legends from both axes
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
# Plot 7: Organ recovery analysis (baseline miss -> pipeline hit)
# ---------------------------------------------------------------------------

# Default pairs: (baseline_experiment, pipeline_experiments) for recovery
# analysis. The baseline is the raw MedSAM2 output; pipeline experiments
# are the ones that add clustering + refinement.
RECOVERY_PAIRS_UNSUP = {
    "unsup_baseline": [
        "unsup_kmeans", "unsup_kmeans_refine",
        "unsup_hdbscan", "unsup_hdbscan_refine",
    ],
}

RECOVERY_PAIRS_FS = {
    "fs_indep_baseline_1ref": [
        "fs_indep_1ref", "fs_indep_refine_1ref",
        "fs_iter_1ref", "fs_iter_refine_1ref",
    ],
    "fs_indep_baseline_3ref": [
        "fs_indep_3ref", "fs_indep_refine_3ref",
        "fs_iter_3ref", "fs_iter_refine_3ref",
    ],
}


def _build_organ_status(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """
    Build a mapping from (image, organ) -> status dict.

    Status dict contains:
        pred_name: str or None (None = organ was not segmented)
        dice: float
    """
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


def _analyze_recovery(
    baseline_rows: list[dict],
    pipeline_rows: list[dict],
) -> dict:
    """
    Compare baseline vs pipeline metrics.csv data and classify each
    (image, organ) pair into: recovered, lost, both_hit, both_miss.
    """
    b_status = _build_organ_status(baseline_rows)
    p_status = _build_organ_status(pipeline_rows)
    all_keys = set(b_status.keys()) | set(p_status.keys())

    recovered = []
    lost = []
    both_hit = []
    both_miss = []

    for key in sorted(all_keys):
        image, organ = key
        b = b_status.get(key, {"pred_name": None, "dice": 0})
        p = p_status.get(key, {"pred_name": None, "dice": 0})

        entry = {
            "image": image,
            "organ": organ,
            "baseline_dice": b["dice"],
            "pipeline_dice": p["dice"],
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

    # Per-organ breakdown of recoveries
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
    recovery_pairs: dict[str, list[str]] | None = None,
):
    """
    For each (version, dataset), compare baseline vs pipeline experiments
    using metrics.csv to count recovered and lost organs.

    Produces:
    - Bar chart of recovered vs lost counts per pipeline variant
    - Per-organ recovery breakdown for the best pipeline variant
    - CSV with detailed per-case recovery data
    """
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

                # --- Bar chart: recovered vs lost per pipeline variant ---
                labels = list(all_results.keys())
                n_rec = [r["n_recovered"] for r in all_results.values()]
                n_lost_vals = [-r["n_lost"] for r in all_results.values()]

                x = np.arange(len(labels))
                width = 0.35

                fig, ax = plt.subplots(
                    figsize=(max(8, len(labels) * 1.8), 6)
                )

                bars_r = ax.bar(
                    x - width / 2, n_rec, width,
                    label="Recovered", color="#4CAF50", alpha=0.85,
                )
                bars_l = ax.bar(
                    x + width / 2, n_lost_vals, width,
                    label="Lost", color="#F44336", alpha=0.85,
                )

                for bar, v in zip(bars_r, n_rec):
                    if v > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.5,
                            str(v), ha="center", va="bottom",
                            fontsize=10, fontweight="bold",
                        )
                for bar, v in zip(bars_l, n_lost_vals):
                    if v < 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            v - 0.5,
                            str(abs(v)), ha="center", va="top",
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

                # --- Per-organ breakdown for best pipeline ---
                best_exp = max(all_results, key=lambda k: all_results[k]["n_recovered"])
                best = all_results[best_exp]

                if best["recovery_by_organ"]:
                    organs = sorted(best["recovery_by_organ"].keys())
                    counts = [
                        len(best["recovery_by_organ"][o]) for o in organs
                    ]
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

                # --- CSV with per-case details ---
                csv_rows = []
                for pipe_exp, result in all_results.items():
                    for r in result["recovered"]:
                        csv_rows.append({
                            "version": version,
                            "dataset": dataset,
                            "baseline": baseline_exp,
                            "pipeline": pipe_exp,
                            "status": "recovered",
                            **r,
                        })
                    for r in result["lost"]:
                        csv_rows.append({
                            "version": version,
                            "dataset": dataset,
                            "baseline": baseline_exp,
                            "pipeline": pipe_exp,
                            "status": "lost",
                            **r,
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
# Plot 8: Box plots — per-organ Dice distribution (baseline vs pipeline)
# ---------------------------------------------------------------------------

BOX_PLOT_PAIRS_UNSUP = {
    "unsup_baseline": [
        "unsup_hdbscan", "unsup_hdbscan_refine",
        "unsup_kmeans_refine",
    ],
}

BOX_PLOT_PAIRS_FS = {
    "fs_indep_baseline_1ref": [
        "fs_iter_1ref", "fs_iter_refine_1ref",
    ],
    "fs_indep_baseline_3ref": [
        "fs_iter_3ref", "fs_iter_refine_3ref",
    ],
}


def plot_box_plots_per_organ(
    results_dir: Path,
    data: dict,
    output_dir: Path,
    box_plot_pairs: dict[str, list[str]] | None = None,
):
    """
    For each (version, dataset, baseline_exp), produce per-organ box plots
    comparing the Dice distribution between baseline and pipeline variants.

    Each organ gets its own figure with side-by-side box plots.
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

                # Baseline
                dice_by_exp[baseline_exp] = defaultdict(list)
                for row in baseline_rows:
                    dice_by_exp[baseline_exp][row["organ"]].append(
                        float(row.get("dice", 0))
                    )

                # Pipeline experiments
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

                if len(dice_by_exp) < 2:
                    continue

                all_organs: set[str] = set()
                for exp_data in dice_by_exp.values():
                    all_organs.update(exp_data.keys())
                organs = sorted(all_organs)

                if not organs:
                    continue

                exp_names = [baseline_exp] + [
                    e for e in pipeline_exps if e in dice_by_exp
                ]
                colors = ["#BDBDBD", "#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

                for organ in organs:
                    box_data = []
                    box_labels = []
                    box_colors = []

                    for idx, exp in enumerate(exp_names):
                        values = dice_by_exp.get(exp, {}).get(organ, [])
                        if values:
                            box_data.append(values)
                            box_labels.append(get_display_name(exp))
                            box_colors.append(colors[idx % len(colors)])

                    if len(box_data) < 2:
                        continue

                    fig, ax = plt.subplots(
                        figsize=(max(6, len(box_data) * 1.5), 5)
                    )

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

                    organ_title = organ.replace("_", " ").title()
                    ax.set_title(
                        f"Dice Distribution — {organ_title}"
                        f" — {dataset} / {version}",
                        fontsize=12, fontweight="bold",
                    )
                    ax.set_ylabel("Dice", fontsize=11)
                    ax.set_ylim(-0.05, 1.05)
                    ax.grid(axis="y", alpha=0.3)
                    ax.tick_params(axis="x", rotation=20)

                    fig.tight_layout()
                    slug = baseline_exp.replace("fs_indep_baseline_", "fs_")
                    out_path = (
                        output_dir
                        / f"box_plot_dice_{dataset}_{version}_{organ}_{slug}.png"
                    )
                    fig.savefig(out_path, dpi=200)
                    plt.close(fig)
                    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Flat CSV export
# ---------------------------------------------------------------------------

def save_full_csv(data: dict, output_dir: Path):
    """
    Flatten all results into a single CSV.

    Includes global quality metrics, P/R/F1 at each available IoU
    threshold, and per-organ recall + dice.
    """
    rows: list[dict] = []

    # Discover IoU thresholds present in the data
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
                row = {
                    "version":    version,
                    "dataset":    dataset,
                    "experiment": exp,
                    "n_images":   summary.get("n_images", 0),
                    "matching":   summary.get("matching", ""),
                    # Overlap quality
                    "dice_mean":  g.get("dice_mean"),
                    "dice_std":   g.get("dice_std"),
                    "iou_mean":   g.get("iou_mean"),
                    "iou_std":    g.get("iou_std"),
                    # Distance quality
                    "hd95_mean":  g.get("hausdorff_95_mean"),
                    "hd95_std":   g.get("hausdorff_95_std"),
                    "hd_mean":    g.get("hausdorff_mean"),
                    "hd_std":     g.get("hausdorff_std"),
                    "assd_mean":  g.get("assd_mean"),
                    "assd_std":   g.get("assd_std"),
                    # Pred / GT counts
                    "n_gt_total":   g.get("n_gt_total"),
                    "n_pred_total": g.get("n_pred_total"),
                    # Detection-level quality (mAP)
                    "map":       g.get("map"),
                    "map_50":    g.get("map_50"),
                    "map_75":    g.get("map_75"),
                }

                # P/R/F1 at each threshold (0.5, 0.7, and any others found)
                for thr in thresholds:
                    row[f"recall@{thr}"]    = g.get(f"recall@{thr}")
                    row[f"precision@{thr}"] = g.get(f"precision@{thr}")
                    row[f"f1@{thr}"]        = g.get(f"f1@{thr}")
                    row[f"n_gt_covered@{thr}"]    = g.get(f"n_gt_covered@{thr}")
                    row[f"n_pred_relevant@{thr}"] = g.get(f"n_pred_relevant@{thr}")

                # Per-organ Dice + recall at each threshold
                for organ, stats in summary.get("per_organ", {}).items():
                    row[f"{organ}_dice"]    = stats.get("dice_mean")
                    row[f"{organ}_missing"] = stats.get("missing", 0)
                    for thr in thresholds:
                        row[f"{organ}_recall@{thr}"] = stats.get(f"recall@{thr}")

                rows.append(row)

    if not rows:
        return

    fieldnames = [
        "version", "dataset", "experiment", "n_images", "matching",
        "dice_mean", "dice_std", "iou_mean", "iou_std",
        "hd95_mean", "hd95_std", "hd_mean", "hd_std",
        "assd_mean", "assd_std",
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
        help="Root results dir (scans for version/dataset/experiment/summary.json)",
    )
    parser.add_argument(
        "--versions", nargs="*",
        help="Versions to compare (default: all subdirectories of results_dir)",
    )
    parser.add_argument(
        "--datasets", nargs="*",
        help="Datasets to include (default: all). Order is preserved for the "
             "cross-dataset heatmap columns, so pass them in the order you "
             "want them displayed (e.g. SunnybrookNicoSent_basal "
             "SunnybrookNicoSent_mid SunnybrookNicoSent_apex).",
    )
    parser.add_argument(
        "--experiments", nargs="*",
        help="Filter to these experiment names only (default: all). "
             "Baseline experiments whose pipeline targets are in the filter "
             "are added automatically. "
             "Example: --experiments unsup_hdbscan_refine fs_iter_refine_1ref",
    )
    parser.add_argument(
        "--reference", default="v0_baseline",
        help="Reference version for delta heatmaps. v0_baseline contains "
             "both unsup and fs baselines, so a single reference version "
             "covers all experiment types.",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=DEFAULT_METRICS,
        help="Metrics to plot (keys as found in summary.json[global])",
    )
    parser.add_argument(
        "--output", default="results/comparison",
        help="Output directory for comparison plots and CSV",
    )
    parser.add_argument(
        "--skip-metric-story", action="store_true",
        help="Skip metric story plots (fix version, compare experiments).",
    )
    parser.add_argument(
        "--skip-recovery", action="store_true",
        help="Skip organ recovery analysis (requires metrics.csv files).",
    )
    parser.add_argument(
        "--skip-box-plots", action="store_true",
        help="Skip per-organ box plot distributions (requires metrics.csv).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    data = load_all_results(
        results_dir,
        versions=args.versions,
        experiments_filter=args.experiments,
    )

    # Optional dataset filter; preserves user-supplied order so cross-dataset
    # plot columns appear in a chosen sequence (e.g. basal / mid / apex).
    if args.datasets:
        data = _filter_datasets(data, args.datasets)
        print(f"\nFiltered to {len(args.datasets)} dataset(s) in order: "
              f"{list(args.datasets)}")

    total = sum(len(exps) for v in data.values() for exps in v.values())
    if total == 0:
        print("No results found.")
        return

    print(f"\nVersions: {list(data.keys())}")
    print(f"Total experiment results: {total}")
    print(f"Metrics to plot: {args.metrics}")
    print(f"Reference version: {args.reference}")
    if args.experiments:
        print(f"Experiment filter (after auto-extension): {sorted(args.experiments)}")

    # ---- Plot 1: per-dataset metric heatmap ----
    print("\n[1/9] Per-dataset metric heatmaps:")
    for metric in args.metrics:
        plot_metric_heatmap_per_dataset(data, output_dir, metric=metric)

    # ---- Plot 2: cross-dataset metric heatmap ----
    # Complement to Plot 1: fixes version and compares datasets side by side.
    # Column order follows --datasets when given, else alphabetical.
    print("\n[2/9] Cross-dataset metric heatmaps:")
    for metric in args.metrics:
        plot_cross_dataset_heatmap(
            data, output_dir,
            metric=metric,
            dataset_order=args.datasets,
        )

    # ---- Plot 3: per-organ metric heatmap (separate file per organ) ----
    print("\n[3/9] Per-organ metric heatmaps:")
    for metric in args.metrics:
        plot_per_organ_heatmap(data, output_dir, metric=metric)

    # ---- Plot 4: multi-metric story per (dataset, version) [optional] ----
    if args.skip_metric_story:
        print("\n[4/9] Metric story plots: SKIPPED (--skip-metric-story)")
    else:
        print("\n[4/9] Multi-metric story plots:")
        plot_metric_story(data, output_dir, metrics=args.metrics)

    # ---- Plot 5: multi-metric story per (dataset, experiment) across versions ----
    print("\n[5/9] Cross-version story plots:")
    plot_experiment_across_versions(data, output_dir, metrics=args.metrics)

    # ---- Plot 6: delta heatmaps vs reference version ----
    # v0_baseline contains both unsup and fs baselines, so a single call with
    # ALL_V0_EQUIVALENCES covers everything. No separate v0_baseline_fs needed.
    print("\n[6/9] Delta heatmaps vs reference version:")
    for metric in args.metrics:
        plot_delta_vs_baseline_heatmap(
            data, output_dir,
            reference_version=args.reference,
            metric=metric,
            name_equivalences=ALL_V0_EQUIVALENCES,
            output_suffix="pipeline",
        )

    # ---- Plot 7: organ recovery analysis ----
    if args.skip_recovery:
        print("\n[7/9] Organ recovery: SKIPPED (--skip-recovery)")
    else:
        print("\n[7/9] Organ recovery analysis:")
        plot_organ_recovery(results_dir, data, output_dir)

    # ---- Plot 8: box plots per organ ----
    if args.skip_box_plots:
        print("\n[8/9] Box plots: SKIPPED (--skip-box-plots)")
    else:
        print("\n[8/9] Per-organ box plots:")
        plot_box_plots_per_organ(results_dir, data, output_dir)

    # ---- CSV ----
    print("\n[9/9] Flat CSV export:")
    save_full_csv(data, output_dir)

    print(f"\nAll comparison artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
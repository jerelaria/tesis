"""
compare_versions.py
-------------------
Generate cross-version, cross-dataset comparison plots.

Scans results/{version}/{dataset}/{experiment}/summary.json and produces
a small, opinionated set of figures designed for the paper:

1. metric_heatmap_{metric}_{dataset}.png
       For each (metric, dataset), a heatmap experiment x version showing
       the absolute metric value. One figure per dataset and metric.

2. per_organ_{metric}_{dataset}.png
       For each (metric, dataset), one subplot per organ, each subplot a
       heatmap experiment x version. Reveals which organs benefit from
       which design choices.

3. metric_story_{dataset}_{version}.png
       For one (dataset, version), grouped bars per experiment showing
       Dice, Recall@0.5, Precision@0.5 and F1@0.5 side by side. This is
       the "why Dice alone hides the gain" plot.

4. delta_vs_{ref}_{metric}_{dataset}.png
       Delta heatmaps vs a reference version (v0_baseline / v0_baseline_fs).
       Side-by-side absolute reference + delta-from-reference.

5. all_versions_summary.csv
       Flat CSV with every (version, dataset, experiment, metric).

Usage:
    python compare_versions.py --results_dir results/ --output results/comparison/

    # Specify versions explicitly:
    python compare_versions.py \\
        --versions v0_baseline v0_baseline_fs v1_baseline v2_standarized \\
        --results_dir results/ \\
        --output results/comparison/

    # Override reference versions:
    python compare_versions.py \\
        --reference v0_baseline_fs --reference_unsup v0_baseline
"""

import argparse
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------

DISPLAY_NAMES = {
    # Unsupervised pipeline
    "unsup_kmeans":           "Unsup KMeans",
    "unsup_hdbscan":          "Unsup HDBSCAN",
    "unsup_kmeans_refine":    "Unsup KMeans+R",
    "unsup_hdbscan_refine":   "Unsup HDBSCAN+R",
    # v0_baseline unsupervised (plain SAM2 grid prompting, no pipeline)
    "unsup_baseline":         "Unsup Baseline",
    # Few-shot independent
    "fs_indep_1ref":          "Indep 1ref",
    "fs_indep_5ref":          "Indep 5ref",
    "fs_indep_10ref":         "Indep 10ref",
    "fs_indep_refine_1ref":   "Indep+R 1ref",
    "fs_indep_refine_5ref":   "Indep+R 5ref",
    "fs_indep_refine_10ref":  "Indep+R 10ref",
    # Few-shot iterative
    "fs_iter_1ref":           "Iter 1ref",
    "fs_iter_5ref":           "Iter 5ref",
    "fs_iter_10ref":          "Iter 10ref",
    "fs_iter_refine_1ref":    "Iter+R 1ref",
    "fs_iter_refine_5ref":    "Iter+R 5ref",
    "fs_iter_refine_10ref":   "Iter+R 10ref",
    # v0_baseline_fs (plain SAM2 few-shot, no pipeline)
    "fs_indep_baseline_1ref":  "Baseline 1ref",
    "fs_indep_baseline_5ref":  "Baseline 5ref",
    "fs_indep_baseline_10ref": "Baseline 10ref",
    # Text-guided
    "tg": "Text Guided",
}

# Canonical display order (unknowns appended alphabetically at the end)
EXPERIMENT_ORDER = [
    "unsup_baseline",
    "unsup_kmeans", "unsup_hdbscan",
    "unsup_kmeans_refine", "unsup_hdbscan_refine",
    "fs_indep_baseline_1ref", "fs_indep_baseline_5ref", "fs_indep_baseline_10ref",
    "fs_indep_1ref",          "fs_indep_refine_1ref",
    "fs_indep_5ref",          "fs_indep_refine_5ref",
    "fs_indep_10ref",         "fs_indep_refine_10ref",
    "fs_iter_1ref",           "fs_iter_refine_1ref",
    "fs_iter_5ref",           "fs_iter_refine_5ref",
    "fs_iter_10ref",          "fs_iter_refine_10ref",
    "tg",
]

# v0 reference -> v1+ equivalences (for delta heatmaps)
V0_FS_NAME_EQUIVALENCES: dict[str, str | list[str]] = {
    "fs_indep_baseline_1ref":  "fs_indep_1ref",
    "fs_indep_baseline_5ref":  "fs_indep_5ref",
    "fs_indep_baseline_10ref": "fs_indep_10ref",
}

V0_UNSUP_NAME_EQUIVALENCES: dict[str, str | list[str]] = {
    "unsup_baseline": [
        "unsup_kmeans",
        "unsup_hdbscan",
        "unsup_kmeans_refine",
        "unsup_hdbscan_refine",
    ],
}


# ---------------------------------------------------------------------------
# Metric metadata
# ---------------------------------------------------------------------------

# Each metric: (display label, colormap, value range for shared colorbar)
METRIC_META = {
    "dice_mean":      ("Dice",         "RdYlGn", (0.0, 1.0)),
    "iou_mean":       ("IoU",          "RdYlGn", (0.0, 1.0)),
    "recall@0.5":     ("Recall@0.5",   "RdYlGn", (0.0, 1.0)),
    "precision@0.5":  ("Precision@0.5", "RdYlGn", (0.0, 1.0)),
    "f1@0.5":         ("F1@0.5",       "RdYlGn", (0.0, 1.0)),
    "recall@0.7":     ("Recall@0.7",   "RdYlGn", (0.0, 1.0)),
    "precision@0.7":  ("Precision@0.7", "RdYlGn", (0.0, 1.0)),
    "f1@0.7":         ("F1@0.7",       "RdYlGn", (0.0, 1.0)),
}

# Default set of metrics produced by the new evaluate.py
DEFAULT_METRICS = ["dice_mean", "recall@0.5", "precision@0.5", "f1@0.5"]


def _metric_label(metric: str) -> str:
    return METRIC_META.get(metric, (metric.replace("_", " "), None, None))[0]


def _metric_cmap(metric: str) -> str:
    return METRIC_META.get(metric, (None, "RdYlGn", None))[1]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _get_display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(
    results_dir: Path,
    versions: list[str] | None = None,
) -> dict[str, dict[str, dict[str, dict]]]:
    """
    Recursively load summary.json files.

    Returns: {version: {dataset: {experiment: summary_dict}}}
    """
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

    Reference versions get a dashed white separator after their column.
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

        fig, ax = plt.subplots(
            figsize=(max(6, len(versions) * 2),
                     max(6, len(experiments) * 0.5)),
        )
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

        for i in range(len(experiments)):
            for j in range(len(versions)):
                _annotate_heatmap_cell(ax, j, i, matrix[i, j], 0, 1)

        for ref_version in ("v0_baseline", "v0_baseline_fs"):
            if ref_version in versions:
                ref_col = versions.index(ref_version)
                ax.axvline(x=ref_col + 0.5, color="white",
                           linewidth=2.5, linestyle="--")

        ax.set_xticks(range(len(versions)))
        ax.set_xticklabels(versions, fontsize=10, rotation=30, ha="right")
        ax.set_yticks(range(len(experiments)))
        ax.set_yticklabels(
            [_get_display_name(e) for e in experiments], fontsize=9,
        )
        ax.set_title(f"Global {label} — {dataset}",
                     fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8, label=label)
        fig.tight_layout()

        out_path = output_dir / f"metric_heatmap_{metric.replace('@','_at_')}_{dataset}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Per-organ heatmap of one metric (one subplot per organ)
# ---------------------------------------------------------------------------

def plot_per_organ_heatmap(
    data: dict,
    output_dir: Path,
    metric: str,
):
    """
    For each dataset, one figure with subplots arranged horizontally: one
    subplot per organ (e.g., heart, left_lung, right_lung) plus a final
    subplot for the global metric.

    Each subplot is a heatmap with experiments on y-axis and versions on
    x-axis. This is the figure that exposes the "heart never recovers,
    lungs do" pattern when present.

    Note: only metrics that exist per-organ in summaries are shown per
    organ. Precision is global-only and is therefore left out of the
    per-organ panels (it shows only as the global subplot if requested).
    """
    versions = list(data.keys())
    if len(versions) < 1:
        return

    label = _metric_label(metric)
    cmap = _metric_cmap(metric)

    # Recall and the quality metrics are well-defined per organ.
    # Precision (and therefore F1) are global-only, so per-organ panels
    # are not produced for them.
    is_per_organ_supported = (
        metric.startswith("dice_") or metric.startswith("iou_")
        or metric.startswith("recall@") or metric.startswith("hausdorff_")
    )
    if not is_per_organ_supported:
        print(f"  [SKIP] per-organ heatmap for {metric}: per-organ N/A")
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

        # Build one matrix per panel: organs + a final 'global' panel
        panels: list[tuple[str, np.ndarray]] = []
        for organ in organs:
            mat = np.full((len(experiments), len(versions)), np.nan)
            for j, v in enumerate(versions):
                for i, exp in enumerate(experiments):
                    summary = data.get(v, {}).get(dataset, {}).get(exp)
                    if summary:
                        val = _get_per_organ(summary, organ, metric)
                        if val is not None:
                            mat[i, j] = val
            panels.append((organ, mat))

        # Global panel
        mat_global = np.full((len(experiments), len(versions)), np.nan)
        for j, v in enumerate(versions):
            for i, exp in enumerate(experiments):
                summary = data.get(v, {}).get(dataset, {}).get(exp)
                if summary:
                    val = _get_global(summary, metric)
                    if val is not None:
                        mat_global[i, j] = val
        panels.append(("GLOBAL", mat_global))

        n_panels = len(panels)
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(max(4, len(versions) * 1.6) * n_panels,
                     max(5, len(experiments) * 0.5) + 1),
            squeeze=False,
        )

        last_im = None
        for k, (panel_name, mat) in enumerate(panels):
            ax = axes[0, k]
            im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=1)
            last_im = im

            for i in range(len(experiments)):
                for j in range(len(versions)):
                    _annotate_heatmap_cell(ax, j, i, mat[i, j], 0, 1)

            for ref_version in ("v0_baseline", "v0_baseline_fs"):
                if ref_version in versions:
                    ref_col = versions.index(ref_version)
                    ax.axvline(x=ref_col + 0.5, color="white",
                               linewidth=2.0, linestyle="--")

            ax.set_xticks(range(len(versions)))
            ax.set_xticklabels(versions, fontsize=9, rotation=35, ha="right")
            if k == 0:
                ax.set_yticks(range(len(experiments)))
                ax.set_yticklabels(
                    [_get_display_name(e) for e in experiments], fontsize=9,
                )
            else:
                ax.set_yticks(range(len(experiments)))
                ax.set_yticklabels([])

            ax.set_title(panel_name, fontsize=11, fontweight="bold")

        fig.suptitle(f"{label} per organ — {dataset}",
                     fontsize=14, fontweight="bold")
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
        fig.colorbar(last_im, cax=cbar_ax, label=label)

        out_path = output_dir / f"per_organ_{metric.replace('@','_at_')}_{dataset}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: Multi-metric story plot (the "why Dice alone hides the gain" plot)
# ---------------------------------------------------------------------------

def plot_metric_story(
    data: dict,
    output_dir: Path,
    metrics: list[str],
):
    """
    For each (dataset, version), one figure with grouped bars per
    experiment showing several metrics side by side.

    The intent is to visualise the gap between Dice (which dilutes recall
    gains across many already-matched entries) and Recall / Precision /
    F1, which respond more sharply to pipeline improvements.
    """
    metric_colors = {
        "dice_mean":      "#5B8DB8",
        "iou_mean":       "#7FA7C9",
        "recall@0.5":     "#6BBF6B",
        "precision@0.5":  "#E8A838",
        "f1@0.5":         "#D45B5B",
        "recall@0.7":     "#4FA84F",
        "precision@0.7":  "#C68E2C",
        "f1@0.7":         "#A84444",
    }

    for version, datasets_dict in data.items():
        for dataset, exps in datasets_dict.items():
            if not exps:
                continue

            experiments = _sort_experiments(list(exps.keys()))
            x = np.arange(len(experiments))
            width = 0.8 / max(len(metrics), 1)

            fig, ax = plt.subplots(figsize=(max(8, len(experiments) * 1.0), 6))

            any_data = False
            for k, metric in enumerate(metrics):
                values = []
                for exp in experiments:
                    val = _get_global(exps.get(exp, {}), metric)
                    values.append(val if val is not None else np.nan)

                if all(np.isnan(v) for v in values):
                    continue
                any_data = True

                offset = (k - len(metrics) / 2 + 0.5) * width
                color = metric_colors.get(metric, f"C{k}")
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
                        ha="center", va="bottom",
                        fontsize=7,
                    )

            if not any_data:
                plt.close(fig)
                continue

            ax.set_ylim(0, 1.10)
            ax.set_ylabel("Metric value", fontsize=11)
            ax.set_title(
                f"Quality vs coverage vs cleanliness — {dataset} / {version}",
                fontsize=13, fontweight="bold",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                [_get_display_name(e) for e in experiments],
                fontsize=8, rotation=35, ha="right",
            )
            ax.legend(fontsize=9, loc="upper left",
                      ncol=min(len(metrics), 4))
            ax.grid(axis="y", alpha=0.3)
            ax.axhline(y=0, color="black", linewidth=0.6)

            fig.tight_layout()
            out_path = output_dir / f"metric_story_{dataset}_{version}.png"
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 4: Delta heatmap vs a reference version
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

    Output filename: delta_vs_{reference}_{suffix}_{metric}_{dataset}.png
    """
    if name_equivalences is None:
        name_equivalences = V0_FS_NAME_EQUIVALENCES

    versions = [v for v in data.keys() if v != reference_version]
    ref_data = data.get(reference_version, {})

    if not ref_data:
        print(f"  [SKIP] reference '{reference_version}' not found")
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
        ax_ref.imshow(ref_vals, cmap=_metric_cmap(metric),
                      aspect="auto", vmin=0, vmax=1)

        for i, (_, _, ref_val) in enumerate(expanded):
            _annotate_heatmap_cell(ax_ref, 0, i, ref_val, 0, 1)

        ax_ref.set_xticks([0])
        ax_ref.set_xticklabels([reference_version],
                               fontsize=9, rotation=30, ha="right")
        ax_ref.set_yticks(range(n_rows))

        y_labels = []
        for ref_exp, equiv_name, _ in expanded:
            if equiv_name != ref_exp:
                y_labels.append(
                    f"{_get_display_name(equiv_name)}\n"
                    f"(ref: {_get_display_name(ref_exp)})"
                )
            else:
                y_labels.append(_get_display_name(ref_exp))
        ax_ref.set_yticklabels(y_labels, fontsize=8)
        ax_ref.set_title(f"Reference\n({label} absolute)", fontsize=10)

        # Right panel: delta heatmap (diverging colormap)
        ax_delta = axes[1]
        max_abs = (
            np.nanmax(np.abs(delta_matrix))
            if not np.all(np.isnan(delta_matrix)) else 0.1
        )
        clim = max(max_abs, 0.05)

        im_delta = ax_delta.imshow(
            delta_matrix, cmap="RdBu", aspect="auto",
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

        cbar = fig.colorbar(im_delta, ax=ax_delta, shrink=0.8)
        cbar.set_label(f"Δ {label} vs {reference_version}", fontsize=9)
        ax_delta.set_title(
            f"Δ {label}  (blue = improvement, red = regression)",
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
            f"{metric.replace('@','_at_')}_{dataset}.png"
        )
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Flat CSV export (one row per (version, dataset, experiment))
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
                    # Quality
                    "dice_mean":  g.get("dice_mean"),
                    "dice_std":   g.get("dice_std"),
                    "iou_mean":   g.get("iou_mean"),
                    "iou_std":    g.get("iou_std"),
                    "hd95_mean":  g.get("hausdorff_95_mean"),
                    "hd95_std":   g.get("hausdorff_95_std"),
                    # Pred / GT counts
                    "n_gt_total":   g.get("n_gt_total"),
                    "n_pred_total": g.get("n_pred_total"),
                }

                # P/R/F1 at each threshold
                for thr in thresholds:
                    row[f"recall@{thr}"]    = g.get(f"recall@{thr}")
                    row[f"precision@{thr}"] = g.get(f"precision@{thr}")
                    row[f"f1@{thr}"]        = g.get(f"f1@{thr}")
                    row[f"n_gt_covered@{thr}"]    = g.get(f"n_gt_covered@{thr}")
                    row[f"n_pred_relevant@{thr}"] = g.get(f"n_pred_relevant@{thr}")

                # Per-organ Dice + recall@thresholds
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
        "hd95_mean", "hd95_std",
        "n_gt_total", "n_pred_total",
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
        "--reference", default="v0_baseline_fs",
        help="Reference version for the few-shot delta heatmap",
    )
    parser.add_argument(
        "--reference_unsup", default="v0_baseline",
        help="Reference version for the unsupervised delta heatmap",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=DEFAULT_METRICS,
        help="Metrics to plot (keys as found in summary.json[global])",
    )
    parser.add_argument(
        "--output", default="results/comparison",
        help="Output directory for comparison plots and CSV",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    data = load_all_results(Path(args.results_dir), versions=args.versions)

    total = sum(len(exps) for v in data.values() for exps in v.values())
    if total == 0:
        print("No results found.")
        return

    print(f"\nVersions: {list(data.keys())}")
    print(f"Total experiment results: {total}")
    print(f"Metrics to plot: {args.metrics}")

    # ---- Plot 1: per-dataset metric heatmap ----
    print("\n[1/4] Per-dataset metric heatmaps:")
    for metric in args.metrics:
        plot_metric_heatmap_per_dataset(data, output_dir, metric=metric)

    # ---- Plot 2: per-organ metric heatmap (skips precision/F1 automatically) ----
    print("\n[2/4] Per-organ metric heatmaps:")
    for metric in args.metrics:
        plot_per_organ_heatmap(data, output_dir, metric=metric)

    # ---- Plot 3: multi-metric story per (dataset, version) ----
    print("\n[3/4] Multi-metric story plots:")
    plot_metric_story(data, output_dir, metrics=args.metrics)

    # ---- Plot 4: delta heatmaps vs both reference versions, per metric ----
    print("\n[4/4] Delta heatmaps vs reference versions:")
    for metric in args.metrics:
        # Few-shot reference
        plot_delta_vs_baseline_heatmap(
            data, output_dir,
            reference_version=args.reference,
            metric=metric,
            name_equivalences=V0_FS_NAME_EQUIVALENCES,
            output_suffix="fs",
        )
        # Unsupervised reference
        plot_delta_vs_baseline_heatmap(
            data, output_dir,
            reference_version=args.reference_unsup,
            metric=metric,
            name_equivalences=V0_UNSUP_NAME_EQUIVALENCES,
            output_suffix="unsup",
        )

    # ---- CSV ----
    print("\n[CSV] Flat export:")
    save_full_csv(data, output_dir)

    print(f"\nAll comparison artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
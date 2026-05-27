"""
plot_distributions.py
---------------------
Per-organ Dice distribution histograms revealing the bimodal failure mode
of co-segmentation: when an organ is detected, Dice is high; when it is
missed, Dice is exactly 0 (pred_name is empty in metrics.csv). The visual
gap between those two masses is the diagnostic signal.

Layout choice
-------------
One figure per (version, dataset, experiment), with a horizontal subplot
grid: one subplot per organ. This produces a manageable number of files
(roughly one per experiment) while keeping per-organ resolution.

For each organ subplot:
  - A separate "missing" bar at the left of the axis, holding all GT
    entries where pred_name is empty (dice == 0 by construction). It is
    visually separated from the continuous histogram so that detection
    failure is not conflated with very-poor segmentation.
  - A continuous histogram over [0, 1] with fixed bins, showing the
    Dice distribution of the detected entries.
  - An annotation with n, missing count, and miss rate.

Overlay mode (--overlay)
------------------------
For each (version, dataset, baseline_exp), overlay a pipeline variant on
top of the baseline histogram (per organ). This is the figure that tells
the story "the refiner shrinks the left mass without touching the right
one".

Usage
-----
    # Per-experiment grid histograms (all experiments).
    python plot_distributions.py \\
        --results_dir results/ \\
        --output results/distributions/

    # Filter to specific versions, datasets, experiments:
    python plot_distributions.py \\
        --results_dir results/ \\
        --versions v3_extended_training_sunny_csf015_msf003 \\
        --datasets SunnybrookNicoSent \\
        --experiments unsup_hdbscan_refine \\
        --output results/distributions/v3_sunny/

    # Baseline vs pipeline overlay figures (extra; use --only-overlay to
    # skip the per-experiment ones).
    python plot_distributions.py \\
        --results_dir results/ \\
        --output results/distributions/ \\
        --overlay
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Display constants (kept in sync with compare_versions.py)
# ---------------------------------------------------------------------------

DISPLAY_NAMES = {
    # Unsupervised
    "unsup_baseline":         "Grid Prompting",
    "unsup_hdbscan":          "HDBSCAN",
    "unsup_hdbscan_refine":   "HDBSCAN + Refine",
    "unsup_kmeans":           "KMeans",
    "unsup_kmeans_refine":    "KMeans + Refine",
    # Few-shot baselines
    "fs_indep_baseline_1ref": "SAM2 Video (1 ref)",
    "fs_indep_baseline_3ref": "SAM2 Video (3 ref)",
    # Few-shot pipelines
    "fs_indep_1ref":          "Independent (1 ref)",
    "fs_indep_3ref":          "Independent (3 ref)",
    "fs_indep_refine_1ref":   "Indep + Refine (1 ref)",
    "fs_indep_refine_3ref":   "Indep + Refine (3 ref)",
    "fs_iter_1ref":           "Iterative (1 ref)",
    "fs_iter_3ref":           "Iterative (3 ref)",
    "fs_iter_refine_1ref":    "Iter + Refine (1 ref)",
    "fs_iter_refine_3ref":    "Iter + Refine (3 ref)",
}

# Pairs for overlay mode: baseline -> pipeline variants to compare against.
OVERLAY_PAIRS_UNSUP = {
    "unsup_baseline": ["unsup_hdbscan_refine"],
}

OVERLAY_PAIRS_FS = {
    "fs_indep_baseline_1ref": ["fs_iter_refine_1ref"],
    "fs_indep_baseline_3ref": ["fs_iter_refine_3ref"],
}


def _get_display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_metrics_csv(csv_path: Path) -> list[dict]:
    """Load a metrics.csv file into a list of dicts."""
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _is_missing(row: dict) -> bool:
    """
    A row is 'missing' when pred_name is empty (no prediction matched
    this GT entry). evaluate.py writes pred_name="" in this case, with
    dice=0, iou=0, hausdorff=inf.
    """
    pred_name = (row.get("pred_name") or "").strip()
    return pred_name == "" or pred_name.lower() == "none"


def _dice_by_organ_split(
    rows: list[dict],
) -> dict[str, tuple[list[float], int]]:
    """
    Group rows by organ and split each into detected vs missing.

    Returns
    -------
    dict[str, tuple[list[float], int]]
        Mapping organ -> (detected_dice_values, n_missing).
    """
    by_organ: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_organ[r["organ"]].append(r)

    out: dict[str, tuple[list[float], int]] = {}
    for organ, entries in by_organ.items():
        detected: list[float] = []
        missing = 0
        for e in entries:
            if _is_missing(e):
                missing += 1
                continue
            try:
                detected.append(float(e["dice"]))
            except (KeyError, ValueError, TypeError):
                pass
        out[organ] = (detected, missing)
    return out


# ---------------------------------------------------------------------------
# Plotting primitives
# ---------------------------------------------------------------------------

MISSING_BAR_X = -0.045
MISSING_BAR_WIDTH = 0.05
N_BINS = 20


def _draw_histogram(
    ax: plt.Axes,
    detected: list[float],
    n_missing: int,
    color: str = "#5B8DB8",
    label: str | None = None,
    alpha: float = 0.75,
) -> None:
    """
    Draw one bimodal histogram on the given axes.

    Renders the missing bar at x = MISSING_BAR_X (left of zero) and a
    continuous histogram over [0, 1]. Counts are raw (not normalized),
    so the missing bar is directly comparable to the histogram bins.
    """
    bins = np.linspace(0.0, 1.0, N_BINS + 1)
    ax.hist(
        detected,
        bins=bins,
        color=color,
        alpha=alpha,
        edgecolor="white",
        linewidth=0.5,
        label=label,
    )

    if n_missing > 0:
        ax.bar(
            x=MISSING_BAR_X,
            height=n_missing,
            width=MISSING_BAR_WIDTH,
            color=color,
            alpha=alpha,
            edgecolor="black",
            linewidth=0.8,
            hatch="//",
        )

    ax.axvline(x=-0.012, color="black", linewidth=0.5, alpha=0.3)


def _style_axes(
    ax: plt.Axes,
    organ_pretty: str,
    n_total: int,
    n_missing: int,
    show_ylabel: bool = True,
) -> None:
    """Common axis styling, with optional Y-label suppression for grids."""
    ax.set_xlim(-0.08, 1.02)
    ax.set_xlabel("Dice", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Count", fontsize=9)
    ax.set_title(organ_pretty, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax.set_xticks([MISSING_BAR_X, 0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(
        ["missing", "0.0", "0.25", "0.5", "0.75", "1.0"], fontsize=8
    )

    if n_total > 0:
        miss_rate = n_missing / n_total
        ax.text(
            0.98, 0.95,
            f"n={n_total}\nmiss={n_missing} ({miss_rate:.0%})",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="#aaa", alpha=0.85),
        )


# ---------------------------------------------------------------------------
# Per-experiment figure (grid of organs)
# ---------------------------------------------------------------------------

def _plot_experiment_grid(
    organ_data: dict[str, tuple[list[float], int]],
    title: str,
    out_path: Path,
    color: str = "#5B8DB8",
) -> None:
    """
    One figure per experiment with one subplot per organ.
    """
    organs = sorted(organ_data.keys())
    n = len(organs)
    if n == 0:
        return

    fig_w = max(5.5 * n, 6.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.2), squeeze=False)
    axes = axes[0]

    for i, organ in enumerate(organs):
        detected, missing = organ_data[organ]
        n_total = len(detected) + missing
        ax = axes[i]
        _draw_histogram(ax, detected, missing, color=color)
        organ_pretty = organ.replace("_", " ").title()
        _style_axes(
            ax, organ_pretty, n_total, missing,
            show_ylabel=(i == 0),
        )

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_per_experiment(
    results_dir: Path,
    output_dir: Path,
    versions_filter: set[str] | None = None,
    datasets_filter: set[str] | None = None,
    experiments_filter: set[str] | None = None,
) -> int:
    """
    For every (version, dataset, experiment) tuple with a metrics.csv,
    produce one figure with one subplot per organ.

    Optional filters restrict which versions, datasets, and experiments
    are processed. Passing None for a filter means "include all".

    Returns the number of figures saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0

    for version_dir in sorted(results_dir.iterdir()):
        if not version_dir.is_dir() or version_dir.name.startswith("."):
            continue
        version = version_dir.name

        if version_dir.resolve() == output_dir.resolve():
            continue

        # Apply version filter
        if versions_filter and version not in versions_filter:
            continue

        for dataset_dir in sorted(version_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name

            # Apply dataset filter
            if datasets_filter and dataset not in datasets_filter:
                continue

            for exp_dir in sorted(dataset_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                exp = exp_dir.name

                # Apply experiment filter
                if experiments_filter and exp not in experiments_filter:
                    continue

                csv_path = exp_dir / "metrics.csv"
                rows = _load_metrics_csv(csv_path)
                if not rows:
                    continue

                organ_data = _dice_by_organ_split(rows)
                if not organ_data:
                    continue

                title = (
                    f"{_get_display_name(exp)}  |  "
                    f"{dataset}  |  {version}"
                )
                out_path = (
                    output_dir
                    / f"hist_{version}_{dataset}_{exp}.png"
                )
                _plot_experiment_grid(organ_data, title, out_path)
                n_saved += 1
                print(f"  Saved: {out_path}")

    return n_saved


# ---------------------------------------------------------------------------
# Overlay figures (baseline vs pipeline)
# ---------------------------------------------------------------------------

OVERLAY_COLORS = {
    "baseline": "#888888",
    "pipeline": "#D45B5B",
}


def _plot_overlay_grid(
    baseline_data: dict[str, tuple[list[float], int]],
    pipeline_data: dict[str, tuple[list[float], int]],
    baseline_label: str,
    pipeline_label: str,
    title: str,
    out_path: Path,
) -> None:
    """
    Overlay figure with one subplot per organ, comparing baseline against
    one pipeline variant on the same axes.
    """
    organs = sorted(baseline_data.keys())
    n = len(organs)
    if n == 0:
        return

    fig_w = max(5.5 * n, 6.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.5), squeeze=False)
    axes = axes[0]

    for i, organ in enumerate(organs):
        ax = axes[i]
        detected_b, missing_b = baseline_data[organ]
        _draw_histogram(
            ax, detected_b, missing_b,
            color=OVERLAY_COLORS["baseline"],
            label=baseline_label,
            alpha=0.55,
        )
        if organ in pipeline_data:
            detected_p, missing_p = pipeline_data[organ]
            _draw_histogram(
                ax, detected_p, missing_p,
                color=OVERLAY_COLORS["pipeline"],
                label=pipeline_label,
                alpha=0.55,
            )

        n_total = len(detected_b) + missing_b
        organ_pretty = organ.replace("_", " ").title()

        ax.set_xlim(-0.08, 1.02)
        ax.set_xlabel("Dice", fontsize=9)
        if i == 0:
            ax.set_ylabel("Count", fontsize=9)
        ax.set_title(f"{organ_pretty} (n={n_total})",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks([MISSING_BAR_X, 0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(
            ["missing", "0.0", "0.25", "0.5", "0.75", "1.0"], fontsize=8
        )
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_overlays(
    results_dir: Path,
    output_dir: Path,
    overlay_pairs: dict[str, list[str]] | None = None,
    versions_filter: set[str] | None = None,
    datasets_filter: set[str] | None = None,
) -> int:
    """
    For each (version, dataset, baseline_exp) with both baseline and at
    least one pipeline variant available, produce one overlay figure per
    (baseline, pipeline) pair.

    Returns the number of figures saved.
    """
    if overlay_pairs is None:
        overlay_pairs = {**OVERLAY_PAIRS_UNSUP, **OVERLAY_PAIRS_FS}

    output_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0

    for version_dir in sorted(results_dir.iterdir()):
        if not version_dir.is_dir() or version_dir.name.startswith("."):
            continue
        version = version_dir.name

        if version_dir.resolve() == output_dir.resolve():
            continue

        if versions_filter and version not in versions_filter:
            continue

        for dataset_dir in sorted(version_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name

            if datasets_filter and dataset not in datasets_filter:
                continue

            for baseline_exp, pipeline_exps in overlay_pairs.items():
                baseline_csv = dataset_dir / baseline_exp / "metrics.csv"
                baseline_rows = _load_metrics_csv(baseline_csv)
                if not baseline_rows:
                    continue
                baseline_data = _dice_by_organ_split(baseline_rows)

                for pipe_exp in pipeline_exps:
                    pipe_csv = dataset_dir / pipe_exp / "metrics.csv"
                    pipe_rows = _load_metrics_csv(pipe_csv)
                    if not pipe_rows:
                        continue
                    pipeline_data = _dice_by_organ_split(pipe_rows)

                    title = (
                        f"{_get_display_name(baseline_exp)}  ->  "
                        f"{_get_display_name(pipe_exp)}  |  "
                        f"{dataset}  |  {version}"
                    )
                    out_path = (
                        output_dir
                        / f"overlay_{version}_{dataset}_"
                          f"{baseline_exp}_vs_{pipe_exp}.png"
                    )
                    _plot_overlay_grid(
                        baseline_data, pipeline_data,
                        baseline_label=_get_display_name(baseline_exp),
                        pipeline_label=_get_display_name(pipe_exp),
                        title=title,
                        out_path=out_path,
                    )
                    n_saved += 1
                    print(f"  Saved: {out_path}")

    return n_saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Per-organ Dice distribution histograms revealing the bimodal "
            "behavior of co-segmentation pipelines."
        )
    )
    parser.add_argument(
        "--results_dir", default="results",
        help="Root results directory (scans for metrics.csv in subdirs).",
    )
    parser.add_argument(
        "--output", default="results/distributions",
        help="Output directory for histogram figures.",
    )
    parser.add_argument(
        "--versions", nargs="*", default=None,
        help="Version directories to include (default: all). "
             "Example: --versions v3_extended_training_sunny_csf015_msf003",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Dataset directories to include (default: all). "
             "Example: --datasets SunnybrookNicoSent",
    )
    parser.add_argument(
        "--experiments", nargs="*", default=None,
        help="Experiment directories to include (default: all). "
             "Example: --experiments unsup_hdbscan_refine unsup_baseline",
    )
    parser.add_argument(
        "--overlay", action="store_true",
        help=(
            "Also produce baseline-vs-pipeline overlay figures. Skipped "
            "by default to keep first-pass output small."
        ),
    )
    parser.add_argument(
        "--only-overlay", action="store_true",
        help="Skip per-experiment figures, only produce overlays.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output).resolve()

    if not results_dir.exists():
        raise SystemExit(f"results_dir does not exist: {results_dir}")

    versions_filter  = set(args.versions)     if args.versions     else None
    datasets_filter  = set(args.datasets)     if args.datasets     else None
    experiments_filter = set(args.experiments) if args.experiments  else None

    n_per_exp = 0
    n_overlay = 0

    if not args.only_overlay:
        print("\n[1/2] Per-experiment grid histograms...")
        n_per_exp = plot_per_experiment(
            results_dir, output_dir,
            versions_filter=versions_filter,
            datasets_filter=datasets_filter,
            experiments_filter=experiments_filter,
        )

    if args.overlay or args.only_overlay:
        print("\n[2/2] Overlay histograms (baseline vs pipeline)...")
        n_overlay = plot_overlays(
            results_dir, output_dir,
            versions_filter=versions_filter,
            datasets_filter=datasets_filter,
        )

    print(
        f"\nDone. {n_per_exp} per-experiment figures, "
        f"{n_overlay} overlay figures."
    )
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
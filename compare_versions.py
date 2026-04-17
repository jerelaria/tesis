"""
compare_versions.py
-------------------
Generate cross-version, cross-dataset comparison plots.

Scans results/{version}/{dataset}/{experiment}/summary.json and produces:
- Heatmap: global Dice per experiment (rows) x version (columns), one per dataset
- Heatmap: global Dice per experiment (rows) x dataset (columns), one per version
- Grouped bar chart: best experiment per version across datasets
- CSV with all results flattened

Usage:
    python compare_versions.py --results_dir results/ --output results/comparison/

    # Or specify versions explicitly:
    python compare_versions.py \
        --versions v1_baseline v1_standardized \
        --results_dir results/ \
        --output results/comparison/
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import csv
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Styling (reuse conventions from plot_results.py)
# ---------------------------------------------------------------------------

DISPLAY_NAMES = {
    "unsup_kmeans": "Unsup KMeans",
    "unsup_hdbscan": "Unsup HDBSCAN",
    "unsup_kmeans_refine": "Unsup KMeans+R",
    "unsup_hdbscan_refine": "Unsup HDBSCAN+R",
    "fs_indep_1ref": "Indep 1ref",
    "fs_indep_1ref_refine": "Indep 1ref+R",
    "fs_indep_4ref": "Indep 4ref",
    "fs_indep_4ref_refine": "Indep 4ref+R",
    "fs_iter_1ref": "Iter 1ref",
    "fs_iter_4ref": "Iter 4ref",
    "fs_iter_4ref_refine": "Iter 4ref+R",
    "tg": "Text Guided",
}

EXPERIMENT_ORDER = [
    "unsup_kmeans", "unsup_hdbscan",
    "unsup_kmeans_refine", "unsup_hdbscan_refine",
    "fs_indep_1ref", "fs_indep_1ref_refine",
    "fs_indep_4ref", "fs_indep_4ref_refine",
    "fs_iter_1ref", "fs_iter_4ref", "fs_iter_4ref_refine",
    "tg",
]

# Supervision mode color palette
MODE_COLORS = {
    "unsup": "#5B8DB8",
    "fs_indep": "#E8A838",
    "fs_iter": "#6BBF6B",
    "tg": "#D45B5B",
}


def _get_display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def _get_color(name: str) -> str:
    for prefix, color in MODE_COLORS.items():
        if name.startswith(prefix):
            return color
    return "#888888"


def _sort_experiments(experiments: list[str]) -> list[str]:
    """Sort experiments in canonical order, unknowns at the end."""
    order_map = {name: i for i, name in enumerate(EXPERIMENT_ORDER)}
    return sorted(experiments, key=lambda e: order_map.get(e, 999))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(
    results_dir: Path,
    versions: list[str] | None = None,
) -> dict[str, dict[str, dict[str, dict]]]:
    """
    Load all summary.json files from the results directory.

    Returns
    -------
    dict: {version: {dataset: {experiment: summary_dict}}}
    """
    data = {}

    if versions:
        version_dirs = [results_dir / v for v in versions]
    else:
        version_dirs = sorted([
            d for d in results_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
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

    # Print summary
    for version in data:
        for dataset in data[version]:
            n = len(data[version][dataset])
            print(f"  {version}/{dataset}: {n} experiments")

    return data


# ---------------------------------------------------------------------------
# Plot 1: Heatmap — experiments x versions (one per dataset)
# ---------------------------------------------------------------------------

def plot_heatmap_by_dataset(
    data: dict, output_dir: Path, metric: str = "dice",
):
    """
    For each dataset, produce a heatmap with experiments on y-axis
    and versions on x-axis. Cell value = global Dice (or IoU).
    """
    versions = list(data.keys())
    all_datasets = set()
    for v in data.values():
        all_datasets.update(v.keys())

    for dataset in sorted(all_datasets):
        # Collect all experiments across versions for this dataset
        all_exps = set()
        for v in versions:
            all_exps.update(data.get(v, {}).get(dataset, {}).keys())
        experiments = _sort_experiments(list(all_exps))

        if not experiments or len(versions) < 2:
            continue

        # Build matrix
        matrix = np.full((len(experiments), len(versions)), np.nan)
        for j, v in enumerate(versions):
            for i, exp in enumerate(experiments):
                summary = data.get(v, {}).get(dataset, {}).get(exp)
                if summary:
                    val = summary["global"].get(f"{metric}_mean")
                    if val is not None:
                        matrix[i, j] = val

        fig, ax = plt.subplots(
            figsize=(max(6, len(versions) * 2), max(6, len(experiments) * 0.5))
        )

        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        # Annotate cells
        for i in range(len(experiments)):
            for j in range(len(versions)):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = "white" if val < 0.3 or val > 0.85 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=9, color=text_color, fontweight="bold")

        ax.set_xticks(range(len(versions)))
        ax.set_xticklabels(versions, fontsize=10, rotation=30, ha="right")
        ax.set_yticks(range(len(experiments)))
        ax.set_yticklabels(
            [_get_display_name(e) for e in experiments], fontsize=9,
        )

        ax.set_title(
            f"Global {metric.capitalize()} — {dataset}",
            fontsize=14, fontweight="bold",
        )
        fig.colorbar(im, ax=ax, shrink=0.8, label=metric.capitalize())
        fig.tight_layout()

        out_path = output_dir / f"heatmap_{metric}_{dataset}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Heatmap — experiments x datasets (one per version)
# ---------------------------------------------------------------------------

def plot_heatmap_by_version(
    data: dict, output_dir: Path, metric: str = "dice",
):
    """
    For each version, produce a heatmap with experiments on y-axis
    and datasets on x-axis.
    """
    for version, datasets_dict in data.items():
        datasets = sorted(datasets_dict.keys())
        all_exps = set()
        for d in datasets:
            all_exps.update(datasets_dict[d].keys())
        experiments = _sort_experiments(list(all_exps))

        if not experiments or len(datasets) < 2:
            continue

        matrix = np.full((len(experiments), len(datasets)), np.nan)
        for j, ds in enumerate(datasets):
            for i, exp in enumerate(experiments):
                summary = datasets_dict.get(ds, {}).get(exp)
                if summary:
                    val = summary["global"].get(f"{metric}_mean")
                    if val is not None:
                        matrix[i, j] = val

        fig, ax = plt.subplots(
            figsize=(max(6, len(datasets) * 2.5), max(6, len(experiments) * 0.5))
        )

        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        for i in range(len(experiments)):
            for j in range(len(datasets)):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = "white" if val < 0.3 or val > 0.85 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=9, color=text_color, fontweight="bold")

        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, fontsize=10, rotation=30, ha="right")
        ax.set_yticks(range(len(experiments)))
        ax.set_yticklabels(
            [_get_display_name(e) for e in experiments], fontsize=9,
        )

        ax.set_title(
            f"Global {metric.capitalize()} — {version}",
            fontsize=14, fontweight="bold",
        )
        fig.colorbar(im, ax=ax, shrink=0.8, label=metric.capitalize())
        fig.tight_layout()

        out_path = output_dir / f"heatmap_{metric}_{version}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: Version delta — how much does each version improve over baseline
# ---------------------------------------------------------------------------

def plot_version_delta(
    data: dict,
    output_dir: Path,
    baseline_version: str | None = None,
    metric: str = "dice",
):
    """
    Bar chart showing the Dice delta (version - baseline) per experiment,
    one subplot per dataset. Baseline is the first version if not specified.
    """
    versions = list(data.keys())
    if len(versions) < 2:
        print("  [SKIP] Need at least 2 versions for delta plot")
        return

    if baseline_version is None:
        baseline_version = versions[0]
    compare_versions = [v for v in versions if v != baseline_version]

    all_datasets = set()
    for v in data.values():
        all_datasets.update(v.keys())
    datasets = sorted(all_datasets)

    if not datasets:
        return

    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(max(8, len(datasets) * 6), 7),
        squeeze=False,
    )

    for ax_idx, dataset in enumerate(datasets):
        ax = axes[0, ax_idx]

        # Get experiments present in baseline for this dataset
        baseline_exps = data.get(baseline_version, {}).get(dataset, {})
        all_exps = set(baseline_exps.keys())
        for cv in compare_versions:
            all_exps.update(data.get(cv, {}).get(dataset, {}).keys())
        experiments = _sort_experiments(list(all_exps))

        if not experiments:
            ax.set_title(f"{dataset}\n(no data)")
            continue

        x = np.arange(len(experiments))
        width = 0.8 / max(len(compare_versions), 1)

        for v_idx, cv in enumerate(compare_versions):
            deltas = []
            for exp in experiments:
                base_val = (
                    baseline_exps.get(exp, {})
                    .get("global", {})
                    .get(f"{metric}_mean")
                )
                comp_val = (
                    data.get(cv, {}).get(dataset, {}).get(exp, {})
                    .get("global", {})
                    .get(f"{metric}_mean")
                )
                if base_val is not None and comp_val is not None:
                    deltas.append(comp_val - base_val)
                else:
                    deltas.append(0)

            offset = (v_idx - len(compare_versions) / 2 + 0.5) * width
            colors = ["#2E7D32" if d >= 0 else "#C62828" for d in deltas]
            bars = ax.bar(
                x + offset, deltas, width,
                label=cv, color=colors, alpha=0.85,
                edgecolor="white", linewidth=0.5,
            )

            # Value labels
            for bar, delta in zip(bars, deltas):
                if abs(delta) > 0.001:
                    sign = "+" if delta >= 0 else ""
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.005 if delta >= 0 else -0.015),
                        f"{sign}{delta:.3f}",
                        ha="center", va="bottom" if delta >= 0 else "top",
                        fontsize=7, fontweight="bold",
                    )

        ax.set_ylabel(f"{metric.capitalize()} Delta", fontsize=11)
        ax.set_title(f"{dataset}", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_get_display_name(e) for e in experiments],
            fontsize=7, rotation=45, ha="right",
        )
        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
        ax.grid(axis="y", alpha=0.3)
        if len(compare_versions) > 1:
            ax.legend(fontsize=8)

    fig.suptitle(
        f"{metric.capitalize()} Delta vs {baseline_version}",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout()

    out_path = output_dir / f"version_delta_{metric}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 4: Per-organ comparison across versions (one dataset at a time)
# ---------------------------------------------------------------------------

def plot_per_organ_across_versions(
    data: dict, output_dir: Path, metric: str = "dice",
):
    """
    For each dataset, grouped bar chart: organ on x-axis, one bar per version,
    using the best experiment's score for that organ.
    """
    all_datasets = set()
    for v in data.values():
        all_datasets.update(v.keys())

    versions = list(data.keys())

    for dataset in sorted(all_datasets):
        # Collect all organs
        all_organs = set()
        for v in versions:
            for exp_summary in data.get(v, {}).get(dataset, {}).values():
                all_organs.update(exp_summary.get("per_organ", {}).keys())
        organs = sorted(all_organs)

        if not organs:
            continue

        # For each version, find best Dice per organ across experiments
        best_by_version = {}
        for v in versions:
            best = {}
            for organ in organs:
                best_val = 0.0
                for exp_summary in data.get(v, {}).get(dataset, {}).values():
                    organ_data = exp_summary.get("per_organ", {}).get(organ, {})
                    val = organ_data.get(f"{metric}_mean")
                    if val is not None and val > best_val:
                        best_val = val
                best[organ] = best_val
            best_by_version[v] = best

        x = np.arange(len(organs))
        width = 0.8 / max(len(versions), 1)

        fig, ax = plt.subplots(figsize=(max(8, len(organs) * 2), 6))

        for v_idx, v in enumerate(versions):
            offset = (v_idx - len(versions) / 2 + 0.5) * width
            values = [best_by_version[v].get(o, 0) for o in organs]
            ax.bar(
                x + offset, values, width,
                label=v, alpha=0.85,
                edgecolor="white", linewidth=0.5,
            )

        ax.set_ylabel(f"Best {metric.capitalize()}", fontsize=12)
        ax.set_title(
            f"Best Per-Organ {metric.capitalize()} by Version — {dataset}",
            fontsize=14, fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(organs, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        out_path = output_dir / f"per_organ_best_{metric}_{dataset}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Full CSV export
# ---------------------------------------------------------------------------

def save_full_csv(data: dict, output_dir: Path):
    """Flatten all results into a single CSV for easy analysis."""
    rows = []
    for version in data:
        for dataset in data[version]:
            for exp, summary in data[version][dataset].items():
                row = {
                    "version": version,
                    "dataset": dataset,
                    "experiment": exp,
                    "dice_mean": summary["global"].get("dice_mean"),
                    "dice_std": summary["global"].get("dice_std"),
                    "iou_mean": summary["global"].get("iou_mean"),
                    "iou_std": summary["global"].get("iou_std"),
                    "hd95_mean": summary["global"].get("hausdorff_95_mean"),
                    "hd95_std": summary["global"].get("hausdorff_95_std"),
                    "n_images": summary.get("n_images", 0),
                }
                for organ, stats in summary.get("per_organ", {}).items():
                    row[f"{organ}_dice"] = stats.get("dice_mean")
                    row[f"{organ}_missing"] = stats.get("missing", 0)
                rows.append(row)

    if not rows:
        return

    # Stable fieldnames
    fieldnames = [
        "version", "dataset", "experiment",
        "dice_mean", "dice_std", "iou_mean", "iou_std",
        "hd95_mean", "hd95_std", "n_images",
    ]
    extra = set()
    for row in rows:
        extra.update(k for k in row if k not in fieldnames)
    fieldnames.extend(sorted(extra))

    csv_path = output_dir / "all_versions_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare results across versions and datasets."
    )
    parser.add_argument(
        "--results_dir", default="results",
        help="Root results directory (scans for version/dataset/experiment/summary.json)",
    )
    parser.add_argument(
        "--versions", nargs="*",
        help="Specific versions to compare (default: all found in results_dir)",
    )
    parser.add_argument(
        "--baseline", default=None,
        help="Baseline version for delta plots (default: first version found)",
    )
    parser.add_argument(
        "--output", default="results/comparison",
        help="Output directory for comparison plots",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    data = load_all_results(
        Path(args.results_dir),
        versions=args.versions,
    )

    total = sum(
        len(exps)
        for v in data.values()
        for exps in v.values()
    )
    if total == 0:
        print("No results found.")
        return

    versions = list(data.keys())
    print(f"\nVersions: {versions}")
    print(f"Total experiment results: {total}")

    print("\nGenerating comparison plots...")
    plot_heatmap_by_dataset(data, output_dir, metric="dice")
    plot_heatmap_by_dataset(data, output_dir, metric="iou")
    plot_heatmap_by_version(data, output_dir, metric="dice")
    plot_version_delta(data, output_dir, baseline_version=args.baseline)
    plot_per_organ_across_versions(data, output_dir)
    save_full_csv(data, output_dir)

    print(f"\nAll comparison plots saved to {output_dir}")


if __name__ == "__main__":
    main()
"""
plot_results.py
---------------
Generate comparative plots across all experiments.

Reads summary.json from each experiment's results directory and produces:
- Bar chart comparing global Dice/IoU across experiments
- Per-organ Dice comparison (grouped bar chart)
- Per-organ missing rate (how often each organ was not detected)
- Summary table as CSV

Usage:
    python plot_results.py --results_dir results/ --output results/plots/

    # Or specify individual experiments:
    python plot_results.py \
        --experiments results/unsup_kmeans results/fs_indep_1ref results/tg \
        --output results/plots/
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import csv
from pathlib import Path


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

# Experiment display names (short labels for plots)
DISPLAY_NAMES = {
    "unsup_kmeans": "Unsup\nKMeans",
    "unsup_hdbscan": "Unsup\nHDBSCAN",
    "unsup_kmeans_refine": "Unsup\nKMeans+R",
    "unsup_hdbscan_refine": "Unsup\nHDBSCAN+R",
    "fs_indep_1ref": "Indep\n1ref",
    "fs_indep_1ref_refine": "Indep\n1ref+R",
    "fs_indep_4ref": "Indep\n4ref",
    "fs_indep_4ref_refine": "Indep\n4ref+R",
    "fs_iter_1ref": "Iter\n1ref",
    "fs_iter_4ref": "Iter\n4ref",
    "fs_iter_4ref_refine": "Iter\n4ref+R",
    "tg": "Text\nGuided",
}

# Color palette: group by supervision mode
MODE_COLORS = {
    "unsup": "#5B8DB8",
    "fs_indep": "#E8A838",
    "fs_iter": "#6BBF6B",
    "tg": "#D45B5B",
}

def _get_color(name: str) -> str:
    for prefix, color in MODE_COLORS.items():
        if name.startswith(prefix):
            return color
    return "#888888"


def _get_display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


# Ordered list for consistent x-axis
EXPERIMENT_ORDER = [
    "unsup_kmeans", "unsup_hdbscan",
    "unsup_kmeans_refine", "unsup_hdbscan_refine",
    "fs_indep_1ref", "fs_indep_1ref_refine",
    "fs_indep_4ref", "fs_indep_4ref_refine",
    "fs_iter_1ref", "fs_iter_4ref", "fs_iter_4ref_refine",
    "tg",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_summaries(results_dir: Path = None, experiments: list[Path] = None) -> dict:
    """
    Load summary.json from each experiment directory.

    Returns dict mapping experiment name to summary dict.
    """
    summaries = {}

    if experiments:
        dirs = [Path(e) for e in experiments]
    elif results_dir:
        dirs = sorted([
            d for d in results_dir.iterdir()
            if d.is_dir() and (d / "summary.json").exists()
        ])
    else:
        raise ValueError("Must provide either --results_dir or --experiments")

    for d in dirs:
        summary_path = d / "summary.json" if d.is_dir() else d
        if not summary_path.exists():
            print(f"  [SKIP] No summary.json in {d}")
            continue

        with open(summary_path) as f:
            summaries[d.name] = json.load(f)

    # Sort by predefined order
    ordered = {}
    for name in EXPERIMENT_ORDER:
        if name in summaries:
            ordered[name] = summaries[name]
    # Add any not in predefined order
    for name in summaries:
        if name not in ordered:
            ordered[name] = summaries[name]

    print(f"  Loaded {len(ordered)} experiments")
    return ordered


# ---------------------------------------------------------------------------
# Plot 1: Global Dice & IoU comparison
# ---------------------------------------------------------------------------

def plot_global_metrics(summaries: dict, output_dir: Path):
    """Bar chart comparing global Dice and IoU across experiments."""
    names = list(summaries.keys())
    n = len(names)

    dice_means = [s["global"]["dice_mean"] or 0 for s in summaries.values()]
    dice_stds = [s["global"]["dice_std"] or 0 for s in summaries.values()]
    iou_means = [s["global"]["iou_mean"] or 0 for s in summaries.values()]
    iou_stds = [s["global"]["iou_std"] or 0 for s in summaries.values()]

    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), 6))

    bars1 = ax.bar(
        x - width / 2, dice_means, width, yerr=dice_stds,
        label="Dice", color=[_get_color(n) for n in names],
        alpha=0.9, capsize=3, edgecolor="white", linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2, iou_means, width, yerr=iou_stds,
        label="IoU", color=[_get_color(n) for n in names],
        alpha=0.5, capsize=3, edgecolor="white", linewidth=0.5,
    )

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.2f}", ha="center", va="bottom", fontsize=7,
            )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Global Dice & IoU by Experiment", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([_get_display_name(n) for n in names], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "global_dice_iou.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Per-organ Dice comparison
# ---------------------------------------------------------------------------

def plot_per_organ_dice(summaries: dict, output_dir: Path):
    """Grouped bar chart: Dice per organ per experiment."""
    names = list(summaries.keys())
    n = len(names)

    # Collect all organs across experiments
    all_organs = set()
    for s in summaries.values():
        all_organs.update(s["per_organ"].keys())
    organs = sorted(all_organs)

    n_organs = len(organs)
    x = np.arange(n)
    width = 0.8 / n_organs

    organ_colors = {
        "heart": "#D45B5B",
        "left_lung": "#5B8DB8",
        "right_lung": "#6BBF6B",
        "lung": "#5BA8D4",
    }

    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), 6))

    for i, organ in enumerate(organs):
        means = []
        stds = []
        for s in summaries.values():
            organ_data = s["per_organ"].get(organ, {})
            means.append(organ_data.get("dice_mean", 0) or 0)
            stds.append(organ_data.get("dice_std", 0) or 0)

        color = organ_colors.get(organ, f"C{i}")
        offset = (i - n_organs / 2 + 0.5) * width
        ax.bar(
            x + offset, means, width, yerr=stds,
            label=organ, color=color, alpha=0.8, capsize=2,
            edgecolor="white", linewidth=0.5,
        )

    ax.set_ylabel("Dice Score", fontsize=12)
    ax.set_title("Per-Organ Dice by Experiment", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([_get_display_name(n) for n in names], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "per_organ_dice.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: Missing rate per organ
# ---------------------------------------------------------------------------

def plot_missing_rate(summaries: dict, output_dir: Path):
    """Bar chart showing fraction of images where each organ was not detected."""
    names = list(summaries.keys())
    n = len(names)

    all_organs = set()
    for s in summaries.values():
        all_organs.update(s["per_organ"].keys())
    organs = sorted(all_organs)

    n_organs = len(organs)
    x = np.arange(n)
    width = 0.8 / n_organs

    organ_colors = {
        "heart": "#D45B5B",
        "left_lung": "#5B8DB8",
        "right_lung": "#6BBF6B",
        "lung": "#5BA8D4",
    }

    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), 6))

    for i, organ in enumerate(organs):
        rates = []
        for s in summaries.values():
            organ_data = s["per_organ"].get(organ, {})
            count = organ_data.get("count", 1)
            missing = organ_data.get("missing", 0)
            rates.append(missing / count if count > 0 else 0)

        color = organ_colors.get(organ, f"C{i}")
        offset = (i - n_organs / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, rates, width,
            label=organ, color=color, alpha=0.8,
            edgecolor="white", linewidth=0.5,
        )

    ax.set_ylabel("Missing Rate", fontsize=12)
    ax.set_title("Per-Organ Missing Rate by Experiment", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([_get_display_name(n) for n in names], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "missing_rate.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 4: Refinement impact (paired comparison)
# ---------------------------------------------------------------------------

def plot_refinement_impact(summaries: dict, output_dir: Path):
    """
    Show before/after refinement pairs for experiments that have both.
    Grouped by base experiment name.
    """
    # Find pairs: name and name_refine
    pairs = []
    for name in summaries:
        refine_name = name + "_refine"
        if refine_name in summaries:
            pairs.append((name, refine_name))

    if not pairs:
        print("  [SKIP] No refinement pairs found for comparison")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(pairs) * 2.5), 6))

    x = np.arange(len(pairs))
    width = 0.35

    base_dice = [summaries[b]["global"]["dice_mean"] or 0 for b, _ in pairs]
    refine_dice = [summaries[r]["global"]["dice_mean"] or 0 for _, r in pairs]

    ax.bar(x - width / 2, base_dice, width, label="Without Refinement",
           color="#5B8DB8", alpha=0.9, edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, refine_dice, width, label="With Refinement",
           color="#E8A838", alpha=0.9, edgecolor="white", linewidth=0.5)

    # Delta labels
    for i, (b, r) in enumerate(zip(base_dice, refine_dice)):
        delta = r - b
        sign = "+" if delta >= 0 else ""
        ax.text(
            i, max(b, r) + 0.03,
            f"{sign}{delta:.3f}", ha="center", fontsize=9, fontweight="bold",
            color="#2E7D32" if delta >= 0 else "#C62828",
        )

    ax.set_ylabel("Global Dice", fontsize=12)
    ax.set_title("Refinement Impact", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([_get_display_name(b) for b, _ in pairs], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "refinement_impact.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary CSV table
# ---------------------------------------------------------------------------

def save_summary_table(summaries: dict, output_dir: Path):
    """Save a single CSV with all experiments' metrics side by side."""
    rows = []
    for name, s in summaries.items():
        row = {
            "experiment": name,
            "dice_mean": s["global"].get("dice_mean"),
            "dice_std": s["global"].get("dice_std"),
            "iou_mean": s["global"].get("iou_mean"),
            "iou_std": s["global"].get("iou_std"),
            "hd95_mean": s["global"].get("hausdorff_95_mean"),
            "hd95_std": s["global"].get("hausdorff_95_std"),
            "n_images": s.get("n_images", 0),
        }
        # Per-organ dice
        for organ, stats in s.get("per_organ", {}).items():
            row[f"{organ}_dice"] = stats.get("dice_mean")
            row[f"{organ}_missing"] = stats.get("missing", 0)
        rows.append(row)

    if not rows:
        return

    # Collect all field names
    fieldnames = ["experiment", "dice_mean", "dice_std", "iou_mean", "iou_std",
                  "hd95_mean", "hd95_std", "n_images"]
    extra_fields = set()
    for row in rows:
        extra_fields.update(k for k in row if k not in fieldnames)
    fieldnames.extend(sorted(extra_fields))

    csv_path = output_dir / "all_experiments_summary.csv"
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
        description="Plot comparative results across experiments."
    )
    parser.add_argument("--results_dir", default="results",
                        help="Root results directory (scans for summary.json in subdirs)")
    parser.add_argument("--experiments", nargs="*",
                        help="Specific experiment directories to compare")
    parser.add_argument("--output", default="results/plots",
                        help="Output directory for plots")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_summaries(
        results_dir=Path(args.results_dir) if not args.experiments else None,
        experiments=args.experiments,
    )

    if not summaries:
        print("No experiment results found.")
        return

    print(f"\n  Generating plots for {len(summaries)} experiments...")
    print(f"  Experiments: {', '.join(summaries.keys())}")

    plot_global_metrics(summaries, output_dir)
    plot_per_organ_dice(summaries, output_dir)
    plot_missing_rate(summaries, output_dir)
    plot_refinement_impact(summaries, output_dir)
    save_summary_table(summaries, output_dir)

    print(f"\n  All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
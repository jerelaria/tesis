"""
plot_results.py
---------------
Generate comparative plots across all experiments.

Reads summary.json from each experiment's results directory and produces:
- Bar chart comparing global Dice/IoU across experiments (with_missing and
  detected_only variants shown side by side)
- Per-organ Dice comparison (grouped bar chart, with_missing)
- Recall@0.5 per organ (replaces raw missing-rate chart)
- Summary table as CSV

Supports both new-format summaries (dice_mean_with_missing / _detected_only)
and old-format summaries (dice_mean flat key), so existing result directories
remain usable before re-evaluation.

Usage:
    python plot_results.py --results_dir results/ --output results/plots/

    # Or specify individual experiments:
    python plot_results.py \\
        --experiments results/unsup_kmeans results/fs_indep_1ref \\
        --output results/plots/
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import csv
from pathlib import Path

from project.evaluation.display_names import (
    EXPERIMENT_ORDER, MODE_COLORS, get_display_name,
)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def _get_color(name: str) -> str:
    for prefix, color in MODE_COLORS.items():
        if name.startswith(prefix):
            return color
    return "#888888"


def _get_display_name(name: str) -> str:
    return get_display_name(name)


# ---------------------------------------------------------------------------
# Schema-compat helpers
#
# New summaries use dice_mean_with_missing / dice_mean_detected_only.
# Old summaries use dice_mean (flat). These helpers try the new key first
# and fall back to the flat key so both formats are handled transparently.
# ---------------------------------------------------------------------------

def _g(summary: dict, key: str):
    """Read a global key from summary, returning None if absent."""
    return summary.get("global", {}).get(key)


def _g_compat(summary: dict, new_key: str, old_key: str | None = None):
    """
    Try new_key in global section; fall back to old_key for legacy summaries.
    """
    val = _g(summary, new_key)
    if val is None and old_key:
        val = _g(summary, old_key)
    return val


def _organ_compat(organ_stats: dict, new_key: str, old_key: str | None = None):
    """Try new_key in per-organ stats; fall back to old_key."""
    val = organ_stats.get(new_key)
    if val is None and old_key:
        val = organ_stats.get(old_key)
    return val


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
    for name in summaries:
        if name not in ordered:
            ordered[name] = summaries[name]

    print(f"  Loaded {len(ordered)} experiments")
    return ordered


# ---------------------------------------------------------------------------
# Plot 1: Global Dice & IoU comparison (with_missing + detected_only)
# ---------------------------------------------------------------------------

def plot_global_metrics(summaries: dict, output_dir: Path):
    """
    Bar chart comparing global Dice and IoU across experiments.

    For each experiment three bars are shown:
      - Dice (incl. missing)  = dice_mean_with_missing
      - Dice (detected)       = dice_mean_detected_only
      - IoU (incl. missing)   = iou_mean_with_missing

    The detected_only bar is plotted with reduced opacity to visually
    express that it is a conditional metric.  Experiments where
    detected_only is None (all instances missed) show only the first bar.
    """
    names = list(summaries.keys())
    n = len(names)

    dice_wm = []
    dice_wm_std = []
    dice_do = []
    iou_wm = []
    iou_wm_std = []

    for s in summaries.values():
        dice_wm.append(
            _g_compat(s, "dice_mean_with_missing", "dice_mean") or 0.0
        )
        dice_wm_std.append(
            _g_compat(s, "dice_std_with_missing", "dice_std") or 0.0
        )
        do_val = _g_compat(s, "dice_mean_detected_only")
        dice_do.append(do_val)  # may be None
        iou_wm.append(
            _g_compat(s, "iou_mean_with_missing", "iou_mean") or 0.0
        )
        iou_wm_std.append(
            _g_compat(s, "iou_std_with_missing", "iou_std") or 0.0
        )

    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, n * 1.5), 6))

    colors = [_get_color(nm) for nm in names]

    bars_wm = ax.bar(
        x - width, dice_wm, width, yerr=dice_wm_std,
        label="Dice (incl. missing)",
        color=colors, alpha=0.90, capsize=3,
        edgecolor="white", linewidth=0.5,
    )
    # detected_only: skip None entries (draw nothing for fully-missed organs)
    do_vals_plot = [v if v is not None else 0.0 for v in dice_do]
    bars_do = ax.bar(
        x, do_vals_plot, width,
        label="Dice (detected)",
        color=colors, alpha=0.45,
        edgecolor="white", linewidth=0.5,
        hatch="//",
    )
    bars_iou = ax.bar(
        x + width, iou_wm, width, yerr=iou_wm_std,
        label="IoU (incl. missing)",
        color=colors, alpha=0.60, capsize=3,
        edgecolor="white", linewidth=0.5,
    )

    for bar, v, do_v in zip(bars_wm, dice_wm, dice_do):
        if v > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7,
            )
        if do_v is not None and do_v > v:
            # Annotate the gap above the detected_only bar
            do_bar = bars_do[list(dice_wm).index(v)]
            ax.text(
                do_bar.get_x() + do_bar.get_width() / 2, do_v + 0.02,
                f"{do_v:.2f}", ha="center", va="bottom", fontsize=6,
                color="#555",
            )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Global Dice & IoU by Experiment\n"
        "(solid = incl. missing, hatched = detected only)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([_get_display_name(nm) for nm in names], fontsize=8)
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "global_dice_iou.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Per-organ Dice comparison (with_missing)
# ---------------------------------------------------------------------------

def plot_per_organ_dice(summaries: dict, output_dir: Path):
    """Grouped bar chart: Dice (incl. missing) per organ per experiment."""
    names = list(summaries.keys())
    n = len(names)

    all_organs = set()
    for s in summaries.values():
        all_organs.update(s["per_organ"].keys())
    organs = sorted(all_organs)

    n_organs = len(organs)
    x = np.arange(n)
    width = 0.8 / max(n_organs, 1)

    organ_colors = {
        "heart":       "#D45B5B",
        "left_lung":   "#5B8DB8",
        "right_lung":  "#6BBF6B",
        "lv_cavity":   "#9C27B0",
        "myocardium":  "#FF9800",
        "lung":        "#5BA8D4",
    }

    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), 6))

    for i, organ in enumerate(organs):
        means = []
        stds = []
        for s in summaries.values():
            od = s["per_organ"].get(organ, {})
            means.append(
                _organ_compat(od, "dice_mean_with_missing", "dice_mean") or 0.0
            )
            stds.append(
                _organ_compat(od, "dice_std_with_missing", "dice_std") or 0.0
            )

        color = organ_colors.get(organ, f"C{i}")
        offset = (i - n_organs / 2 + 0.5) * width
        ax.bar(
            x + offset, means, width, yerr=stds,
            label=organ.replace("_", " ").title(),
            color=color, alpha=0.8, capsize=2,
            edgecolor="white", linewidth=0.5,
        )

    ax.set_ylabel("Dice (incl. missing)", fontsize=12)
    ax.set_title(
        "Per-Organ Dice by Experiment (incl. missing organs penalised as 0)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([_get_display_name(nm) for nm in names], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "per_organ_dice.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: Recall@0.5 per organ
# ---------------------------------------------------------------------------

def plot_missing_rate(summaries: dict, output_dir: Path):
    """
    Grouped bar chart: Recall@0.5 per organ per experiment.

    Recall@0.5 is the fraction of GT organ instances matched by at least one
    prediction with IoU >= 0.5.  It is the direct counterpart of missing-rate
    (recall@0.5 = 1 - missing_rate) and is easier to interpret: higher = better.

    Falls back to (1 - missing/count) when recall@0.5 is absent (legacy run).
    """
    names = list(summaries.keys())
    n = len(names)

    all_organs = set()
    for s in summaries.values():
        all_organs.update(s["per_organ"].keys())
    organs = sorted(all_organs)

    n_organs = len(organs)
    x = np.arange(n)
    width = 0.8 / max(n_organs, 1)

    organ_colors = {
        "heart":       "#D45B5B",
        "left_lung":   "#5B8DB8",
        "right_lung":  "#6BBF6B",
        "lv_cavity":   "#9C27B0",
        "myocardium":  "#FF9800",
        "lung":        "#5BA8D4",
    }

    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), 6))

    for i, organ in enumerate(organs):
        recalls = []
        for s in summaries.values():
            od = s["per_organ"].get(organ, {})
            val = od.get("recall@0.5")
            if val is None:
                # Legacy fallback: derive from missing / count
                count = od.get("count", 1) or 1
                missing = od.get("missing", 0)
                val = 1.0 - missing / count
            recalls.append(val)

        color = organ_colors.get(organ, f"C{i}")
        offset = (i - n_organs / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, recalls, width,
            label=organ.replace("_", " ").title(),
            color=color, alpha=0.8,
            edgecolor="white", linewidth=0.5,
        )
        for bar, v in zip(bars, recalls):
            if v < 1.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6,
                )

    ax.set_ylabel("Recall @ IoU ≥ 0.5", fontsize=12)
    ax.set_title(
        "Per-Organ Recall@0.5 by Experiment\n"
        "(fraction of GT organs matched by a prediction, higher = better)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([_get_display_name(nm) for nm in names], fontsize=8)
    ax.set_ylim(0, 1.10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "recall_per_organ.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 4: Refinement impact (paired comparison)
# ---------------------------------------------------------------------------

def plot_refinement_impact(summaries: dict, output_dir: Path):
    """
    Show before/after refinement pairs for experiments that have both.
    Uses dice_mean_with_missing (falls back to dice_mean for legacy runs).
    """
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

    base_dice = [
        _g_compat(summaries[b], "dice_mean_with_missing", "dice_mean") or 0.0
        for b, _ in pairs
    ]
    refine_dice = [
        _g_compat(summaries[r], "dice_mean_with_missing", "dice_mean") or 0.0
        for _, r in pairs
    ]

    ax.bar(x - width / 2, base_dice, width, label="Without Refinement",
           color="#5B8DB8", alpha=0.9, edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, refine_dice, width, label="With Refinement",
           color="#E8A838", alpha=0.9, edgecolor="white", linewidth=0.5)

    for i, (b, r) in enumerate(zip(base_dice, refine_dice)):
        delta = r - b
        sign = "+" if delta >= 0 else ""
        ax.text(
            i, max(b, r) + 0.03,
            f"{sign}{delta:.3f}", ha="center", fontsize=9, fontweight="bold",
            color="#2E7D32" if delta >= 0 else "#C62828",
        )

    ax.set_ylabel("Global Dice (incl. missing)", fontsize=12)
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
    """Save a CSV with all experiments' metrics. Includes both split variants."""
    rows = []
    for name, s in summaries.items():
        g = s.get("global", {})
        row = {
            "experiment": name,
            # New-format keys (may be None for old summaries)
            "dice_mean_with_missing":    g.get("dice_mean_with_missing"),
            "dice_std_with_missing":     g.get("dice_std_with_missing"),
            "dice_mean_detected_only":   g.get("dice_mean_detected_only"),
            "dice_std_detected_only":    g.get("dice_std_detected_only"),
            "iou_mean_with_missing":     g.get("iou_mean_with_missing"),
            "iou_std_with_missing":      g.get("iou_std_with_missing"),
            "iou_mean_detected_only":    g.get("iou_mean_detected_only"),
            # Legacy flat keys (populated by old runs; None for new runs)
            "dice_mean":  g.get("dice_mean"),
            "dice_std":   g.get("dice_std"),
            "iou_mean":   g.get("iou_mean"),
            "iou_std":    g.get("iou_std"),
            "hd95_mean":  g.get("hausdorff_95_mean_with_missing")
                          or g.get("hausdorff_95_mean"),
            "hd95_std":   g.get("hausdorff_95_std_with_missing")
                          or g.get("hausdorff_95_std"),
            "recall_05":  g.get("recall@0.5"),
            "precision_05": g.get("precision@0.5"),
            "f1_05":      g.get("f1@0.5"),
            "recall_07":  g.get("recall@0.7"),
            "precision_07": g.get("precision@0.7"),
            "f1_07":      g.get("f1@0.7"),
            "map":        g.get("map"),
            "map_50":     g.get("map_50"),
            "map_75":     g.get("map_75"),
            "n_images":   s.get("n_images", 0),
        }
        for organ, stats in s.get("per_organ", {}).items():
            row[f"{organ}_dice_with_missing"] = _organ_compat(
                stats, "dice_mean_with_missing", "dice_mean"
            )
            row[f"{organ}_dice_detected_only"] = stats.get(
                "dice_mean_detected_only"
            )
            row[f"{organ}_missing"] = stats.get("missing", 0)
            row[f"{organ}_recall_05"] = stats.get("recall@0.5")
        rows.append(row)

    if not rows:
        return

    fieldnames = [
        "experiment",
        "dice_mean_with_missing", "dice_std_with_missing",
        "dice_mean_detected_only", "dice_std_detected_only",
        "iou_mean_with_missing", "iou_std_with_missing",
        "iou_mean_detected_only",
        "dice_mean", "dice_std", "iou_mean", "iou_std",
        "hd95_mean", "hd95_std",
        "recall_05", "precision_05", "f1_05",
        "recall_07", "precision_07", "f1_07",
        "map", "map_50", "map_75",
        "n_images",
    ]
    extra_fields = sorted({k for row in rows for k in row if k not in fieldnames})
    fieldnames.extend(extra_fields)

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

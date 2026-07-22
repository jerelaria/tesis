#!/usr/bin/env python3
"""
scripts/plot_feature_violins_gt.py
-----------------------------------
Violin plots for all 16 moment features coloured by GT organ.

One subplot per feature arranged in a 4x4 grid. Features that were removed
from the pipeline's 12-feature set (ecc, hu1, hu2, orientation) are framed
in red with a "eliminado" annotation so the reader can see why they were
dropped (poor inter-organ discriminability or redundancy).

Requires: seg-cache + gt-dir (same inputs as diagnose_clustering.py).

Usage:
    PYTHONPATH=. python scripts/plot_feature_violins_gt.py \
        --seg-cache  results/_segmentation/Sunnybrook/<hash>/ \
        --gt-dir     data/processed/Sunnybrook/masks \
        --output     results/figures/feature_violins_Sunnybrook.png \
        [--no-standardize]  # plot raw values instead of z-scored
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS))
from diagnose_clustering import (  # noqa: E402
    assign_gt_organ,
    load_objects,
    FEATURE_NAMES,
)
from sklearn.preprocessing import StandardScaler

# Features kept by the pipeline (12-feature set used in experiments).
PIPELINE_FEATURES = {
    "V", "Cx", "Cy", "Dx", "Dy", "L",
    "solidity", "extent", "compact", "hu0",
    "intensity_mean", "intensity_std",
}
REMOVED_FEATURES = set(FEATURE_NAMES) - PIPELINE_FEATURES

ORGAN_COLORS = {
    "lv_cavity":  "#E74C3C",
    "myocardium": "#3498DB",
    "rv_cavity":  "#2ECC71",
    "heart":      "#9B59B6",
    "left_lung":  "#F39C12",
    "right_lung": "#1ABC9C",
}
GRID_ROWS, GRID_COLS = 4, 4

# Display-only relabeling (1-indexed hu moments for the reader), does not
# affect the underlying feature identifiers used elsewhere in the pipeline.
DISPLAY_NAMES = {"hu0": "hu1", "hu1": "hu2", "hu2": "hu3"}


def _display_name(feat_name: str) -> str:
    return DISPLAY_NAMES.get(feat_name, feat_name)


def _organ_color(organ: str) -> str:
    return ORGAN_COLORS.get(organ, "#888888")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--seg-cache", required=True, type=Path)
    ap.add_argument("--gt-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--dataset-name", default=None,
                    help="Dataset name shown in the figure title. "
                         "Defaults to the gt-dir parent folder name.")
    ap.add_argument("--gt-iou-min", type=float, default=0.10)
    ap.add_argument("--no-standardize", action="store_true",
                    help="Plot raw feature values instead of z-scored.")
    ap.add_argument("--exclude-none", action="store_true",
                    help="Exclude objects with gt_organ='none' from violins.")
    args = ap.parse_args()

    dataset_name = args.dataset_name or args.gt_dir.parent.name
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading objects ...")
    objects = load_objects(args.seg_cache)
    ids = list(objects.keys())

    print("Assigning GT organs ...")
    assign_gt_organ(objects, args.gt_dir, args.gt_iou_min)

    # Build feature matrix (N x 16).
    X = np.stack([objects[i]["moments"] for i in ids])
    if not args.no_standardize:
        X = StandardScaler().fit_transform(X)

    organs = np.array([objects[i]["gt_organ"] for i in ids])
    organ_cats = sorted(o for o in set(organs) if not (args.exclude_none and o == "none"))

    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS,
                             figsize=(GRID_COLS * 3.2, GRID_ROWS * 3.2))
    axes_flat = axes.flatten()

    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        ax = axes_flat[feat_idx]
        removed = feat_name in REMOVED_FEATURES

        data_per_organ = []
        labels_for_plot = []
        colors_for_plot = []
        for organ in organ_cats:
            mask = organs == organ
            vals = X[mask, feat_idx]
            if len(vals) == 0:
                continue
            data_per_organ.append(vals)
            labels_for_plot.append(organ)
            colors_for_plot.append(_organ_color(organ))

        if data_per_organ:
            parts = ax.violinplot(data_per_organ, positions=range(len(data_per_organ)),
                                  showmedians=True, showextrema=False)
            for i, (pc, col) in enumerate(zip(parts["bodies"], colors_for_plot)):
                pc.set_facecolor(col)
                pc.set_alpha(0.7)
                pc.set_edgecolor("#333")
                pc.set_linewidth(0.5)
            parts["cmedians"].set_color("#111")
            parts["cmedians"].set_linewidth(1.5)

        ax.set_xticks(range(len(labels_for_plot)))
        ax.set_xticklabels(
            [l.replace("_", "\n") for l in labels_for_plot],
            fontsize=7,
        )
        ax.set_title(_display_name(feat_name), fontsize=9,
                     fontweight="bold",
                     color="#CC0000" if removed else "#000000")

        label_text = "eliminado" if removed else "pipeline"
        edge_col = "#CC0000" if removed else "#2980B9"
        lw = 2.0 if removed else 0.8

        for spine in ax.spines.values():
            spine.set_edgecolor(edge_col)
            spine.set_linewidth(lw)

        if removed:
            ax.text(0.98, 0.97, "✗ eliminado", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color="#CC0000",
                    style="italic")

        ax.tick_params(axis="y", labelsize=7)
        scale_label = "z-score" if not args.no_standardize else "valor crudo"
        ax.set_ylabel(scale_label, fontsize=6)

    # Hide unused subplots if any (16 features fills 4x4 exactly).
    for ax in axes_flat[len(FEATURE_NAMES):]:
        ax.set_visible(False)

    # Legend for organs.
    patches = [mpatches.Patch(color=_organ_color(o), label=o) for o in organ_cats
               if o != "none"]
    if "none" not in organ_cats or not args.exclude_none:
        patches.append(mpatches.Patch(color=_organ_color("none"), label="none (fondo)"))
    fig.legend(handles=patches, loc="lower center",
               ncol=min(len(patches), 6), fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    scale_title = "estandarizadas (z-score)" if not args.no_standardize else "valores crudos"
    fig.suptitle(
        f"{dataset_name} — Distribución de features por órgano GT ({scale_title})\n"
        f"(rojo = eliminado del pipeline; azul = incluido)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

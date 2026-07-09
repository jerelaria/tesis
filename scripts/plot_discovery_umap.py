#!/usr/bin/env python3
"""
scripts/plot_discovery_umap.py
------------------------------
Clean two-panel UMAP figure for the results section:
  left panel  — coloured by HDBSCAN cluster
  right panel — coloured by GT organ

The cluster whose dominant GT organ matches --highlight-organ is drawn with
star markers and on top of all other points in the left panel, and the same
organ is highlighted in the right panel. Designed for Sunnybrook.

Reuses loading, clustering, and projection helpers from
tools/diagnose_clustering.py without duplicating them.

CLI
---
  python scripts/plot_discovery_umap.py \\
      --seg-cache   results/_segmentation/Sunnybrook/<hash>/ \\
      --gt-dir      data/processed/Sunnybrook/masks \\
      --features-csv results_leaf/embeddings_raw/Sunnybrook/unsup_hdbscan_propagation/phase2_clustering/clustering_features.csv \\
      --output      figures/discovery_umap.png \\
      --params      "default:min_cluster_size=136,min_samples=27,method=leaf" \\
      --highlight-organ rv_cavity
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Import helpers from tools/diagnose_clustering (not a package; add dir to path).
_TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS))
from diagnose_clustering import (  # noqa: E402
    NOISE_LABEL,
    assign_gt_organ,
    build_feature_sets,
    cluster_labels,
    load_objects,
    parse_params,
    project_2d,
    resolve_set_params,
)

# Fixed colours for known Sunnybrook organs.
_ORGAN_COLORS = {
    "lv_cavity":  "#E74C3C",
    "myocardium": "#3498DB",
    "rv_cavity":  "#2ECC71",
    "none":       "#CCCCCC",
}


def _organ_color(organ: str) -> str:
    return _ORGAN_COLORS.get(organ, "#9B59B6")


def _cluster_cmap(n: int):
    cmap = plt.cm.tab20 if n > 10 else plt.cm.tab10
    return [cmap(i % cmap.N) for i in range(n)]


def _find_highlight_clusters(clusters, organs, organ: str | None) -> set[int]:
    """Return cluster IDs whose dominant GT label is `organ`."""
    if not organ:
        return set()
    by_cluster: dict[int, Counter] = {}
    for c, o in zip(clusters, organs):
        if c == NOISE_LABEL:
            continue
        by_cluster.setdefault(c, Counter())[o] += 1
    return {
        c for c, counts in by_cluster.items()
        if counts.most_common(1)[0][0] == organ
    }


def scatter_by_cluster(ax, xy: np.ndarray, clusters, highlight: set[int],
                       title: str) -> None:
    """Scatter coloured by cluster; highlighted cluster(s) drawn on top with stars."""
    unique = sorted({c for c in clusters if c != NOISE_LABEL})
    colors = _cluster_cmap(len(unique))
    color_of = {c: colors[i] for i, c in enumerate(unique)}

    noise_mask = np.array([c == NOISE_LABEL for c in clusters])
    if noise_mask.any():
        ax.scatter(xy[noise_mask, 0], xy[noise_mask, 1],
                   s=4, alpha=0.25, color="#BBBBBB",
                   label="noise", rasterized=True)

    regular = [c for c in unique if c not in highlight]
    for c in regular:
        m = np.array([x == c for x in clusters])
        ax.scatter(xy[m, 0], xy[m, 1],
                   s=8, alpha=0.55, color=color_of[c], marker="o",
                   linewidths=0, label=f"C{c}", rasterized=True)

    # Highlighted cluster(s) drawn last (on top).
    for c in sorted(highlight):
        m = np.array([x == c for x in clusters])
        ax.scatter(xy[m, 0], xy[m, 1],
                   s=28, alpha=0.9, color=color_of[c], marker="*",
                   linewidths=0.5, edgecolors="#222",
                   label=f"C{c} ★", rasterized=True)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=1.8, fontsize=8, loc="best",
              framealpha=0.7, handletextpad=0.3, ncol=2)


def scatter_by_organ(ax, xy: np.ndarray, organs, highlight_organ: str | None,
                     title: str) -> None:
    """Scatter coloured by GT organ; highlighted organ drawn on top with stars."""
    cats = sorted(set(organs), key=lambda x: (x == "none", x))

    for cat in cats:
        if cat == highlight_organ:
            continue  # draw highlighted organ last
        m = np.array([o == cat for o in organs])
        is_none = cat == "none"
        ax.scatter(xy[m, 0], xy[m, 1],
                   s=8, alpha=0.3 if is_none else 0.6,
                   color=_organ_color(cat), marker="o",
                   linewidths=0, label=cat, rasterized=True)

    if highlight_organ and highlight_organ in set(organs):
        m = np.array([o == highlight_organ for o in organs])
        ax.scatter(xy[m, 0], xy[m, 1],
                   s=28, alpha=0.9, color=_organ_color(highlight_organ),
                   marker="*", linewidths=0.5, edgecolors="#222",
                   label=f"{highlight_organ} ★", rasterized=True)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=1.8, fontsize=8, loc="best",
              framealpha=0.7, handletextpad=0.3)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--seg-cache", required=True, type=Path,
                    help="Phase 1 cache (segmented.json + segmented.npz).")
    ap.add_argument("--gt-dir", required=True, type=Path,
                    help="GT masks root: <gt-dir>/<stem>/<organ>.png")
    ap.add_argument("--features-csv", type=Path, default=None,
                    help="clustering_features.csv (enables the 'as_clustered' set).")
    ap.add_argument("--feature-set", default="embeddings_raw",
                    choices=["moments_all", "moments_geom",
                             "embeddings_raw", "as_clustered"],
                    help="Feature set to project. Default: embeddings_raw.")
    ap.add_argument("--params", action="append", default=None,
                    help="HDBSCAN params: 'SET:key=val,...' (repeatable). "
                         "SET in {default, moments_all, moments_geom, "
                         "embeddings_raw, as_clustered}.")
    ap.add_argument("--gt-iou-min", type=float, default=0.10,
                    help="Min IoU to assign an object to a GT organ. Default 0.10.")
    ap.add_argument("--umap-neighbors", type=int, default=30,
                    help="UMAP n_neighbors. Default 30.")
    ap.add_argument("--highlight-organ", default="rv_cavity",
                    help="Organ to highlight (right ventricle by default). "
                         "The cluster(s) dominated by this organ are starred in "
                         "the left panel; the organ itself is starred in the right.")
    ap.add_argument("--highlight-cluster", type=int, action="append", default=[],
                    help="Cluster ID to highlight directly in the left panel, "
                         "bypassing the GT-based lookup. Repeatable. "
                         "Use when the organ has no GT (e.g. rv_cavity in Sunnybrook).")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output PNG path.")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading objects ...")
    objects = load_objects(args.seg_cache)
    ids = list(objects.keys())

    print("Assigning GT organs ...")
    assign_gt_organ(objects, args.gt_dir, args.gt_iou_min)
    gt_organs = [objects[i]["gt_organ"] for i in ids]

    print("Building feature sets ...")
    feature_sets = build_feature_sets(objects, ids, args.features_csv)
    if args.feature_set not in feature_sets:
        raise ValueError(
            f"Feature set '{args.feature_set}' not available. "
            f"Available: {sorted(feature_sets)}. "
            f"Pass --features-csv to enable 'as_clustered'."
        )
    X = feature_sets[args.feature_set]

    print("Clustering ...")
    default_over, per_set = parse_params(args.params)
    params = resolve_set_params(args.feature_set, default_over, per_set, len(ids))
    clusters = cluster_labels(X, params)
    n_cl = len({c for c in clusters if c != NOISE_LABEL})
    n_noise = int((clusters == NOISE_LABEL).sum())
    print(f"  {n_cl} clusters, {n_noise}/{len(ids)} noise")

    print("Projecting to 2D ...")
    proj = project_2d(X, args.umap_neighbors)
    xy = proj["umap"] if proj.get("umap") is not None else proj["pca"]
    method = "UMAP" if proj.get("umap") is not None else "PCA"

    if args.highlight_cluster:
        highlight_cl = set(args.highlight_cluster)
        print(f"  Highlighting cluster(s) {sorted(highlight_cl)} (explicit override)")
    else:
        highlight_cl = _find_highlight_clusters(clusters, gt_organs, args.highlight_organ)
        if highlight_cl:
            print(f"  Highlighting cluster(s) {sorted(highlight_cl)} "
                  f"→ dominant organ '{args.highlight_organ}'")
        else:
            print(f"  No cluster dominantly labelled '{args.highlight_organ}' "
                  f"(organ absent from GT or below iou_min). "
                  f"Use --highlight-organ or --highlight-cluster to change.")

    fig, (ax_cl, ax_gt) = plt.subplots(1, 2, figsize=(13, 5.5))

    scatter_by_cluster(
        ax_cl, xy, clusters, highlight_cl,
        f"{method} — coloured by cluster",
    )
    scatter_by_organ(
        ax_gt, xy, gt_organs, args.highlight_organ,
        f"{method} — coloured by GT organ",
    )

    fig.suptitle(
        f"Feature-space structure ({args.feature_set})",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

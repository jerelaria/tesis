"""
Shared cluster visualization primitives.

Imported by both the pipeline (project/pipeline/phase2_debug.py) and the
standalone diagnostic script (tools/explore_leaf_clustering.py).

All functions are pure: they write PNGs or JSON to disk and return nothing.
No imports from project/pipeline or project/segmentation at module level —
this keeps the dependency edge clean.
"""

import logging
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Re-export so callers don't need two imports
from project.evaluation.visualizer import _get_cluster_color  # noqa: F401


# ---------------------------------------------------------------------------
# Generic mask panel
# ---------------------------------------------------------------------------

def save_mask_panel(
    entries: list[tuple[np.ndarray, np.ndarray, int, str]],
    output_path: Path,
    suptitle: str = "",
    n_cols: int = 5,
) -> None:
    """
    Save a grid of image + mask subplots to output_path.

    Parameters
    ----------
    entries : list of (image, mask, cluster_id, title)
        image   : (H, W[, C]) uint8 or float32 array
        mask    : (H, W) bool
        cluster_id : used only for overlay colour (−1 = noise/gray)
        title   : subplot title string
    n_cols : int
        Number of columns in the grid.
    """
    if not entries:
        return

    n_rows = (len(entries) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(5, 4 * n_cols), max(4, 4 * n_rows)),
        squeeze=False,
    )

    for i, (img, mask, cluster_id, title) in enumerate(entries):
        ax = axes[i // n_cols][i % n_cols]
        ax.imshow(img, cmap="gray")

        color = _get_cluster_color(cluster_id, alpha=0.45)
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask] = color
        ax.imshow(overlay)
        ls = "dashed" if cluster_id == -1 else "solid"
        ax.contour(mask, levels=[0.5], colors=[color[:3]], linewidths=1.5, linestyles=ls)

        ax.set_title(title, fontsize=7)
        ax.axis("off")

    for j in range(len(entries), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    logger.debug(f"Saved -> {output_path.name}")


# ---------------------------------------------------------------------------
# Feature violin plots
# ---------------------------------------------------------------------------

def save_feature_violin(
    cluster_id_by_object_id: dict[str, int],
    features_csv: Path,
    output_dir: Path,
    method_label: str = "pipeline",
) -> None:
    """
    Violin plots per feature, one violin per cluster (noise included).

    Also writes a 2-panel scatter (V vs intensity_mean, V vs ecc, Cx vs Cy)
    for sub-structure inspection.

    Parameters
    ----------
    cluster_id_by_object_id : dict[str, int]
        object_id (UUID string) → cluster_id (−1 = noise).
    features_csv : Path
        clustering_features.csv written by ClusteringLabeler.debug_dir.
    output_dir : Path
        Destination directory (created if absent).
    method_label : str
        Used in figure titles and filenames.
    """
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)

    feat_df = pd.read_csv(features_csv)
    feat_df["cluster_id"] = feat_df["object_id"].map(
        lambda oid: cluster_id_by_object_id.get(oid, -1)
    )

    feature_cols = [c for c in feat_df.columns if c not in ("object_id", "cluster_id")]
    unique_ids = sorted(feat_df["cluster_id"].unique())
    labels_map = {cid: ("noise" if cid == -1 else f"cluster_{cid}") for cid in unique_ids}
    label_order = ["noise"] + [f"cluster_{c}" for c in unique_ids if c != -1]

    # tab10 colour palette matching overlay plots
    palette = {
        labels_map[cid]: (
            "#888888" if cid == -1
            else "#{:02x}{:02x}{:02x}".format(
                *[int(255 * v) for v in plt.cm.tab10(cid % plt.cm.tab10.N)[:3]]
            )
        )
        for cid in unique_ids
    }

    feat_df["label"] = feat_df["cluster_id"].map(labels_map)

    # --- Panel 1: violin per feature ---
    n_cols_p = 4
    n_rows_p = max(1, (len(feature_cols) + n_cols_p - 1) // n_cols_p)
    fig, axes = plt.subplots(
        n_rows_p, n_cols_p,
        figsize=(5 * n_cols_p, 3.5 * n_rows_p),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, feat in enumerate(feature_cols):
        ax = axes_flat[i]
        groups = [feat_df.loc[feat_df["label"] == lbl, feat].dropna().values
                  for lbl in label_order]
        # Skip empty groups
        pos = [j for j, g in enumerate(groups) if len(g) > 0]
        grps = [g for g in groups if len(g) > 0]
        ticks = [label_order[j] for j in pos]
        if not grps:
            ax.set_visible(False)
            continue
        parts = ax.violinplot(grps, positions=range(len(grps)),
                              showmedians=True, showextrema=True)
        for pc, lbl in zip(parts["bodies"], ticks):
            pc.set_facecolor(palette.get(lbl, "#888888"))
            pc.set_alpha(0.7)
        ax.set_xticks(range(len(grps)))
        ax.set_xticklabels(ticks, fontsize=7, rotation=20, ha="right")
        ax.set_title(feat, fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    for ax in axes_flat[len(feature_cols):]:
        ax.set_visible(False)

    fig.suptitle(f"Feature distributions by cluster  [{method_label}]", fontsize=12)
    fig.tight_layout()
    out = output_dir / "feature_distributions_violin.png"
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)
    logger.info(f"Violin plot -> {out.name}")

    # --- Panel 2: scatter for sub-structure inspection ---
    noise_df = feat_df[feat_df["cluster_id"] == -1]
    real_df = feat_df[feat_df["cluster_id"] != -1]

    scatter_pairs = [
        (f, g) for f, g in [("V", "intensity_mean"), ("V", "ecc"), ("Cx", "Cy")]
        if f in feature_cols and g in feature_cols
    ]
    if scatter_pairs:
        fig, axes = plt.subplots(1, len(scatter_pairs),
                                 figsize=(6 * len(scatter_pairs), 5), squeeze=False)
        for ax, (fx, fy) in zip(axes[0], scatter_pairs):
            if not real_df.empty:
                ax.scatter(real_df[fx], real_df[fy], c="#3399cc", alpha=0.25, s=8,
                           label="assigned clusters")
            if not noise_df.empty:
                ax.scatter(noise_df[fx], noise_df[fy], c="#cc3333", alpha=0.2, s=8,
                           label="noise")
            ax.set_xlabel(fx)
            ax.set_ylabel(fy)
            ax.set_title(f"{fx} vs {fy}")
            ax.legend(fontsize=7, markerscale=2)
            ax.grid(alpha=0.3)
        fig.suptitle(
            f"Noise sub-structure check  [{method_label}]\n"
            "Red = noise,  Blue = assigned clusters",
            fontsize=10,
        )
        fig.tight_layout()
        out = output_dir / "noise_scatter.png"
        fig.savefig(out, bbox_inches="tight", dpi=130)
        plt.close(fig)
        logger.info(f"Scatter -> {out.name}")


# ---------------------------------------------------------------------------
# SAM memory composition figure
# ---------------------------------------------------------------------------

def save_memory_composition_figure(
    memory_composition: list[dict],
    output_path: Path,
    n_cols: int = 6,
) -> None:
    """
    Grid figure showing every reference frame seeded into SAM2's video predictor.

    Each subplot shows:
    - Source MRI (grayscale)
    - Prototype mask overlay (coloured by cluster_id)
    - Title: frame_idx, obj_id, cluster_id, source filename, area, combined_score

    Parameters
    ----------
    memory_composition : list[dict]
        Each entry: frame_idx, obj_id, cluster_id, source_path,
        mask (np.ndarray), combined_score, area.
    output_path : Path
        Where to save the figure.
    n_cols : int
        Number of columns in the grid.
    """
    if not memory_composition:
        logger.warning("memory_composition is empty; skipping figure.")
        return

    entries = []
    for entry in memory_composition:
        label = entry.get("label", str(entry.get("obj_id", "?")))
        obj_id = entry["obj_id"]
        frame_idx = entry["frame_idx"]
        source_path = entry.get("source_path", "")
        mask = entry["mask"]
        score = entry["combined_score"]
        area = entry["area"]

        if source_path:
            try:
                img = np.array(plt.imread(source_path))
            except Exception:
                img = np.zeros((*mask.shape, 3), dtype=np.uint8)
        else:
            img = np.zeros((*mask.shape, 3), dtype=np.uint8)

        src_name = Path(source_path).name if source_path else "?"
        title = (
            f"frame={frame_idx}  obj_id={obj_id}  label={label}\n"
            f"{src_name}\n"
            f"area={area}  score={score:.3f}"
        )
        entries.append((img, mask, obj_id, title))

    n_frames = len(entries)
    suptitle = (
        f"SAM2 memory composition  —  {n_frames} reference frames\n"
        f"(obj_id = sorted_position(cluster_id) + 1,  SAM rejects obj_id=0)"
    )
    save_mask_panel(entries, output_path, suptitle=suptitle, n_cols=n_cols)
    logger.info(f"Memory composition figure -> {output_path.name}")

"""
Visualization utilities for pipeline debug and output.

Provides two visualization functions:

1. save_segmentation_vis() — Phase 1 output: raw masks overlaid on image,
   colored by index, no clustering info. Used before labeling.

2. save_visualization() — Post-labeling output: masks colored by cluster ID,
   with organ names in legend. Used after clustering, semantic mapping,
   refinement, and filtering.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

from project.core.data_types import SegmentedObject


NOISE_COLOR = (0.5, 0.5, 0.5, 0.3)  # gray for noise (cluster_id == -1)


def _get_cluster_color(cluster_id: int, alpha: float = 0.4) -> tuple:
    """Generate a distinct color for a given cluster_id using a fixed colormap.
    cluster_id == -1 (noise) always returns gray.
    """
    if cluster_id == -1:
        return NOISE_COLOR
    cmap = plt.cm.tab10
    r, g, b, _ = cmap(cluster_id % cmap.N)
    return (r, g, b, alpha)


def _get_object_color(index: int, alpha: float = 0.4) -> tuple:
    """Generate a distinct color for a segmented object by index."""
    cmap = plt.cm.Set3
    r, g, b, _ = cmap(index % cmap.N)
    return (r, g, b, alpha)


# ---------------------------------------------------------------------------
# Phase 1: raw segmentation visualization (before clustering)
# ---------------------------------------------------------------------------

def save_segmentation_vis(
    path: Path,
    objects: list[SegmentedObject],
    output_dir: Path,
    suffix: str = "_seg",
):
    """
    Overlay raw segmentation masks on the original image.
    Each mask is colored by its index (no cluster info at this stage).

    Used to visualize Phase 1 output for debugging: are the masks
    reasonable? Are important organs captured?

    Parameters
    ----------
    path : Path
        Path to the original image file.
    objects : list[SegmentedObject]
        Segmented objects (before clustering).
    output_dir : Path
        Directory to save the visualization.
    suffix : str
        Filename suffix before .png extension.
    """
    image = np.array(plt.imread(str(path)))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image, cmap="gray")

    legend_handles = {}

    for i, obj in enumerate(objects):
        color = _get_object_color(i)
        mask = obj.mask

        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask] = color
        ax.imshow(overlay)

        ax.contour(mask, levels=[0.5], colors=[color[:3]], linewidths=1.0)

        # Label at centroid
        rows, cols = np.where(mask)
        if len(rows) > 0:
            cy, cx = rows.mean(), cols.mean()
            label_text = obj.label if obj.label else str(i)
            conf_text = f"{obj.confidence:.2f}" if obj.confidence else ""
            ax.text(
                cx, cy, f"{label_text}\n{conf_text}",
                color="white", fontsize=7, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc=color[:3], alpha=0.7),
            )

            display_label = obj.label if obj.label else f"obj_{i}"
            if display_label not in legend_handles:
                legend_handles[display_label] = mpatches.Patch(
                    color=color, label=display_label
                )

    handles = sorted(legend_handles.values(), key=lambda h: h.get_label())
    ax.legend(handles=handles, loc="upper right", fontsize=7)
    ax.set_title(f"{path.name} — {len(objects)} objects")
    ax.axis("off")

    out_path = output_dir / f"{path.stem}{suffix}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Post-labeling visualization (after clustering / refinement / filtering)
# ---------------------------------------------------------------------------

def save_visualization(
    path: Path,
    labeled: list,
    results_dir: Path,
    suffix: str = "_labeled",
):
    """
    Overlay colored masks on the original image with cluster labels.

    Parameters
    ----------
    path : Path
        Path to the original image file.
    labeled : list[LabeledObject]
        Labeled objects with cluster IDs and organ names.
    results_dir : Path
        Directory to save the visualization.
    suffix : str
        Filename suffix before .png extension.
    """
    image = np.array(plt.imread(str(path)))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image, cmap="gray")

    legend_handles = {}

    for obj in labeled:
        cid   = -1 if obj.is_noise else obj.organ_id
        label = "noise" if obj.is_noise else obj.organ_name
        color = _get_cluster_color(cid)

        mask = obj.segmented_object.mask
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask] = color
        ax.imshow(overlay)

        if obj.is_noise:
            ax.contour(
                mask, levels=[0.5], colors=["white"],
                linewidths=1.5, linestyles="dashed",
            )
        else:
            ax.contour(mask, levels=[0.5], colors=[color[:3]], linewidths=1.5)

        # Cluster id label at centroid
        rows, cols = np.where(mask)
        cy, cx = rows.mean(), cols.mean()
        label_text = "-" if obj.is_noise else str(obj.organ_id)
        ax.text(
            cx, cy, label_text,
            color="white", fontsize=9, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc=color[:3], alpha=0.7),
        )

        if cid not in legend_handles:
            legend_handles[cid] = mpatches.Patch(color=color, label=label)

    legend_handles = sorted(legend_handles.values(), key=lambda h: h.get_label())
    ax.legend(handles=legend_handles, loc="upper right")
    ax.set_title(path.name)
    ax.axis("off")

    out_path = results_dir / f"{path.stem}{suffix}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    Saved -> {out_path}")
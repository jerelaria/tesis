import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path


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


def save_visualization(path: Path, labeled: list, results_dir: Path):
    """Overlay colored masks on the original image and save to results_dir."""
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
            ax.contour(mask, levels=[0.5], colors=["white"], linewidths=1.5, linestyles="dashed")
        else:
            ax.contour(mask, levels=[0.5], colors=[color[:3]], linewidths=1.5)

        # Cluster id label at centroid
        rows, cols = np.where(mask)
        cy, cx = rows.mean(), cols.mean()
        label_text = "-" if obj.is_noise else str(obj.organ_id)
        ax.text(cx, cy, label_text,
                color="white", fontsize=9, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc=color[:3], alpha=0.7))

        if cid not in legend_handles:
            legend_handles[cid] = mpatches.Patch(color=color, label=label)

    legend_handles = sorted(legend_handles.values(), key=lambda h: h.get_label())
    ax.legend(handles=legend_handles, loc="upper right")
    ax.set_title(path.name)
    ax.axis("off")

    out_path = results_dir / f"{path.stem}_labeled.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")
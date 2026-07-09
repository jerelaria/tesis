"""make_mask_overlay_figure.py

Save a single figure with several binary organ masks overlaid (as colored,
semi-transparent regions) on top of one grayscale image, with a legend.

Usage:
    python make_mask_overlay_figure.py \
        --image data/processed/ACDC/images/patient001_frame01_slice_1.png \
        --mask lv_cavity:results/acdc_transfer/masks/patient001_frame01_slice_1/lv_cavity.png:"Cavidad VI" \
        --mask myocardium:results/acdc_transfer/masks/patient001_frame01_slice_1/myocardium.png:"Miocardio" \
        --mask rv_cavity:results/acdc_transfer/masks/patient001_frame01_slice_1/rv_cavity.png:"Cavidad VD" \
        --output results/figures/acdc_transfer.png \
        --title "Transferencia Sunnybrook -> ACDC (few-shot)"
"""
import argparse
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3"]


def parse_mask_arg(spec):
    key, path, label = spec.split(":", 2)
    return key, path, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Grayscale base image")
    ap.add_argument(
        "--mask", action="append", required=True,
        help='One per organ, format "key:mask_path:legend label" (repeatable)',
    )
    ap.add_argument("--output", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--alpha", type=float, default=0.45)
    args = ap.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(image, cmap="gray")

    legend_handles = []
    for i, spec in enumerate(args.mask):
        key, mask_path, label = parse_mask_arg(spec)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        color = COLORS[i % len(COLORS)]

        rgba = np.zeros((*mask.shape, 4))
        rgb = np.array(plt.matplotlib.colors.to_rgb(color))
        present = mask > 0
        rgba[present, :3] = rgb
        rgba[present, 3] = args.alpha
        ax.imshow(rgba)

        legend_handles.append(mpatches.Patch(color=color, label=label))

    ax.set_title(args.title)
    ax.axis("off")
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.85)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()

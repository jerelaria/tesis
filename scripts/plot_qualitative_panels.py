#!/usr/bin/env python3
"""
scripts/plot_qualitative_panels.py
-----------------------------------
Qualitative result panels for the thesis. For each selected case, one row:

    Image  |  GT (organ overlay)  |  Best-match prediction (overlay)

The prediction with the highest IoU against the requested GT organ is
selected automatically, so prediction file naming does not matter.

Works with any dataset that follows the standard mask layout:
    <run-dir>/<stem>/<pred>.png      — predictions (one or more per stem)
    <gt-dir>/<stem>/<organ>.png      — GT organ masks

Point --run-dir at ACDC results (same layout) to produce the RV-propagation
figure; only --organ and --images-dir need to change.

CLI
---
  python scripts/plot_qualitative_panels.py \\
      --run-dir    results_leaf/embeddings_raw/Sunnybrook/unsup_hdbscan_propagation/masks \\
      --gt-dir     data/processed/Sunnybrook/masks \\
      --images-dir data/raw/Sunnybrook/images \\
      --organ      lv_cavity \\
      --n          6 \\
      --output     figures/qualitative_Sunnybrook_lv.png

  # Explicit stems (useful for picking specific thesis cases):
      --stems SCD0000901_IM_0003_0140 SCD0001501_IM_0002_0060

  # RV propagation to ACDC (same script, different paths):
      --run-dir    results_leaf/.../masks \\
      --gt-dir     data/processed/ACDC/masks \\
      --images-dir data/raw/ACDC/images \\
      --organ      rv_cavity
"""

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from project.evaluation.io import load_masks_from_dir

# RGBA overlays (values in [0, 1]).
_GT_COLOR = (0.13, 0.70, 0.17, 0.45)     # green
_PRED_COLOR = (0.90, 0.30, 0.07, 0.45)   # orange-red


def _load_image(images_dir: Path, stem: str) -> np.ndarray | None:
    """Try common extensions; return HxWx3 uint8 array or None."""
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        p = images_dir / (stem + ext)
        if p.exists():
            return np.array(Image.open(p).convert("RGB"))
    return None


def _overlay(img: np.ndarray, mask: np.ndarray, rgba) -> np.ndarray:
    """Alpha-composite a binary mask on an HxWx3 uint8 image."""
    r, g, b, a = rgba
    out = img.astype(np.float32) / 255.0
    out[mask, 0] = out[mask, 0] * (1 - a) + r * a
    out[mask, 1] = out[mask, 1] * (1 - a) + g * a
    out[mask, 2] = out[mask, 2] * (1 - a) + b * a
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def _resize_to(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.array(
        Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
    ) > 127


def _best_prediction(
    preds: dict[str, np.ndarray],
    gt: np.ndarray,
) -> tuple[str | None, np.ndarray | None, float]:
    """Return (name, mask, iou) for the prediction with highest IoU vs gt."""
    h, w = gt.shape
    best_name, best_mask, best_iou = None, None, -1.0
    for name, pm in preds.items():
        if pm.shape != (h, w):
            pm = _resize_to(pm, h, w)
        v = _iou(pm, gt)
        if v > best_iou:
            best_name, best_mask, best_iou = name, pm, v
    return best_name, best_mask, best_iou


def collect_stems(run_dir: Path, gt_dir: Path, organ: str) -> list[str]:
    """
    Return stems that have a predictions directory in run_dir AND a GT file
    for `organ` in gt_dir, sorted alphabetically.
    """
    stems = []
    for d in sorted(run_dir.iterdir()):
        if not d.is_dir():
            continue
        if (gt_dir / d.name / f"{organ}.png").exists():
            stems.append(d.name)
    return stems


def _annotate(ax, text: str, loc: str = "tl", fontsize: int = 7) -> None:
    """Add a small text annotation inside an axis (tl=top-left, bc=bottom-center)."""
    x, y, ha, va = {
        "tl": (0.03, 0.97, "left",   "top"),
        "bc": (0.50, 0.03, "center", "bottom"),
    }[loc]
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=fontsize, color="white",
            ha=ha, va=va,
            bbox=dict(boxstyle="round,pad=0.2", fc="#00000077", ec="none"))


def render_row(axes, stem: str, images_dir: Path, run_dir: Path,
               gt_dir: Path, organ: str) -> float:
    """
    Fill axes[0..2] (image / GT overlay / prediction overlay) for one case.
    Returns the IoU of the best-matching prediction (0.0 if none found).
    """
    ax_img, ax_gt, ax_pred = axes

    img = _load_image(images_dir, stem)
    gt_masks = load_masks_from_dir(gt_dir / stem)
    pred_masks = load_masks_from_dir(run_dir / stem)

    gt_mask = gt_masks.get(organ)
    if gt_mask is None:
        raise KeyError(
            f"Organ '{organ}' not in GT for stem '{stem}'. "
            f"Available: {sorted(gt_masks)}."
        )

    h, w = gt_mask.shape
    if img is not None and img.shape[:2] != (h, w):
        img = np.array(Image.fromarray(img).resize((w, h)))

    # Panel 1: raw image (or grey placeholder when missing).
    if img is not None:
        ax_img.imshow(img)
    else:
        ax_img.imshow(np.full((h, w, 3), 180, dtype=np.uint8))
        _annotate(ax_img, "no image", "tl")
    ax_img.axis("off")
    _annotate(ax_img, stem[:22], "tl")

    # Panel 2: GT overlay on image.
    base = img if img is not None else np.full((h, w, 3), 180, dtype=np.uint8)
    ax_gt.imshow(_overlay(base, gt_mask, _GT_COLOR))
    ax_gt.axis("off")

    # Panel 3: best-match prediction overlay.
    pred_name, pred_mask, iou = _best_prediction(pred_masks, gt_mask)
    if pred_mask is not None:
        ax_pred.imshow(_overlay(base, pred_mask, _PRED_COLOR))
        _annotate(ax_pred, f"IoU={iou:.2f}", "bc")
    else:
        ax_pred.imshow(base)
        ax_pred.text(0.5, 0.5, "no prediction", ha="center", va="center",
                     transform=ax_pred.transAxes, fontsize=9, color="#aaa")
    ax_pred.axis("off")

    return iou


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="Prediction masks root: <stem>/<pred>.png per image.")
    ap.add_argument("--gt-dir", required=True, type=Path,
                    help="GT masks root: <stem>/<organ>.png per image.")
    ap.add_argument("--images-dir", required=True, type=Path,
                    help="Raw images directory: <stem>.<ext>")
    ap.add_argument("--organ", required=True,
                    help="GT organ to focus on (e.g. lv_cavity, rv_cavity).")
    ap.add_argument("--n", type=int, default=6,
                    help="Number of cases to display. Default: 6.")
    ap.add_argument("--sort", choices=["alpha", "random"], default="alpha",
                    help="How to order cases when more than --n are available. "
                         "Default: alpha.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for --sort random. Default: 42.")
    ap.add_argument("--stems", nargs="*", default=None,
                    help="Explicit stem list; overrides --n and --sort.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output PNG path.")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.stems:
        stems = args.stems
    else:
        stems = collect_stems(args.run_dir, args.gt_dir, args.organ)
        if not stems:
            raise RuntimeError(
                f"No stems found in '{args.run_dir}' that also have GT organ "
                f"'{args.organ}' in '{args.gt_dir}'."
            )
        if args.sort == "random":
            random.seed(args.seed)
            random.shuffle(stems)
        stems = stems[: args.n]

    n = len(stems)
    organ_title = args.organ.replace("_", " ").title()

    fig, axes = plt.subplots(n, 3, figsize=(9.5, n * 3.1), squeeze=False)

    # Column headers on row 0.
    for col, hdr in enumerate(["Image", f"GT: {organ_title}", "Prediction"]):
        axes[0][col].set_title(hdr, fontsize=11, fontweight="bold", pad=5)

    ious = []
    for row, stem in enumerate(stems):
        iou = render_row(axes[row], stem, args.images_dir,
                         args.run_dir, args.gt_dir, args.organ)
        ious.append(iou)

    # Legend.
    gt_patch = mpatches.Patch(color=_GT_COLOR[:3], alpha=0.7, label="GT organ")
    pred_patch = mpatches.Patch(color=_PRED_COLOR[:3], alpha=0.7, label="Prediction")
    fig.legend(handles=[gt_patch, pred_patch], fontsize=9,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.005))

    mean_iou = float(np.mean([v for v in ious if v > 0])) if ious else 0.0
    fig.suptitle(
        f"Qualitative results — {organ_title}  (mean IoU={mean_iou:.2f})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}  ({n} cases, mean IoU={mean_iou:.3f})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Figure 4 - Per-dataset example panel, one row per dataset.

    JSRT        chest X-ray with left lung, right lung and heart annotations
                (use an example where the heart IS annotated).
    Sunnybrook  short-axis cardiac MRI with LV cavity and myocardium, plus the
                right ventricle marked as visible-but-absent-from-GT (dashed).
    ACDC        short-axis slice with the three annotations including the right
                ventricle, to show what it adds over Sunnybrook.

Two sources (see SOURCE):
    "synthetic"  self-contained phantoms, runs with no data. Locks the layout.
    "real"       loads images and masks from the paths in DATASETS.

Mask inputs per dataset accept either a single integer label map (with a
label->structure dict) or one binary mask file per structure. Supported file
types: .png/.jpg (PIL), .npy (numpy), .nii/.nii.gz (nibabel, needs SLICE).

Outputs a single vector PDF (fig4_ejemplos_datasets.pdf).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SOURCE = "real"               # global default; each DATASETS entry may override
                              # it with its own "source": "synthetic" | "real"
OUT = Path("figuras_out/fig4_ejemplos_datasets.pdf")
DPI = 300

# Per-structure colors (RGB 0-1).
COLORS = {
    "left_lung":  (0.20, 0.55, 0.90),
    "right_lung": (0.95, 0.60, 0.15),
    "heart":      (0.85, 0.20, 0.25),
    "lv_cavity":  (0.20, 0.55, 0.90),
    "myocardium": (0.30, 0.75, 0.40),
    "rv_cavity":  (0.70, 0.35, 0.80),
}
LABELS_ES = {
    "left_lung": "Pulmón izquierdo", "right_lung": "Pulmón derecho",
    "heart": "Corazón", "lv_cavity": "Cavidad VI", "myocardium": "Miocardio",
    "rv_cavity": "Cavidad VD",
}

# Real-data configuration. Fill in your own paths and choose ONE of:
#   "labelmap": path + "label_to_structure": {int: name}
#   "masks":    {structure: path}  (one binary file each)
# For NIfTI volumes set "slice" to the 2D index to show.
DATASETS = [
    {
        # JSRT chest X-ray (JPCLN* = JSRT). Per-structure binary masks.
        "name": "JSRT",
        "image": "data/few_shot/XRay/307400577352691476839033094654629005292_2_eyg2tu/image.png",
        "masks": {
            "left_lung":  "data/few_shot/XRay/307400577352691476839033094654629005292_2_eyg2tu/left_lung.png",
            "right_lung": "data/few_shot/XRay/307400577352691476839033094654629005292_2_eyg2tu/right_lung.png",
            "heart":      "data/few_shot/XRay/307400577352691476839033094654629005292_2_eyg2tu/heart.png",
        },
        "structures": ["left_lung", "right_lung", "heart"],
        "rv_outline": None,
    },
    {
        "name": "Sunnybrook",
        "image": "data/raw/Sunnybrook/images/SCD0000401_IM_0002_0120.png",
        # Sunnybrook GT is a single label map: 1 = LV cavity, 2 = myocardium.
        "masks": {
            "labelmap": "data/raw/Sunnybrook/masks/SCD0000401_IM_0002_0120.png",
            "label_to_structure": {1: "lv_cavity", 2: "myocardium"},
        },
        "structures": ["lv_cavity", "myocardium"],
        # RV is visible but absent from the Sunnybrook GT. We borrow its location
        # from the unsupervised segmentation (cluster_4 of this same slice) to
        # draw the dashed "visible but unannotated" outline.
        "rv_outline": {
            "mask": "results/moments/Sunnybrook/unsup_hdbscan_propagation/"
                    "masks/SCD0000401_IM_0002_0120/cluster_4.png",
        },
    },
    {
        # ACDC short-axis slice with all three structures as per-structure masks.
        "name": "ACDC",
        "image": "data/raw/ACDC/images/patient001_frame01_slice_3.png",
        "masks": {
            "lv_cavity":  "data/raw/ACDC/masks/patient001_frame01_slice_3/lv_cavity.png",
            "myocardium": "data/raw/ACDC/masks/patient001_frame01_slice_3/myocardium.png",
            "rv_cavity":  "data/raw/ACDC/masks/patient001_frame01_slice_3/rv_cavity.png",
        },
        "structures": ["lv_cavity", "myocardium", "rv_cavity"],
        "rv_outline": None,
    },
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_2d(path, slice_idx=None):
    """Load a 2D array from png/jpg/npy/nifti."""
    path = str(path)
    if path.endswith((".png", ".jpg", ".jpeg")):
        from PIL import Image
        return np.asarray(Image.open(path).convert("L"), dtype=float)
    if path.endswith(".npy"):
        arr = np.load(path)
    elif path.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        arr = np.asarray(nib.load(path).dataobj)
        if arr.ndim == 3:
            arr = arr[:, :, slice_idx]
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return arr.astype(float)


def load_real(ds):
    """Return (image, {structure: bool_mask}) for one real dataset entry."""
    sl = ds.get("slice")
    img = _load_2d(ds["image"], sl)
    masks = {}
    mconf = ds["masks"]
    if "labelmap" in mconf:
        lab = _load_2d(mconf["labelmap"], sl)
        for value, name in mconf["label_to_structure"].items():
            masks[name] = lab == value
    else:
        for name, path in mconf.items():
            masks[name] = _load_2d(path, sl) > 0
    return img, masks


# ---------------------------------------------------------------------------
# Synthetic phantoms
# ---------------------------------------------------------------------------
def _ellipse(h, w, cx, cy, rx, ry, angle=0.0):
    yy, xx = np.mgrid[0:h, 0:w]
    xr = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
    yr = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
    return (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0


def _bg(h, w, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = 0.25 + 0.12 * np.exp(-(((xx - w / 2) / (0.5 * w)) ** 2
                                 + ((yy - h / 2) / (0.5 * h)) ** 2))
    return np.clip(img + 0.03 * rng.standard_normal((h, w)), 0, 1)


def synth_jsrt(h=320, w=320):
    img = _bg(h, w, 1)
    return img, {
        "left_lung":  _ellipse(h, w, 0.34 * w, 0.5 * h, 0.13 * w, 0.26 * h, 0.1),
        "right_lung": _ellipse(h, w, 0.66 * w, 0.5 * h, 0.13 * w, 0.26 * h, -0.1),
        "heart":      _ellipse(h, w, 0.5 * w, 0.62 * h, 0.10 * w, 0.12 * h),
    }


def synth_sunnybrook(h=256, w=256):
    img = _bg(h, w, 2)
    lv = _ellipse(h, w, 0.42 * w, 0.52 * h, 0.10 * w, 0.10 * h)
    myo = _ellipse(h, w, 0.42 * w, 0.52 * h, 0.16 * w, 0.16 * h) & ~lv
    return img, {"lv_cavity": lv, "myocardium": myo}


def synth_acdc(h=256, w=256):
    img = _bg(h, w, 3)
    lv = _ellipse(h, w, 0.42 * w, 0.52 * h, 0.10 * w, 0.10 * h)
    myo = _ellipse(h, w, 0.42 * w, 0.52 * h, 0.16 * w, 0.16 * h) & ~lv
    rv = _ellipse(h, w, 0.62 * w, 0.50 * h, 0.14 * w, 0.20 * h, 0.4) & ~myo & ~lv
    return img, {"lv_cavity": lv, "myocardium": myo, "rv_cavity": rv}


SYNTH = {"JSRT": synth_jsrt, "Sunnybrook": synth_sunnybrook, "ACDC": synth_acdc}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def draw_row(ax, img, masks, structures, name, rv_outline):
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel(name, fontsize=11, rotation=90, labelpad=8)

    handles = []
    for s in structures:
        if s not in masks:
            continue
        color = COLORS[s]
        overlay = np.zeros((*img.shape, 4))
        overlay[masks[s]] = (*color, 0.40)
        ax.imshow(overlay, interpolation="nearest")
        ax.contour(masks[s], levels=[0.5], colors=[color], linewidths=1.4)
        handles.append(Line2D([0], [0], color=color, lw=3, label=LABELS_ES[s]))

    # Right ventricle marked as visible-but-unannotated (Sunnybrook).
    if rv_outline is not None:
        if "ellipse" in rv_outline:
            cx, cy, rx, ry = rv_outline["ellipse"]
            rv = _ellipse(img.shape[0], img.shape[1], cx, cy, rx, ry)
        else:
            rv = _load_2d(rv_outline["mask"]) > 0
        ax.contour(rv, levels=[0.5], colors=["#d0d0d0"], linewidths=1.6,
                   linestyles="dashed")
        ys, xs = np.where(rv)
        cy, cx = ys.mean(), xs.mean()
        ax.annotate("VD: visible,\nausente del GT", xy=(cx, cy),
                    xytext=(cx + 0.30 * img.shape[1], cy - 0.18 * img.shape[0]),
                    fontsize=8.5, color="white", ha="left",
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.4),
                    bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.6))
        handles.append(Line2D([0], [0], color="#d0d0d0", lw=2, ls="--",
                              label="Cavidad VD (no anotado)"))

    ax.legend(handles=handles, loc="lower right", fontsize=7.5,
              framealpha=0.85, handlelength=1.2)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(5.0, 13.5))
    if len(DATASETS) == 1:
        axes = [axes]

    for ax, ds in zip(axes, DATASETS):
        # Each row may override the global SOURCE via its own "source" key.
        src = ds.get("source", SOURCE)
        if src == "synthetic":
            img, masks = SYNTH[ds["name"]]()
        else:
            img, masks = load_real(ds)
        draw_row(ax, img, masks, ds["structures"], ds["name"], ds.get("rv_outline"))

    fig.tight_layout()
    fig.savefig(OUT, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT.resolve()}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Figure 1 - Grid prompting illustration in three SEPARATE panels.

Generates three independent PNG files:
    panel_a_grid.png      image with a uniform grid of prompt points
    panel_b_raw.png       raw candidate masks (overlaps + spurious visible)
    panel_c_clean.png     depurated set (low-score dropped + overlap NMS)

The three files are meant to be composed later on a canvas and connected
with arrows by hand.

Two input sources are supported (see SOURCE below):
    "synthetic"  self-contained illustrative phantom, runs with no data/model.
                 Use it to lock the layout, then switch to "real".
    "real"       load a real example image and the raw masks/scores produced
                 by the MedSAM2 grid stage. Wire `load_real_inputs` to your
                 own segmentation cache.

Only numpy and matplotlib are required.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SOURCE = "medsam2"       # "synthetic" | "real" | "medsam2"
OUT_DIR = Path("figuras_out")  # output directory for the three PNGs
GRID_N = 6                    # points per side of the uniform grid
SCORE_THR = 0.50             # drop candidates below this segmentation score
NMS_IOU_THR = 0.50           # two masks above this IoU are considered redundant
DPI = 300
SHOW_GRID_IN_B = True        # overlay the prompt grid on the raw-masks panel

# ---- "real" source: read masks back from the depurated segmentation cache ----
# The cache is a (segmented.json, segmented.npz) pair:
#   - segmented.json lists every image with its `objects` (each object has a
#     `confidence` and a `mask_key`).
#   - segmented.npz stores one boolean (H, W) mask per `mask_key`.
# NOTE: this cache is ALREADY post score-threshold + NMS, so panels B and C
# come out identical. Use SOURCE == "medsam2" to get the raw per-point masks.
REAL_CACHE_DIR = "/media/apoloml/DATOS_2/Tesis_Cosegmentacion/results/_segmentation/XRay/unsupervised_grid6_st0.50_iou0.50_n911__18298beb"
REAL_IMAGE_STEM = "CHNCXR_0159_0"   # stem of the image to use from the cache

# ---- "medsam2" source: run the grid prompting live and keep EVERY mask -------
# One mask per grid point, before any depuration -> a genuine "raw" panel B.
MEDSAM2_IMAGE_PATH = "/media/apoloml/DATOS_2/Tesis_Cosegmentacion/data/raw/XRay/images/CHNCXR_0159_0.png"
MEDSAM2_DEVICE = "cuda"             # "cuda" | "mps" | "cpu"
MEDSAM2_RAW_NPZ = None             # set to a path to cache/reuse raw masks; None recomputes


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def make_grid_points(h, w, n, margin=0.08):
    """Uniform n x n grid of (x, y) points inside a margin of the image."""
    xs = np.linspace(margin * w, (1 - margin) * w, n)
    ys = np.linspace(margin * h, (1 - margin) * h, n)
    return np.array([(x, y) for y in ys for x in xs], dtype=float)


def iou(a, b):
    """Intersection over union between two boolean masks."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def depurate(masks, scores, score_thr, nms_iou_thr):
    """Drop low-score masks, then greedy IoU non-maximum suppression."""
    # 1) score threshold
    keep_idx = [i for i, s in enumerate(scores) if s >= score_thr]
    keep_idx.sort(key=lambda i: scores[i], reverse=True)
    # 2) overlap NMS
    selected = []
    for i in keep_idx:
        if all(iou(masks[i], masks[j]) < nms_iou_thr for j in selected):
            selected.append(i)
    return selected


# ---------------------------------------------------------------------------
# Synthetic illustrative inputs
# ---------------------------------------------------------------------------
def _ellipse_mask(h, w, cx, cy, rx, ry, angle=0.0):
    """Boolean ellipse mask."""
    yy, xx = np.mgrid[0:h, 0:w]
    xr = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
    yr = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
    return (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0


def make_synthetic_inputs(h=512, w=512, seed=0):
    """Return (image, masks, scores) for a chest-like illustrative phantom."""
    rng = np.random.default_rng(seed)

    # Soft background that reads as a medical image.
    yy, xx = np.mgrid[0:h, 0:w]
    img = 0.25 + 0.15 * np.exp(-(((xx - w / 2) / (0.55 * w)) ** 2
                                 + ((yy - h / 2) / (0.55 * h)) ** 2))
    img += 0.03 * rng.standard_normal((h, w))
    img = np.clip(img, 0, 1)

    # "True" structures: two lungs and a heart-like ellipse.
    left_lung = _ellipse_mask(h, w, 0.34 * w, 0.50 * h, 0.13 * w, 0.26 * h, 0.1)
    right_lung = _ellipse_mask(h, w, 0.66 * w, 0.50 * h, 0.13 * w, 0.26 * h, -0.1)
    heart = _ellipse_mask(h, w, 0.50 * w, 0.62 * h, 0.10 * w, 0.12 * h, 0.0)
    true_masks = [left_lung, right_lung, heart]

    masks, scores = [], []
    # Correct candidates (high score).
    for m in true_masks:
        masks.append(m)
        scores.append(rng.uniform(0.88, 0.97))
    # Redundant near-duplicates of the lungs (high score, heavy overlap).
    masks.append(_ellipse_mask(h, w, 0.35 * w, 0.49 * h, 0.14 * w, 0.27 * h, 0.1))
    scores.append(rng.uniform(0.80, 0.90))
    masks.append(_ellipse_mask(h, w, 0.65 * w, 0.51 * h, 0.12 * w, 0.25 * h, -0.1))
    scores.append(rng.uniform(0.80, 0.90))
    # Spurious low-score blobs (background, rib gaps, etc.).
    for _ in range(4):
        cx, cy = rng.uniform(0.15, 0.85) * w, rng.uniform(0.15, 0.85) * h
        rx, ry = rng.uniform(0.04, 0.08) * w, rng.uniform(0.04, 0.08) * h
        masks.append(_ellipse_mask(h, w, cx, cy, rx, ry))
        scores.append(rng.uniform(0.20, 0.50))
    # One large spurious mask spanning the thorax (low score).
    masks.append(_ellipse_mask(h, w, 0.50 * w, 0.50 * h, 0.40 * w, 0.40 * h))
    scores.append(rng.uniform(0.25, 0.45))

    return img, masks, np.array(scores)


# ---------------------------------------------------------------------------
# Real inputs (wire this to your own MedSAM2 grid cache)
# ---------------------------------------------------------------------------
def load_real_inputs():
    """Return (image, masks, scores) for REAL_IMAGE_STEM from the seg cache.

    Reads the (segmented.json, segmented.npz) pair produced by the MedSAM2
    grid stage. Each cached image carries a list of `objects`, and every object
    has a `confidence` (used as score) and a `mask_key` indexing the NPZ.
    """
    import json

    from PIL import Image

    cache_dir = Path(REAL_CACHE_DIR)
    meta = json.loads((cache_dir / "segmented.json").read_text())

    # Locate the requested image by its stem.
    entry = next((im for im in meta["images"] if im["stem"] == REAL_IMAGE_STEM), None)
    if entry is None:
        stems = [im["stem"] for im in meta["images"]]
        raise ValueError(
            f"Stem {REAL_IMAGE_STEM!r} not found in cache. "
            f"Available examples include: {stems[:5]} ... ({len(stems)} total)"
        )

    data = np.load(cache_dir / "segmented.npz")
    masks = [data[obj["mask_key"]].astype(bool) for obj in entry["objects"]]
    scores = np.asarray([obj["confidence"] for obj in entry["objects"]], dtype=float)

    # Render against the source image, resized to the mask resolution so the
    # overlays line up exactly.
    img = Image.open(entry["source_path"]).convert("L")
    if masks:
        h, w = masks[0].shape
        img = img.resize((w, h), Image.BILINEAR)
    img = np.asarray(img, dtype=float) / 255.0
    return img, masks, scores, None


# ---------------------------------------------------------------------------
# Live MedSAM2 inputs (run the grid prompting and keep every raw mask)
# ---------------------------------------------------------------------------
def load_medsam2_inputs():
    """Run MedSAM2 grid prompting live and return EVERY raw per-point mask.

    Reuses the project pipeline (same model, grid and predictor used in
    Phase 1) but stops before the score-threshold + NMS depuration, so panel B
    shows the genuine raw candidate set (one mask per grid point). The actual
    prompt grid is returned as well so it can be drawn on top of panel B.

    Returns (image, masks, scores, points).
    """
    from PIL import Image

    # Optional cache: skip the GPU run if a saved raw NPZ is available.
    if MEDSAM2_RAW_NPZ and Path(MEDSAM2_RAW_NPZ).exists():
        data = np.load(MEDSAM2_RAW_NPZ)
        masks = [m.astype(bool) for m in data["masks"]]
        scores = np.asarray(data["scores"], dtype=float)
        points = np.asarray(data["points"], dtype=float)
        img = np.asarray(
            Image.open(MEDSAM2_IMAGE_PATH).convert("L").resize(
                masks[0].shape[::-1], Image.BILINEAR
            ),
            dtype=float,
        ) / 255.0
        return img, masks, scores, points

    from project.data_io.reader import MedicalImageReader
    from project.segmentation.utils import make_point_grid, to_uint8
    from project.segmentation.medsam2.segmenter import MedSAM2Segmenter, MedSAM2Config

    segmenter = MedSAM2Segmenter(MedSAM2Config(device=MEDSAM2_DEVICE, grid_side=GRID_N))
    medimg = MedicalImageReader().load(MEDSAM2_IMAGE_PATH)

    img_uint8 = to_uint8(medimg.volume)
    segmenter._predictor.set_image(img_uint8)
    h, w = img_uint8.shape[:2]
    points = make_point_grid(h, w, GRID_N).astype(float)

    # Raw: one (mask, score) per grid point, NO threshold and NO NMS.
    raw = segmenter._predict_all_points(points)
    masks = [m for m, _ in raw]
    scores = np.asarray([s for _, s in raw], dtype=float)

    if MEDSAM2_RAW_NPZ:
        Path(MEDSAM2_RAW_NPZ).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            MEDSAM2_RAW_NPZ,
            masks=np.stack(masks), scores=scores, points=points,
        )

    img = np.asarray(
        Image.open(MEDSAM2_IMAGE_PATH).convert("L").resize((w, h), Image.BILINEAR),
        dtype=float,
    ) / 255.0
    return img, masks, scores, points


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _new_axes(img):
    """Square figure showing the image with no axes or padding."""
    h, w = img.shape
    fig, ax = plt.subplots(figsize=(5, 5 * h / w))
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def _draw_masks(ax, masks, idxs, cmap_name="tab10"):
    """Translucent fill + solid contour for each selected mask."""
    cmap = plt.get_cmap(cmap_name)
    for k, i in enumerate(idxs):
        color = cmap(k % 10)
        overlay = np.zeros((*masks[i].shape, 4))
        overlay[masks[i]] = (*color[:3], 0.40)
        ax.imshow(overlay, interpolation="nearest")
        ax.contour(masks[i], levels=[0.5], colors=[color], linewidths=1.6)


def panel_a(img, points, out):
    fig, ax = _new_axes(img)
    _draw_grid(ax, points, img)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _draw_grid(ax, points, img):
    for (x, y) in points:
        ax.add_patch(Circle((x, y), radius=0.012 * img.shape[1],
                            facecolor="#ffd400", edgecolor="black", linewidth=0.6))


def panel_b(img, masks, out, points=None):
    fig, ax = _new_axes(img)
    _draw_masks(ax, masks, list(range(len(masks))))
    if points is not None:
        _draw_grid(ax, points, img)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def panel_c(img, masks, kept, out):
    fig, ax = _new_axes(img)
    _draw_masks(ax, masks, kept)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    points = None
    if SOURCE == "synthetic":
        img, masks, scores = make_synthetic_inputs()
    elif SOURCE == "real":
        img, masks, scores, points = load_real_inputs()
    elif SOURCE == "medsam2":
        img, masks, scores, points = load_medsam2_inputs()
    else:
        raise ValueError(f"Unknown SOURCE: {SOURCE}")

    h, w = img.shape
    # Use the actual prompt grid when the source provides it, else a layout grid.
    if points is None:
        points = make_grid_points(h, w, GRID_N)
    kept = depurate(masks, scores, SCORE_THR, NMS_IOU_THR)

    grid_for_b = points if SHOW_GRID_IN_B else None
    panel_a(img, points, OUT_DIR / "panel_a_grid.png")
    panel_b(img, masks, OUT_DIR / "panel_b_raw.png", points=grid_for_b)
    panel_c(img, masks, kept, OUT_DIR / "panel_c_clean.png")

    print(f"Candidates: {len(masks)} -> kept after depuration: {len(kept)}")
    print(f"Wrote three panels to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
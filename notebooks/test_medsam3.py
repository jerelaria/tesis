import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("..")

from project.segmentation.medsam3 import MedSAM3Segmenter, MedSAM3Config
from project.data_io.reader import MedicalImageReader

# ---------------------------------------------------------------------------
# Output directory — same folder as this script
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGE_NAME_XRAY = "1256842362861431725328351539259305635_u1qifz.png"
_IMAGE_NAME_SUNNYBROOK = "SCD0000101_CINESAX_300.png"

dataset = "xray"  # change this to "sunnybrook" to test the other image
if dataset == "xray":
    _IMAGE_NAME = _IMAGE_NAME_XRAY
elif dataset == "sunnybrook":
    _IMAGE_NAME = _IMAGE_NAME_SUNNYBROOK

_IMAGE_STEM = os.path.splitext(_IMAGE_NAME)[0]  # filename without extension

# ---------------------------------------------------------------------------
# Load image
# ---------------------------------------------------------------------------

reader = MedicalImageReader()

image = reader.load(os.path.join("..", "data", "raw", dataset, _IMAGE_NAME))

print(f"Image shape : {image.volume.shape}")
print(f"Image dtype : {image.volume.dtype}")
print(f"Source path : {image.source_path}")

# ---------------------------------------------------------------------------
# Initialize segmenter
# ---------------------------------------------------------------------------

prompt_xray = ["lung", "heart"]
prompt_sunnybrook = ["left ventricle", "right ventricle", "myocardium"]

config = MedSAM3Config(
    prompts=prompt_xray if dataset == "xray" else prompt_sunnybrook,
    score_threshold=0.5,
    nms_iou=0.5,
    resolution=1008,
    device="cuda",
)

segmenter = MedSAM3Segmenter(config)
print("Segmenter ready.")

# ---------------------------------------------------------------------------
# Run segmentation
# ---------------------------------------------------------------------------

objects = segmenter.segment(image)
print(f"Objects detected: {len(objects)}")

for obj in objects:
    print(f"  id={obj.id}  confidence={obj.confidence:.3f}  mask_shape={obj.mask.shape}")

# ---------------------------------------------------------------------------
# Plot 1: original + individual masks side by side
# ---------------------------------------------------------------------------

n_cols = len(objects) + 1
fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))

axes[0].imshow(image.volume.astype(np.uint8))
axes[0].set_title("Original")
axes[0].axis("off")

for i, obj in enumerate(objects):
    axes[i + 1].imshow(obj.mask, cmap="gray")
    axes[i + 1].set_title(f"{obj.label}\nscore {obj.confidence:.2f}")
    axes[i + 1].axis("off")

plt.suptitle(f"MedSAM3 — prompts: {config.prompts}", fontsize=12)
plt.tight_layout()

out_masks = os.path.join(_SCRIPT_DIR, f"{_IMAGE_STEM}_masks.png")
plt.savefig(out_masks, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_masks}")

# ---------------------------------------------------------------------------
# Plot 2: overlay masks on original image
# ---------------------------------------------------------------------------

COLORS = [
    [1, 0, 0, 0.4],    # red
    [0, 0, 1, 0.4],    # blue
    [0, 1, 0, 0.4],    # green
    [1, 1, 0, 0.4],    # yellow
    [0, 1, 1, 0.4],    # cyan
    [1, 0, 1, 0.4],    # magenta
]

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.imshow(image.volume.astype(np.uint8))

for i, obj in enumerate(objects):
    color = COLORS[i % len(COLORS)]
    overlay = np.zeros((*obj.mask.shape, 4))
    overlay[obj.mask] = color
    ax.imshow(overlay)
    ax.text(
        5, 20 + i * 20,
        f"{obj.label} — score {obj.confidence:.2f}",
        color=color[:3], fontsize=10,
        bbox=dict(facecolor="black", alpha=0.5, pad=2)
    )

ax.set_title(f"MedSAM3 overlay — prompts: {config.prompts}")
ax.axis("off")
plt.tight_layout()

out_overlay = os.path.join(_SCRIPT_DIR, f"{_IMAGE_STEM}_overlay.png")
plt.savefig(out_overlay, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_overlay}")
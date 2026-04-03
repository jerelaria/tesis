"""
Loads few-shot reference data: one or more reference images, each with its
associated organ masks (one binary PNG per organ per image).

Supports two directory structures:

A) Single reference image:
    data/few_shot/xray/
    ├── reference.png
    ├── left_lung.png
    ├── right_lung.png
    └── heart.png

B) Multiple reference images:
    data/few_shot/xray/
    ├── ref1/
    │   ├── image.png
    │   ├── left_lung.png
    │   └── heart.png
    └── ref2/
        ├── image.png
        ├── left_lung.png
        └── right_lung.png

YAML config:
    few_shot:
      references_dir: "data/few_shot/xray"
      references:
        - image: "reference.png"
          masks:
            left_lung: "left_lung.png"
            right_lung: "right_lung.png"
            heart: "heart.png"
        - image: "ref2/image.png"
          masks:
            left_lung: "ref2/left_lung.png"
            right_lung: "ref2/right_lung.png"
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from PIL import Image


@dataclass
class FewShotReference:
    """A single reference image with its organ masks."""
    volume: np.ndarray                    # (H, W, 3) float32
    masks: dict[str, np.ndarray]          # organ_name -> (H, W) bool
    source_path: str                      # for traceability


def load_few_shot_references(
    references_dir: str,
    references_config: list[dict],
) -> list[FewShotReference]:
    """
    Load one or more reference images and their masks from disk.

    Parameters
    ----------
    references_dir : str
        Base directory containing all reference data.
    references_config : list[dict]
        List of reference specs, each with:
        - "image": filename relative to references_dir
        - "masks": dict of organ_name -> mask filename

    Returns
    -------
    list[FewShotReference]
        One FewShotReference per reference image.

    Raises
    ------
    FileNotFoundError
        If any image or mask file is not found.
    ValueError
        If a mask shape doesn't match its image, or no valid data loaded.
    """
    ref_dir = Path(references_dir)
    results: list[FewShotReference] = []

    for i, ref_spec in enumerate(references_config):
        image_filename = ref_spec["image"]
        mask_filenames = ref_spec["masks"]

        image_path = ref_dir / image_filename
        if not image_path.exists():
            raise FileNotFoundError(
                f"Reference image not found: {image_path}"
            )

        img = Image.open(image_path).convert("RGB")
        volume = np.array(img, dtype=np.float32)
        h, w = volume.shape[:2]

        print(f"  Reference {i+1}: {image_path} ({w}x{h})")

        masks: dict[str, np.ndarray] = {}
        for organ_name, mask_filename in mask_filenames.items():
            mask_path = ref_dir / mask_filename
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask not found for '{organ_name}': {mask_path}"
                )

            mask_img = Image.open(mask_path).convert("L")
            mask_array = np.array(mask_img)

            if mask_array.shape != (h, w):
                raise ValueError(
                    f"Mask '{organ_name}' shape {mask_array.shape} != "
                    f"image shape ({h}, {w}) in reference {i+1}"
                )

            binary = mask_array > 127
            if not binary.any():
                print(f"    [WARN] Mask '{organ_name}' is empty, skipping")
                continue

            masks[organ_name] = binary
            pct = 100 * binary.sum() / (h * w)
            print(f"    {organ_name}: {binary.sum()} px ({pct:.1f}%)")

        if not masks:
            print(f"    [WARN] No valid masks in reference {i+1}, skipping")
            continue

        results.append(FewShotReference(
            volume=volume,
            masks=masks,
            source_path=str(image_path),
        ))

    if not results:
        raise ValueError("No valid references loaded")

    all_organs: dict[str, int] = {}
    for ref in results:
        for organ in ref.masks:
            all_organs[organ] = all_organs.get(organ, 0) + 1

    print(f"  Loaded {len(results)} references covering: "
          f"{', '.join(f'{k}(x{v})' for k, v in sorted(all_organs.items()))}")

    return results
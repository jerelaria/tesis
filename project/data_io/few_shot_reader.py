"""
Loads few-shot reference data: one or more reference images, each with its
associated organ masks (one binary PNG per organ per image).

Supports two modes:

A) Auto-discovery:
    discover_few_shot_references() scans data/few_shot/{dataset}/ for
    subdirectories, each containing image.png and organ mask PNGs.

B) Explicit config:
    load_few_shot_references() loads references from a YAML spec list.

Expected directory structure for auto-discovery:
    data/few_shot/{dataset_short}/
    ├── ref_001/
    │   ├── image.png
    │   ├── left_lung.png
    │   ├── right_lung.png
    │   └── heart.png
    ├── ref_002/
    │   ├── image.png
    │   └── ...
    └── ref_003/
        └── ...

Convention: every .png file in a reference directory that is NOT image.png
is treated as an organ mask. The filename stem becomes the organ name.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from PIL import Image


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class FewShotReference:
    """A single reference image with its organ masks."""
    volume: np.ndarray                    # (H, W, 3) float32
    masks: dict[str, np.ndarray]          # organ_name -> (H, W) bool
    source_path: str                      # for traceability


# ======================================================================
# AUTO-DISCOVERY (new primary interface)
# ======================================================================

def discover_few_shot_references(
    dataset_name: str,
    num_refs: int,
    ref_names: list[str] | None = None,
) -> list[FewShotReference]:
    """
    Auto-discover reference images from data/few_shot/{dataset_short}/.

    Each subdirectory is a reference containing image.png and organ mask PNGs.
    Masks are auto-detected: every .png that is not image.png is an organ mask,
    with the filename stem as the organ name.

    Reference selection logic:
    - If ref_names is given with len >= num_refs: use the first num_refs
      entries from ref_names.
    - If ref_names is given with len < num_refs: use all ref_names first,
      then fill remaining slots from available directories (alphabetically)
      that are not already in ref_names.
    - If ref_names is None: take the first num_refs directories alphabetically.

    This ensures that for a sweep K=1,4,7,10 with the same ref_names list,
    K=4 is always a subset of K=7 (first-N selection).

    Parameters
    ----------
    dataset_name : str
        Dataset identifier (e.g., "XRayNicoSent/images"). Only the first
        component before '/' is used to locate the few-shot directory.
    num_refs : int
        Number of reference images to load.
    ref_names : list[str] | None
        Specific reference directory names to prioritize. If fewer than
        num_refs are given, remaining slots are filled alphabetically
        from available directories not in ref_names.

    Returns
    -------
    list[FewShotReference]
        Exactly num_refs references (or fewer if not enough are available).

    Raises
    ------
    FileNotFoundError
        If the few-shot directory does not exist or is empty.
    ValueError
        If no valid references could be loaded.
    """
    dataset_short = dataset_name.split("/")[0]
    fs_dir = _PROJECT_ROOT / "data" / "few_shot" / dataset_short

    if not fs_dir.is_dir():
        raise FileNotFoundError(
            f"Few-shot directory not found: {fs_dir}\n"
            f"Expected structure: data/few_shot/{dataset_short}/<ref_name>/image.png"
        )

    # Discover all valid reference directories
    available_dirs = _discover_reference_dirs(fs_dir)

    if not available_dirs:
        raise FileNotFoundError(
            f"No valid reference directories in {fs_dir}. "
            f"Each subdirectory must contain image.png and at least one mask PNG."
        )

    available_names = [d.name for d in available_dirs]
    available_by_name = {d.name: d for d in available_dirs}

    # Build the ordered selection
    selected_names = _select_references(available_names, num_refs, ref_names)

    if not selected_names:
        raise ValueError(
            f"Could not select any references. "
            f"Available: {available_names}, requested: {ref_names}"
        )

    if len(selected_names) < num_refs:
        print(f"  [WARN] Requested {num_refs} references but only "
              f"{len(selected_names)} available. Using all.")

    # Load each selected reference
    print(f"\n  Loading {len(selected_names)} few-shot references "
          f"from {fs_dir}:")
    results: list[FewShotReference] = []

    for i, name in enumerate(selected_names):
        ref_dir = available_by_name[name]
        ref = _load_single_reference(ref_dir, index=i + 1)
        if ref is not None:
            results.append(ref)

    if not results:
        raise ValueError("No valid references loaded after processing all directories")

    # Summary
    all_organs: dict[str, int] = {}
    for ref in results:
        for organ in ref.masks:
            all_organs[organ] = all_organs.get(organ, 0) + 1

    print(f"  Loaded {len(results)} references covering: "
          f"{', '.join(f'{k}(x{v})' for k, v in sorted(all_organs.items()))}")

    return results


def get_few_shot_reference_stems(
    dataset_name: str,
    num_refs: int,
    ref_names: list[str] | None = None,
) -> list[str]:
    """
    Return the directory names of the references that would be selected,
    without loading any image data. Useful for excluding these images
    from the evaluation dataset.

    Parameters are identical to discover_few_shot_references().

    Returns
    -------
    list[str]
        Reference directory names that would be selected.
    """
    dataset_short = dataset_name.split("/")[0]
    fs_dir = _PROJECT_ROOT / "data" / "few_shot" / dataset_short

    if not fs_dir.is_dir():
        return []

    available_dirs = _discover_reference_dirs(fs_dir)
    available_names = [d.name for d in available_dirs]

    return _select_references(available_names, num_refs, ref_names)


# ======================================================================
# EXPLICIT CONFIG (legacy interface, kept for backwards compatibility)
# ======================================================================

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


# ======================================================================
# INTERNAL HELPERS
# ======================================================================

def _discover_reference_dirs(fs_dir: Path) -> list[Path]:
    """
    Find all valid reference subdirectories (contain image.png + at least
    one mask PNG). Returns sorted list of Paths.
    """
    valid = []
    for child in sorted(fs_dir.iterdir()):
        if not child.is_dir():
            continue
        image_path = child / "image.png"
        if not image_path.exists():
            # Also check for .jpg
            image_path = child / "image.jpg"
            if not image_path.exists():
                continue

        # Check there is at least one mask file
        mask_files = [
            f for f in child.iterdir()
            if f.suffix.lower() == ".png" and f.name != "image.png"
        ]
        if mask_files:
            valid.append(child)

    return valid


def _select_references(
    available_names: list[str],
    num_refs: int,
    ref_names: list[str] | None = None,
) -> list[str]:
    """
    Build an ordered selection of reference names.

    Priority: ref_names first (in order), then fill remaining slots
    from available_names (alphabetically) excluding already-selected ones.
    """
    if ref_names is None:
        return available_names[:num_refs]

    available_set = set(available_names)
    selected = []

    # First: add requested names that actually exist
    for name in ref_names:
        if name in available_set and name not in selected:
            selected.append(name)
        elif name not in available_set:
            print(f"  [WARN] Requested reference '{name}' not found, skipping")

    # Fill remaining slots from available (alphabetically), excluding selected
    if len(selected) < num_refs:
        selected_set = set(selected)
        for name in available_names:
            if name not in selected_set:
                selected.append(name)
                selected_set.add(name)
                if len(selected) >= num_refs:
                    break

    return selected[:num_refs]


def _load_single_reference(ref_dir: Path, index: int) -> FewShotReference | None:
    """
    Load a single reference from a directory.

    Auto-detects image (image.png or image.jpg) and masks (all other .png files).
    """
    # Find the image file
    image_path = ref_dir / "image.png"
    if not image_path.exists():
        image_path = ref_dir / "image.jpg"
        if not image_path.exists():
            print(f"    [WARN] No image.png/jpg in {ref_dir.name}, skipping")
            return None

    img = Image.open(image_path).convert("RGB")
    volume = np.array(img, dtype=np.float32)
    h, w = volume.shape[:2]

    print(f"  Reference {index}: {ref_dir.name} ({w}x{h})")

    # Auto-detect masks: every .png that is not image.png
    masks: dict[str, np.ndarray] = {}
    for mask_file in sorted(ref_dir.iterdir()):
        if mask_file.suffix.lower() != ".png":
            continue
        if mask_file.name == "image.png":
            continue

        organ_name = mask_file.stem

        mask_img = Image.open(mask_file).convert("L")
        mask_array = np.array(mask_img)

        if mask_array.shape != (h, w):
            print(f"    [WARN] Mask '{organ_name}' shape {mask_array.shape} != "
                  f"image ({h}, {w}), skipping")
            continue

        binary = mask_array > 127
        if not binary.any():
            print(f"    [WARN] Mask '{organ_name}' is empty, skipping")
            continue

        masks[organ_name] = binary
        pct = 100 * binary.sum() / (h * w)
        print(f"    {organ_name}: {binary.sum()} px ({pct:.1f}%)")

    if not masks:
        print(f"    [WARN] No valid masks in {ref_dir.name}, skipping")
        return None

    return FewShotReference(
        volume=volume,
        masks=masks,
        source_path=str(image_path),
    )
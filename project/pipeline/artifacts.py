"""Save and load Phase 1 segmentation artifacts (segmented.npz + segmented.json)."""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from project.core.data_types import MedicalImage, SegmentedObject

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


def save_segmented(
    objects_by_image: dict[Path, list[SegmentedObject]],
    out_dir: Path,
    hash8: str,
    dataset_short: str,
    mode: str,
    propagation_mode: str,
) -> None:
    """Persist Phase 1 output to segmented.npz and segmented.json.

    The NPZ is written atomically (temp file + rename) to guard against
    partial writes on shared filesystems.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    images_meta: list[dict] = []

    for img_idx, (path, objects) in enumerate(objects_by_image.items()):
        src = objects[0].source_image if objects else None
        img_entry: dict = {
            "source_path": str(path),
            "stem": path.stem,
            "modality": src.modality if src else "",
            "metadata": (src.metadata if src else {}) or {},
            "objects": [],
        }

        for obj_idx, obj in enumerate(objects):
            prefix = f"{img_idx:04d}_{obj_idx:03d}"
            mask_key = f"{prefix}_mask"
            feat_key = f"{prefix}_feat"
            emb_key = f"{prefix}_emb" if obj.embedding is not None else None

            arrays[mask_key] = obj.mask.astype(bool)
            arrays[feat_key] = obj.features.astype(np.float32)
            if emb_key is not None:
                arrays[emb_key] = obj.embedding.astype(np.float32)

            img_entry["objects"].append({
                "id": obj.id,
                "confidence": obj.confidence,
                "label": obj.label,
                "mask_key": mask_key,
                "features_key": feat_key,
                "embedding_key": emb_key,
            })

        images_meta.append(img_entry)

    # Atomic write: create temp file ending in .npz so numpy doesn't append suffix
    npz_path = out_dir / "segmented.npz"
    tmp_fd, tmp_path_str = tempfile.mkstemp(dir=str(out_dir), suffix=".npz")
    os.close(tmp_fd)
    tmp_path = Path(tmp_path_str)
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, npz_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    meta = {
        "schema_version": _SCHEMA_VERSION,
        "hash8": hash8,
        "dataset_short": dataset_short,
        "mode": mode,
        "propagation_mode": propagation_mode,
        "num_images": len(objects_by_image),
        "images": images_meta,
    }
    with open(out_dir / "segmented.json", "w") as f:
        json.dump(meta, f, indent=2)

    total = sum(len(v) for v in objects_by_image.values())
    logger.info(
        f"Segmentation artifact written: {total} objects "
        f"across {len(objects_by_image)} images -> {out_dir}"
    )


def load_segmented(
    in_dir: Path,
) -> tuple[list[SegmentedObject], dict[Path, list[SegmentedObject]]]:
    """Load Phase 1 output from disk. Restores object UUIDs exactly.

    Returns (all_objects, objects_by_image). MedicalImage.volume is set to
    None for all reconstructed objects; downstream phases do not read it.
    """
    with open(in_dir / "segmented.json") as f:
        meta = json.load(f)

    arrays = np.load(in_dir / "segmented.npz", allow_pickle=False)

    all_objects: list[SegmentedObject] = []
    objects_by_image: dict[Path, list[SegmentedObject]] = {}

    for img_entry in meta["images"]:
        path = Path(img_entry["source_path"])
        # volume=None: shared per-image object; downstream phases use source_path
        med_img = MedicalImage(
            volume=None,
            modality=img_entry["modality"],
            metadata=img_entry.get("metadata", {}),
            source_path=img_entry["source_path"],
        )

        objs: list[SegmentedObject] = []
        for obj_entry in img_entry["objects"]:
            mask = arrays[obj_entry["mask_key"]]
            features = arrays[obj_entry["features_key"]]
            emb_key = obj_entry.get("embedding_key")
            embedding = arrays[emb_key] if emb_key is not None else None

            obj = SegmentedObject(
                mask=mask,
                source_image=med_img,
                features=features,
                embedding=embedding,
                confidence=obj_entry.get("confidence"),
                label=obj_entry.get("label"),
            )
            obj.id = obj_entry["id"]  # restore exactly; never regenerate
            objs.append(obj)

        objects_by_image[path] = objs
        all_objects.extend(objs)

    logger.info(
        f"Loaded segmentation cache: {len(all_objects)} objects "
        f"from {len(objects_by_image)} images <- {in_dir}"
    )
    return all_objects, objects_by_image


def write_cache_key(cache_dir: Path, payload: dict, hash8: str) -> None:
    """Write cache_key.json for human-readable debug inspection."""
    data = {
        "payload": payload,
        "hash8": hash8,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(cache_dir / "cache_key.json", "w") as f:
        json.dump(data, f, indent=2)

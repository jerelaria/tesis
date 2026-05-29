import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def save_predicted_masks(
    labeled_by_image: dict[Path, list],
    results_dir: Path,
    score_alpha: float = 0.5,
) -> None:
    """Save final predicted masks as binary PNGs for evaluation.

    Alongside masks, write a `scores.json` per image with:
        {filename.png: {"sam": float, "labeling": float, "combined": float}}

    The `combined` score is used by evaluate.py to order predictions for
    mAP computation. The other two are kept for potential downstream
    analysis (e.g., recalibration studies).
    """
    from PIL import Image as PILImage

    masks_dir = results_dir / "masks"
    logger.info(f"Saving predicted masks -> {masks_dir}")

    count = 0
    for path, labeled in labeled_by_image.items():
        image_dir = masks_dir / path.stem
        image_dir.mkdir(parents=True, exist_ok=True)

        by_name: dict[str, list] = {}
        for obj in labeled:
            if obj.is_noise:
                continue
            by_name.setdefault(obj.organ_name, []).append(obj)

        scores_payload: dict[str, dict[str, float]] = {}

        for name, objs in by_name.items():
            for i, obj in enumerate(objs, start=1):
                filename = f"{name}_{i}.png" if len(objs) > 1 else f"{name}.png"
                mask_uint8 = (obj.segmented_object.mask * 255).astype(np.uint8)
                PILImage.fromarray(mask_uint8, mode="L").save(image_dir / filename)
                sam = float(obj.segmented_object.confidence or 0.0)
                lab = float(obj.labeling_confidence or 0.0)
                combined = score_alpha * sam + (1.0 - score_alpha) * lab
                scores_payload[filename] = {
                    "sam": sam, "labeling": lab, "combined": combined,
                }
                count += 1

        if scores_payload:
            with open(image_dir / "scores.json", "w") as f:
                json.dump(scores_payload, f, indent=2)

    logger.info(f"Saved {count} masks across {len(labeled_by_image)} images")


def print_summary(all_objects, labeled_by_image) -> None:
    """Log a summary of segmentation, clustering, and feature statistics."""
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)

    total_labeled = sum(len(v) for v in labeled_by_image.values())
    noise_count = sum(
        1 for objs in labeled_by_image.values()
        for obj in objs if obj.is_noise
    )
    organ_names = set(
        obj.organ_name for objs in labeled_by_image.values()
        for obj in objs if not obj.is_noise
    )
    method_counts: dict[str, int] = {}
    for objs in labeled_by_image.values():
        for obj in objs:
            m = obj.method_used
            method_counts[m] = method_counts.get(m, 0) + 1

    logger.info(f"  Total segmented: {len(all_objects)}")
    logger.info(f"  Total labeled: {total_labeled}")
    logger.info(f"  Noise: {noise_count}")
    logger.info(f"  Organs: {sorted(organ_names)}")
    logger.info(f"  Images: {len(labeled_by_image)}")
    logger.info(f"  Methods: {method_counts}")

    features = [o.features for o in all_objects if o.features is not None]
    if features:
        X = np.stack(features)
        logger.info(f"  Moment features: {X.shape}")
        logger.debug(f"  Min: {X.min(axis=0)}")
        logger.debug(f"  Max: {X.max(axis=0)}")
        logger.debug(f"  Std: {X.std(axis=0)}")

    embeddings = [o.embedding for o in all_objects if o.embedding is not None]
    if embeddings:
        E = np.stack(embeddings)
        logger.info(f"  Embeddings: {E.shape} "
                    f"(mean_norm={np.linalg.norm(E, axis=1).mean():.3f})")
    elif any(o.embedding is not None for o in all_objects):
        n_with = sum(1 for o in all_objects if o.embedding is not None)
        logger.info(f"  Embeddings: {n_with}/{len(all_objects)} objects have embeddings")

import logging

from project.evaluation.visualizer import save_segmentation_vis
from project.pipeline.feature_helpers import _extract_features

logger = logging.getLogger(__name__)


def run_phase1_few_shot_independent(
    image_paths, reader, segmenter, extractor, references, phase1_dir,
    extract_embeddings=False,
):
    """
    Few-shot independent: per-image (K+1)-frame video.
    K references + 1 target per call. Order-invariant.
    """
    all_objects = []
    objects_by_image = {}

    for idx, path in enumerate(image_paths):
        logger.info(f"[{idx+1}/{len(image_paths)}] {path.name}")

        image = reader.load(str(path))

        fs_objects = segmenter.segment_with_video_prompts(
            target_image=image,
            references=references,
        )
        labels = ', '.join(o.label for o in fs_objects if o.label)
        logger.debug(f"  Independent ({len(references)} refs): "
                     f"{len(fs_objects)} objects ({labels})")

        image_embed = (
            segmenter.encode_image(image) if extract_embeddings else None
        )

        valid = _extract_features(fs_objects, extractor, image_embed)
        if valid:
            objects_by_image[path] = valid
            all_objects.extend(valid)
            save_segmentation_vis(path, valid, phase1_dir)
        else:
            logger.warning(f"[SKIP] {path.name}: no valid objects")

    return all_objects, objects_by_image


def run_phase1_few_shot_iterative(
    image_paths, reader, segmenter, extractor, references, phase1_dir,
    extract_embeddings=False,
):
    """
    Few-shot iterative: single (K+N)-frame video.
    K references + N targets. Memory accumulates.
    """
    logger.info("Loading all images...")
    images_by_path = {}
    for path in image_paths:
        images_by_path[path] = reader.load(str(path))

    logger.info("Running iterative video predictor...")
    target_entries = [(path, images_by_path[path]) for path in image_paths]
    fs_by_image = segmenter.segment_batch_iterative(
        target_entries=target_entries,
        references=references,
    )

    logger.info("Extracting features...")
    all_objects = []
    objects_by_image = {}

    for path in image_paths:
        fs_objs = fs_by_image.get(path, [])
        if fs_objs:
            labels = ', '.join(o.label for o in fs_objs if o.label)
            logger.debug(f"  {path.name}: {len(fs_objs)} objects ({labels})")
        else:
            logger.debug(f"  {path.name}: 0 objects")

        image_embed = (
            segmenter.encode_image(images_by_path[path])
            if extract_embeddings else None
        )

        valid = _extract_features(fs_objs, extractor, image_embed)
        if valid:
            objects_by_image[path] = valid
            all_objects.extend(valid)
            save_segmentation_vis(path, valid, phase1_dir)
        else:
            logger.warning(f"[SKIP] {path.name}: no valid objects")

    return all_objects, objects_by_image

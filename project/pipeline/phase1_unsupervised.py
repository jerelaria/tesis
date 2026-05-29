import logging

from project.evaluation.visualizer import save_segmentation_vis
from project.pipeline.feature_helpers import _extract_features

logger = logging.getLogger(__name__)


def run_phase1_unsupervised(
    image_paths, reader, segmenter, extractor, phase1_dir,
    extract_embeddings=False,
):
    """Grid-based segmentation (unsupervised mode)."""
    all_objects = []
    objects_by_image = {}

    for idx, path in enumerate(image_paths):
        logger.info(f"[{idx+1}/{len(image_paths)}] {path.name}")

        image = reader.load(str(path))
        grid_objects = segmenter.segment(image)
        logger.debug(f"  Grid: {len(grid_objects)} objects")

        image_embed = (
            segmenter.encode_image(image) if extract_embeddings else None
        )

        valid = _extract_features(grid_objects, extractor, image_embed)
        if valid:
            objects_by_image[path] = valid
            all_objects.extend(valid)
            save_segmentation_vis(path, valid, phase1_dir)
        else:
            logger.warning(f"[SKIP] {path.name}: no valid objects")

    return all_objects, objects_by_image

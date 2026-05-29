import logging

import numpy as np

from project.feature_extraction.embedding import extract_sam2_embedding

logger = logging.getLogger(__name__)


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a binary mask."""
    from scipy.ndimage import label as cc_label
    labeled, n = cc_label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    return labeled == sizes.argmax()


def _extract_features(objects, extractor, image_embed=None) -> list:
    """
    Extract moment features (and optionally SAM2 embeddings) for each object.

    Parameters
    ----------
    objects : list[SegmentedObject]
    extractor : FeatureExtractor
    image_embed : torch.Tensor, optional
        SAM2 image encoder output (1, 256, 64, 64). If provided,
        embeddings are extracted via masked average pooling per object.
    """
    valid = []
    for obj in objects:
        if obj.mask is None or not obj.mask.any():
            continue
        try:
            obj.mask = _keep_largest_component(obj.mask)
            obj.features = extractor.extract(obj)

            if image_embed is not None:
                obj.embedding = extract_sam2_embedding(obj, image_embed)

            valid.append(obj)
        except ValueError as e:
            logger.warning(f"[SKIP] obj {obj.id[:8]}...: {e}")
    return valid

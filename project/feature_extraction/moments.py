import numpy as np
from skimage.measure import regionprops

from project.core.interfaces import FeatureExtractor
from project.core.data_types import SegmentedObject


class MomentFeatureExtractor(FeatureExtractor):
    """
    Extracts the 6 shape descriptors from Kervadec et al.

    All values are normalized to image size, making them resolution-independent
    and comparable across images.

    Example
    -------
    >>> extractor = MomentFeatureExtractor()
    >>> features = extractor.extract(segmented_obj)  # np.ndarray (6,)
    >>> print(features)  # [V, Cx, Cy, Dx, Dy, L]
    """

    FEATURE_NAMES = ["V", "Cx", "Cy", "Dx", "Dy", "L"]

    def extract(self, obj: SegmentedObject) -> np.ndarray:
        """
        Currently supports 2D masks only.

        Parameters
        ----------
        obj : SegmentedObject
            Must have `mask` (H, W) bool and `source_image` with `volume` (H, W, 3).

        Returns
        -------
        np.ndarray (6,) float32
            [V, Cx, Cy, Dx, Dy, L], all normalized.
            Returns zeros if the mask is empty.
        """
        mask = obj.mask.astype(bool)

        if mask.ndim != 2:
            raise ValueError(f"mask must be 2D, got shape {mask.shape}")

        H, W = mask.shape
        diagonal = np.sqrt(H**2 + W**2)

        # Empty mask → raise ValueError
        if not mask.any():
            raise ValueError("Empty mask")

        # regionprops requires a labeled image
        # Since each SegmentedObject is a single object, we label directly
        labeled = mask.astype(np.int32)
        props = regionprops(labeled)

        if not props:
            raise ValueError("No regions found in the mask")

        region = max(props, key=lambda r: r.area) # Dado que algunas mascaras pueden incluir ruido, tomamos la más grande

        # --- V: normalized area ---
        V = region.area / (H * W)                 # Se reescalan las imagenes antes de comenzar? Para evalauar la necesidad de normalizar

        # --- C: normalized centroid ---
        # regionprops returns (row, col) → equivalent to (y, x)
        cy_px, cx_px = region.centroid
        Cx = cx_px / W
        Cy = cy_px / H

        # --- D: average distance to centroid (spatial std dev per axis) ---
        mu = region.moments_central  # 3x3 moments matrix
        Dx = np.sqrt(mu[0, 2] / (region.area + 1e-8)) / W 
        Dy = np.sqrt(mu[2, 0] / (region.area + 1e-8)) / H

        # --- L: major axis length, normalized by image diagonal ---
        L = region.perimeter / diagonal

        return np.array([V, Cx, Cy, Dx, Dy, L], dtype=np.float32)
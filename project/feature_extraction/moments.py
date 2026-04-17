import numpy as np
from skimage.measure import regionprops

from project.core.interfaces import FeatureExtractor
from project.core.data_types import SegmentedObject


class MomentFeatureExtractor(FeatureExtractor):
    """
    Extracts shape, moment, and intensity descriptors from segmented objects.

    Extends the original 6 descriptors from Kervadec et al. with additional
    shape descriptors (eccentricity, solidity, extent, compactness),
    Hu moments (top 3, log-transformed), and intensity statistics.

    All values are normalized to be resolution-independent and comparable
    across images. The extractor always computes ALL features; downstream
    selection for clustering is controlled via YAML config.

    Feature vector layout (16 features):
        [V, Cx, Cy, Dx, Dy, L,                    # Kervadec (6)
         ecc, solidity, extent, compact,            # Shape (4)
         hu0, hu1, hu2,                             # Hu moments (3)
         intensity_mean, intensity_std, orientation] # Intensity + orient (3)
    """

    FEATURE_NAMES = [
        # --- Kervadec et al. (original 6) ---
        "V", "Cx", "Cy", "Dx", "Dy", "L",
        # --- Shape descriptors ---
        "ecc", "solidity", "extent", "compact",
        # --- Hu moments (log-transformed) ---
        "hu0", "hu1", "hu2",
        # --- Intensity statistics ---
        "intensity_mean", "intensity_std",
        # --- Orientation ---
        "orientation",
    ]

    def extract(self, obj: SegmentedObject) -> np.ndarray:
        """
        Extract all 16 features from a segmented object.

        Parameters
        ----------
        obj : SegmentedObject
            Must have `mask` (H, W) bool and `source_image` with
            `volume` (H, W, 3).

        Returns
        -------
        np.ndarray (16,) float32
            Full feature vector, all normalized.

        Raises
        ------
        ValueError
            If mask is not 2D, is empty, or contains no regions.
        """
        mask = obj.mask.astype(bool)

        if mask.ndim != 2:
            raise ValueError(f"mask must be 2D, got shape {mask.shape}")

        H, W = mask.shape
        diagonal = np.sqrt(H**2 + W**2)

        if not mask.any():
            raise ValueError("Empty mask")

        # regionprops on single-object labeled mask
        labeled = mask.astype(np.int32)
        props = regionprops(labeled)

        if not props:
            raise ValueError("No regions found in the mask")

        # Take largest region (some masks may include noise fragments)
        region = max(props, key=lambda r: r.area)

        # =============================================================
        # Kervadec et al. features (original 6)
        # =============================================================

        V = region.area / (H * W)

        # regionprops returns (row, col) = (y, x)
        cy_px, cx_px = region.centroid
        Cx = cx_px / W
        Cy = cy_px / H

        mu = region.moments_central
        Dx = np.sqrt(mu[0, 2] / (region.area + 1e-8)) / W
        Dy = np.sqrt(mu[2, 0] / (region.area + 1e-8)) / H

        L = region.perimeter / diagonal

        # =============================================================
        # Shape descriptors (all from regionprops, naturally in [0, 1])
        # =============================================================

        ecc = region.eccentricity                    # 0 = circle, 1 = line
        solidity = region.solidity                   # area / convex_area
        extent = region.extent                       # area / bbox_area
        perimeter = region.perimeter
        compact = (4 * np.pi * region.area) / (perimeter**2 + 1e-8)  # isoperimetric ratio

        # =============================================================
        # Hu moments (log-transformed, top 3)
        # =============================================================

        hu = region.moments_hu
        eps = 1e-12
        hu0 = np.log(np.abs(hu[0]) + eps)
        hu1 = np.log(np.abs(hu[1]) + eps)
        hu2 = np.log(np.abs(hu[2]) + eps)

        # =============================================================
        # Intensity statistics (from source image, normalized by 255)
        # =============================================================

        volume = obj.source_image.volume  # (H, W, 3) uint8
        if volume.ndim == 3:
            gray = np.mean(volume, axis=2)  # simple average to grayscale
        else:
            gray = volume.astype(np.float64)

        masked_pixels = gray[mask]
        intensity_mean = np.mean(masked_pixels) / 255.0
        intensity_std = np.std(masked_pixels) / 255.0

        # =============================================================
        # Orientation (angle of major axis, normalized to [0, 1])
        # =============================================================

        # regionprops.orientation is in [-pi/2, pi/2]
        orientation = (region.orientation + np.pi / 2) / np.pi  # -> [0, 1]

        return np.array([
            V, Cx, Cy, Dx, Dy, L,
            ecc, solidity, extent, compact,
            hu0, hu1, hu2,
            intensity_mean, intensity_std,
            orientation,
        ], dtype=np.float32)
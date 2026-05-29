"""Single source of truth for moment-feature names and indices."""

FEATURE_NAMES = [
    # Kervadec et al. (original 6)
    "V", "Cx", "Cy", "Dx", "Dy", "L",
    # Shape descriptors
    "ecc", "solidity", "extent", "compact",
    # Hu moments (log-transformed)
    "hu0", "hu1", "hu2",
    # Intensity statistics
    "intensity_mean", "intensity_std",
    # Orientation
    "orientation",
]

FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}

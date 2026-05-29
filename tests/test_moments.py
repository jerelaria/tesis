"""Tests for MomentFeatureExtractor."""
import numpy as np
import pytest
from project.core.data_types import MedicalImage, SegmentedObject
from project.feature_extraction.moments import MomentFeatureExtractor


def _make_obj(mask: np.ndarray) -> SegmentedObject:
    image = MedicalImage(
        volume=np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32),
        modality="synthetic",
    )
    return SegmentedObject(mask=mask, source_image=image)


def test_disc_features(disc_mask):
    """A perfect disc should have V≈π·25²/100²≈0.196, Cx=Cy≈0.5, ecc<0.05."""
    extractor = MomentFeatureExtractor()
    obj = _make_obj(disc_mask)
    features = extractor.extract(obj)

    V, Cx, Cy = features[0], features[1], features[2]
    ecc = features[6]

    expected_V = np.pi * 25**2 / 100**2
    assert abs(V - expected_V) < 0.01, f"V={V:.4f}, expected≈{expected_V:.4f}"
    assert abs(Cx - 0.5) < 0.02, f"Cx={Cx:.4f}"
    assert abs(Cy - 0.5) < 0.02, f"Cy={Cy:.4f}"
    assert ecc < 0.05, f"ecc={ecc:.4f} (disc should be near-circular)"


def test_output_dimension(disc_mask):
    """Feature vector must always be exactly 16 elements."""
    extractor = MomentFeatureExtractor()
    obj = _make_obj(disc_mask)
    features = extractor.extract(obj)
    assert features.shape == (16,)


def test_empty_mask_raises(synthetic_image):
    """extract() on an empty mask must raise ValueError."""
    extractor = MomentFeatureExtractor()
    empty_mask = np.zeros((100, 100), dtype=bool)
    obj = SegmentedObject(mask=empty_mask, source_image=synthetic_image)
    with pytest.raises(ValueError, match="[Ee]mpty"):
        extractor.extract(obj)

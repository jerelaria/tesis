"""Shared fixtures for the test suite."""
import numpy as np
import pytest
from project.core.data_types import MedicalImage, SegmentedObject, LabeledObject


@pytest.fixture
def synthetic_image():
    """Plain 100x100 grayscale image as MedicalImage."""
    return MedicalImage(
        volume=np.zeros((100, 100, 3), dtype=np.float32),
        modality="synthetic",
        source_path="/tmp/synthetic.png",
    )


@pytest.fixture
def disc_mask():
    """50x50 disc centered in a 100x100 mask."""
    yy, xx = np.mgrid[:100, :100]
    return ((yy - 50) ** 2 + (xx - 50) ** 2) <= 25 ** 2


@pytest.fixture
def square_mask():
    """30x30 square in a 100x100 mask."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:60, 30:60] = True
    return mask

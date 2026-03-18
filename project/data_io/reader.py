import numpy as np
from PIL import Image
from project.core.data_types import MedicalImage
from project.core.interfaces import ImageReader


class MedicalImageReader(ImageReader):

    def load(self, path: str) -> MedicalImage:
        img = Image.open(path).convert("RGB")
        volume = np.array(img, dtype=np.float32)
        return MedicalImage(
            volume=volume,
            modality="unknown",
            metadata={},
            source_path=path
        )
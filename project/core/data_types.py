from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import uuid

@dataclass
class MedicalImage:
    volume: np.ndarray
    modality: str
    metadata: dict = field(default_factory=dict)
    source_path: Optional[str] = None


@dataclass
class SegmentedObject:
    mask: np.ndarray
    # bbox: tuple, en caso de querer incluir coordenadas de bounding box (x_min, y_min, x_max, y_max)
    source_image: MedicalImage
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    features: Optional[np.ndarray] = None
    confidence: Optional[float] = None
    label: Optional[str] = None  # text prompt that generated this mask

@dataclass
class LabeledObject:
    segmented_object: SegmentedObject
    organ_id: int
    organ_name: str
    labeling_confidence: float
    method_used: str
    is_noise: bool = False
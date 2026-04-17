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
    source_image: MedicalImage
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    features: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None  # SAM2 encoder embedding (256,)
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
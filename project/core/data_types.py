from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import uuid

@dataclass
class MedicalImage:
    """A single medical image loaded from disk.

    Fields
    ------
    volume : np.ndarray
        Image data as (H, W, 3) float32, filled by MedicalImageReader (Phase 1).
    modality : str
        Imaging modality (e.g. "xray", "ct"), set by reader.
    metadata : dict
        Arbitrary key-value metadata forwarded from the reader.
    source_path : str or None
        Absolute path to the source file on disk.
    """

    volume: np.ndarray
    modality: str
    metadata: dict = field(default_factory=dict)
    source_path: Optional[str] = None


@dataclass
class SegmentedObject:
    """A single object segmented from a MedicalImage.

    Fields
    ------
    mask : np.ndarray
        Binary segmentation mask (H, W) bool, produced by MedSAM2Segmenter
        (Phase 1).
    source_image : MedicalImage
        The parent image this object was segmented from.
    id : str
        Auto-generated UUID for stable object identity across passes.
    features : np.ndarray or None
        Moment feature vector extracted by MomentFeatureExtractor (Phase 1).
        Required for clustering (Phase 2); None until extraction runs.
    embedding : np.ndarray or None
        SAM2 encoder embedding of shape (256,), extracted optionally in Phase 1
        when embedding-based clustering is enabled.
    confidence : float or None
        SAM2 segmentation confidence score produced by the segmenter (Phase 1).
    label : str or None
        Text prompt that generated this mask; set in few-shot mode only.
    """

    mask: np.ndarray
    source_image: MedicalImage
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    features: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    confidence: Optional[float] = None
    label: Optional[str] = None


@dataclass
class LabeledObject:
    """A SegmentedObject with a cluster/organ assignment from Phase 2 onward.

    Fields
    ------
    segmented_object : SegmentedObject
        The underlying segmented object (mask, features, confidence).
    organ_id : int
        Numeric cluster identifier assigned by the labeler (Phase 2).
    organ_name : str
        Human-readable organ name. In unsupervised mode: "cluster_N" after
        Phase 2, updated to the propagated label after Phase 4.
        In few-shot mode: the organ name from the reference annotations.
    labeling_confidence : float
        Clustering assignment confidence from Phase 2 (e.g. soft-assignment
        probability or distance-based score).
    method_used : str
        Tag indicating how this object was produced, e.g. "kmeans",
        "propagation", "few_shot_baseline", "unsupervised_baseline".
    is_noise : bool
        True if this object is excluded from quality scoring and prototype
        selection. HDBSCAN noise points use organ_id=-1 by convention; this
        flag is reserved for explicit per-object exclusion.
    """

    segmented_object: SegmentedObject
    organ_id: int
    organ_name: str
    labeling_confidence: float
    method_used: str
    is_noise: bool = False
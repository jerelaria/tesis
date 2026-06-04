from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol
import numpy as np

from project.core.data_types import MedicalImage, SegmentedObject, LabeledObject


class ImageReader(ABC):
    @abstractmethod
    def load(self, path: str) -> MedicalImage:
        """Read a medical image from disk and return a MedicalImage object"""
        ...


class Segmenter(ABC):
    @abstractmethod
    def segment(self, image: MedicalImage) -> list[SegmentedObject]:
        """Segment an image and return a list of segmented objects"""
        ...


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, obj: SegmentedObject) -> np.ndarray:
        """Extract a feature vector from a segmented object"""
        ...


class Labeler(ABC):
    @abstractmethod
    def fit(self, objects: list[SegmentedObject]) -> None:
        """
        Adjust the labeler based on all objects.
        For clustering: calculate centroids.
        For coregistration: can be left empty (pass).
        """
        ...

    @abstractmethod
    def label(self, objects: list[SegmentedObject]) -> list[LabeledObject]:
        """Assign an organ label to each segmented object"""
        ...


class VideoSegmenter(Protocol):
    """Protocol for segmenters that support independent multi-reference propagation.

    Each call to segment_with_video_prompts runs a fresh (K+1)-frame video
    session using only the provided references — no state is shared between
    calls.  Decouples PrototypePropagator from MedSAM2Segmenter so that stubs
    satisfy the interface in tests without loading the GPU model.
    """

    def segment_with_video_prompts(
        self,
        target_image: "MedicalImage",
        references: list,
    ) -> "list[SegmentedObject]": ...


class Propagator(ABC):
    """Abstract base class for prototype-based batch propagation.

    Identifies good clusters from clustering output, selects top-K prototypes
    per cluster, and propagates all clusters to every target image in a single
    batch video pass.
    """

    @abstractmethod
    def propagate(
        self,
        labeled_by_image: "dict[Path, list[LabeledObject]]",
        target_paths: "list[Path]",
        reader: "ImageReader",
    ) -> "tuple[dict[Path, list[LabeledObject]], list[dict]]":
        """
        Parameters
        ----------
        labeled_by_image : dict[Path, list[LabeledObject]]
            Clustering result from Phase 2 (used to identify good clusters
            and select prototypes).
        target_paths : list[Path]
            Images to propagate to.  May overlap with the clustering images.
        reader : ImageReader
            Used to reload pixel data when source_image.volume is None
            (e.g. when Phase 1 output was loaded from cache).

        Returns
        -------
        labeled_by_image : dict[Path, list[LabeledObject]]
            Fresh dict containing only the propagated masks.
        memory_composition : list[dict]
            Per reference-frame metadata: frame_idx, obj_id, cluster_id,
            source_path, mask (np.ndarray), combined_score, area.
        """
        ...

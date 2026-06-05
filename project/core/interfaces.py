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

    Each call runs a fresh video session using only the provided references —
    no state is shared between calls.  Decouples PrototypePropagator from
    MedSAM2Segmenter so that stubs satisfy the interface in tests without
    loading the GPU model.
    """

    def segment_with_video_prompts(
        self,
        target_image: "MedicalImage",
        references: list,
    ) -> "list[SegmentedObject]": ...

    def segment_with_multi_reference(
        self,
        target_image: "MedicalImage",
        reference_entries: "list[tuple[np.ndarray, np.ndarray]]",
        organ_name: str,
    ) -> "SegmentedObject | None":
        """Run a single-object (K+1)-frame video session for one organ.

        reference_entries is a list of (volume_array, mask_array) pairs.
        Returns a SegmentedObject for the target frame, or None if SAM2
        produces an empty mask.
        """
        ...

    def segment_batch_iterative_per_cluster(
        self,
        target_entries: "list[tuple]",
        reference_entries: "list[tuple[np.ndarray, np.ndarray]]",
        organ_name: str,
    ) -> "dict":
        """Run a single-organ iterative (K+N)-frame session.

        reference_entries: list of (volume_array, mask_array) pairs (K frames).
        Memory accumulates from each target prediction into the next.
        Returns {path: SegmentedObject | None} for each target.
        """
        ...


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

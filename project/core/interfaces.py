from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
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


class Refiner(ABC):
    """
    Abstract base class for retroactive refinement.
 
    A Refiner attempts to recover organs that were missed during initial
    segmentation by using evidence from other images in the dataset.
    It operates after clustering and labeling, and produces new objects
    that inherit cluster labels directly (no re-clustering).
    """
 
    @abstractmethod
    def refine(
        self,
        objects_by_image: dict[Path, list[SegmentedObject]],
        labeled_by_image: dict[Path, list[LabeledObject]],
        reader: ImageReader,
    ) -> tuple[dict[Path, list[SegmentedObject]], dict[Path, list[LabeledObject]]]:
        """
        Attempt to recover missing clusters in each image.
 
        Parameters
        ----------
        objects_by_image : dict[Path, list[SegmentedObject]]
            All segmented objects grouped by image path.
        labeled_by_image : dict[Path, list[LabeledObject]]
            All labeled objects grouped by image path.
        reader : ImageReader
            Used to reload images from disk for the video predictor.
 
        Returns
        -------
        tuple of (updated objects_by_image, updated labeled_by_image)
            Both dicts are updated in-place and returned for convenience.
        """
        ...
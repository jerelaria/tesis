from abc import ABC, abstractmethod
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
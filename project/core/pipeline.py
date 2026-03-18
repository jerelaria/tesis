from project.core.interfaces import ImageReader, Segmenter, FeatureExtractor, Labeler
from project.core.data_types import LabeledObject


class Pipeline:
    """
    Orchestrates the full segmentation and labeling pipeline.

    Steps:
        1. Load image from disk
        2. Segment image into objects
        3. Extract features from each object
        4. Label objects using the fitted labeler

    NOTE: The labeler must be fitted externally before calling run().
    fit() should be called once with objects from the full dataset,
    not per image.

    Example
    -------
    >>> pipeline = Pipeline(reader, segmenter, extractor, labeler)
    >>> all_objects = []
    >>> for path in image_paths:
    ...     image = pipeline.reader.load(path)
    ...     objects = pipeline.segmenter.segment(image)
    ...     for obj in objects:
    ...         obj.features = pipeline.extractor.extract(obj)
    ...     all_objects.extend(objects)
    >>> pipeline.labeler.fit(all_objects)
    >>> results = pipeline.run("image.png")
    """

    def __init__(
        self,
        reader: ImageReader,
        segmenter: Segmenter,
        extractor: FeatureExtractor,
        labeler: Labeler,
    ):
        self.reader = reader
        self.segmenter = segmenter
        self.extractor = extractor
        self.labeler = labeler

    def run(self, image_path: str) -> list[LabeledObject]:
        """
        Run the full pipeline on a single image.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        list[LabeledObject]
            One LabeledObject per segmented region.
        """
        image = self.reader.load(image_path)

        objects = self.segmenter.segment(image)

        for obj in objects:
            obj.features = self.extractor.extract(obj)

        return self.labeler.label(objects)
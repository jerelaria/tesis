from dataclasses import dataclass
from pathlib import Path

from project.core.data_types import LabeledObject


@dataclass
class ClusterFilterConfig:
    min_image_frequency: float = 0.3    # cluster must appear in at least this fraction of images
    min_avg_labeling_confidence: float = 0.5   # minimum average clustering confidence
    min_avg_sam_confidence: float = 0.75       # minimum average SAM segmentation score


class ClusterFilter:
    """
    Evaluates clusters globally across all images and marks low-quality
    clusters as noise based on configurable thresholds.

    A cluster is marked as noise if ANY of the following conditions hold:
        - It appears in fewer images than min_image_frequency * total_images
        - Its average labeling confidence is below min_avg_labeling_confidence
        - Its average SAM confidence is below min_avg_sam_confidence

    Objects belonging to noise clusters have is_noise set to True.
    The organ_id and organ_name are preserved for traceability.

    Parameters
    ----------
    config : ClusterFilterConfig
        Thresholds for cluster quality evaluation.
    """

    def __init__(self, config: ClusterFilterConfig = ClusterFilterConfig()):
        self.config = config

    @classmethod
    def from_config(cls, yaml_path: str) -> "ClusterFilter":
        """Load thresholds from a project YAML config file."""
        import yaml
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        config = ClusterFilterConfig(**raw.get("cluster_filter", {}))
        return cls(config)

    def filter(self, labeled_by_image: dict[Path, list[LabeledObject]]) -> dict[Path, list[LabeledObject]]:
        """
        Evaluate cluster quality globally and mark noise objects in-place.

        Parameters
        ----------
        labeled_by_image : dict[Path, list[LabeledObject]]
            All labeled objects grouped by image path.

        Returns
        -------
        dict[Path, list[LabeledObject]]
            Same structure with is_noise set on low-quality cluster objects.
        """
        total_images = len(labeled_by_image)
        all_labeled = [obj for objects in labeled_by_image.values() for obj in objects]

        bad_cluster_ids = self._find_bad_clusters(all_labeled, total_images)

        if bad_cluster_ids:
            print(f"  Clusters marked as noise: {sorted(bad_cluster_ids)}")

        # Mark noise in-place
        for obj in all_labeled:
            if obj.organ_id in bad_cluster_ids:
                obj.is_noise = True

        return labeled_by_image

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_bad_clusters(self, all_labeled: list[LabeledObject], total_images: int) -> set[int]:
        """
        Compute per-cluster metrics and return the set of cluster IDs
        that fail at least one quality threshold.
        """
        cluster_ids = {obj.organ_id for obj in all_labeled}
        bad = set()

        for cid in cluster_ids:
            objects = [obj for obj in all_labeled if obj.organ_id == cid]
            metrics = self._compute_cluster_metrics(objects, total_images)

            print(f"  cluster_{cid}: "
                  f"image_freq={metrics['image_frequency']:.2f}, "
                  f"avg_labeling_confidence={metrics['avg_labeling_confidence']:.2f}, "
                  f"avg_sam_confidence={metrics['avg_sam_confidence']:.2f}")

            failed = []
            if metrics["image_frequency"] < self.config.min_image_frequency:
                failed.append(f"image_frequency={metrics['image_frequency']:.2f} < {self.config.min_image_frequency}")
            if metrics["avg_labeling_confidence"] < self.config.min_avg_labeling_confidence:
                failed.append(f"avg_labeling_confidence={metrics['avg_labeling_confidence']:.2f} < {self.config.min_avg_labeling_confidence}")
            if metrics["avg_sam_confidence"] < self.config.min_avg_sam_confidence:
                failed.append(f"avg_sam_confidence={metrics['avg_sam_confidence']:.2f} < {self.config.min_avg_sam_confidence}")

            if failed:
                print(f"  cluster_{cid} discarded → {', '.join(failed)}")
                bad.add(cid)

        return bad

    def _compute_cluster_metrics(self, objects: list[LabeledObject], total_images: int) -> dict:
        """
        Compute quality metrics for a single cluster.

        Returns
        -------
        dict with keys:
            image_frequency       : fraction of images where this cluster appears
            avg_labeling_confidence : mean clustering assignment confidence
            avg_sam_confidence    : mean SAM segmentation score
        """
        # Count unique images where this cluster appears
        unique_images = {
            obj.segmented_object.source_image.source_path
            for obj in objects
        }
        image_frequency = len(unique_images) / total_images

        avg_labeling_confidence = sum(obj.labeling_confidence for obj in objects) / len(objects)

        # SAM confidence may be None if segmenter did not provide it
        sam_scores = [
            obj.segmented_object.confidence
            for obj in objects
            if obj.segmented_object.confidence is not None
        ]
        avg_sam_confidence = sum(sam_scores) / len(sam_scores) if sam_scores else 1.0

        return {
            "image_frequency": image_frequency,
            "avg_labeling_confidence": avg_labeling_confidence,
            "avg_sam_confidence": avg_sam_confidence,
        }
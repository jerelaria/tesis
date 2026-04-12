from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

from project.core.data_types import LabeledObject


@dataclass
class ClusterFilterConfig:
    min_image_frequency: float = 0.3    # cluster must appear in at least this fraction of images
    min_avg_labeling_confidence: float = 0.5   # minimum average clustering confidence
    min_avg_sam_confidence: float = 0.75       # minimum average SAM segmentation score
    deduplicate_per_image: bool = False
    dedup_sam_score_weight: float = 0.5  # alpha for combined score: alpha * sam + (1-alpha) * cluster_conf

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


    # ------------------------------------------------------------------
    # Per-image deduplication
    # ------------------------------------------------------------------
 
    def deduplicate_per_image(
        self, labeled_by_image: dict[Path, list[LabeledObject]]
    ) -> dict[Path, list[LabeledObject]]:
        """
        For each image, if a cluster has multiple non-noise objects, keep only
        the one with the highest combined score and mark the rest as noise.
 
        This assumes organs are anatomically unique per image (e.g., one heart,
        one left lung). Should NOT be used for datasets where multiple instances
        of the same structure are valid (e.g., vertebrae, lesions).
 
        The combined score is:
            alpha * sam_score + (1 - alpha) * labeling_confidence
 
        Parameters
        ----------
        labeled_by_image : dict[Path, list[LabeledObject]]
            All labeled objects grouped by image path.
 
        Returns
        -------
        dict[Path, list[LabeledObject]]
            Same structure with duplicates marked as noise in-place.
        """
        alpha = self.config.dedup_sam_score_weight
        total_removed = 0
 
        for path, objects in labeled_by_image.items():
            # Group non-noise objects by cluster id
            cluster_groups: dict[int, list[LabeledObject]] = defaultdict(list)
            for obj in objects:
                if not obj.is_noise:
                    cluster_groups[obj.organ_id].append(obj)
 
            for cid, group in cluster_groups.items():
                if len(group) <= 1:
                    continue
 
                # Sort by combined score descending, keep the best
                group.sort(
                    key=lambda o: self._combined_score(o, alpha),
                    reverse=True,
                )
                best = group[0]
                duplicates = group[1:]
 
                for dup in duplicates:
                    dup.is_noise = True
                    total_removed += 1
 
                best_score = self._combined_score(best, alpha)
                print(
                    f"  {path.name}: cluster_{cid} had {len(group)} objects, "
                    f"kept best (score={best_score:.3f}), "
                    f"marked {len(duplicates)} as noise"
                )
 
        if total_removed > 0:
            print(f"  Deduplication: {total_removed} duplicate objects marked as noise")
        else:
            print("  Deduplication: no duplicates found")
 
        return labeled_by_image
 
    @staticmethod
    def _combined_score(obj: LabeledObject, alpha: float) -> float:
        """Compute alpha * sam_score + (1 - alpha) * labeling_confidence."""
        sam_score = obj.segmented_object.confidence or 0.0
        return alpha * sam_score + (1.0 - alpha) * obj.labeling_confidence
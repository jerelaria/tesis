"""
Retroactive refinement: recovers organs missed during initial segmentation.

After clustering, some images may be missing certain organs (e.g., a lung
obscured by a pacemaker). The refiner identifies which clusters are absent
from each image, selects K best reference masks from other images, and
uses SAM2's video predictor with all K as context frames to recover the
organ in the target image.

Key design decisions:
- Single refinement pass (not iterative): deterministic, easy to analyze.
- No re-fit after refinement: centroids from Phase 2 are used as-is.
- Label inheritance: new objects inherit the cluster label directly.
- Multi-reference video: K references as frames 0..K-1, target as frame K.
  SAM2 accumulates evidence from all K references before propagating,
  which is more robust than K independent propagations + NMS.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass

from project.core.data_types import LabeledObject, SegmentedObject
from project.core.interfaces import ImageReader, FeatureExtractor, Refiner
from project.segmentation.medsam2 import MedSAM2Segmenter
from project.segmentation.quality import (
    MaskSelectionConfig,
    find_absent_clusters,
    select_reference_masks,
    get_good_cluster_ids,
)


@dataclass
class RefinementConfig:
    """Configuration for retroactive refinement."""
    enabled: bool = False
    min_cluster_confidence: float = 0.6  # theta: threshold for "present" in image
    min_image_frequency: float = 0.3     # clusters below this are not worth refining
    mask_selection: MaskSelectionConfig = None
    improve_existing: bool = False   # Improve existing masks: re-segment objects below this combined score
    improve_min_combined_score: float = 0.6  # objects below this are candidates for improvement
    improve_sam_score_weight: float = 0.5    # alpha for evaluating existing object quality


    def __post_init__(self):
        if self.mask_selection is None:
            self.mask_selection = MaskSelectionConfig()
        elif isinstance(self.mask_selection, dict):
            self.mask_selection = MaskSelectionConfig(**self.mask_selection)


class RetroactiveRefiner(Refiner):
    """
    Recovers missing organs using SAM2's video predictor with multiple
    reference images as context.

    For each cluster C that is absent from image I:
    1. Select top-K reference objects from C in other images.
    2. Build a (K+1)-frame video: K references + target.
    3. SAM2 accumulates evidence from all K references, then propagates.
    4. New object inherits cluster C label (no re-clustering).
    5. Features are extracted for the new object.

    Parameters
    ----------
    segmenter : MedSAM2Segmenter
        Provides access to segment_with_multi_reference().
    extractor : FeatureExtractor
        Used to extract features for recovered objects.
    config : RefinementConfig
        Refinement hyperparameters.
    """

    def __init__(
        self,
        segmenter: MedSAM2Segmenter,
        extractor: FeatureExtractor,
        config: RefinementConfig,
    ):
        self.segmenter = segmenter
        self.extractor = extractor
        self.config = config

    def refine(
        self,
        objects_by_image: dict[Path, list[SegmentedObject]],
        labeled_by_image: dict[Path, list[LabeledObject]],
        reader: ImageReader,
    ) -> tuple[dict[Path, list[SegmentedObject]], dict[Path, list[LabeledObject]]]:
        """
        Single refinement pass over all images.

        For each image, identifies absent clusters and attempts to recover
        them using the video predictor with K reference frames.
        """
        # Determine which clusters are worth refining
        good_clusters = get_good_cluster_ids(
            labeled_by_image, self.config.min_image_frequency
        )

        if not good_clusters:
            print("  No good clusters found for refinement.")
            return objects_by_image, labeled_by_image

        print(f"  Good clusters for refinement: {sorted(good_clusters)}")

        total_recovered = 0
        total_attempts = 0

        for path, labeled_objects in list(labeled_by_image.items()):
            absent = find_absent_clusters(
                labeled_objects, good_clusters, self.config.min_cluster_confidence
            )

            if not absent:
                continue

            print(f"  {path.name}: absent clusters = {sorted(absent)}")

            for cluster_id in sorted(absent):
                total_attempts += 1
                recovered = self._recover_cluster(
                    cluster_id=cluster_id,
                    target_path=path,
                    labeled_by_image=labeled_by_image,
                    reader=reader,
                )

                if recovered is None:
                    continue

                # Extract features for the recovered object
                try:
                    recovered.features = self.extractor.extract(recovered)
                except ValueError as e:
                    print(f"    [SKIP] Recovered object features failed: {e}")
                    continue

                # Create LabeledObject with inherited label
                labeled_recovered = LabeledObject(
                    segmented_object=recovered,
                    organ_id=cluster_id,
                    organ_name=self._get_cluster_name(
                        cluster_id, labeled_by_image
                    ),
                    labeling_confidence=recovered.confidence or 0.0,
                    method_used="refinement_multi_reference",
                )

                # Add to both dicts
                objects_by_image[path].append(recovered)
                labeled_by_image[path].append(labeled_recovered)

                total_recovered += 1
                print(
                    f"    Recovered cluster_{cluster_id} in {path.name} "
                    f"(confidence={recovered.confidence:.3f}, "
                    f"refs={self.config.mask_selection.num_reference_frames})"
                )

        print(
            f"  Refinement complete: {total_recovered}/{total_attempts} "
            f"clusters recovered."
        )

        # ------------------------------------------------------------------
        # Step 2: Improve existing low-quality masks
        # ------------------------------------------------------------------
        if self.config.improve_existing:
            self._improve_existing_masks(
                objects_by_image, labeled_by_image, good_clusters, reader
            )

        return objects_by_image, labeled_by_image

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _combined_score(self, obj: LabeledObject, alpha: float) -> float:
        """Compute alpha * sam_score + (1 - alpha) * labeling_confidence."""
        sam_score = obj.segmented_object.confidence or 0.0
        return alpha * sam_score + (1.0 - alpha) * obj.labeling_confidence
 
    def _improve_existing_masks(
        self,
        objects_by_image: dict[Path, list[SegmentedObject]],
        labeled_by_image: dict[Path, list[LabeledObject]],
        good_clusters: set[int],
        reader: ImageReader,
    ) -> None:
        """
        Step 2: Re-segment low-quality existing objects using multi-reference
        video. If the new mask has a higher combined score, replace the original.
 
        Only considers non-noise objects in good clusters whose combined score
        is below improve_min_combined_score.
 
        Operates in-place on both dicts.
        """
        alpha = self.config.improve_sam_score_weight
        threshold = self.config.improve_min_combined_score
 
        print(f"\n  Step 2: Improving existing masks "
              f"(threshold={threshold}, alpha={alpha})")
 
        total_improved = 0
        total_candidates = 0
 
        for path, labeled_objects in list(labeled_by_image.items()):
            for labeled_obj in labeled_objects:
                if labeled_obj.is_noise:
                    continue
                if labeled_obj.organ_id not in good_clusters:
                    continue
 
                original_score = self._combined_score(labeled_obj, alpha)
                if original_score >= threshold:
                    continue
 
                total_candidates += 1
                cluster_id = labeled_obj.organ_id
 
                # Re-segment using multi-reference video
                new_obj = self._recover_cluster(
                    cluster_id=cluster_id,
                    target_path=path,
                    labeled_by_image=labeled_by_image,
                    reader=reader,
                )
 
                if new_obj is None:
                    continue
 
                # Extract features for the new object
                try:
                    new_obj.features = self.extractor.extract(new_obj)
                except ValueError as e:
                    print(f"    [SKIP] Improved object features failed: {e}")
                    continue
 
                # Evaluate new mask quality using SAM confidence only
                # (labeling_confidence is from clustering, not applicable here)
                new_sam_score = new_obj.confidence or 0.0
                # Use original labeling_confidence for comparison since
                # the cluster assignment hasn't changed
                new_score = (
                    alpha * new_sam_score
                    + (1.0 - alpha) * labeled_obj.labeling_confidence
                )
 
                if new_score <= original_score:
                    print(
                        f"    {path.name}: cluster_{cluster_id} kept original "
                        f"(original={original_score:.3f} >= new={new_score:.3f})"
                    )
                    continue
 
                # Replace: swap the mask and confidence in the existing objects
                old_seg = labeled_obj.segmented_object
                old_seg.mask = new_obj.mask
                old_seg.confidence = new_obj.confidence
                old_seg.features = new_obj.features
                labeled_obj.method_used = "refinement_improved"
 
                total_improved += 1
                print(
                    f"    {path.name}: cluster_{cluster_id} improved "
                    f"(score {original_score:.3f} -> {new_score:.3f})"
                )
 
        print(
            f"  Improvement complete: {total_improved}/{total_candidates} "
            f"masks improved."
        )


    def _recover_cluster(
        self,
        cluster_id: int,
        target_path: Path,
        labeled_by_image: dict[Path, list[LabeledObject]],
        reader: ImageReader,
    ) -> SegmentedObject | None:
        """
        Attempt to recover a single cluster in a target image by building
        a multi-reference video with K best examples from other images.
        """
        references = select_reference_masks(
            cluster_id=cluster_id,
            labeled_by_image=labeled_by_image,
            target_path=target_path,
            config=self.config.mask_selection,
        )

        if not references:
            print(f"    No valid references for cluster_{cluster_id}")
            return None

        # Load target image
        target_image = reader.load(str(target_path))

        # Build reference entries: (volume, mask) pairs
        reference_entries: list[tuple[np.ndarray, np.ndarray]] = []
        for ref_obj in references:
            ref_source = ref_obj.segmented_object.source_image
            ref_mask = ref_obj.segmented_object.mask
            reference_entries.append((ref_source.volume, ref_mask))

        organ_name = self._get_cluster_name(cluster_id, labeled_by_image)

        # Single multi-reference video call
        recovered = self.segmenter.segment_with_multi_reference(
            target_image=target_image,
            reference_entries=reference_entries,
            organ_name=organ_name,
        )

        return recovered

    def _get_cluster_name(
        self,
        cluster_id: int,
        labeled_by_image: dict[Path, list[LabeledObject]],
    ) -> str:
        """Get the organ_name assigned to a cluster (from existing objects)."""
        for objects in labeled_by_image.values():
            for obj in objects:
                if obj.organ_id == cluster_id and not obj.is_noise:
                    return obj.organ_name
        return f"cluster_{cluster_id}"
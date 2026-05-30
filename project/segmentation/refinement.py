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

import copy
import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from project.core.data_types import LabeledObject, SegmentedObject, MedicalImage
from project.core.interfaces import (
    ImageReader, FeatureExtractor, Refiner, VideoSegmenter,
)
from project.segmentation.quality import (
    MaskSelectionConfig,
    find_absent_clusters,
    select_reference_masks,
    get_good_cluster_ids,
    ReferenceIndex,
    build_reference_index,
    select_from_index,
)
from project.segmentation.utils import reference_centroid, select_closest_component


@dataclass
class RefinementConfig:
    """Configuration for retroactive refinement."""
    enabled: bool = False
    min_cluster_confidence: float = 0.6  # theta: threshold for "present" in image
    min_image_frequency: float = 0.3     # clusters below this are not worth refining
    mask_selection: MaskSelectionConfig = None
    improve_existing: bool = False   # Re-segment objects below combined score threshold
    improve_min_combined_score: float = 1.0
    improve_sam_score_weight: float = 0.2  # alpha for combined quality score

    def __post_init__(self):
        if self.mask_selection is None:
            self.mask_selection = MaskSelectionConfig()
        elif isinstance(self.mask_selection, dict):
            self.mask_selection = MaskSelectionConfig(**self.mask_selection)


# ---------------------------------------------------------------------------
# Pure module-level helpers
# ---------------------------------------------------------------------------

def _snapshot_labeled(
    labeled_by_image: dict[Path, list[LabeledObject]],
) -> dict[Path, list[LabeledObject]]:
    """Return a copy of labeled_by_image with all mask arrays deep-copied.

    Prevents in-place mask mutations during an improvement pass from
    contaminating the reference pool for subsequent iterations (cascade
    via shared mutable state).
    """
    snapshot: dict[Path, list[LabeledObject]] = {}
    for path, objs in labeled_by_image.items():
        snapped = []
        for obj in objs:
            obj_copy = copy.copy(obj)
            seg_copy = copy.copy(obj.segmented_object)
            seg_copy.mask = obj.segmented_object.mask.copy()
            obj_copy.segmented_object = seg_copy
            snapped.append(obj_copy)
        snapshot[path] = snapped
    return snapshot


def _collect_improvement_candidates(
    labeled_by_image: dict[Path, list[LabeledObject]],
    good_clusters: set[int],
    alpha: float,
    threshold: float,
) -> list[tuple[Path, LabeledObject, float]]:
    """Return (path, obj, score) for every object below the combined score threshold."""
    candidates = []
    for path, labeled_objects in labeled_by_image.items():
        for labeled_obj in labeled_objects:
            if labeled_obj.is_noise:
                continue
            if labeled_obj.organ_id not in good_clusters:
                continue
            sam_score = labeled_obj.segmented_object.confidence or 0.0
            score = alpha * sam_score + (1.0 - alpha) * labeled_obj.labeling_confidence
            if score < threshold:
                candidates.append((path, labeled_obj, score))
    return candidates


# ---------------------------------------------------------------------------
# Refiner
# ---------------------------------------------------------------------------

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
    segmenter : VideoSegmenter
        Provides segment_with_multi_reference() and encode_image().
    extractor : FeatureExtractor
        Used to extract features for recovered objects.
    config : RefinementConfig
        Refinement hyperparameters.
    extract_embeddings : bool
        If True, also extract SAM2 embeddings for recovered/improved objects.
    """

    def __init__(
        self,
        segmenter: VideoSegmenter,
        extractor: FeatureExtractor,
        config: RefinementConfig,
        extract_embeddings: bool = False,
        debug_dir: Path | None = None,
    ):
        self.segmenter = segmenter
        self.extractor = extractor
        self.config = config
        self.extract_embeddings = extract_embeddings
        self.debug_dir = debug_dir
        self._refinement_log = None  # Opened as a JSONL file handle in refine()

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
        good_clusters = get_good_cluster_ids(
            labeled_by_image, self.config.min_image_frequency
        )

        if not good_clusters:
            logger.info("No good clusters found for refinement.")
            return objects_by_image, labeled_by_image

        logger.info(f"Good clusters for refinement: {sorted(good_clusters)}")

        # Build reference index once for the entire recovery pass.
        # Replaces O(N) scan inside select_reference_masks with O(K) lookups.
        ref_index = build_reference_index(
            labeled_by_image, self.config.mask_selection
        )
        logger.info(
            f"Reference index built: "
            f"{sum(len(v) for v in ref_index._data.values())} candidates "
            f"across {len(ref_index._data)} clusters"
        )

        log_path = self.debug_dir / "refinement_log.jsonl" if self.debug_dir else None
        self._refinement_log = open(log_path, "w") if log_path else None  # noqa: SIM115

        try:
            total_recovered = 0
            total_attempts = 0

            for path, labeled_objects in list(labeled_by_image.items()):
                absent = find_absent_clusters(
                    labeled_objects, good_clusters, self.config.min_cluster_confidence
                )
                if not absent:
                    continue

                logger.info(f"{path.name}: absent clusters = {sorted(absent)}")

                target_image = reader.load(str(path))
                image_embed = (
                    self.segmenter.encode_image(target_image)
                    if self.extract_embeddings else None
                )

                for cluster_id in sorted(absent):
                    total_attempts += 1
                    recovered = self._recover_cluster(
                        cluster_id=cluster_id,
                        target_path=path,
                        labeled_by_image=labeled_by_image,
                        reader=reader,
                        context="recover",
                        reference_index=ref_index,
                    )

                    if recovered is None:
                        continue

                    if not self._extract_obj_features(recovered, image_embed):
                        continue

                    labeled_recovered = LabeledObject(
                        segmented_object=recovered,
                        organ_id=cluster_id,
                        organ_name=self._get_cluster_name(
                            cluster_id, labeled_by_image
                        ),
                        labeling_confidence=recovered.confidence or 0.0,
                        method_used="refinement_multi_reference",
                    )

                    objects_by_image[path].append(recovered)
                    labeled_by_image[path].append(labeled_recovered)

                    total_recovered += 1
                    logger.info(
                        f"Recovered cluster_{cluster_id} in {path.name} "
                        f"(confidence={recovered.confidence:.3f}, "
                        f"refs={self.config.mask_selection.num_reference_frames})"
                    )

            logger.info(
                f"Refinement complete: {total_recovered}/{total_attempts} "
                f"clusters recovered."
            )

            if self.config.improve_existing:
                self._improve_existing_masks(
                    objects_by_image, labeled_by_image, good_clusters, reader
                )

        finally:
            if self._refinement_log is not None:
                self._refinement_log.close()
                self._refinement_log = None
                logger.info(f"Refinement log saved -> {log_path}")

        return objects_by_image, labeled_by_image

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _extract_obj_features(
        self, obj: SegmentedObject, image_embed=None,
    ) -> bool:
        """
        Extract moment features and optionally SAM2 embedding for an object.

        Returns True if features were extracted successfully, False otherwise.
        """
        try:
            obj.features = self.extractor.extract(obj)
        except ValueError as e:
            logger.warning(f"[SKIP] Feature extraction failed: {e}")
            return False

        if image_embed is not None:
            from project.feature_extraction.embedding import extract_sam2_embedding
            try:
                obj.embedding = extract_sam2_embedding(obj, image_embed)
            except Exception as e:
                logger.warning(f"Embedding extraction failed: {e}")
                # Non-fatal: moment features are still available.

        return True

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

        A snapshot of labeled_by_image is taken before iteration begins so that
        in-place mask updates during this pass cannot contaminate the reference
        pool for subsequent iterations.
        """
        alpha = self.config.improve_sam_score_weight
        threshold = self.config.improve_min_combined_score

        logger.info(
            f"Step 2: Improving existing masks "
            f"(threshold={threshold}, alpha={alpha})"
        )

        snapshot = _snapshot_labeled(labeled_by_image)
        snap_index = build_reference_index(snapshot, self.config.mask_selection)
        candidates = _collect_improvement_candidates(
            labeled_by_image, good_clusters, alpha, threshold
        )

        total_improved = 0
        _current_path = None
        target_image = None
        image_embed = None

        for path, labeled_obj, original_score in candidates:
            if path is not _current_path:
                target_image = reader.load(str(path))
                image_embed = (
                    self.segmenter.encode_image(target_image)
                    if self.extract_embeddings else None
                )
                _current_path = path

            cluster_id = labeled_obj.organ_id
            new_obj = self._recover_cluster(
                cluster_id=cluster_id,
                target_path=path,
                labeled_by_image=labeled_by_image,
                reader=reader,
                context="improve",
                reference_source=snapshot,
                reference_index=snap_index,
            )

            if new_obj is None:
                continue

            if not self._extract_obj_features(new_obj, image_embed):
                continue

            new_sam_score = new_obj.confidence or 0.0
            new_score = (
                alpha * new_sam_score
                + (1.0 - alpha) * labeled_obj.labeling_confidence
            )

            if new_score <= original_score:
                logger.debug(
                    f"{path.name}: cluster_{cluster_id} kept original "
                    f"(original={original_score:.3f} >= new={new_score:.3f})"
                )
                continue

            # Replace mask in-place (snapshot is unaffected).
            old_seg = labeled_obj.segmented_object
            old_seg.mask = new_obj.mask
            old_seg.confidence = new_obj.confidence
            old_seg.features = new_obj.features
            old_seg.embedding = new_obj.embedding
            labeled_obj.method_used = "refinement_improved"

            total_improved += 1
            logger.info(
                f"{path.name}: cluster_{cluster_id} improved "
                f"(score {original_score:.3f} -> {new_score:.3f})"
            )

        logger.info(
            f"Improvement complete: {total_improved}/{len(candidates)} "
            f"masks improved."
        )

    def _recover_cluster(
        self,
        cluster_id: int,
        target_path: Path,
        labeled_by_image: dict[Path, list[LabeledObject]],
        reader: ImageReader,
        context: str = "recover",
        reference_source: dict[Path, list[LabeledObject]] | None = None,
        reference_index: ReferenceIndex | None = None,
    ) -> SegmentedObject | None:
        """
        Attempt to recover a single cluster in a target image by building
        a multi-reference video with K best examples from other images.

        After propagation, a connected-component filter keeps only the
        component whose centroid is closest to the mean centroid of the K
        reference masks, discarding spurious activations on nearby objects.

        The `context` parameter ("recover" or "improve") is recorded in the
        refinement log for post-hoc analysis.
        """
        _ref_source = reference_source if reference_source is not None \
            else labeled_by_image

        if reference_index is not None:
            references = select_from_index(
                cluster_id=cluster_id,
                target_path=target_path,
                index=reference_index,
                config=self.config.mask_selection,
            )
        else:
            references = select_reference_masks(
                cluster_id=cluster_id,
                labeled_by_image=_ref_source,
                target_path=target_path,
                config=self.config.mask_selection,
            )

        if not references:
            logger.debug(f"No valid references for cluster_{cluster_id}")
            self._log({"target": target_path.name, "cluster_id": int(cluster_id),
                       "context": context, "references": [], "outcome": "no_references"})
            return None

        alpha = self.config.mask_selection.sam_score_weight
        ref_records = []
        reference_entries: list[tuple[np.ndarray, np.ndarray]] = []

        for i, ref_obj in enumerate(references):
            ref_source = ref_obj.segmented_object.source_image
            ref_mask = ref_obj.segmented_object.mask
            # volume is None when source_image was reconstructed from cache;
            # reload from disk via source_path in that case.
            vol = ref_source.volume
            if vol is None:
                vol = reader.load(str(ref_source.source_path)).volume
            reference_entries.append((vol, ref_mask))

            sam = float(ref_obj.segmented_object.confidence or 0.0)
            conf = float(ref_obj.labeling_confidence)
            combined = alpha * sam + (1.0 - alpha) * conf
            ref_src_path = getattr(ref_source, "source_path", None)
            ref_name = Path(ref_src_path).name if ref_src_path else f"ref_{i}"
            ref_records.append({
                "index": i,
                "source": ref_name,
                "sam_score": sam,
                "labeling_confidence": conf,
                "combined_score": float(combined),
                "mask_area": int(ref_mask.sum()),
            })

        ref_cent = reference_centroid(reference_entries)

        logger.debug(
            f"[REFS] cluster_{cluster_id} target={target_path.name} ctx={context}:"
        )
        for r in ref_records:
            logger.debug(
                f"  ref_{r['index']}: {r['source']:<24}  "
                f"sam={r['sam_score']:.3f}  "
                f"conf={r['labeling_confidence']:.3f}  "
                f"combined={r['combined_score']:.3f}  "
                f"area={r['mask_area']}"
            )

        target_image = reader.load(str(target_path))
        organ_name = self._get_cluster_name(cluster_id, labeled_by_image)

        recovered = self.segmenter.segment_with_multi_reference(
            target_image=target_image,
            reference_entries=reference_entries,
            organ_name=organ_name,
        )

        if recovered is not None:
            filtered_mask = select_closest_component(recovered.mask, ref_cent)
            if filtered_mask is None or not filtered_mask.any():
                logger.debug(
                    f"[CC-FILTER] No valid component for "
                    f"cluster_{cluster_id} in {target_path.name}"
                )
                recovered = None
            elif filtered_mask.sum() < recovered.mask.sum():
                n_before = int(recovered.mask.sum())
                n_after = int(filtered_mask.sum())
                logger.debug(
                    f"[CC-FILTER] cluster_{cluster_id} {target_path.name}: "
                    f"area {n_before} -> {n_after} "
                    f"({n_before - n_after} px discarded)"
                )
                recovered.mask = filtered_mask

        log_entry: dict = {
            "target": target_path.name,
            "cluster_id": int(cluster_id),
            "context": context,
            "references": ref_records,
        }
        if recovered is None:
            log_entry["outcome"] = "propagation_failed"
        else:
            log_entry["outcome"] = "ok"
            log_entry["recovered_mask_area"] = int(recovered.mask.sum())
            log_entry["recovered_sam_confidence"] = float(recovered.confidence or 0.0)
        self._log(log_entry)

        return recovered

    def _log(self, entry: dict) -> None:
        """Write one JSON line to the refinement log if it is open."""
        if self._refinement_log is not None:
            self._refinement_log.write(json.dumps(entry) + "\n")

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

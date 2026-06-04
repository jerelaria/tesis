"""
Prototype-based propagation stage (Phase 4).

Two operating modes, selected via PropagationConfig.mode:

  independent (default)
      One (K+1)-frame video session per target image: K reference frames +
      1 target.  No temporal state is shared between images, so processing
      order does not affect results.

  iterative
      A single (K+N)-frame video session: K references followed by N targets.
      Memory accumulates from references AND previously segmented targets.

References can be:
  - Mono-organ frames built from clustering output (unsupervised Phase 3),
    one mask key per frame, interleaved by prototype index.
  - Multi-organ frames provided directly by the caller (few-shot), where
    each frame carries one mask per annotated organ.

obj_id assignment
-----------------
SAM2's video predictor does not accept obj_id=0.  Labels are mapped to
obj_ids by sorting the unique labels present in references and assigning
obj_id = position + 1:

    sorted_labels = sorted(unique_labels)   # e.g. ["cluster_0", "cluster_2"]
    label_to_obj_id = {"cluster_0": 1, "cluster_2": 2}

This mapping is stable across reference frames and is recorded in the
memory_composition output for traceability.

Reference frame ordering
------------------------
Frames should be pre-ordered by the caller (e.g. interleaved by prototype
index).  This propagator accepts any ordering and processes them as-is.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from project.core.data_types import LabeledObject, MedicalImage
from project.core.interfaces import VideoSegmenter, ImageReader, Propagator
from project.data_io.few_shot_reader import FewShotReference
from project.segmentation.quality import (
    ClusterQualityConfig,
    MaskSelectionConfig,
)
from project.segmentation.utils import reference_centroid, select_closest_component

logger = logging.getLogger(__name__)


@dataclass
class PropagationConfig:
    """Configuration for prototype-based batch propagation.

    Parameters
    ----------
    references_per_cluster : int
        Maximum number of prototype masks to use per cluster (k).
        Clusters with fewer qualifying objects use however many are available.
    quality : ClusterQualityConfig
        Three thresholds that determine which clusters represent real organs.
    mask_selection : MaskSelectionConfig
        Governs the combined_score formula (sam_score_weight) and the minimum
        score gate (min_combined_score) for prototype selection.
    mode : str
        "independent" — one video session per target image (default).
        "iterative"   — single (K+N)-frame video session for all targets.
    """
    references_per_cluster: int = 5
    quality: ClusterQualityConfig = field(default_factory=ClusterQualityConfig)
    mask_selection: MaskSelectionConfig = field(default_factory=MaskSelectionConfig)
    mode: str = "independent"

    def __post_init__(self):
        if isinstance(self.quality, dict):
            self.quality = ClusterQualityConfig(**self.quality)
        if isinstance(self.mask_selection, dict):
            self.mask_selection = MaskSelectionConfig(**self.mask_selection)
        if self.mode not in ("independent", "iterative"):
            raise ValueError(
                f"PropagationConfig.mode must be 'independent' or 'iterative', "
                f"got {self.mode!r}"
            )


class PrototypePropagator(Propagator):
    """
    Propagates prototype masks to every target image independently.

    Each target image is segmented in its own (K+1)-frame video session using
    only the K reference frames.  No temporal state is shared between images,
    so processing order does not affect results.

    Parameters
    ----------
    segmenter : VideoSegmenter
        Must implement segment_with_video_prompts(target_image, references).
        MedSAM2Segmenter satisfies this interface.
    config : PropagationConfig
        Prototype selection and cluster quality thresholds.
    """

    def __init__(self, segmenter: VideoSegmenter, config: PropagationConfig):
        self.segmenter = segmenter
        self.config = config

    def propagate(
        self,
        references: list[FewShotReference],
        frame_scores: list[float],
        target_paths: list[Path],
        reader: ImageReader,
    ) -> tuple[dict[Path, list[LabeledObject]], list[dict]]:
        """
        Propagate pre-built reference frames to every target image.

        Parameters
        ----------
        references
            Mono-organ FewShotReference frames (one mask key each).
        frame_scores
            combined_score for each frame in references (parallel list).
        target_paths
            Images to segment.
        reader
            Used to load target image volumes.

        Returns
        -------
        result
            {path: [LabeledObject, ...]} for every target path.
        memory_composition
            Per-frame metadata list for traceability and debug output.
        """
        if not references:
            logger.warning("No reference frames; propagation produced no results.")
            return {path: [] for path in target_paths}, []

        # Collect unique labels in sorted order; assign SAM2-compatible obj_ids
        labels_seen: list[str] = []
        for ref in references:
            for label in ref.masks:
                if label not in labels_seen:
                    labels_seen.append(label)
        unique_labels = sorted(labels_seen)
        label_to_obj_id: dict[str, int] = {
            label: i + 1 for i, label in enumerate(unique_labels)
        }

        # Pre-compute per-label centroid across all reference masks for CC filter
        _dummy = np.empty(0, dtype=np.float32)
        ref_centroids: dict[str, tuple[float, float]] = {}
        for label in unique_labels:
            mask_pairs = [
                (_dummy, ref.masks[label])
                for ref in references
                if label in ref.masks and ref.masks[label].any()
            ]
            ref_centroids[label] = (
                reference_centroid(mask_pairs) if mask_pairs else (0.0, 0.0)
            )

        # Build memory composition log (one entry per reference frame)
        memory_composition: list[dict] = []
        for frame_idx, (ref, score) in enumerate(zip(references, frame_scores)):
            label = next(iter(ref.masks))  # mono-organ: exactly one key
            mask = ref.masks[label]
            memory_composition.append({
                "frame_idx": frame_idx,
                "obj_id": label_to_obj_id[label],
                "label": label,
                "source_path": ref.source_path,
                "mask": mask,
                "combined_score": score,
                "area": int(mask.sum()),
            })

        # Load target images
        target_entries: list[tuple[Path, MedicalImage]] = [
            (path, reader.load(str(path))) for path in target_paths
        ]

        result: dict[Path, list[LabeledObject]] = {}
        if self.config.mode == "independent":
            n_targets = len(target_entries)
            for target_idx, (path, source_image) in enumerate(target_entries):
                if target_idx % 50 == 0 or target_idx == n_targets - 1:
                    logger.info(
                        f"  Propagating {target_idx + 1}/{n_targets}: {path.name}"
                    )
                raw_objects = self.segmenter.segment_with_video_prompts(
                    source_image, references
                )
                result[path] = self._apply_cc_filter(
                    raw_objects, label_to_obj_id, ref_centroids, path
                )
        else:  # iterative
            raw_by_path = self.segmenter.segment_batch_iterative(
                target_entries, references
            )
            for path, raw_objects in raw_by_path.items():
                result[path] = self._apply_cc_filter(
                    raw_objects, label_to_obj_id, ref_centroids, path
                )
            # Ensure every requested target is present in the output
            for path in target_paths:
                result.setdefault(path, [])

        total = sum(len(v) for v in result.values())
        logger.info(
            f"Propagation complete: {total} objects "
            f"across {len(target_paths)} images"
        )
        return result, memory_composition

    def _apply_cc_filter(
        self,
        raw_objects: list,
        label_to_obj_id: dict[str, int],
        ref_centroids: dict[str, tuple[float, float]],
        path: Path,
    ) -> list[LabeledObject]:
        """Apply connected-component filter and build LabeledObjects."""
        labeled: list[LabeledObject] = []
        for obj in raw_objects:
            if obj.label is None or obj.label not in label_to_obj_id:
                continue
            centroid = ref_centroids.get(obj.label, (0.0, 0.0))
            filtered = select_closest_component(obj.mask, centroid)
            if filtered is None or not filtered.any():
                logger.debug(f"{path.name}: {obj.label} discarded by CC filter")
                continue
            obj.mask = filtered
            labeled.append(LabeledObject(
                segmented_object=obj,
                organ_id=label_to_obj_id[obj.label],
                organ_name=obj.label,
                labeling_confidence=obj.confidence or 0.0,
                method_used="propagation",
            ))
        return labeled

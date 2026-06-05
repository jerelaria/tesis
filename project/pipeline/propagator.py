"""
Prototype-based propagation stage (Phase 4).

Two operating modes, selected via PropagationConfig.mode:

  independent (default)
      One single-object video session per (cluster, target) pair.
      K reference frames conditioned on that cluster + 1 target frame.
      No temporal state is shared between sessions.

  iterative
      One single-object video session per cluster, spanning all targets.
      K reference frames conditioned on that cluster + N target frames.
      Memory accumulates from each predicted target into the next.

In both modes every cluster gets its own isolated video session, so SAM2's
memory is never contaminated by reference frames or predictions from other
clusters.

obj_id assignment
-----------------
SAM2's video predictor does not accept obj_id=0.  Labels are mapped to
obj_ids by sorting the unique labels present in references and assigning
obj_id = position + 1:

    sorted_labels = sorted(unique_labels)   # e.g. ["cluster_0", "cluster_2"]
    label_to_obj_id = {"cluster_0": 1, "cluster_2": 2}

This mapping is recorded in the memory_composition output for traceability.

memory_composition frame_idx
-----------------------------
frame_idx in each memory_composition entry is the index of that reference
frame within its own per-cluster video session (0..K-1).  Entries are
grouped by label so the per-cluster frame sequence is always contiguous.
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
        "independent" — one session per (cluster, target) pair (default).
        "iterative"   — one session per cluster spanning all N targets;
                        memory accumulates from predicted targets.
    debug_collect_reference_predictions : bool
        When True, collect SAM2 predictions on reference frames (not just the
        target) for each per-cluster video session.  Used to visualize memory
        contamination.  Has significant runtime overhead; disable in production
        runs.
    """
    references_per_cluster: int = 5
    quality: ClusterQualityConfig = field(default_factory=ClusterQualityConfig)
    mask_selection: MaskSelectionConfig = field(default_factory=MaskSelectionConfig)
    mode: str = "independent"
    debug_collect_reference_predictions: bool = False

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
    Propagates prototype masks to every target image using per-cluster sessions.

    Each cluster runs its own isolated video session so SAM2's memory contains
    only that cluster's reference frames.  In independent mode one session is
    opened per (cluster, target) pair; in iterative mode one session per
    cluster spans all N targets, accumulating memory as targets are processed.

    Parameters
    ----------
    segmenter : VideoSegmenter
        Must implement segment_with_multi_reference and
        segment_batch_iterative_per_cluster.  MedSAM2Segmenter satisfies this.
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
    ) -> tuple[dict[Path, list[LabeledObject]], list[dict], list[dict]]:
        """
        Propagate pre-built reference frames to every target image.

        Parameters
        ----------
        references
            FewShotReference frames (mono-organ for unsupervised, may be
            multi-organ for few-shot).
        frame_scores
            combined_score for each frame in references (parallel list).
        target_paths
            Images to segment.
        reader
            Used to load image volumes when ref.volume is None.

        Returns
        -------
        result
            {path: [LabeledObject, ...]} for every target path.
        memory_composition
            Per-cluster, per-frame metadata list.  frame_idx is the index
            within that cluster's video session (0..K-1).  Entries are grouped
            by label so each cluster's sequence is contiguous.
        debug_predictions
            Per-frame SAM2 predictions on reference frames when
            debug_collect_reference_predictions is True; empty list otherwise.
        """
        if not references:
            logger.warning("No reference frames; propagation produced no results.")
            return {path: [] for path in target_paths}, [], []

        # Step 1: collect unique labels in sorted order; assign SAM2-compatible obj_ids
        labels_seen: list[str] = []
        for ref in references:
            for label in ref.masks:
                if label not in labels_seen:
                    labels_seen.append(label)
        unique_labels = sorted(labels_seen)
        label_to_obj_id: dict[str, int] = {
            label: i + 1 for i, label in enumerate(unique_labels)
        }

        # Step 2: build cluster_ref_data: label → [(vol, mask, score, source_path)]
        # One consolidated loop over references; volumes are loaded here if missing.
        cluster_ref_data: dict[str, list[tuple[np.ndarray, np.ndarray, float, str]]] = {
            label: [] for label in unique_labels
        }
        for ref, score in zip(references, frame_scores):
            vol = ref.volume
            if vol is None:
                vol = reader.load(str(ref.source_path)).volume
            for label, mask in ref.masks.items():
                cluster_ref_data[label].append(
                    (vol, mask, float(score), ref.source_path or "")
                )

        # Derived views used by the segmenter calls
        cluster_ref_entries: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
            label: [(vol, mask) for vol, mask, _, _ in data]
            for label, data in cluster_ref_data.items()
        }

        # Step 3: pre-compute per-label centroid for the CC filter
        _dummy = np.empty(0, dtype=np.float32)
        ref_centroids: dict[str, tuple[float, float]] = {}
        for label in unique_labels:
            mask_pairs = [
                (_dummy, mask)
                for _, mask, _, _ in cluster_ref_data[label]
                if mask.any()
            ]
            ref_centroids[label] = (
                reference_centroid(mask_pairs) if mask_pairs else (0.0, 0.0)
            )

        # Step 4: build memory_composition grouped by label, frame_idx per-cluster
        memory_composition: list[dict] = []
        for label in unique_labels:
            for cluster_frame_idx, (_, mask, score, source_path) in enumerate(
                cluster_ref_data[label]
            ):
                memory_composition.append({
                    "frame_idx": cluster_frame_idx,
                    "obj_id": label_to_obj_id[label],
                    "label": label,
                    "source_path": source_path,
                    "mask": mask,
                    "combined_score": score,
                    "area": int(mask.sum()),
                })

        # Step 5: load target images
        target_entries: list[tuple[Path, MedicalImage]] = [
            (path, reader.load(str(path))) for path in target_paths
        ]

        result: dict[Path, list[LabeledObject]] = {path: [] for path in target_paths}
        debug_predictions: list[dict] = []

        if self.config.mode == "independent":
            n_targets = len(target_entries)
            for target_idx, (path, target_image) in enumerate(target_entries):
                if target_idx % 50 == 0 or target_idx == n_targets - 1:
                    logger.info(
                        f"  Propagating {target_idx + 1}/{n_targets}: {path.name}"
                    )
                for label in unique_labels:
                    entries = cluster_ref_entries[label]
                    if not entries:
                        continue

                    if not self.config.debug_collect_reference_predictions:
                        obj = self.segmenter.segment_with_multi_reference(
                            target_image, entries, label
                        )
                    else:
                        obj = self._segment_with_debug(
                            target_image, entries, label,
                            [sp for _, _, _, sp in cluster_ref_data[label]],
                            debug_predictions,
                        )

                    labeled_obj = self._make_labeled_object(
                        obj, label, label_to_obj_id, ref_centroids, path
                    )
                    if labeled_obj is not None:
                        result[path].append(labeled_obj)

        else:  # iterative: one per-cluster session spanning all N targets
            n_targets = len(target_entries)
            for label in unique_labels:
                entries = cluster_ref_entries[label]
                if not entries:
                    continue
                logger.info(
                    f"  [{label}] iterative session: "
                    f"{len(entries)} refs + {n_targets} targets"
                )
                per_cluster = self.segmenter.segment_batch_iterative_per_cluster(
                    target_entries, entries, label
                )
                for path, obj in per_cluster.items():
                    labeled_obj = self._make_labeled_object(
                        obj, label, label_to_obj_id, ref_centroids, path
                    )
                    if labeled_obj is not None:
                        result[path].append(labeled_obj)

        total = sum(len(v) for v in result.values())
        logger.info(
            f"Propagation complete: {total} objects "
            f"across {len(target_paths)} images"
        )
        return result, memory_composition, debug_predictions

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_labeled_object(
        self,
        obj,
        label: str,
        label_to_obj_id: dict[str, int],
        ref_centroids: dict[str, tuple[float, float]],
        path: Path,
    ) -> LabeledObject | None:
        """Apply CC filter and wrap in a LabeledObject.  Returns None to skip."""
        if obj is None or not obj.mask.any():
            return None
        centroid = ref_centroids.get(label, (0.0, 0.0))
        filtered = select_closest_component(obj.mask, centroid)
        if filtered is None or not filtered.any():
            logger.debug(f"{path.name}: {label} discarded by CC filter")
            return None
        obj.mask = filtered
        return LabeledObject(
            segmented_object=obj,
            organ_id=label_to_obj_id[label],
            organ_name=label,
            labeling_confidence=obj.confidence or 0.0,
            method_used="propagation",
        )

    def _segment_with_debug(
        self,
        target_image: MedicalImage,
        entries: list[tuple[np.ndarray, np.ndarray]],
        label: str,
        source_paths: list[str],
        debug_predictions: list[dict],
    ):
        """Manual video session that also collects reference-frame predictions."""
        from project.segmentation.medsam2.video import (
            _video_session,
            collect_frame_results,
        )
        from project.segmentation.utils import to_uint8

        video_pred = self.segmenter._get_video_predictor()
        obj_id = 1
        K = len(entries)
        ref_uint8s = [to_uint8(vol) for vol, _ in entries]
        tgt_uint8 = to_uint8(target_image.volume)

        def _register(vp, st):
            for fi, (_, mask) in enumerate(entries):
                vp.add_new_mask(
                    inference_state=st,
                    frame_idx=fi,
                    obj_id=obj_id,
                    mask=mask.astype(np.float32),
                )
            return {obj_id: label}

        result = None
        with _video_session(video_pred, ref_uint8s, [tgt_uint8], _register) as (state, organ_map):
            for fi, obj_ids, masks in video_pred.propagate_in_video(state):
                if fi < K:
                    for pos, oid in enumerate(obj_ids):
                        logits = masks[pos].cpu().numpy().squeeze()
                        debug_predictions.append({
                            "cluster_id": label,
                            "frame_idx": fi,
                            "source_path": source_paths[fi] if fi < len(source_paths) else "",
                            "predicted_mask": logits > 0.0,
                            "is_conditioning_frame": True,
                            "is_target": False,
                        })
                elif fi == K:
                    objects = collect_frame_results(masks, obj_ids, organ_map, target_image)
                    if objects:
                        result = objects[0]
        return result

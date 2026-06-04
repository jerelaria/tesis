"""
Prototype-based batch propagation stage.

Replaces the RetroactiveRefiner + ClusterFilter pair with a single linear step:

  clustering result
      → identify good clusters (three-threshold criterion)
      → select top-K prototypes per cluster
      → build interleaved mono-organ reference frames
      → one batch video pass (all clusters × all targets)
      → CC filter per result
      → labeled_by_image (propagated masks only)

obj_id assignment
-----------------
SAM2's video predictor does not accept obj_id=0.  Cluster IDs are mapped to
obj_ids by sorting good_clusters and assigning obj_id = position + 1:

    sorted_good = sorted(good_clusters)   # e.g. [0, 2, 5]
    cluster_to_obj_id = {0: 1, 2: 2, 5: 3}

This mapping is stable across prototype index iterations and is recorded in
the memory_composition output for traceability.

Reference frame ordering
------------------------
Frames are interleaved by prototype index:

    [cid_a proto0, cid_b proto0, ..., cid_a proto1, cid_b proto1, ...]

This ensures every cluster has at least one context frame before any cluster
gets a second frame, maximising early anatomical diversity.

Chunking note
-------------
segment_batch_iterative sends all N targets in a single video session (no
chunking).  run_phase1_few_shot_iterative follows the same convention.
If VRAM is insufficient for large datasets the minimum change is to add a
chunk_size parameter to MedSAM2Segmenter.segment_batch_iterative and call
it in a loop here; this interface does not need to change.
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
    compute_cluster_quality,
    identify_good_clusters,
    select_prototypes,
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
        num_reference_frames is ignored here; references_per_cluster is used.
    """
    references_per_cluster: int = 5
    quality: ClusterQualityConfig = field(default_factory=ClusterQualityConfig)
    mask_selection: MaskSelectionConfig = field(default_factory=MaskSelectionConfig)

    def __post_init__(self):
        if isinstance(self.quality, dict):
            self.quality = ClusterQualityConfig(**self.quality)
        if isinstance(self.mask_selection, dict):
            self.mask_selection = MaskSelectionConfig(**self.mask_selection)


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
        labeled_by_image: dict[Path, list[LabeledObject]],
        target_paths: list[Path],
        reader: ImageReader,
    ) -> tuple[dict[Path, list[LabeledObject]], list[dict]]:
        # --- Step 1: identify good clusters, log rejection reasons ---
        quality_report = compute_cluster_quality(labeled_by_image, self.config.quality)
        good_clusters = {cid for cid, info in quality_report.items() if info["good"]}

        for cid, info in sorted(quality_report.items()):
            if info["good"]:
                logger.info(
                    f"  cluster_{cid}: GOOD  "
                    f"freq={info['image_frequency']:.3f}  "
                    f"lconf={info['avg_labeling_confidence']:.3f}  "
                    f"sam={info['avg_sam_confidence']}  "
                    f"n={info['n_objects']}"
                )
            else:
                logger.info(
                    f"  cluster_{cid}: FILTERED — {'; '.join(info['failed'])}"
                )

        if not good_clusters:
            logger.warning("No good clusters found; propagation produced no results.")
            return {path: [] for path in target_paths}, []
        logger.info(f"Good clusters for propagation: {sorted(good_clusters)}")

        # --- Step 2: select top-K prototypes ---
        prototypes = select_prototypes(
            labeled_by_image,
            good_clusters,
            self.config.references_per_cluster,
            self.config.mask_selection,
        )
        for cid, p in sorted(prototypes.items()):
            logger.info(f"  cluster_{cid}: {len(p)} prototypes")

        # --- Step 3: build interleaved mono-organ reference frames ---
        # Order: [cid_a p0, cid_b p0, ..., cid_a p1, cid_b p1, ...]
        sorted_clusters = sorted(good_clusters)
        cluster_to_obj_id = {cid: i + 1 for i, cid in enumerate(sorted_clusters)}

        ref_frames: list[tuple[FewShotReference, int, np.ndarray, float]] = []
        for proto_idx in range(self.config.references_per_cluster):
            for cluster_id in sorted_clusters:
                cluster_protos = prototypes[cluster_id]
                if proto_idx >= len(cluster_protos):
                    continue
                labeled_obj = cluster_protos[proto_idx]
                seg = labeled_obj.segmented_object
                vol = seg.source_image.volume
                if vol is None:
                    vol = reader.load(str(seg.source_image.source_path)).volume
                organ_name = f"cluster_{cluster_id}"
                alpha = self.config.mask_selection.sam_score_weight
                sam = seg.confidence or 0.0
                score = alpha * sam + (1.0 - alpha) * labeled_obj.labeling_confidence
                ref = FewShotReference(
                    volume=vol,
                    masks={organ_name: seg.mask},
                    source_path=seg.source_image.source_path or "",
                )
                ref_frames.append((ref, cluster_id, seg.mask, score))

        if not ref_frames:
            logger.warning("No reference frames built (all prototypes failed score gate).")
            return {path: [] for path in target_paths}, []

        logger.info(
            f"Reference frames: {len(ref_frames)} "
            f"(max {self.config.references_per_cluster}×{len(sorted_clusters)})"
        )

        references = [rf[0] for rf in ref_frames]

        # --- Step 4: memory composition log ---
        memory_composition: list[dict] = []
        for frame_idx, (ref, cluster_id, mask, score) in enumerate(ref_frames):
            memory_composition.append({
                "frame_idx": frame_idx,
                "obj_id": cluster_to_obj_id[cluster_id],
                "cluster_id": cluster_id,
                "source_path": ref.source_path,
                "mask": mask,
                "combined_score": float(score),
                "area": int(mask.sum()),
            })

        # --- Step 5: pre-compute per-cluster prototype centroids for CC filter ---
        # reference_centroid ignores the first element of each tuple
        _dummy = np.empty(0, dtype=np.float32)
        ref_centroids: dict[int, tuple[float, float]] = {}
        for cluster_id in sorted_clusters:
            proto_mask_pairs = [
                (_dummy, p.segmented_object.mask)
                for p in prototypes[cluster_id]
                if p.segmented_object.mask.any()
            ]
            ref_centroids[cluster_id] = (
                reference_centroid(proto_mask_pairs) if proto_mask_pairs else (0.0, 0.0)
            )

        # --- Step 6: load target images ---
        target_entries: list[tuple[Path, MedicalImage]] = [
            (path, reader.load(str(path))) for path in target_paths
        ]

        # --- Step 7: independent per-image propagation ---
        # Each call is a fresh (K+1)-frame video session; no state bleeds
        # between images regardless of dataset order.
        result: dict[Path, list[LabeledObject]] = {}
        n_targets = len(target_entries)
        for target_idx, (path, source_image) in enumerate(target_entries):
            if target_idx % 50 == 0 or target_idx == n_targets - 1:
                logger.info(f"  Propagating {target_idx + 1}/{n_targets}: {path.name}")
            raw_objects = self.segmenter.segment_with_video_prompts(source_image, references)

            # --- Step 8: CC filter + build LabeledObjects ---
            labeled: list[LabeledObject] = []
            for obj in raw_objects:
                if obj.label is None:
                    continue
                try:
                    cluster_id = int(obj.label.split("_")[1])
                except (IndexError, ValueError):
                    logger.warning(f"Cannot parse cluster_id from '{obj.label}', skipping")
                    continue
                if cluster_id not in good_clusters:
                    continue

                centroid = ref_centroids.get(cluster_id, (0.0, 0.0))
                filtered = select_closest_component(obj.mask, centroid)
                if filtered is None or not filtered.any():
                    logger.debug(
                        f"{path.name}: cluster_{cluster_id} discarded by CC filter"
                    )
                    continue
                obj.mask = filtered

                labeled.append(LabeledObject(
                    segmented_object=obj,
                    organ_id=cluster_id,
                    organ_name=f"cluster_{cluster_id}",
                    labeling_confidence=obj.confidence or 0.0,
                    method_used="propagation",
                ))
            result[path] = labeled

        total = sum(len(v) for v in result.values())
        logger.info(
            f"Propagation complete: {total} objects "
            f"across {len(target_paths)} images"
        )
        return result, memory_composition

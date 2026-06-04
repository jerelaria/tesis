"""Build reference frames for prototype-based propagation."""

import logging
from pathlib import Path

from project.core.data_types import LabeledObject
from project.core.interfaces import ImageReader
from project.data_io.few_shot_reader import FewShotReference
from project.pipeline.propagator import PropagationConfig
from project.segmentation.quality import (
    compute_cluster_quality,
    select_prototypes,
)

logger = logging.getLogger(__name__)


def build_fewshot_references(
    references: list[FewShotReference],
) -> tuple[list[FewShotReference], list[float]]:
    """Return human references with ground-truth frame scores of 1.0.

    Skips all clustering and quality filtering — references are already named
    by real organ names and are treated as perfect prototypes.

    Parameters
    ----------
    references
        Human-annotated FewShotReference objects loaded by
        discover_few_shot_references (may be multi-organ per frame).

    Returns
    -------
    references
        The input list unchanged.
    frame_scores
        1.0 for every frame (ground-truth confidence).
    """
    if not references:
        logger.warning("No few-shot references provided.")
        return [], []

    unique_organs = sorted({
        organ
        for ref in references
        for organ in ref.masks
    })
    logger.info(
        f"Few-shot references: {len(references)} frames, organs: {unique_organs}"
    )
    return references, [1.0] * len(references)


def build_unsupervised_references(
    labeled_by_image: dict[Path, list[LabeledObject]],
    config: PropagationConfig,
    reader: ImageReader,
) -> tuple[list[FewShotReference], list[float]]:
    """Build interleaved mono-organ reference frames from clustering output.

    Implements Steps 1–4 of the unsupervised propagation pipeline: cluster
    quality filtering, prototype selection, and interleaved frame construction.

    Parameters
    ----------
    labeled_by_image
        Clustering output: one list of LabeledObjects per source image.
    config
        Propagation config providing quality thresholds, k, and score weights.
    reader
        Used to load image volumes when source_image.volume is None.

    Returns
    -------
    references
        Mono-organ FewShotReference frames ordered by prototype index then by
        sorted cluster_id.  Each frame has masks = {"cluster_<cid>": mask}.
    frame_scores
        combined_score for each reference frame (parallel to references).
    """
    # Step 1: identify good clusters, log rejection reasons
    quality_report = compute_cluster_quality(labeled_by_image, config.quality)
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
        return [], []
    logger.info(f"Good clusters for propagation: {sorted(good_clusters)}")

    # Step 2: select top-K prototypes
    prototypes = select_prototypes(
        labeled_by_image,
        good_clusters,
        config.references_per_cluster,
        config.mask_selection,
    )
    for cid, p in sorted(prototypes.items()):
        logger.info(f"  cluster_{cid}: {len(p)} prototypes")

    # Step 3: build interleaved mono-organ reference frames
    # Order: [cid_a p0, cid_b p0, ..., cid_a p1, cid_b p1, ...]
    sorted_clusters = sorted(good_clusters)
    alpha = config.mask_selection.sam_score_weight

    references: list[FewShotReference] = []
    frame_scores: list[float] = []

    for proto_idx in range(config.references_per_cluster):
        for cluster_id in sorted_clusters:
            cluster_protos = prototypes[cluster_id]
            if proto_idx >= len(cluster_protos):
                continue
            labeled_obj = cluster_protos[proto_idx]
            seg = labeled_obj.segmented_object
            vol = seg.source_image.volume
            if vol is None:
                vol = reader.load(str(seg.source_image.source_path)).volume
            sam = seg.confidence or 0.0
            score = alpha * sam + (1.0 - alpha) * labeled_obj.labeling_confidence
            references.append(FewShotReference(
                volume=vol,
                masks={f"cluster_{cluster_id}": seg.mask},
                source_path=seg.source_image.source_path or "",
            ))
            frame_scores.append(float(score))

    if not references:
        logger.warning("No reference frames built (all prototypes failed score gate).")
        return [], []

    logger.info(
        f"Reference frames: {len(references)} "
        f"(max {config.references_per_cluster}×{len(sorted_clusters)})"
    )

    return references, frame_scores

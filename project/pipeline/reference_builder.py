"""Build reference frames for prototype-based propagation."""

import dataclasses
import logging
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from project.core.data_types import LabeledObject
from project.core.interfaces import ImageReader, VideoSegmenter
from project.data_io.few_shot_reader import FewShotReference
from project.pipeline.propagator import PropagationConfig, PrototypePropagator
from project.segmentation.quality import (
    compute_cluster_quality,
    select_prototypes,
    suppress_overlapping_clusters,
)
from project.segmentation.utils import mask_iou

logger = logging.getLogger(__name__)


def _apply_cluster_nms(
    prototypes: dict[int, list[LabeledObject]],
    quality_report: dict[int, dict],
    config: PropagationConfig,
) -> tuple[dict[int, list[LabeledObject]], list[dict]]:
    """Drop duplicate clusters (same organ) before propagation.

    Runs cluster-level NMS over the selected prototypes and logs which clusters
    were suppressed and by which survivor.  Returns the surviving prototypes and
    the suppression report (empty when NMS is disabled or nothing overlaps).
    """
    cluster_sizes = {cid: quality_report[cid]["n_objects"] for cid in prototypes}
    survivors, nms_report = suppress_overlapping_clusters(
        prototypes, cluster_sizes, config.mask_selection, config.cluster_nms,
    )
    for entry in nms_report:
        logger.info(
            f"  NMS: cluster_{entry['cluster']} suppressed by "
            f"cluster_{entry['suppressed_by']} (IoU={entry['iou']:.3f})"
        )
    if nms_report:
        logger.info(
            f"Cluster NMS suppressed {len(nms_report)} cluster(s); "
            f"survivors={sorted(survivors)}"
        )
    return survivors, nms_report


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
    for i, ref in enumerate(references):
        organ_areas = {
            organ: int(mask.sum()) for organ, mask in ref.masks.items()
        }
        logger.info(
            f"  frame_{i}: {Path(ref.source_path).name}  "
            f"organs={list(ref.masks.keys())}  "
            f"areas={organ_areas}"
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

    # Step 2b: cluster-level NMS — drop duplicate clusters of the same organ.
    prototypes, _ = _apply_cluster_nms(prototypes, quality_report, config)
    good_clusters = set(prototypes)

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


def export_references_to_fewshot(
    references: list[FewShotReference],
    out_root: Path,
) -> int:
    """Persist built reference frames as a few-shot reference directory tree.

    Writes one ``ref_NNN/`` subdirectory per reference frame, each containing
    ``image.png`` (the source slice) and one ``<organ>.png`` per mask.  The
    resulting layout under ``out_root`` is exactly what
    ``discover_few_shot_references`` expects, so the exported memory can be
    reused as the prototype pool for a few-shot run on any target dataset.

    The source image is copied verbatim from ``ref.source_path`` when that file
    exists (preserving original pixels); otherwise it is reconstructed from
    ``ref.volume``.  Masks are saved as binary 0/255 PNGs named by organ.

    Parameters
    ----------
    references
        Reference frames produced by build_unsupervised_references or
        build_unsupervised_multiorgan_references.  Mono-organ frames (one mask
        each) and multi-organ frames are both supported.
    out_root
        Destination directory, e.g. data/few_shot/ACDC.  Created if missing.

    Returns
    -------
    int
        Number of reference directories written.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    written = 0
    for i, ref in enumerate(references, start=1):
        ref_dir = out_root / f"ref_{i:03d}"
        ref_dir.mkdir(exist_ok=True)

        src = Path(ref.source_path) if ref.source_path else None
        if src is not None and src.is_file():
            shutil.copy(src, ref_dir / "image.png")
        else:
            Image.fromarray(
                ref.volume.round().clip(0, 255).astype(np.uint8)
            ).save(ref_dir / "image.png")

        for organ, mask in ref.masks.items():
            mask_img = (mask.astype(np.uint8) * 255)
            Image.fromarray(mask_img, mode="L").save(ref_dir / f"{organ}.png")

        organs = ", ".join(sorted(ref.masks))
        logger.info(f"  {ref_dir.name}: {Path(ref.source_path).name}  organs=[{organs}]")
        written += 1

    logger.info(f"Exported {written} reference frames to {out_root}")
    return written


def _merge_clusters_by_iou(
    anchor_to_masks: dict[Path, dict[str, tuple[np.ndarray, float]]],
    iou_threshold: float,
) -> tuple[dict[Path, dict[str, tuple[np.ndarray, float]]], list[dict]]:
    """Collapse clusters that represent the same organ (over-clustering).

    Two labels are connected when, over the anchors where BOTH are present, the
    median per-anchor IoU is >= iou_threshold.  Connected components (union-find)
    form one organ.  For each component with more than one member the canonical
    label is the one with the highest mean SAM (over the anchors where it is
    present); in every anchor, among the present members of the component, the
    highest-SAM mask is kept under the canonical label and the rest are dropped.

    Returns the recollapsed anchor_to_masks (keys = canonical labels) and a
    merge_report listing each collapsed group, its canonical label and the
    discarded labels.
    """
    labels = sorted({label for masks in anchor_to_masks.values() for label in masks})

    parent = {label: label for label in labels}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            l1, l2 = labels[i], labels[j]
            ious = [
                mask_iou(masks[l1][0], masks[l2][0])
                for masks in anchor_to_masks.values()
                if l1 in masks and l2 in masks
            ]
            if ious and float(np.median(ious)) >= iou_threshold:
                union(l1, l2)

    def mean_sam(label: str) -> float:
        sams = [
            masks[label][1]
            for masks in anchor_to_masks.values()
            if label in masks
        ]
        return sum(sams) / len(sams) if sams else 0.0

    components: dict[str, list[str]] = defaultdict(list)
    for label in labels:
        components[find(label)].append(label)

    canonical_of: dict[str, str] = {}
    merge_report: list[dict] = []
    for members in components.values():
        members = sorted(members)
        canonical = max(members, key=mean_sam)
        for member in members:
            canonical_of[member] = canonical
        if len(members) > 1:
            merge_report.append({
                "members": members,
                "canonical": canonical,
                "discarded": [m for m in members if m != canonical],
            })

    merged: dict[Path, dict[str, tuple[np.ndarray, float]]] = {}
    for anchor, masks in anchor_to_masks.items():
        by_canonical: dict[str, list[tuple[np.ndarray, float]]] = defaultdict(list)
        for label, entry in masks.items():
            by_canonical[canonical_of[label]].append(entry)
        merged[anchor] = {
            canonical: max(entries, key=lambda e: e[1])
            for canonical, entries in by_canonical.items()
        }

    return merged, merge_report


def _select_complete_topk_anchors(
    anchor_to_masks: dict[Path, dict[str, tuple[np.ndarray, float]]],
    organs: set[str],
    k: int,
) -> list[tuple[Path, float]]:
    """Pick the top-k anchors that carry a mask for every organ.

    An anchor is complete when it has a mask for all `organs`; this guarantees
    the final memory holds every organ.  Anchors are scored by the minimum SAM
    over their organs and ranked descending.

    Returns up to k (anchor, min_sam) pairs, best first.  Raises ValueError when
    no complete anchor exists; warns and returns all of them when fewer than k.
    """
    complete = [
        (anchor, min(masks[organ][1] for organ in organs))
        for anchor, masks in anchor_to_masks.items()
        if organs.issubset(masks.keys())
    ]
    if not complete:
        raise ValueError(
            f"No complete anchor covers all {len(organs)} organs {sorted(organs)}; "
            f"cannot build a multi-organ memory."
        )
    complete.sort(key=lambda x: -x[1])
    if len(complete) < k:
        logger.warning(
            f"Only {len(complete)} complete anchors available (< k={k}); using all."
        )
    return complete[:k]


def build_unsupervised_multiorgan_references(
    labeled_by_image: dict[Path, list[LabeledObject]],
    config: PropagationConfig,
    reader: ImageReader,
    segmenter: VideoSegmenter,
) -> tuple[list[FewShotReference], list[float]]:
    """Build K multi-organ reference frames from clustering output.

    Multi-organ variant of build_unsupervised_references (which stays the
    mono-organ path).  Prototypes are selected exactly as there; each cluster is
    then cross-propagated over the pool of prototype images so every anchor can
    carry every organ.  Native prototype masks always win over propagated ones;
    propagated masks must clear the segmenter score gate.  Over-clustered organs
    are merged by IoU, and the top-k anchors that cover all surviving organs
    become the final memory.

    Parameters
    ----------
    labeled_by_image
        Clustering output: one list of LabeledObjects per source image.
    config
        Propagation config providing quality thresholds, k, and score weights.
    reader
        Used to load image volumes when source_image.volume is None.
    segmenter
        Video segmenter; its config supplies the score and IoU thresholds used
        for the propagation gate and the cluster-merge criterion.

    Returns
    -------
    references
        Multi-organ FewShotReference frames, one per selected anchor, each with
        masks = {canonical_label: mask} for every surviving organ.
    frame_scores
        min-SAM per selected anchor (parallel to references).
    """
    # Step 1: identify good clusters and select top-K prototypes (as in mono).
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
        logger.warning("No good clusters found; no multi-organ references built.")
        return [], []
    logger.info(f"Good clusters for propagation: {sorted(good_clusters)}")

    prototypes = select_prototypes(
        labeled_by_image,
        good_clusters,
        config.references_per_cluster,
        config.mask_selection,
    )

    # Cluster-level NMS — drop duplicate clusters before the cross-fill so the
    # same organ is never propagated under two cluster ids.
    prototypes, _ = _apply_cluster_nms(prototypes, quality_report, config)
    good_clusters = set(prototypes)

    # Derive natives, the unique pool of anchors, and mono-organ refs for filling.
    native_lookup: dict[tuple[Path, str], tuple[np.ndarray, float]] = {}
    mono_refs: list[FewShotReference] = []
    frame_scores_mono: list[float] = []
    pool_paths: list[Path] = []
    seen_paths: set[Path] = set()

    logger.info("Prototypes per organ:")
    for cid in sorted(prototypes):
        label = f"cluster_{cid}"
        for labeled_obj in prototypes[cid]:
            seg = labeled_obj.segmented_object
            anchor = Path(seg.source_image.source_path or "")
            sam = seg.confidence or 0.0
            logger.info(f"  {label}: {anchor.name}  sam={sam:.3f}")

            key = (anchor, label)
            if key not in native_lookup or sam > native_lookup[key][1]:
                native_lookup[key] = (seg.mask, sam)

            vol = seg.source_image.volume
            if vol is None:
                vol = reader.load(str(anchor)).volume
            mono_refs.append(FewShotReference(
                volume=vol,
                masks={label: seg.mask},
                source_path=str(anchor),
            ))
            frame_scores_mono.append(float(sam))

            if anchor not in seen_paths:
                seen_paths.add(anchor)
                pool_paths.append(anchor)

    if not mono_refs:
        logger.warning("No prototypes passed the score gate; no references built.")
        return [], []

    # Step 2: cross-fill — propagate every cluster over every pool anchor.
    indep_config = dataclasses.replace(config, mode="independent")
    propagator = PrototypePropagator(segmenter, indep_config)
    prop_result, _, _ = propagator.propagate(
        mono_refs, frame_scores_mono, pool_paths, reader
    )

    propagated_lookup: dict[tuple[Path, str], tuple[np.ndarray, float]] = {}
    for anchor, objs in prop_result.items():
        for obj in objs:
            seg = obj.segmented_object
            propagated_lookup[(anchor, obj.organ_name)] = (
                seg.mask, seg.confidence or 0.0
            )

    # Step 3: assemble per-anchor masks (native wins; propagated must pass gate).
    all_labels = sorted({label for _, label in native_lookup})
    score_threshold = segmenter.config.score_threshold

    anchor_to_masks: dict[Path, dict[str, tuple[np.ndarray, float]]] = {}
    origin_by_anchor_label: dict[Path, dict[str, str]] = {}
    n_gate_discarded = 0
    for anchor in pool_paths:
        per_label: dict[str, tuple[np.ndarray, float]] = {}
        origins: dict[str, str] = {}
        for label in all_labels:
            native = native_lookup.get((anchor, label))
            if native is not None:
                per_label[label] = native
                origins[label] = "native"
                continue
            propagated = propagated_lookup.get((anchor, label))
            if propagated is not None:
                if propagated[1] >= score_threshold:
                    per_label[label] = propagated
                    origins[label] = "propagated"
                else:
                    n_gate_discarded += 1
        anchor_to_masks[anchor] = per_label
        origin_by_anchor_label[anchor] = origins

    # Step 4: merge over-clustered organs.
    pre_merge = anchor_to_masks
    anchor_to_masks, merge_report = _merge_clusters_by_iou(
        pre_merge, segmenter.config.iou_threshold
    )
    canonical_of: dict[str, str] = {}
    for group in merge_report:
        for member in group["members"]:
            canonical_of[member] = group["canonical"]

    # Step 5: select complete anchors and keep the top-k by min-SAM.
    organs = {label for masks in anchor_to_masks.values() for label in masks}
    n_complete = sum(
        1 for masks in anchor_to_masks.values() if organs.issubset(masks.keys())
    )
    n_incomplete = len(anchor_to_masks) - n_complete
    chosen = _select_complete_topk_anchors(
        anchor_to_masks, organs, config.references_per_cluster
    )

    # Step 6: build the final multi-organ references.
    references: list[FewShotReference] = []
    frame_scores: list[float] = []
    for anchor, min_sam in chosen:
        masks = anchor_to_masks[anchor]
        references.append(FewShotReference(
            volume=reader.load(str(anchor)).volume,
            masks={label: mask for label, (mask, _) in masks.items()},
            source_path=str(anchor),
        ))
        frame_scores.append(min_sam)

    # Logging: final memory composition, merge report and discard counts.
    def _origin(anchor: Path, canonical: str, mask: np.ndarray) -> str:
        for label, (m, _) in pre_merge[anchor].items():
            if canonical_of.get(label, label) == canonical and m is mask:
                return origin_by_anchor_label[anchor][label]
        return "propagated"

    logger.info("Final memory:")
    for anchor, min_sam in chosen:
        masks = anchor_to_masks[anchor]
        parts = [
            f"{label}({_origin(anchor, label, mask)},sam={sam:.3f})"
            for label, (mask, sam) in sorted(masks.items())
        ]
        logger.info(
            f"  {anchor.name}: organs={sorted(masks)} "
            f"{' '.join(parts)} min_sam={min_sam:.3f}"
        )
    logger.info(f"Merge report: {merge_report}")
    logger.info(
        f"Discarded anchors: {n_incomplete} incomplete; "
        f"{n_gate_discarded} (anchor, organ) propagations gated."
    )

    return references, frame_scores

"""Pure, GPU-free consolidation of over-clustered organ labels.

Given per-image predicted masks keyed by an arbitrary cluster/label name,
merge labels that consistently refer to the same organ (high mask IoU across
the images where both co-occur) into a single canonical label per organ.

This module must stay free of torch/GPU imports so it can run on an
already-produced results directory without a GPU. It only knows about
MaskRecord; it does not depend on project.core.data_types.LabeledObject.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MaskRecord:
    """A single predicted mask file for one image.

    label : str
        Cluster/organ label for this mask (file stem).
    mask : np.ndarray
        2D boolean mask.
    sam : float
        Local SAM score of this mask on this image.
    scores : dict
        Original score payload for this file (kept for write-back).
    """

    label: str
    mask: np.ndarray
    sam: float
    scores: dict


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union between two boolean masks. 0.0 if union is empty."""
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


class _UnionFind:
    def __init__(self, items):
        self._parent = {item: item for item in items}

    def find(self, x):
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def _empty_report(iou_threshold: float, min_overlap_images: int) -> dict:
    return {
        "iou_threshold": iou_threshold,
        "min_overlap_images": min_overlap_images,
        "n_labels_before": 0,
        "n_organs_after": 0,
        "n_masks_dropped_dedup": 0,
        "organ_id_map": {},
        "label_to_canonical": {},
        "groups": [],
        "pairs_with_insufficient_overlap": [],
    }


def consolidate(
    records_by_image: dict[str, list[MaskRecord]],
    iou_threshold: float = 0.5,
    min_overlap_images: int = 1,
) -> tuple[dict[str, list[MaskRecord]], dict]:
    """Collapse over-clustered labels into one canonical label per organ.

    Returns (consolidated_records_by_image, report). See module docstring
    and project design notes for the full algorithm description.
    """
    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError(f"iou_threshold must be in [0.0, 1.0], got {iou_threshold}")
    if min_overlap_images < 1:
        raise ValueError(f"min_overlap_images must be >= 1, got {min_overlap_images}")

    # Drop records with empty masks.
    filtered_by_image: dict[str, list[MaskRecord]] = {}
    for image_stem, records in records_by_image.items():
        kept = [r for r in records if r.mask.any()]
        if kept:
            filtered_by_image[image_stem] = kept

    if not filtered_by_image:
        logger.warning("consolidate: no non-empty masks in input, returning empty result")
        return {}, _empty_report(iou_threshold, min_overlap_images)

    # Unique labels + mean SAM over all remaining records.
    sam_values: dict[str, list[float]] = {}
    for records in filtered_by_image.values():
        for r in records:
            sam_values.setdefault(r.label, []).append(r.sam)
    labels = sorted(sam_values.keys())
    mean_sam = {label: float(np.mean(values)) for label, values in sam_values.items()}

    # Per-(label, image) best mask, used only for pairwise IoU computation.
    best_by_image_label: dict[str, dict[str, MaskRecord]] = {}
    for image_stem, records in filtered_by_image.items():
        best: dict[str, MaskRecord] = {}
        for r in records:
            current = best.get(r.label)
            if current is None or r.sam > current.sam:
                best[r.label] = r
        best_by_image_label[image_stem] = best

    # Pairwise IoU across co-occurring images -> merge edges.
    edges: list[tuple[str, str]] = []
    merged_median_iou: dict[tuple[str, str], float] = {}
    pairs_with_insufficient_overlap: list[list] = []
    for i, l1 in enumerate(labels):
        for l2 in labels[i + 1:]:
            ious = [
                _iou(best[l1].mask, best[l2].mask)
                for best in best_by_image_label.values()
                if l1 in best and l2 in best
            ]
            if not ious:
                continue
            if len(ious) >= min_overlap_images:
                median_iou = float(np.median(ious))
                if median_iou >= iou_threshold:
                    edges.append((l1, l2))
                    merged_median_iou[(l1, l2)] = median_iou
            else:
                pairs_with_insufficient_overlap.append([l1, l2, len(ious)])

    # Union-find over labels (sorted iteration order for determinism).
    uf = _UnionFind(labels)
    for l1, l2 in edges:
        uf.union(l1, l2)

    components: dict[str, list[str]] = {}
    for label in labels:
        root = uf.find(label)
        components.setdefault(root, []).append(label)

    # Canonical label per component: highest mean_sam, tie-break smallest label.
    label_to_canonical: dict[str, str] = {}
    members_by_canonical: dict[str, list[str]] = {}
    for members in components.values():
        canonical = sorted(members, key=lambda l: (-mean_sam[l], l))[0]
        members_by_canonical[canonical] = sorted(members)
        for label in members:
            label_to_canonical[label] = canonical

    # Contiguous, 1-based organ ids over sorted canonical labels.
    canonical_labels_sorted = sorted(members_by_canonical.keys())
    organ_id_map = {canonical: idx + 1 for idx, canonical in enumerate(canonical_labels_sorted)}

    # Multi-member groups for the report.
    groups = []
    for canonical in canonical_labels_sorted:
        members = members_by_canonical[canonical]
        if len(members) <= 1:
            continue
        discarded = sorted(m for m in members if m != canonical)
        median_iou_map = {
            f"{l1}|{l2}": iou_val
            for (l1, l2), iou_val in merged_median_iou.items()
            if l1 in members and l2 in members
        }
        groups.append({
            "canonical": canonical,
            "members": members,
            "discarded": discarded,
            "mean_sam": {m: mean_sam[m] for m in members},
            "median_iou": median_iou_map,
        })

    # Build the consolidated output: per image, dedup within each canonical group.
    consolidated_by_image: dict[str, list[MaskRecord]] = {}
    n_masks_dropped_dedup = 0
    for image_stem, records in filtered_by_image.items():
        group_records: dict[str, list[MaskRecord]] = {}
        for r in records:
            canonical = label_to_canonical[r.label]
            group_records.setdefault(canonical, []).append(r)

        survivors = []
        for canonical, candidates in group_records.items():
            if len(candidates) > 1:
                n_masks_dropped_dedup += len(candidates) - 1
            survivor = min(
                candidates,
                key=lambda r: (-r.sam, -int(r.mask.sum()), r.label),
            )
            survivors.append(MaskRecord(
                label=canonical,
                mask=survivor.mask,
                sam=survivor.sam,
                scores=survivor.scores,
            ))
        consolidated_by_image[image_stem] = survivors

    report = {
        "iou_threshold": iou_threshold,
        "min_overlap_images": min_overlap_images,
        "n_labels_before": len(labels),
        "n_organs_after": len(canonical_labels_sorted),
        "n_masks_dropped_dedup": n_masks_dropped_dedup,
        "organ_id_map": organ_id_map,
        "label_to_canonical": label_to_canonical,
        "groups": groups,
        "pairs_with_insufficient_overlap": pairs_with_insufficient_overlap,
    }
    return consolidated_by_image, report

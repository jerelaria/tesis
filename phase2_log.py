"""
phase2_log.py — Phase 2 clustering stats, no GPU, no images.

Loads Phase 1 from cache, runs clustering, logs cluster quality,
prototype details, and memory-composition frame order.
Saves summary.json and umap.png under <output-dir>/phase2_clustering/.

Requires Phase 1 cache. Generate it first:
  python main.py --config ... --dataset ... --output-dir ... --segmentation-only

Usage:
  python phase2_log.py \\
    --config configs/experiments/unsup_hdbscan_propagation.yaml \\
    --dataset Sunnybrook/images \\
    --output-dir results/phase2_test \\
    --gt-dir data/processed/Sunnybrook/masks
"""

import argparse
import logging
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from project.core.config_utils import apply_config_overrides
from project.core.logging_setup import setup_logging
from project.data_io.utils import load_image_paths
from project.labeling.clustering import ClusteringLabeler, ClusteringConfig
from project.pipeline.artifacts import load_segmented
from project.pipeline.cache_key import (
    build_fingerprint_payload,
    cache_dir_name,
    segmentation_fingerprint,
)
from project.pipeline.phase2_debug import ClusteringDebugWriter
from project.pipeline.propagator import PropagationConfig
from project.segmentation.quality import (
    compute_cluster_quality,
    select_prototypes,
    suppress_overlapping_clusters,
)

logger = logging.getLogger(__name__)

_NOISE_LABEL = -1

_ORGAN_COLORS = {
    "lv_cavity":  "#E74C3C",
    "myocardium": "#3498DB",
    "rv_cavity":  "#2ECC71",
    "none":       "#CCCCCC",
}


# ---------------------------------------------------------------------------
# UMAP helpers
# ---------------------------------------------------------------------------

def _project_2d(X: np.ndarray, umap_neighbors: int) -> dict:
    from sklearn.decomposition import PCA
    out = {"pca": PCA(n_components=2, random_state=42).fit_transform(X)}
    try:
        import umap
        out["umap"] = umap.UMAP(
            n_components=2, n_neighbors=umap_neighbors, min_dist=0.1, random_state=42
        ).fit_transform(X)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"UMAP unavailable ({e}); falling back to PCA. Install: pip install umap-learn")
        out["umap"] = None
    return out


def _strip_instance(stem: str) -> str:
    m = re.match(r"^(.+)_(\d+)$", stem)
    return m.group(1) if m else stem


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def _load_gt_for_stem(gt_dir: Path, stem: str) -> dict:
    d = gt_dir / stem
    if not d.is_dir():
        return {}
    out: dict = {}
    for f in sorted(d.iterdir()):
        if f.suffix.lower() != ".png" or f.name.lower() == "image.png":
            continue
        from PIL import Image as PILImage
        organ = _strip_instance(f.stem)
        mask = np.array(PILImage.open(f).convert("L")) > 127
        out.setdefault(organ, []).append(mask)
    return {organ: np.logical_or.reduce(masks) for organ, masks in out.items()}


def _assign_gt_organs(object_ids: list, obj_by_id: dict, gt_dir: Path, iou_min: float = 0.10) -> list[str]:
    from PIL import Image as PILImage
    gt_cache: dict = {}
    organs = []
    for oid in object_ids:
        obj = obj_by_id[oid]
        stem = Path(obj.source_image.source_path).stem
        if stem not in gt_cache:
            gt_cache[stem] = _load_gt_for_stem(gt_dir, stem)
        gt_masks = gt_cache[stem]
        if not gt_masks:
            organs.append("none")
            continue
        h, w = obj.mask.shape
        best_organ, best_iou = "none", 0.0
        for organ, gm in gt_masks.items():
            if gm.shape != (h, w):
                gm = np.array(
                    PILImage.fromarray(gm.astype(np.uint8) * 255).resize((w, h), PILImage.NEAREST)
                ) > 127
            v = _iou(obj.mask, gm)
            if v > best_iou:
                best_organ, best_iou = organ, v
        organs.append(best_organ if best_iou >= iou_min else "none")
    n_named = sum(1 for o in organs if o != "none")
    logger.info(f"GT match: {n_named}/{len(organs)} objects mapped to an organ (iou_min={iou_min})")
    return organs


def _find_highlight_clusters(clusters: np.ndarray, organs: list, organ: str | None) -> set:
    if not organ or not organs:
        return set()
    by_cluster: dict[int, Counter] = {}
    for c, o in zip(clusters, organs):
        if c == _NOISE_LABEL:
            continue
        by_cluster.setdefault(c, Counter())[o] += 1
    return {
        c for c, counts in by_cluster.items()
        if counts.most_common(1)[0][0] == organ
    }


def _cluster_cmap(n: int):
    cmap = plt.cm.tab20 if n > 10 else plt.cm.tab10
    return [cmap(i % cmap.N) for i in range(n)]


def _scatter_by_cluster(ax, xy: np.ndarray, clusters: np.ndarray, highlight: set, title: str) -> None:
    unique = sorted({c for c in clusters if c != _NOISE_LABEL})
    colors = _cluster_cmap(len(unique))
    color_of = {c: colors[i] for i, c in enumerate(unique)}

    noise_mask = clusters == _NOISE_LABEL
    if noise_mask.any():
        ax.scatter(xy[noise_mask, 0], xy[noise_mask, 1],
                   s=4, alpha=0.25, color="#BBBBBB", label="noise", rasterized=True)

    for c in unique:
        if c in highlight:
            continue
        m = clusters == c
        ax.scatter(xy[m, 0], xy[m, 1],
                   s=8, alpha=0.55, color=color_of[c], marker="o",
                   linewidths=0, label=f"C{c}", rasterized=True)

    for c in sorted(highlight):
        m = clusters == c
        ax.scatter(xy[m, 0], xy[m, 1],
                   s=28, alpha=0.9, color=color_of[c], marker="*",
                   linewidths=0.5, edgecolors="#222", label=f"C{c} ★", rasterized=True)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=1.8, fontsize=8, loc="best", framealpha=0.7, ncol=2)


def _scatter_by_organ(ax, xy: np.ndarray, organs: list, highlight_organ: str | None, title: str) -> None:
    cats = sorted(set(organs), key=lambda x: (x == "none", x))
    for cat in cats:
        if cat == highlight_organ:
            continue
        m = np.array([o == cat for o in organs])
        ax.scatter(xy[m, 0], xy[m, 1],
                   s=8, alpha=0.3 if cat == "none" else 0.6,
                   color=_ORGAN_COLORS.get(cat, "#9B59B6"), marker="o",
                   linewidths=0, label=cat, rasterized=True)
    if highlight_organ and highlight_organ in set(organs):
        m = np.array([o == highlight_organ for o in organs])
        ax.scatter(xy[m, 0], xy[m, 1],
                   s=28, alpha=0.9, color=_ORGAN_COLORS.get(highlight_organ, "#9B59B6"),
                   marker="*", linewidths=0.5, edgecolors="#222",
                   label=f"{highlight_organ} ★", rasterized=True)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=1.8, fontsize=8, loc="best", framealpha=0.7)


def _plot_umap(
    all_objects: list,
    labeler,
    phase2_dir: Path,
    gt_dir: Path | None,
    umap_neighbors: int,
    highlight_organ: str | None,
) -> None:
    import pandas as pd

    csv_path = phase2_dir / "clustering_features.csv"
    if not csv_path.exists():
        logger.warning("clustering_features.csv not found; skipping UMAP plot")
        return

    df = pd.read_csv(csv_path)
    object_ids = df["object_id"].tolist()
    X = df.drop(columns=["object_id"]).values.astype(np.float32)

    clusters = np.array([labeler._id_to_cluster.get(oid, _NOISE_LABEL) for oid in object_ids])

    gt_organs: list | None = None
    if gt_dir is not None:
        obj_by_id = {obj.id: obj for obj in all_objects}
        gt_organs = _assign_gt_organs(object_ids, obj_by_id, gt_dir)

    logger.info("Projecting features to 2D ...")
    proj = _project_2d(X, umap_neighbors)
    xy = proj["umap"] if proj.get("umap") is not None else proj["pca"]
    method = "UMAP" if proj.get("umap") is not None else "PCA"

    highlight_cl = _find_highlight_clusters(clusters, gt_organs or [], highlight_organ)
    if highlight_cl:
        logger.info(f"Highlighting cluster(s) {sorted(highlight_cl)} → dominant organ '{highlight_organ}'")

    n_panels = 2 if gt_organs is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]

    _scatter_by_cluster(axes[0], xy, clusters, highlight_cl, f"{method} — by cluster")
    if gt_organs is not None:
        _scatter_by_organ(axes[1], xy, gt_organs, highlight_organ, f"{method} — by GT organ")

    fig.suptitle("Feature-space structure", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    feature_analysis_dir = phase2_dir / "feature_analysis"
    feature_analysis_dir.mkdir(exist_ok=True)
    out = feature_analysis_dir / "umap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"UMAP plot -> {out}")


def _dataset_short(dataset_arg: str) -> str:
    p = Path(dataset_arg)
    return p.parent.name if p.name in ("images", "imgs", "data") else p.name


def _locate_cache(cfg: dict, image_paths: list, dataset_short: str, seg_cache_dir: str) -> tuple[Path, str]:
    payload = build_fingerprint_payload(
        dataset_short=dataset_short,
        image_paths=image_paths,
        mode="unsupervised",
        propagation_mode=None,
        segmenter_cfg=cfg["segmenter"],
        reference_stems=[],
    )
    dir_name = cache_dir_name(payload)
    cache_dir = Path(seg_cache_dir) / dataset_short / dir_name
    return cache_dir, dir_name


def main(args) -> None:
    results_dir = Path(args.output_dir)
    phase2_dir = results_dir / "phase2_clustering"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(verbose=args.verbose, log_file=str(phase2_dir / "phase2.log"))

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.override:
        apply_config_overrides(cfg, args.override)

    mode = cfg.get("mode", "unsupervised")
    if mode != "unsupervised":
        raise SystemExit(f"Only unsupervised mode supported, got '{mode}'")
    if not cfg.get("unsupervised", {}).get("clustering_enabled", True):
        raise SystemExit("clustering_enabled=false — nothing to cluster")

    extensions = cfg.get("dataset", {}).get("extensions", ["png"])
    image_paths = load_image_paths(args.dataset, extensions)
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found in '{args.dataset}'")
    logger.info(f"Dataset: {args.dataset}  ({len(image_paths)} images)")

    ds = _dataset_short(args.dataset)
    cache_dir, dir_name = _locate_cache(cfg, image_paths, ds, args.seg_cache_dir)
    if not cache_dir.exists():
        raise SystemExit(
            f"Phase 1 cache not found: {cache_dir}\n"
            "Run first: python main.py ... --segmentation-only"
        )

    logger.info(f"Phase 1 cache: {dir_name}")
    all_objects, objects_by_image = load_segmented(cache_dir)
    logger.info(f"Loaded {len(all_objects)} objects from {len(objects_by_image)} images")

    # -----------------------------------------------------------------------
    # Phase 2: clustering
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 2: Clustering")
    logger.info("=" * 60)

    labeler = ClusteringLabeler(ClusteringConfig(**cfg["labeler"]))
    labeler.resolve_adaptive_params(len(image_paths))
    labeler.debug_dir = phase2_dir  # enables clustering_features.csv

    labeler.fit(all_objects)

    labeled_by_image: dict[Path, list] = {}
    for path, objects in objects_by_image.items():
        labeled_by_image[path] = labeler.label(objects)

    # -----------------------------------------------------------------------
    # Quality filtering + prototype selection
    # -----------------------------------------------------------------------
    prop_cfg = PropagationConfig(**cfg.get("propagation", {}))
    quality_report = compute_cluster_quality(labeled_by_image, prop_cfg.quality)
    good_clusters = {cid for cid, info in quality_report.items() if info["good"]}
    prototypes = select_prototypes(
        labeled_by_image,
        good_clusters,
        prop_cfg.references_per_cluster,
        prop_cfg.mask_selection,
    )
    alpha = prop_cfg.mask_selection.sam_score_weight

    # -----------------------------------------------------------------------
    # Log: cluster quality
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Cluster Quality Report")
    logger.info("=" * 60)
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
    logger.info(f"Quality-good clusters: {sorted(good_clusters)}")

    # -----------------------------------------------------------------------
    # Cluster-level NMS: drop duplicate clusters (same organ) before
    # propagation, so the prototypes and memory below reflect exactly what
    # would be propagated in the final phase.
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Cluster NMS")
    logger.info("=" * 60)
    cluster_sizes = {cid: quality_report[cid]["n_objects"] for cid in prototypes}
    prototypes, nms_report = suppress_overlapping_clusters(
        prototypes, cluster_sizes, prop_cfg.mask_selection, prop_cfg.cluster_nms
    )
    if not prop_cfg.cluster_nms.enabled:
        logger.info("  disabled")
    elif not nms_report:
        logger.info("  no overlapping clusters suppressed")
    else:
        for entry in nms_report:
            logger.info(
                f"  cluster_{entry['cluster']} suppressed by "
                f"cluster_{entry['suppressed_by']} (IoU={entry['iou']:.3f})"
            )
    good_clusters = set(prototypes)
    logger.info(f"Clusters to propagate: {sorted(good_clusters)}")

    # -----------------------------------------------------------------------
    # Log: prototypes
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Prototypes")
    logger.info("=" * 60)
    for cid, proto_list in sorted(prototypes.items()):
        logger.info(f"  cluster_{cid}: {len(proto_list)} prototypes")
        for i, obj in enumerate(proto_list):
            seg = obj.segmented_object
            sam = seg.confidence or 0.0
            combined = alpha * sam + (1.0 - alpha) * obj.labeling_confidence
            src = Path(seg.source_image.source_path or "?").name
            ys, xs = np.where(seg.mask)
            area = int(seg.mask.sum())
            cx = float(xs.mean()) if len(xs) else 0.0
            cy = float(ys.mean()) if len(ys) else 0.0
            logger.info(
                f"    [{i}] src={src}  "
                f"sam={sam:.3f}  lconf={obj.labeling_confidence:.3f}  "
                f"combined={combined:.3f}  area={area}  "
                f"centroid=({cx:.0f},{cy:.0f})"
            )

    # -----------------------------------------------------------------------
    # Log: memory composition (interleaved frame order as passed to SAM2)
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Memory Composition (interleaved SAM2 reference frame order)")
    logger.info("=" * 60)
    sorted_clusters = sorted(good_clusters)
    frame_idx = 0
    for proto_idx in range(prop_cfg.references_per_cluster):
        for cid in sorted_clusters:
            cluster_protos = prototypes[cid]
            if proto_idx >= len(cluster_protos):
                continue
            obj = cluster_protos[proto_idx]
            seg = obj.segmented_object
            sam = seg.confidence or 0.0
            combined = alpha * sam + (1.0 - alpha) * obj.labeling_confidence
            src = Path(seg.source_image.source_path or "?").name
            area = int(seg.mask.sum())
            logger.info(
                f"  frame_{frame_idx:02d}: cluster_{cid}  "
                f"src={src}  combined={combined:.3f}  area={area}"
            )
            frame_idx += 1
    logger.info(f"Total reference frames: {frame_idx}")

    # -----------------------------------------------------------------------
    # Reconstruct memory_composition from prototypes (no GPU needed)
    # -----------------------------------------------------------------------
    cluster_to_obj_id = {cid: i + 1 for i, cid in enumerate(sorted(good_clusters))}
    memory_composition = []
    for cid, proto_list in sorted(prototypes.items()):
        for frame_idx, obj in enumerate(proto_list):
            seg = obj.segmented_object
            sam = seg.confidence or 0.0
            combined = alpha * sam + (1.0 - alpha) * obj.labeling_confidence
            memory_composition.append({
                "frame_idx": frame_idx,
                "obj_id": cluster_to_obj_id[cid],
                "label": cid,
                "source_path": seg.source_image.source_path,
                "mask": seg.mask,
                "combined_score": combined,
                "area": int(seg.mask.sum()),
            })

    # -----------------------------------------------------------------------
    # Save all Phase 2 outputs
    # -----------------------------------------------------------------------
    # phase2_log reconstructs the mono-organ memory from prototypes (no GPU /
    # no propagation). The real multi-organ memory needs the full pipeline, so
    # force the mono-organ memory layout here regardless of reference_mode.
    if prop_cfg.reference_mode != "monoorgan":
        logger.warning(
            "reference_mode=%s requested, but phase2_log cannot build the "
            "multi-organ memory without propagation (GPU). Rendering the "
            "mono-organ per-cluster memory figures instead.",
            prop_cfg.reference_mode,
        )
    writer_cfg = replace(prop_cfg, reference_mode="monoorgan")
    ClusteringDebugWriter(phase2_dir).write_all(
        labeled_by_image_clustering=labeled_by_image,
        labeled_by_image_propagated=labeled_by_image,  # clustering as proxy (no propagation)
        memory_composition=memory_composition,
        propagation_config=writer_cfg,
        features_csv=phase2_dir / "clustering_features.csv",
        image_paths=image_paths,
    )

    # -----------------------------------------------------------------------
    # UMAP plot
    # -----------------------------------------------------------------------
    _plot_umap(
        all_objects=all_objects,
        labeler=labeler,
        phase2_dir=phase2_dir,
        gt_dir=Path(args.gt_dir) if args.gt_dir else None,
        umap_neighbors=args.umap_neighbors,
        highlight_organ=args.highlight_organ,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase 2 clustering stats (no GPU, no images).")
    p.add_argument("--config", required=True, help="Experiment YAML config path")
    p.add_argument("--dataset", required=True, help="Dataset path under data/raw/")
    p.add_argument("--output-dir", required=True, help="Results directory")
    p.add_argument("--verbose", action="store_true", help="DEBUG-level logging")
    p.add_argument("--max-images", type=int, help="Limit dataset images")
    p.add_argument(
        "--seg-cache-dir",
        default="results/_segmentation",
        metavar="PATH",
        help="Root directory for segmentation caches (default: results/_segmentation)",
    )
    p.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="K=V",
        help="Dot-notation config overrides: key.sub=value",
    )
    p.add_argument(
        "--gt-dir",
        default=None,
        metavar="PATH",
        help="GT masks root for UMAP organ panel: <gt-dir>/<stem>/<organ>.png. "
             "Omit to generate cluster-only plot.",
    )
    p.add_argument(
        "--umap-neighbors",
        type=int,
        default=30,
        metavar="N",
        help="UMAP n_neighbors (default: 30).",
    )
    p.add_argument(
        "--highlight-organ",
        default=None,
        metavar="ORGAN",
        help="Organ name to highlight with stars in the GT panel (e.g. rv_cavity).",
    )
    main(p.parse_args())

#!/usr/bin/env python3
"""
prototype_transfer.py
---------------------
Transfer cluster prototypes from a completed Sunnybrook unsupervised run
onto ACDC images, then evaluate with semantic matching against ACDC GT.

The script reconstructs reference frames (Phase 3) by re-running Phase 2
clustering (CPU-only) from the Phase 1 segmentation cache of the source run,
then propagates those prototypes onto ACDC images using the existing
PrototypePropagator (GPU, Phase 4).

Cluster-to-organ mapping note
------------------------------
The unsupervised method produces anonymous cluster labels ("cluster_0",
"cluster_1", ...).  The --cluster-map argument provides an explicit
integer cluster_id → organ_name translation that is applied AFTER
propagation so that predictions can be evaluated with semantic matching
against named GT masks.  This mapping is an evaluation decision made by the
researcher; it is not inferred by the method itself.  Clusters absent from
the map are excluded from the output (they produce no mask files and do not
participate in evaluation).

Usage
-----
# Full run
python scripts/prototype_transfer.py \\
    --source-run results/momentos/Sunnybrook/unsup_hdbscan_propagation \\
    --acdc-images data/processed/ACDC/images \\
    --acdc-gt     data/processed/ACDC/masks \\
    --cluster-map "0=rv_cavity 1=lv_cavity 2=myocardium" \\
    --output      results/transfer_v1/ACDC/prototype_transfer

# Smoke test with 5 images (no GPU — will fail at propagation unless GPU present)
python scripts/prototype_transfer.py \\
    --source-run results/momentos/Sunnybrook/unsup_hdbscan_propagation \\
    --acdc-images data/processed/ACDC/images \\
    --acdc-gt     data/processed/ACDC/masks \\
    --cluster-map "1=lv_cavity" \\
    --max-images  5 \\
    --output      results/_smoke/ACDC/prototype_transfer
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from project.core.logging_setup import setup_logging
from project.core.data_types import LabeledObject
from project.data_io.reader import MedicalImageReader
from project.evaluation.cluster_map import parse_cluster_map
from project.evaluation.io import save_results, print_summary
from project.evaluation.runner import evaluate, _DEFAULT_IOU_THRESHOLDS
from project.labeling.clustering import ClusteringLabeler, ClusteringConfig
from project.pipeline.artifacts import load_segmented
from project.pipeline.persistence import save_predicted_masks
from project.pipeline.propagator import PrototypePropagator, PropagationConfig
from project.pipeline.reference_builder import build_unsupervised_references
from project.segmentation.medsam2 import MedSAM2Segmenter, MedSAM2Config

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SEG_CACHE_DIR = _PROJECT_ROOT / "results" / "_segmentation"


# Cluster-map parsing lives in project.evaluation.cluster_map (parse_cluster_map).


# ---------------------------------------------------------------------------
# Source-run loading
# ---------------------------------------------------------------------------

def load_source_run(source_run: Path) -> dict:
    """Load and validate resolved_config.json from a completed source run.

    Returns the config dict.  Raises if required fields are absent.
    """
    cfg_path = source_run / "resolved_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"resolved_config.json not found in source run: {source_run}\n"
            f"Expected path: {cfg_path}"
        )
    with open(cfg_path) as f:
        cfg = json.load(f)

    for required in ("seg_fingerprint", "labeler", "propagation"):
        if required not in cfg:
            raise ValueError(
                f"resolved_config.json is missing required key '{required}'. "
                f"Was this run an unsupervised propagation run?"
            )
    return cfg


def locate_seg_cache(
    cfg: dict,
    dataset_short: str,
    seg_cache_dir: Path,
) -> Path:
    """Find the Phase 1 segmentation cache directory for the source run.

    Returns the path to the directory containing segmented.npz + segmented.json.
    Raises FileNotFoundError if the cache directory or required files are absent.
    """
    fingerprint = cfg["seg_fingerprint"]
    cache_dir = seg_cache_dir / dataset_short / fingerprint

    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"Segmentation cache not found: {cache_dir}\n"
            f"Searched under seg_cache_dir={seg_cache_dir}, "
            f"dataset={dataset_short}, fingerprint={fingerprint}"
        )
    for fname in ("segmented.npz", "segmented.json"):
        if not (cache_dir / fname).exists():
            raise FileNotFoundError(
                f"Segmentation cache incomplete: {cache_dir / fname} missing."
            )
    return cache_dir


# ---------------------------------------------------------------------------
# Phase 2 re-run (CPU)
# ---------------------------------------------------------------------------

def rebuild_clustering(
    all_objects,
    objects_by_image,
    labeler_cfg: dict,
    n_images: int,
) -> dict[Path, list[LabeledObject]]:
    """Re-run Phase 2 clustering from a loaded Phase 1 cache.

    Parameters
    ----------
    all_objects
        Flat list of SegmentedObjects from load_segmented().
    objects_by_image
        Dict {path: [SegmentedObject, ...]} from load_segmented().
    labeler_cfg
        The 'labeler' sub-dict from resolved_config.json.
    n_images
        Number of images in the source dataset (used by adaptive-param
        resolution for HDBSCAN fraction-based settings).

    Returns
    -------
    labeled_by_image
        Dict {path: [LabeledObject, ...]} with cluster_N organ names.
    """
    # Remove non-ClusteringConfig keys that may be present in saved configs
    labeler_kwargs = {k: v for k, v in labeler_cfg.items()
                     if k != "standardize" or True}
    config = ClusteringConfig(**labeler_cfg)
    labeler = ClusteringLabeler(config)
    labeler.resolve_adaptive_params(n_images)

    logger.info(f"  Clustering algorithm: {config.algorithm.value}")
    if config.algorithm.value == "hdbscan":
        logger.info(
            f"  HDBSCAN params (resolved for n_images={n_images}): "
            f"min_cluster_size={labeler.config.hdbscan.min_cluster_size}, "
            f"min_samples={labeler.config.hdbscan.min_samples}"
        )

    labeler.fit(all_objects)

    labeled_by_image: dict[Path, list[LabeledObject]] = {}
    for path, objects in objects_by_image.items():
        labeled_by_image[path] = labeler.label(objects)
    return labeled_by_image


# ---------------------------------------------------------------------------
# Cluster-map application
# ---------------------------------------------------------------------------

def apply_cluster_map(
    result: dict[Path, list[LabeledObject]],
    cluster_map: dict[int, str],
) -> dict[Path, list[LabeledObject]]:
    """Rename organ_names from cluster_N to anatomical names, drop unmapped.

    For each LabeledObject whose organ_name is "cluster_N":
    - If N is in cluster_map, rename organ_name to cluster_map[N].
    - If N is not in cluster_map, exclude the object from the output.

    This renaming is applied in-place on copied LabeledObject instances.
    The original result dict is consumed; do not use it after this call.
    """
    from copy import copy

    remapped: dict[Path, list[LabeledObject]] = {}
    drop_count = 0

    for path, labeled in result.items():
        kept: list[LabeledObject] = []
        for obj in labeled:
            name = obj.organ_name
            if name.startswith("cluster_"):
                try:
                    cid = int(name[len("cluster_"):])
                except ValueError:
                    logger.warning(f"Cannot parse cluster ID from organ_name {name!r}; dropping.")
                    drop_count += 1
                    continue
                if cid not in cluster_map:
                    drop_count += 1
                    continue
                new_obj = copy(obj)
                new_obj.organ_name = cluster_map[cid]
                kept.append(new_obj)
            else:
                # Not a cluster label; keep as-is (shouldn't happen in unsup mode)
                kept.append(obj)
        remapped[path] = kept

    if drop_count > 0:
        logger.info(
            f"  Dropped {drop_count} predictions from unmapped/unparseable clusters."
        )
    return remapped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer Sunnybrook cluster prototypes onto ACDC and evaluate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-run", required=True, type=Path,
        help="Results directory of the completed Sunnybrook unsupervised run "
             "(must contain resolved_config.json with seg_fingerprint).",
    )
    parser.add_argument(
        "--acdc-images", required=True, type=Path,
        help="Directory of ACDC PNG images (flat: {stem}.png).",
    )
    parser.add_argument(
        "--acdc-gt", required=True, type=Path,
        help="Directory of ACDC GT masks (structure: {stem}/{organ}.png).",
    )
    parser.add_argument(
        "--cluster-map", required=True,
        help='Cluster ID → organ name mapping, e.g. "0=rv_cavity 1=lv_cavity 2=myocardium". '
             "This is an evaluation decision (not part of the unsupervised method).",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output directory for masks/, summary.json, metrics.csv.",
    )
    parser.add_argument(
        "--seg-cache-dir", type=Path, default=_DEFAULT_SEG_CACHE_DIR,
        help="Root of the segmentation cache (contains {dataset}/{fingerprint}/).",
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Limit the number of ACDC images processed (useful for smoke tests).",
    )
    parser.add_argument(
        "--match-threshold", type=float, default=0.5,
        help="IoU gate for semantic quality matching.",
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Skip the evaluation step (propagation only).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    # ------------------------------------------------------------------
    # 1. Parse cluster map
    # ------------------------------------------------------------------
    cluster_map = parse_cluster_map(args.cluster_map)
    logger.info(f"Cluster map: {cluster_map}")

    # ------------------------------------------------------------------
    # 2. Load source run config
    # ------------------------------------------------------------------
    source_run: Path = args.source_run.resolve()
    if not source_run.is_dir():
        raise FileNotFoundError(f"Source run not found: {source_run}")

    logger.info(f"Loading source run: {source_run}")
    source_cfg = load_source_run(source_run)

    # Derive dataset_short from the source-run path:
    # results/{version}/{DatasetShort}/{experiment} → parent = {DatasetShort}
    dataset_short = source_run.parent.name
    logger.info(f"Source dataset: {dataset_short}")

    # ------------------------------------------------------------------
    # 3. Locate and load Phase 1 segmentation cache
    # ------------------------------------------------------------------
    seg_cache_dir: Path = args.seg_cache_dir.resolve()
    cache_dir = locate_seg_cache(source_cfg, dataset_short, seg_cache_dir)
    logger.info(f"Loading Phase 1 cache: {cache_dir}")
    all_objects, objects_by_image = load_segmented(cache_dir)

    n_images_source = len(objects_by_image)
    logger.info(f"Loaded {len(all_objects)} objects from {n_images_source} source images")

    # ------------------------------------------------------------------
    # 4. Re-run Phase 2 clustering (CPU-only)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 2: Clustering (re-run from cache, CPU)")
    logger.info("=" * 60)
    labeled_by_image = rebuild_clustering(
        all_objects,
        objects_by_image,
        labeler_cfg=source_cfg["labeler"],
        n_images=n_images_source,
    )

    # ------------------------------------------------------------------
    # 5. Build reference frames from prototypes (Phase 3)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 3: Building reference frames from prototypes")
    logger.info("=" * 60)
    reader = MedicalImageReader()
    prop_cfg = PropagationConfig(**source_cfg["propagation"])
    references, frame_scores = build_unsupervised_references(
        labeled_by_image, prop_cfg, reader
    )
    if not references:
        raise RuntimeError(
            "No reference frames could be built from the source run. "
            "Check that the Sunnybrook run produced valid clusters and that "
            "the Phase 1 cache is intact."
        )
    logger.info(
        f"Built {len(references)} reference frames covering labels: "
        f"{sorted({k for r in references for k in r.masks})}"
    )

    # ------------------------------------------------------------------
    # 6. Load ACDC target images
    # ------------------------------------------------------------------
    acdc_images_dir: Path = args.acdc_images.resolve()
    if not acdc_images_dir.is_dir():
        raise FileNotFoundError(f"ACDC images directory not found: {acdc_images_dir}")

    image_paths = sorted(acdc_images_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in: {acdc_images_dir}")

    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    logger.info(f"ACDC target images: {len(image_paths)}")

    # ------------------------------------------------------------------
    # 7. Build segmenter + propagator, run Phase 4
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 4: Prototype Propagation onto ACDC (GPU)")
    logger.info("=" * 60)

    seg_cfg_raw = source_cfg.get("segmenter", {})
    seg_cfg_kwargs = {k: v for k, v in seg_cfg_raw.items() if k != "model"}
    segmenter = MedSAM2Segmenter(MedSAM2Config(**seg_cfg_kwargs))
    propagator = PrototypePropagator(segmenter=segmenter, config=prop_cfg)

    result, _memory, _debug = propagator.propagate(
        references=references,
        frame_scores=frame_scores,
        target_paths=image_paths,
        reader=reader,
    )

    # ------------------------------------------------------------------
    # 8. Apply cluster → organ rename; drop unmapped clusters
    # ------------------------------------------------------------------
    logger.info("Applying cluster map ...")
    result = apply_cluster_map(result, cluster_map)

    n_total = sum(len(v) for v in result.values())
    organs_found = sorted({obj.organ_name for v in result.values() for obj in v})
    logger.info(f"  Total predictions after mapping: {n_total}")
    logger.info(f"  Organs present: {organs_found}")

    # ------------------------------------------------------------------
    # 9. Save predicted masks
    # ------------------------------------------------------------------
    output_dir: Path = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving masks -> {output_dir}/masks/")
    save_predicted_masks(result, output_dir)

    # ------------------------------------------------------------------
    # 10. Save transfer metadata (cluster map + source run info)
    # ------------------------------------------------------------------
    meta = {
        "source_run": str(source_run),
        "source_dataset": dataset_short,
        "seg_fingerprint": source_cfg["seg_fingerprint"],
        "cluster_map": {str(k): v for k, v in cluster_map.items()},
        "n_source_images": n_images_source,
        "n_acdc_images": len(image_paths),
        "n_references": len(references),
        "organs_produced": organs_found,
        "note": (
            "cluster_map is an evaluation decision: the researcher maps "
            "anonymous cluster IDs to anatomical organ names post-hoc. "
            "The unsupervised method itself does not assign organ names."
        ),
    }
    with open(output_dir / "transfer_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved transfer_meta.json -> {output_dir / 'transfer_meta.json'}")

    # ------------------------------------------------------------------
    # 11. Evaluate against ACDC GT (semantic matching)
    # ------------------------------------------------------------------
    if args.no_eval:
        logger.info("Evaluation skipped (--no-eval).")
        return

    acdc_gt: Path = args.acdc_gt.resolve()
    if not acdc_gt.is_dir():
        raise FileNotFoundError(f"ACDC GT directory not found: {acdc_gt}")

    pred_dir = output_dir / "masks"
    if not pred_dir.is_dir():
        raise RuntimeError(f"Predictions directory not found after save: {pred_dir}")

    logger.info("=" * 60)
    logger.info("Evaluation: semantic matching vs ACDC GT")
    logger.info("=" * 60)
    logger.info(f"  GT:   {acdc_gt}")
    logger.info(f"  Pred: {pred_dir}")

    all_results, summary = evaluate(
        gt_dir=acdc_gt,
        pred_dir=pred_dir,
        matching="semantic",
        iou_thresholds=_DEFAULT_IOU_THRESHOLDS,
        match_threshold=args.match_threshold,
    )

    save_results(all_results, summary, output_dir)
    print_summary(summary)
    logger.info(f"Results saved -> {output_dir}")


if __name__ == "__main__":
    main()

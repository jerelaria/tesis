"""Segmentation cache resolution: hit/miss/compute/raise state machine."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from project.pipeline.artifacts import load_segmented, save_segmented, write_cache_key
from project.pipeline.cache_key import (
    build_fingerprint_payload,
    cache_dir_name,
    segmentation_fingerprint,
)

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResolution:
    """Result of resolving the segmentation cache for a run.

    preloaded   – (all_objects, objects_by_image) on cache hit; None otherwise.
    seg_fingerprint – cache dir name to record in resolved_config.json.
    need_segmenter  – feed to build_pipeline_from_config; True when Phase 1
                      or propagation will use the GPU model.
    compute_fn  – call as compute_fn(pipeline) after building the pipeline when
                  not None; runs Phase 1 and writes the cache, then returns
                  (all_objects, objects_by_image). None on cache hit or bypass.
    """

    preloaded: tuple | None
    seg_fingerprint: str | None
    need_segmenter: bool
    compute_fn: Callable | None = field(default=None, repr=False)


def resolve_segmentation(
    cfg: dict,
    image_paths: list[Path],
    references: list | None,
    mode: str,
    propagation_mode: str | None,
    args,
    dataset_short: str,
) -> SegmentationResolution:
    """Determine segmentation cache state and return a resolution.

    Owns: payload/fingerprint construction, hit/miss/compute/raise decision,
    and the single compute-and-write block (captured in compute_fn).
    """
    if args.no_seg_cache or mode == "few_shot":
        # few_shot skips Phase 1 entirely; no dataset segmentation cache needed.
        return SegmentationResolution(
            preloaded=None,
            seg_fingerprint=None,
            need_segmenter=True,
        )

    reference_stems = (
        [Path(r.source_path).stem for r in references] if references else []
    )
    payload = build_fingerprint_payload(
        dataset_short=dataset_short,
        image_paths=image_paths,
        mode=mode,
        propagation_mode=propagation_mode,
        segmenter_cfg=cfg["segmenter"],
        reference_stems=reference_stems,
    )
    hash8 = segmentation_fingerprint(payload)
    dir_name = cache_dir_name(payload)
    cache_dir = Path(args.seg_cache_dir) / dataset_short / dir_name

    # Propagation always needs the GPU model; only skip loading it when
    # clustering is disabled (baseline mode, early-exit path).
    _mode = cfg.get("mode", "unsupervised")
    if _mode == "few_shot":
        clustering_enabled = cfg.get("few_shot", {}).get("clustering_enabled", True)
    else:
        clustering_enabled = cfg.get("unsupervised", {}).get("clustering_enabled", True)

    cache_hit = cache_dir.exists() and not args.refresh_seg
    will_compute = args.segmentation_only or args.compute_seg_if_missing or args.refresh_seg

    if cache_hit:
        logger.info(f"Segmentation cache hit: {dir_name}")
        if args.segmentation_only:
            logger.info(
                f"Segmentation already cached (--segmentation-only is no-op): {cache_dir}"
            )
        return SegmentationResolution(
            preloaded=load_segmented(cache_dir),
            seg_fingerprint=dir_name,
            need_segmenter=clustering_enabled,
        )

    if not will_compute:
        raise SystemExit(
            f"Segmentation cache not found for fingerprint {dir_name} "
            f"under {args.seg_cache_dir}. "
            f"Generate it first with --segmentation-only, or pass "
            f"--compute-seg-if-missing to compute it intentionally."
        )

    logger.info(
        "Segmentation cache will be refreshed"
        if args.refresh_seg and cache_dir.exists()
        else f"Segmentation cache miss: {dir_name}"
    )

    # Capture all cache state in the closure so the call site is pipeline-only.
    def compute_fn(pipeline):
        all_objects, objects_by_image = pipeline.phase1(
            image_paths, force_embeddings=True
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        save_segmented(
            objects_by_image, cache_dir, hash8,
            dataset_short, mode, propagation_mode or "-",
        )
        write_cache_key(cache_dir, payload, hash8)
        logger.info(f"Segmentation cache written: {cache_dir}")
        return all_objects, objects_by_image

    return SegmentationResolution(
        preloaded=None,
        seg_fingerprint=dir_name,
        need_segmenter=True,
        compute_fn=compute_fn,
    )

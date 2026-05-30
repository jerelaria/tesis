"""CLI entry point. Loads YAML, builds Pipeline, runs it, saves results."""

import argparse
import logging
from pathlib import Path

import yaml

from project.core.config_utils import apply_config_overrides, save_resolved_config
from project.core.logging_setup import setup_logging
from project.data_io.utils import load_image_paths
from project.pipeline.orchestrator import build_pipeline_from_config

logger = logging.getLogger(__name__)


def _exclude_few_shot_images(image_paths: list[Path], references: list) -> list[Path]:
    ref_ids = set()
    for ref in references:
        p = Path(ref.source_path)
        ref_ids.add(p.parent.name)
        ref_ids.add(p.stem)
    original = len(image_paths)
    filtered = [p for p in image_paths if p.stem not in ref_ids]
    excluded = original - len(filtered)
    if excluded > 0:
        logger.info(f"Excluded {excluded} images matching few-shot references")
    else:
        logger.info("No dataset images matched few-shot references (0 excluded)")
    return filtered


def main(args) -> None:
    setup_logging(verbose=args.verbose)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.override:
        apply_config_overrides(cfg, args.override)
    mode = cfg.get("mode", "unsupervised")
    logger.info(f"Mode: {mode}")
    logger.info(f"Experiment: {cfg.get('experiment', {}).get('name', 'unnamed')}")
    logger.info(f"Dataset: {args.dataset}")

    extensions = cfg.get("dataset", {}).get("extensions", ["png"])
    image_paths = load_image_paths(args.dataset, extensions)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found for dataset '{args.dataset}'")
    logger.info(f"Found {len(image_paths)} images.")

    # TODO(text_guided): Text-guided dispatch was removed in Iteration 1 to keep the
    # CLI decoupled from the concrete segmenter. Re-introduce an `elif mode ==
    # "text_guided":` block here once a VideoSegmenter-compatible text backend exists.

    references, references_info = None, None
    if mode == "few_shot":
        if args.num_refs is None:
            raise ValueError("--num-refs is required for few_shot mode.")
        from project.data_io.few_shot_reader import discover_few_shot_references
        logger.info(f"Loading few-shot references (K={args.num_refs})...")
        references = discover_few_shot_references(
            dataset_name=args.dataset,
            num_refs=args.num_refs,
            ref_names=args.ref_images,
        )
        image_paths = _exclude_few_shot_images(image_paths, references)
        logger.info(f"Dataset after exclusion: {len(image_paths)} images")
        references_info = {
            "num_refs": len(references),
            "ref_sources": [r.source_path for r in references],
            "ref_organs": {
                Path(r.source_path).parent.name: list(r.masks.keys())
                for r in references
            },
        }

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    pipeline = build_pipeline_from_config(cfg, results_dir, len(image_paths), references)
    pipeline.run(image_paths, references=references)
    save_resolved_config(
        cfg=cfg,
        results_dir=results_dir,
        config_path=args.config,
        dataset_name=args.dataset,
        num_images=len(image_paths),
        references_info=references_info,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Co-segmentation pipeline.")
    p.add_argument("--config", required=True, help="Experiment YAML config path")
    p.add_argument("--dataset", required=True, help="Dataset path under data/raw/")
    p.add_argument("--output-dir", required=True, help="Results directory")
    p.add_argument("--verbose", action="store_true", help="DEBUG-level logging")
    p.add_argument("--max-images", type=int, help="Limit dataset images")
    p.add_argument("--num-refs", type=int, help="Few-shot reference count")
    p.add_argument("--ref-images", nargs="*", help="Reference directory names")
    p.add_argument("--override", nargs="*", default=[], metavar="K=V",
                   help="Dot-notation config overrides: key.sub=value")
    main(p.parse_args())

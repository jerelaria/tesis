"""
Main entry point for the co-segmentation and anatomical labeling pipeline.

Supports three modes configured via YAML:
- unsupervised: grid-based segmentation + clustering + optional refinement
- few_shot: video predictor with K reference images + clustering + semantic mapping
- text_guided: MedSAM3 text prompts + optional clustering

Few-shot propagation modes:
- independent: (K+1)-frame video per target (K refs + 1 target). Order-invariant.
- iterative: (K+N)-frame video (K refs + all targets). Memory accumulates.

Usage:
    python -m main --config configs/experiments/experiment.yaml
"""

import yaml
import argparse
import numpy as np
from pathlib import Path

from project.data_io.reader import MedicalImageReader
from project.data_io.utils import load_image_paths
from project.segmentation.medsam2 import MedSAM2Segmenter, MedSAM2Config
from project.feature_extraction.moments import MomentFeatureExtractor
from project.labeling.clustering import ClusteringLabeler, ClusteringConfig
from project.labeling.clustering_filter import ClusterFilter, ClusterFilterConfig
from project.evaluation.visualizer import (
    save_segmentation_vis,
    save_visualization,
)


def main(config_path: str, output_dir_override: str = None, dataset_override: str = None, max_images_override: int = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    mode = cfg.get("mode", "unsupervised")
    print(f"Mode: {mode}")
    print(f"Experiment: {cfg.get('experiment', {}).get('name', 'unnamed')}")

    dataset_name = dataset_override or cfg["dataset"]["name"]
    image_paths = load_image_paths(
        dataset_name,
        cfg["dataset"]["extensions"],
    )

    max_images = max_images_override or cfg["dataset"].get("max_images")
    if max_images is not None:
        image_paths = image_paths[:max_images]

    if not image_paths:
        raise FileNotFoundError(
            f"No images found for dataset '{cfg['dataset']['name']}'"
        )

    print(f"Found {len(image_paths)} images.")

    results_dir = Path(output_dir_override) if output_dir_override else Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    if mode in ("unsupervised", "few_shot"):
        _run_unsupervised_or_fewshot(cfg, mode, image_paths, results_dir)
    elif mode == "text_guided":
        _run_text_guided(cfg, image_paths, results_dir)
    else:
        raise ValueError(
            f"Unknown mode: '{mode}'. "
            "Must be one of: unsupervised, few_shot, text_guided"
        )


# ======================================================================
# UNSUPERVISED / FEW-SHOT PIPELINE
# ======================================================================

def _run_unsupervised_or_fewshot(
    cfg: dict, mode: str, image_paths: list[Path], results_dir: Path
):
    reader    = MedicalImageReader()
    extractor = MomentFeatureExtractor()
    labeler   = ClusteringLabeler(ClusteringConfig(**cfg["labeler"]))

    seg_cfg = dict(cfg["segmenter"])
    seg_cfg.pop("model", None)
    segmenter = MedSAM2Segmenter(MedSAM2Config(**seg_cfg))

    # Load few-shot references
    references = None
    propagation_mode = "independent"
    if mode == "few_shot":
        from project.data_io.few_shot_reader import load_few_shot_references

        fs_cfg = cfg["few_shot"]
        propagation_mode = fs_cfg.get("propagation_mode", "independent")

        print("Loading few-shot references...")
        references = load_few_shot_references(
            references_dir=fs_cfg["references_dir"],
            references_config=fs_cfg["references"],
        )
        print(f"  Propagation mode: {propagation_mode}")

    # ------------------------------------------------------------------
    # Phase 1: Segmentation + Feature Extraction
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Phase 1: Segmentation + Feature Extraction")
    print("="*60)

    phase1_dir = results_dir / "phase1_segmentation"
    phase1_dir.mkdir(exist_ok=True)

    if mode == "unsupervised":
        all_objects, objects_by_image = _phase1_unsupervised(
            image_paths, reader, segmenter, extractor, phase1_dir,
        )
    elif propagation_mode == "iterative":
        all_objects, objects_by_image = _phase1_few_shot_iterative(
            image_paths, reader, segmenter, extractor,
            references, phase1_dir,
        )
    else:
        all_objects, objects_by_image = _phase1_few_shot_independent(
            image_paths, reader, segmenter, extractor,
            references, phase1_dir,
        )

    print(f"\nTotal objects: {len(all_objects)}")
    print(f"Unique IDs: {len(set(obj.id for obj in all_objects))}")

    # ------------------------------------------------------------------
    # Phase 2: Global Clustering
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Phase 2: Global Clustering")
    print("="*60)

    labeler.fit(all_objects)

    phase2_dir = results_dir / "phase2_clustering"
    phase2_dir.mkdir(exist_ok=True)

    labeled_by_image: dict[Path, list] = {}
    for path, objects in objects_by_image.items():
        labeled = labeler.label(objects)
        labeled_by_image[path] = labeled

        for obj in labeled:
            print(
                f"  {path.name} | {obj.segmented_object.id[:8]}... -> "
                f"{obj.organ_name} (conf={obj.labeling_confidence:.2f})"
            )
        save_visualization(path, labeled, phase2_dir, suffix="_clustered")

    # ------------------------------------------------------------------
    # Phase 3: Semantic Cluster Mapping (few-shot only)
    # ------------------------------------------------------------------
    if mode == "few_shot":
        print("\n" + "="*60)
        print("Phase 3: Semantic Cluster Mapping")
        print("="*60)

        from project.labeling.semantic_mapper import ClusterSemanticMapper

        phase3_dir = results_dir / "phase3_semantic"
        phase3_dir.mkdir(exist_ok=True)

        semantic_mapper = ClusterSemanticMapper()
        labeled_by_image = semantic_mapper.map(labeled_by_image)

        for path, labeled in labeled_by_image.items():
            save_visualization(path, labeled, phase3_dir, suffix="_semantic")

    # ------------------------------------------------------------------
    # Phase 4: Retroactive Refinement (if enabled)
    # ------------------------------------------------------------------
    refinement_cfg = cfg.get("refinement", {})
    if refinement_cfg.get("enabled", False):
        print("\n" + "="*60)
        print("Phase 4: Retroactive Refinement")
        print("="*60)

        from project.segmentation.refinement import (
            RetroactiveRefiner, RefinementConfig,
        )

        phase4_dir = results_dir / "phase4_refinement"
        phase4_dir.mkdir(exist_ok=True)

        refiner = RetroactiveRefiner(
            segmenter=segmenter,
            extractor=extractor,
            config=RefinementConfig(**refinement_cfg),
        )
        objects_by_image, labeled_by_image = refiner.refine(
            objects_by_image, labeled_by_image, reader
        )

        for path, labeled in labeled_by_image.items():
            save_visualization(path, labeled, phase4_dir, suffix="_refined")
    else:
        print("\nPhase 4: Refinement skipped (disabled)")

    # ------------------------------------------------------------------
    # Phase 5: Cluster Filtering
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Phase 5: Cluster Filtering")
    print("="*60)
 
    phase5_dir = results_dir / "phase5_filtered"
    phase5_dir.mkdir(exist_ok=True)
 
    cluster_filter = ClusterFilter(ClusterFilterConfig(**cfg["cluster_filter"]))
    labeled_by_image = cluster_filter.filter(labeled_by_image)
 
    if cfg["cluster_filter"].get("deduplicate_per_image", False):
        print("\n  Per-image deduplication:")
        labeled_by_image = cluster_filter.deduplicate_per_image(labeled_by_image)
 
    for path, labeled in labeled_by_image.items():
        save_visualization(path, labeled, phase5_dir, suffix="_final")

    _save_predicted_masks(labeled_by_image, results_dir)

    _print_summary(all_objects, labeled_by_image)


# ======================================================================
# PHASE 1 VARIANTS
# ======================================================================

def _phase1_unsupervised(
    image_paths, reader, segmenter, extractor, phase1_dir,
):
    """Grid-based segmentation (unsupervised mode)."""
    all_objects = []
    objects_by_image = {}

    for idx, path in enumerate(image_paths):
        print(f"\n  [{idx+1}/{len(image_paths)}] {path.name}")

        image = reader.load(str(path))
        grid_objects = segmenter.segment(image)
        print(f"    Grid: {len(grid_objects)} objects")

        valid = _extract_features(grid_objects, extractor)
        if valid:
            objects_by_image[path] = valid
            all_objects.extend(valid)
            save_segmentation_vis(path, valid, phase1_dir)
        else:
            print(f"    [SKIP] no valid objects")

    return all_objects, objects_by_image


def _phase1_few_shot_independent(
    image_paths, reader, segmenter, extractor, references, phase1_dir,
):
    """
    Few-shot independent: per-image (K+1)-frame video.
    K references + 1 target per call. No grid.
    """
    all_objects = []
    objects_by_image = {}

    for idx, path in enumerate(image_paths):
        print(f"\n  [{idx+1}/{len(image_paths)}] {path.name}")

        image = reader.load(str(path))

        fs_objects = segmenter.segment_with_video_prompts(
            target_image=image,
            references=references,
        )
        labels = ', '.join(o.label for o in fs_objects if o.label)
        print(f"    Independent ({len(references)} refs): "
              f"{len(fs_objects)} objects ({labels})")

        valid = _extract_features(fs_objects, extractor)
        if valid:
            objects_by_image[path] = valid
            all_objects.extend(valid)
            save_segmentation_vis(path, valid, phase1_dir)
        else:
            print(f"    [SKIP] no valid objects")

    return all_objects, objects_by_image


def _phase1_few_shot_iterative(
    image_paths, reader, segmenter, extractor, references, phase1_dir,
):
    """
    Few-shot iterative: single (K+N)-frame video.
    K references + N targets. Memory accumulates. No grid.
    """
    print("  Loading all images...")
    images_by_path = {}
    for path in image_paths:
        images_by_path[path] = reader.load(str(path))

    print("  Running iterative video predictor...")
    target_entries = [(path, images_by_path[path]) for path in image_paths]
    fs_by_image = segmenter.segment_batch_iterative(
        target_entries=target_entries,
        references=references,
    )

    print("  Extracting features...")
    all_objects = []
    objects_by_image = {}

    for path in image_paths:
        fs_objs = fs_by_image.get(path, [])
        if fs_objs:
            labels = ', '.join(o.label for o in fs_objs if o.label)
            print(f"    {path.name}: {len(fs_objs)} objects ({labels})")
        else:
            print(f"    {path.name}: 0 objects")

        valid = _extract_features(fs_objs, extractor)
        if valid:
            objects_by_image[path] = valid
            all_objects.extend(valid)
            save_segmentation_vis(path, valid, phase1_dir)
        else:
            print(f"    [SKIP] {path.name}")

    return all_objects, objects_by_image


# ======================================================================
# TEXT-GUIDED PIPELINE
# ======================================================================

def _run_text_guided(cfg: dict, image_paths: list[Path], results_dir: Path):
    from project.segmentation.medsam3 import MedSAM3Segmenter, MedSAM3Config

    reader    = MedicalImageReader()
    extractor = MomentFeatureExtractor()

    seg_cfg = dict(cfg["segmenter"])
    seg_cfg.pop("model", None)
    tg_cfg = cfg.get("text_guided", {})
    if "prompts" in tg_cfg:
        seg_cfg["prompts"] = tg_cfg["prompts"]

    segmenter = MedSAM3Segmenter(MedSAM3Config(**seg_cfg))
    clustering_enabled = tg_cfg.get("clustering_enabled", True)

    # Phase 1
    print("\n" + "="*60)
    print("Phase 1: MedSAM3 Segmentation")
    print("="*60)

    phase1_dir = results_dir / "phase1_segmentation"
    phase1_dir.mkdir(exist_ok=True)

    all_objects = []
    objects_by_image = {}

    for idx, path in enumerate(image_paths):
        print(f"\n  [{idx+1}/{len(image_paths)}] {path.name}")
        image = reader.load(str(path))
        objects = segmenter.segment(image)
        print(f"    MedSAM3: {len(objects)} objects")

        valid = _extract_features(objects, extractor)
        if valid:
            objects_by_image[path] = valid
            all_objects.extend(valid)
            save_segmentation_vis(path, valid, phase1_dir)

    print(f"\nTotal objects: {len(all_objects)}")

    if clustering_enabled:
        print("\n" + "="*60)
        print("Phase 2: Clustering")
        print("="*60)

        phase2_dir = results_dir / "phase2_clustering"
        phase2_dir.mkdir(exist_ok=True)

        labeler = ClusteringLabeler(ClusteringConfig(**cfg["labeler"]))
        labeler.fit(all_objects)

        labeled_by_image = {}
        for path, objects in objects_by_image.items():
            labeled = labeler.label(objects)
            labeled_by_image[path] = labeled
            save_visualization(path, labeled, phase2_dir, suffix="_clustered")

        print("\n" + "="*60)
        print("Phase 3: Cluster Filtering")
        print("="*60)

        phase3_dir = results_dir / "phase3_filtered"
        phase3_dir.mkdir(exist_ok=True)

        cluster_filter = ClusterFilter(
            ClusterFilterConfig(**cfg["cluster_filter"])
        )
        labeled_by_image = cluster_filter.filter(labeled_by_image)

        for path, labeled in labeled_by_image.items():
            save_visualization(path, labeled, phase3_dir, suffix="_final")
    else:
        print("\nPhase 2-3: Clustering skipped")
        from project.core.data_types import LabeledObject

        final_dir = results_dir / "final"
        final_dir.mkdir(exist_ok=True)
        labeled_by_image = {}

        for path, objects in objects_by_image.items():
            labeled = [
                LabeledObject(
                    segmented_object=obj, organ_id=i,
                    organ_name=obj.label or "unknown",
                    labeling_confidence=obj.confidence or 0.0,
                    method_used="text_guided_medsam3",
                )
                for i, obj in enumerate(objects)
            ]
            labeled_by_image[path] = labeled
            save_visualization(path, labeled, final_dir, suffix="_final")

    _save_predicted_masks(labeled_by_image, results_dir)

    _print_summary(all_objects, labeled_by_image)


# ======================================================================
# Shared
# ======================================================================

def _extract_features(objects, extractor) -> list:
    valid = []
    for obj in objects:
        try:
            obj.features = extractor.extract(obj)
            obj.mask = _keep_largest_component(obj.mask)
            valid.append(obj)
        except ValueError as e:
            print(f"    [SKIP] obj {obj.id[:8]}...: {e}")
    return valid


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Replace mask with its largest connected component."""
    from scipy.ndimage import label as cc_label
    labeled, n = cc_label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    return labeled == sizes.argmax()

def _save_predicted_masks(
    labeled_by_image: dict[Path, list], results_dir: Path,
) -> None:
    """Save final predicted masks as binary PNGs for evaluation."""
    from PIL import Image as PILImage

    masks_dir = results_dir / "masks"
    print(f"\n  Saving predicted masks -> {masks_dir}")

    count = 0
    for path, labeled in labeled_by_image.items():
        image_dir = masks_dir / path.stem
        image_dir.mkdir(parents=True, exist_ok=True)

        # Group objects by organ name to determine if numbering is needed
        by_name: dict[str, list] = {}
        for obj in labeled:
            if obj.is_noise:
                continue
            by_name.setdefault(obj.organ_name, []).append(obj)

        for name, objs in by_name.items():
            for i, obj in enumerate(objs, start=1):
                # Only add suffix if there are multiple masks with the same name
                filename = f"{name}_{i}.png" if len(objs) > 1 else f"{name}.png"
                mask_uint8 = (obj.segmented_object.mask * 255).astype(np.uint8)
                PILImage.fromarray(mask_uint8, mode="L").save(image_dir / filename)
                count += 1

    print(f"  Saved {count} masks across {len(labeled_by_image)} images")

def _print_summary(all_objects, labeled_by_image):
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    total_labeled = sum(len(v) for v in labeled_by_image.values())
    noise_count = sum(
        1 for objs in labeled_by_image.values()
        for obj in objs if obj.is_noise
    )
    organ_names = set(
        obj.organ_name for objs in labeled_by_image.values()
        for obj in objs if not obj.is_noise
    )
    method_counts: dict[str, int] = {}
    for objs in labeled_by_image.values():
        for obj in objs:
            m = obj.method_used
            method_counts[m] = method_counts.get(m, 0) + 1

    print(f"  Total segmented: {len(all_objects)}")
    print(f"  Total labeled: {total_labeled}")
    print(f"  Noise: {noise_count}")
    print(f"  Organs: {sorted(organ_names)}")
    print(f"  Images: {len(labeled_by_image)}")
    print(f"  Methods: {method_counts}")

    features = [o.features for o in all_objects if o.features is not None]
    if features:
        X = np.stack(features)
        print(f"\n  Features: {X.shape}")
        print(f"  Min: {X.min(axis=0)}")
        print(f"  Max: {X.max(axis=0)}")
        print(f"  Std: {X.std(axis=0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None,
                        help="Override output.results_dir from YAML")
    parser.add_argument("--dataset", default=None,
                        help="Override dataset.name from YAML")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Override dataset.max_images from YAML")
    args = parser.parse_args()
    main(args.config, output_dir_override=args.output_dir,
            dataset_override=args.dataset, max_images_override=args.max_images)
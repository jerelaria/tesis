"""Pipeline orchestrator: executes the five-phase co-segmentation pipeline."""

import logging
from pathlib import Path

from project.core.data_types import LabeledObject
from project.data_io.reader import MedicalImageReader
from project.evaluation.visualizer import save_visualization
from project.feature_extraction.moments import MomentFeatureExtractor
from project.labeling.clustering import ClusteringLabeler, ClusteringConfig
from project.labeling.clustering_filter import ClusterFilter, ClusterFilterConfig
from project.segmentation.medsam2 import MedSAM2Segmenter, MedSAM2Config
from project.pipeline.phase1_unsupervised import run_phase1_unsupervised
from project.pipeline.phase1_few_shot import (
    run_phase1_few_shot_independent,
    run_phase1_few_shot_iterative,
)
from project.pipeline.persistence import save_predicted_masks, print_summary

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        mode: str,
        reader: MedicalImageReader,
        segmenter: MedSAM2Segmenter | None,
        extractor: MomentFeatureExtractor,
        labeler: ClusteringLabeler | None,
        refiner,
        cluster_filter: ClusterFilter,
        results_dir: Path,
        extract_embeddings: bool = False,
        propagation_mode: str = "independent",
    ):
        self.mode = mode
        self.reader = reader
        self.segmenter = segmenter
        self.extractor = extractor
        self.labeler = labeler
        self.refiner = refiner
        self.cluster_filter = cluster_filter
        self.results_dir = results_dir
        self.extract_embeddings = extract_embeddings
        self.propagation_mode = propagation_mode

    def run(
        self,
        image_paths,
        references=None,
        preloaded: tuple | None = None,
    ) -> dict[Path, list[LabeledObject]]:
        if preloaded is not None:
            all_objects, objects_by_image = preloaded
            logger.info(
                f"Phase 1 skipped: using {len(all_objects)} cached objects "
                f"from {len(objects_by_image)} images"
            )
        else:
            all_objects, objects_by_image = self.phase1(image_paths, references)
        logger.info(f"Total objects: {len(all_objects)}")

        if self.labeler is None:
            return self._baseline_label(all_objects, objects_by_image, references)

        labeled_by_image = self.phase2(all_objects, objects_by_image)

        if self.mode == "few_shot":
            labeled_by_image = self.phase3(labeled_by_image)

        if self.refiner is not None:
            objects_by_image, labeled_by_image = self.phase4(
                objects_by_image, labeled_by_image
            )

        labeled_by_image = self.phase5(labeled_by_image)
        save_predicted_masks(labeled_by_image, self.results_dir)
        print_summary(all_objects, labeled_by_image)
        return labeled_by_image

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    def phase1(self, image_paths, references, force_embeddings: bool = False):
        logger.info("=" * 60)
        logger.info("Phase 1: Segmentation + Feature Extraction")
        logger.info("=" * 60)

        if self.segmenter is None:
            raise RuntimeError(
                "Segmenter not available for Phase 1. "
                "Pass need_segmenter=True to build_pipeline_from_config."
            )

        phase1_dir = self.results_dir / "phase1_segmentation"
        phase1_dir.mkdir(exist_ok=True)

        extract_embeddings = self.extract_embeddings or force_embeddings

        if self.mode == "unsupervised":
            return run_phase1_unsupervised(
                image_paths, self.reader, self.segmenter, self.extractor,
                phase1_dir, extract_embeddings=extract_embeddings,
            )
        if self.propagation_mode == "independent":
            return run_phase1_few_shot_independent(
                image_paths, self.reader, self.segmenter, self.extractor,
                references, phase1_dir, extract_embeddings=extract_embeddings,
            )
        return run_phase1_few_shot_iterative(
            image_paths, self.reader, self.segmenter, self.extractor,
            references, phase1_dir, extract_embeddings=extract_embeddings,
        )

    def phase2(self, all_objects, objects_by_image):
        logger.info("=" * 60)
        logger.info("Phase 2: Clustering")
        logger.info("=" * 60)

        phase2_dir = self.results_dir / "phase2_clustering"
        phase2_dir.mkdir(exist_ok=True)
        self.labeler.debug_dir = phase2_dir

        self.labeler.fit(all_objects)

        labeled_by_image = {}
        for path, objects in objects_by_image.items():
            labeled = self.labeler.label(objects)
            labeled_by_image[path] = labeled
            save_visualization(path, labeled, phase2_dir, suffix="_clustered")

        return labeled_by_image

    def phase3(self, labeled_by_image):
        logger.info("=" * 60)
        logger.info("Phase 3: Semantic Cluster Mapping")
        logger.info("=" * 60)

        from project.labeling.semantic_mapper import ClusterSemanticMapper

        phase3_dir = self.results_dir / "phase3_semantic"
        phase3_dir.mkdir(exist_ok=True)

        labeled_by_image = ClusterSemanticMapper().map(labeled_by_image)

        for path, labeled in labeled_by_image.items():
            save_visualization(path, labeled, phase3_dir, suffix="_semantic")

        return labeled_by_image

    def phase4(self, objects_by_image, labeled_by_image):
        """Phase 4 updates both dicts: refiner may add recovered objects to both."""
        logger.info("=" * 60)
        logger.info("Phase 4: Retroactive Refinement")
        logger.info("=" * 60)

        phase4_dir = self.results_dir / "phase4_refinement"
        phase4_dir.mkdir(exist_ok=True)

        objects_by_image, labeled_by_image = self.refiner.refine(
            objects_by_image, labeled_by_image, self.reader
        )

        for path, labeled in labeled_by_image.items():
            save_visualization(path, labeled, phase4_dir, suffix="_refined")

        return objects_by_image, labeled_by_image

    def phase5(self, labeled_by_image):
        logger.info("=" * 60)
        logger.info("Phase 5: Cluster Filtering")
        logger.info("=" * 60)

        phase5_dir = self.results_dir / "phase5_filtered"
        phase5_dir.mkdir(exist_ok=True)

        labeled_by_image = self.cluster_filter.filter(labeled_by_image)

        if self.cluster_filter.config.deduplicate_per_image:
            logger.info("Per-image deduplication:")
            labeled_by_image = self.cluster_filter.deduplicate_per_image(
                labeled_by_image
            )

        for path, labeled in labeled_by_image.items():
            save_visualization(path, labeled, phase5_dir, suffix="_final")

        return labeled_by_image

    # ------------------------------------------------------------------
    # Baseline early-exit (clustering disabled)
    # ------------------------------------------------------------------

    def _baseline_label(self, all_objects, objects_by_image, references):
        logger.info("[BASELINE] clustering_enabled=false -> "
                    "skipping phases 2-5, using MedSAM2 labels directly")

        organ_name_to_id: dict[str, int] = {}
        if self.mode == "few_shot" and references is not None:
            for ref in references:
                for name in ref.masks:
                    if name not in organ_name_to_id:
                        organ_name_to_id[name] = len(organ_name_to_id) + 1

        baseline_tag = (
            "few_shot_baseline" if self.mode == "few_shot"
            else "unsupervised_baseline"
        )

        labeled_by_image: dict[Path, list[LabeledObject]] = {}
        for path, objects in objects_by_image.items():
            labeled = []
            for local_idx, obj in enumerate(objects):
                if self.mode == "few_shot":
                    organ_name = obj.label or "unknown"
                    organ_id = organ_name_to_id.get(obj.label, 0)
                else:
                    # Unsupervised: positional name avoids PNG collisions.
                    organ_name = f"obj_{local_idx:03d}"
                    organ_id = local_idx

                labeled.append(LabeledObject(
                    segmented_object=obj,
                    organ_id=organ_id,
                    organ_name=organ_name,
                    labeling_confidence=obj.confidence or 0.0,
                    method_used=baseline_tag,
                ))
            labeled_by_image[path] = labeled

        save_predicted_masks(labeled_by_image, self.results_dir)
        print_summary(all_objects, labeled_by_image)
        return labeled_by_image


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

def build_pipeline_from_config(
    cfg: dict,
    results_dir: Path,
    num_images: int,
    references: list | None,
    need_segmenter: bool = True,
) -> Pipeline:
    """Construct a Pipeline from a resolved YAML config dict.

    Set need_segmenter=False when Phase 1 output is loaded from cache and
    refinement is disabled; this skips the expensive GPU model load.
    """
    mode = cfg.get("mode", "unsupervised")

    reader = MedicalImageReader()
    extractor = MomentFeatureExtractor()

    if need_segmenter:
        seg_cfg = dict(cfg["segmenter"])
        seg_cfg.pop("model", None)
        segmenter: MedSAM2Segmenter | None = MedSAM2Segmenter(MedSAM2Config(**seg_cfg))
    else:
        segmenter = None
        logger.info("Segmenter not loaded (using cached segmentation)")

    # Clustering can be disabled to benchmark raw MedSAM2 output as a
    # minimal-pipeline baseline.
    clustering_enabled = True
    if mode == "few_shot":
        clustering_enabled = cfg.get("few_shot", {}).get("clustering_enabled", True)
    elif mode == "unsupervised":
        clustering_enabled = cfg.get("unsupervised", {}).get("clustering_enabled", True)

    labeler = None
    if clustering_enabled:
        if mode == "few_shot" and references:
            organ_names = set()
            for ref in references:
                organ_names.update(ref.masks.keys())
            n_clusters = len(organ_names)
            cfg.setdefault("labeler", {})
            cfg["labeler"].setdefault("kmeans", {})
            cfg["labeler"]["kmeans"]["n_clusters"] = n_clusters
            logger.info(
                f"Inferred n_clusters={n_clusters} from references: "
                f"{sorted(organ_names)}"
            )

        labeler = ClusteringLabeler(ClusteringConfig(**cfg["labeler"]))
        labeler.resolve_adaptive_params(num_images)

        labeler_features = cfg.get("labeler", {}).get("features", None)
        if labeler_features is None:
            logger.info("Moment features: NONE (embeddings-only mode)")
        else:
            logger.info(f"Moment features: {len(labeler.config.features)} selected")
    else:
        logger.info("[BASELINE] clustering_enabled=false -> "
                    "skipping phases 2-5, using MedSAM2 labels directly")

    extract_embeddings = (
        cfg.get("labeler", {}).get("embedding", {}).get("enabled", False)
    )
    if extract_embeddings:
        logger.info("Embedding extraction: ENABLED")

    propagation_mode = "independent"
    if mode == "few_shot":
        propagation_mode = cfg.get("few_shot", {}).get(
            "propagation_mode", "independent"
        )
        logger.info(f"Propagation mode: {propagation_mode}")

    refiner = None
    refinement_cfg = cfg.get("refinement", {})
    if refinement_cfg.get("enabled", False):
        from project.segmentation.refinement import RetroactiveRefiner, RefinementConfig
        refiner = RetroactiveRefiner(
            segmenter=segmenter,
            extractor=extractor,
            config=RefinementConfig(**refinement_cfg),
            labeler=labeler,
            extract_embeddings=extract_embeddings,
            debug_dir=results_dir / "phase4_refinement",
        )

    cluster_filter = ClusterFilter(ClusterFilterConfig(**cfg.get("cluster_filter", {})))

    return Pipeline(
        mode=mode,
        reader=reader,
        segmenter=segmenter,
        extractor=extractor,
        labeler=labeler,
        refiner=refiner,
        cluster_filter=cluster_filter,
        results_dir=results_dir,
        extract_embeddings=extract_embeddings,
        propagation_mode=propagation_mode,
    )

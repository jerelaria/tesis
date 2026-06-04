"""Pipeline orchestrator: clustering → propagation."""

import logging
from pathlib import Path

from project.core.data_types import LabeledObject
from project.data_io.reader import MedicalImageReader
from project.evaluation.visualizer import save_visualization
from project.feature_extraction.moments import MomentFeatureExtractor
from project.labeling.clustering import ClusteringLabeler, ClusteringConfig
from project.segmentation.medsam2 import MedSAM2Segmenter, MedSAM2Config
from project.pipeline.phase1_unsupervised import run_phase1_unsupervised
from project.pipeline.persistence import save_predicted_masks, print_summary
from project.pipeline.propagator import PropagationConfig, PrototypePropagator
from project.pipeline.reference_builder import (
    build_fewshot_references,
    build_unsupervised_references,
)
from project.pipeline.phase2_debug import ClusteringDebugWriter

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Linear co-segmentation pipeline:

      Unsupervised: Phase 1 Segmentation+Features → Phase 2 Clustering
                    → Phase 3 Reference Selection → Phase 4 Propagation
      Few-shot:     Phase 4 Propagation only (references provided by caller)
    """

    def __init__(
        self,
        mode: str,
        reader: MedicalImageReader,
        segmenter: MedSAM2Segmenter | None,
        extractor: MomentFeatureExtractor,
        labeler: ClusteringLabeler | None,
        propagator: PrototypePropagator | None,
        results_dir: Path,
        extract_embeddings: bool = False,
    ):
        self.mode = mode
        self.reader = reader
        self.segmenter = segmenter
        self.extractor = extractor
        self.labeler = labeler
        self.propagator = propagator
        self.results_dir = results_dir
        self.extract_embeddings = extract_embeddings
        self.debug_writer: ClusteringDebugWriter | None = None

    def run(
        self,
        image_paths,
        references=None,
        preloaded: tuple | None = None,
    ) -> dict[Path, list[LabeledObject]]:
        if self.mode == "few_shot":
            return self._run_few_shot(image_paths, references)

        if preloaded is not None:
            all_objects, objects_by_image = preloaded
            logger.info(
                f"Phase 1 skipped: using {len(all_objects)} cached objects "
                f"from {len(objects_by_image)} images"
            )
        else:
            all_objects, objects_by_image = self.phase1(image_paths)
        logger.info(f"Total objects: {len(all_objects)}")

        if self.labeler is None:
            return self._baseline_label(all_objects, objects_by_image, references)

        labeled_by_image_clustering = self.phase2(all_objects, objects_by_image)

        labeled_by_image, memory = self.phase4_propagate(
            labeled_by_image_clustering, list(objects_by_image.keys())
        )

        if self.debug_writer is not None:
            features_csv = self.results_dir / "phase2_clustering" / "clustering_features.csv"
            self.debug_writer.write_all(
                labeled_by_image_clustering=labeled_by_image_clustering,
                labeled_by_image_propagated=labeled_by_image,
                memory_composition=memory,
                propagation_config=self.propagator.config,
                features_csv=features_csv,
                image_paths=list(objects_by_image.keys()),
            )

        save_predicted_masks(labeled_by_image, self.results_dir)
        print_summary(all_objects, labeled_by_image)
        return labeled_by_image

    # ------------------------------------------------------------------
    # Few-shot direct propagation (no Phase 1/2/3)
    # ------------------------------------------------------------------

    def _run_few_shot(
        self,
        image_paths,
        references: list,
    ) -> dict[Path, list[LabeledObject]]:
        """Propagate human references directly — no segmentation or clustering."""
        logger.info("=" * 60)
        logger.info("Phase 4: Prototype Propagation (few-shot)")
        logger.info("=" * 60)

        phase4_dir = self.results_dir / "phase4_propagation"
        phase4_dir.mkdir(exist_ok=True)

        refs, frame_scores = build_fewshot_references(references or [])
        if not refs:
            logger.warning("No references; returning empty result.")
            return {p: [] for p in image_paths}

        labeled_by_image, _ = self.propagator.propagate(
            references=refs,
            frame_scores=frame_scores,
            target_paths=list(image_paths),
            reader=self.reader,
        )

        for path, labeled in labeled_by_image.items():
            save_visualization(path, labeled, phase4_dir, suffix="_propagated")

        save_predicted_masks(labeled_by_image, self.results_dir)
        print_summary([], labeled_by_image)
        return labeled_by_image

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    def phase1(self, image_paths, force_embeddings: bool = False):
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

        return run_phase1_unsupervised(
            image_paths, self.reader, self.segmenter, self.extractor,
            phase1_dir, extract_embeddings=extract_embeddings,
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

    def phase4_propagate(
        self,
        labeled_by_image: dict[Path, list[LabeledObject]],
        target_paths: list[Path],
    ) -> tuple[dict[Path, list[LabeledObject]], list[dict]]:
        logger.info("=" * 60)
        logger.info("Phase 4: Prototype Propagation")
        logger.info("=" * 60)

        phase4_dir = self.results_dir / "phase4_propagation"
        phase4_dir.mkdir(exist_ok=True)

        references, frame_scores = build_unsupervised_references(
            labeled_by_image, self.propagator.config, self.reader
        )
        labeled_by_image, memory = self.propagator.propagate(
            references=references,
            frame_scores=frame_scores,
            target_paths=target_paths,
            reader=self.reader,
        )

        for path, labeled in labeled_by_image.items():
            save_visualization(path, labeled, phase4_dir, suffix="_propagated")

        return labeled_by_image, memory

    # ------------------------------------------------------------------
    # Baseline early-exit (clustering disabled)
    # ------------------------------------------------------------------

    def _baseline_label(self, all_objects, objects_by_image, references):
        logger.info("[BASELINE] clustering_enabled=false -> "
                    "skipping phases 2-4, using MedSAM2 labels directly")

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

    Set need_segmenter=False only when Phase 1 is cached AND clustering is
    disabled (baseline mode).  Propagation always requires the GPU model.
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
        logger.info("Segmenter not loaded (using cached segmentation, baseline mode)")

    # Clustering only applies to unsupervised mode; few_shot always propagates directly.
    if mode == "few_shot":
        clustering_enabled = False
    elif mode == "unsupervised":
        clustering_enabled = cfg.get("unsupervised", {}).get("clustering_enabled", True)
    else:
        clustering_enabled = True

    labeler = None
    if clustering_enabled:
        labeler = ClusteringLabeler(ClusteringConfig(**cfg["labeler"]))
        labeler.resolve_adaptive_params(num_images)

        labeler_features = cfg.get("labeler", {}).get("features", None)
        if labeler_features is None:
            logger.info("Moment features: NONE (embeddings-only mode)")
        else:
            logger.info(f"Moment features: {len(labeler.config.features)} selected")
    elif mode != "few_shot":
        logger.info("[BASELINE] clustering_enabled=false -> "
                    "skipping phases 2-4, using MedSAM2 labels directly")

    extract_embeddings = (
        cfg.get("labeler", {}).get("embedding", {}).get("enabled", False)
    )
    if extract_embeddings:
        logger.info("Embedding extraction: ENABLED")

    propagation_mode = cfg.get("propagation", {}).get("mode", "independent")
    logger.info(f"Propagation mode: {propagation_mode}")

    propagator = None
    debug_writer = None
    if clustering_enabled or mode == "few_shot":
        if segmenter is None:
            raise RuntimeError(
                "Segmenter required for propagation but not loaded. "
                "This is a configuration error: need_segmenter must be True "
                "when propagation is enabled."
            )
        propagation_cfg = dict(cfg.get("propagation", {}))
        if mode == "few_shot" and "mode" not in propagation_cfg:
            # few_shot configs express propagation mode under few_shot.propagation_mode
            propagation_cfg["mode"] = (
                cfg.get("few_shot", {}).get("propagation_mode", "independent")
            )
        propagator = PrototypePropagator(
            segmenter=segmenter,
            config=PropagationConfig(**propagation_cfg),
        )
        logger.info(
            f"Propagator: references_per_cluster="
            f"{propagator.config.references_per_cluster}, "
            f"mode={propagator.config.mode}"
        )

        if clustering_enabled:
            debug_cfg = cfg.get("debug", {})
            n_preview = debug_cfg.get("n_preview", 10)
            seed = debug_cfg.get("seed", 42)
            debug_writer = ClusteringDebugWriter(
                phase2_dir=results_dir / "phase2_clustering",
                n_preview=n_preview,
                seed=seed,
            )

    pipeline = Pipeline(
        mode=mode,
        reader=reader,
        segmenter=segmenter,
        extractor=extractor,
        labeler=labeler,
        propagator=propagator,
        results_dir=results_dir,
        extract_embeddings=extract_embeddings,
    )
    pipeline.debug_writer = debug_writer
    return pipeline

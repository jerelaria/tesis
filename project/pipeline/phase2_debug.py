"""
Clustering debug output writer.

Generates 6 diagnostic outputs under phase2_clustering/ after propagation:
  prototypes/            — top-K prototype masks per good cluster
  top10_debug/           — top-10 masks per good cluster (regardless of K)
  feature_analysis/      — violin plots of shape features by cluster
  memory_composition/    — SAM2 reference frame grid
  result_previews/       — propagated masks on sampled images
  summary.json           — cluster quality stats and prototype metadata
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from project.core.data_types import LabeledObject
from project.evaluation.cluster_vis import (
    save_feature_violin,
    save_mask_panel,
    save_memory_composition_figure,
)
from project.evaluation.visualizer import save_visualization
from project.segmentation.quality import (
    compute_cluster_quality,
    identify_good_clusters,
    select_prototypes,
)

logger = logging.getLogger(__name__)


def _load_image(source_image) -> np.ndarray:
    """Return HxWx3 uint8 array, loading from disk if volume is None."""
    if source_image.volume is not None:
        vol = source_image.volume
        arr = (vol * 255).astype(np.uint8) if vol.max() <= 1.0 else vol.astype(np.uint8)
        return arr
    return np.array(plt.imread(str(source_image.source_path)))


class ClusteringDebugWriter:
    """
    Writes all 6 clustering/propagation debug outputs under phase2_dir.

    Parameters
    ----------
    phase2_dir : Path
        Root of phase2_clustering/ in the experiment results.
    n_preview : int
        Number of images sampled for result_previews/.
    seed : int
        RNG seed for image sampling (reproducible across runs).
    """

    def __init__(self, phase2_dir: Path, n_preview: int = 10, seed: int = 42):
        self.phase2_dir = Path(phase2_dir)
        self.n_preview = n_preview
        self.seed = seed

    def write_all(
        self,
        labeled_by_image_clustering: dict[Path, list[LabeledObject]],
        labeled_by_image_propagated: dict[Path, list[LabeledObject]],
        memory_composition: list[dict],
        propagation_config,
        features_csv: Path,
        image_paths: list[Path],
    ) -> None:
        """
        Generate all 6 debug outputs.

        Parameters
        ----------
        labeled_by_image_clustering
            Phase-2 clustering result (used for prototypes, violin, summary).
        labeled_by_image_propagated
            Phase-4 propagation result (used for previews, summary counts).
        memory_composition
            List of per-frame dicts returned by PrototypePropagator.propagate().
        propagation_config : PropagationConfig
            Config used for propagation (provides quality thresholds and k).
        features_csv
            Path to phase2_clustering/clustering_features.csv.
        image_paths
            All image paths processed by the pipeline.
        """
        alpha = propagation_config.mask_selection.sam_score_weight

        quality_report = compute_cluster_quality(
            labeled_by_image_clustering, propagation_config.quality
        )
        good_clusters = {cid for cid, info in quality_report.items() if info["good"]}
        k = propagation_config.references_per_cluster
        prototypes = select_prototypes(
            labeled_by_image_clustering,
            good_clusters,
            k,
            propagation_config.mask_selection,
        )
        cluster_to_obj_id = {
            cid: i + 1 for i, cid in enumerate(sorted(good_clusters))
        }

        logger.info(
            f"[DEBUG] Writing outputs for {len(good_clusters)} good clusters, "
            f"{len(quality_report) - len(good_clusters)} filtered, "
            f"k={k}, {len(memory_composition)} reference frames"
        )

        self._write_prototypes(prototypes, cluster_to_obj_id, alpha, k)
        self._write_top10_debug(labeled_by_image_clustering, good_clusters, alpha)
        self._write_filtered_clusters(labeled_by_image_clustering, quality_report, alpha)
        self._write_feature_violin(labeled_by_image_clustering, features_csv)
        self._write_memory_composition(memory_composition)
        self._write_result_previews(labeled_by_image_propagated, image_paths)
        self._write_summary_json(
            good_clusters=good_clusters,
            labeled_by_image_clustering=labeled_by_image_clustering,
            labeled_by_image_propagated=labeled_by_image_propagated,
            prototypes=prototypes,
            k=k,
            cluster_to_obj_id=cluster_to_obj_id,
            alpha=alpha,
            num_images=len(image_paths),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_prototypes(self, prototypes, cluster_to_obj_id, alpha, k):
        out_dir = self.phase2_dir / "prototypes"
        out_dir.mkdir(exist_ok=True)

        for cid, proto_list in sorted(prototypes.items()):
            if not proto_list:
                logger.debug(f"cluster_{cid}: no qualifying prototypes")
                continue

            entries = []
            for obj in proto_list:
                seg = obj.segmented_object
                img = _load_image(seg.source_image)
                sam = seg.confidence or 0.0
                combined = alpha * sam + (1.0 - alpha) * obj.labeling_confidence
                ys, xs = np.where(seg.mask)
                area = int(seg.mask.sum())
                cx = float(xs.mean()) if len(xs) else 0.0
                cy = float(ys.mean()) if len(ys) else 0.0
                src_name = Path(seg.source_image.source_path or "?").name
                title = (
                    f"{src_name}\n"
                    f"sam={sam:.3f}  lconf={obj.labeling_confidence:.3f}\n"
                    f"combined={combined:.3f}  area={area}\n"
                    f"centroid=({cx:.0f},{cy:.0f})"
                )
                entries.append((img, seg.mask, cid, title))

            obj_id = cluster_to_obj_id.get(cid, "?")
            save_mask_panel(
                entries,
                out_dir / f"cluster_{cid}_prototypes.png",
                suptitle=(
                    f"cluster_{cid}  [obj_id={obj_id}]  "
                    f"— top-{len(entries)} prototypes (k={k})"
                ),
                n_cols=min(k, 5),
            )
        logger.info(f"Prototypes -> {out_dir.name}/")

    def _write_top10_debug(self, labeled_by_image, good_clusters, alpha, k_debug=10):
        out_dir = self.phase2_dir / "top10_debug"
        out_dir.mkdir(exist_ok=True)

        for cid in sorted(good_clusters):
            scored = []
            for objs in labeled_by_image.values():
                for obj in objs:
                    if obj.organ_id != cid or obj.is_noise:
                        continue
                    sam = obj.segmented_object.confidence or 0.0
                    score = alpha * sam + (1.0 - alpha) * obj.labeling_confidence
                    scored.append((score, obj))
            scored.sort(key=lambda x: -x[0])
            top = scored[:k_debug]

            if not top:
                continue

            entries = []
            for combined, obj in top:
                seg = obj.segmented_object
                img = _load_image(seg.source_image)
                src_name = Path(seg.source_image.source_path or "?").name
                area = int(seg.mask.sum())
                title = (
                    f"{src_name}\n"
                    f"combined={combined:.3f}  area={area}"
                )
                entries.append((img, seg.mask, cid, title))

            save_mask_panel(
                entries,
                out_dir / f"cluster_{cid}_top{k_debug}.png",
                suptitle=f"cluster_{cid} — top-{k_debug} by combined_score (debug)",
                n_cols=5,
            )
        logger.info(f"Top-10 debug -> {out_dir.name}/")

    def _write_feature_violin(self, labeled_by_image, features_csv):
        if not features_csv.exists():
            logger.warning(f"features_csv not found: {features_csv} — skipping violin")
            return

        out_dir = self.phase2_dir / "feature_analysis"
        out_dir.mkdir(exist_ok=True)

        cluster_id_by_object_id = {
            obj.segmented_object.id: (-1 if obj.is_noise else obj.organ_id)
            for objs in labeled_by_image.values()
            for obj in objs
        }
        save_feature_violin(cluster_id_by_object_id, features_csv, out_dir)
        logger.info(f"Feature violin -> {out_dir.name}/")

    def _write_memory_composition(self, memory_composition):
        if not memory_composition:
            logger.debug("Empty memory_composition — skipping figure")
            return

        out_dir = self.phase2_dir / "memory_composition"
        out_dir.mkdir(exist_ok=True)
        save_memory_composition_figure(
            memory_composition,
            out_dir / "sam_memory.png",
        )

    def _write_result_previews(self, labeled_by_image_propagated, image_paths):
        out_dir = self.phase2_dir / "result_previews"
        out_dir.mkdir(exist_ok=True)

        available = [p for p in image_paths if p in labeled_by_image_propagated]
        rng = np.random.default_rng(self.seed)
        n = min(self.n_preview, len(available))
        indices = sorted(rng.choice(len(available), size=n, replace=False).tolist())

        for idx in indices:
            path = available[idx]
            labeled = labeled_by_image_propagated.get(path, [])
            save_visualization(path, labeled, out_dir, suffix="_propagated")

        logger.info(f"Result previews ({n}) -> {out_dir.name}/")

    def _write_filtered_clusters(
        self,
        labeled_by_image: dict[Path, list[LabeledObject]],
        quality_report: dict[int, dict],
        alpha: float,
        n_show: int = 6,
    ) -> None:
        """Save a mask panel for each cluster that was filtered out, with the
        rejection reason in the figure title."""
        filtered = {
            cid: info for cid, info in quality_report.items() if not info["good"]
        }
        if not filtered:
            return

        out_dir = self.phase2_dir / "filtered_clusters"
        out_dir.mkdir(exist_ok=True)

        for cid, info in sorted(filtered.items()):
            scored = []
            for objs in labeled_by_image.values():
                for obj in objs:
                    if obj.organ_id != cid or obj.is_noise:
                        continue
                    sam = obj.segmented_object.confidence or 0.0
                    score = alpha * sam + (1.0 - alpha) * obj.labeling_confidence
                    scored.append((score, obj))
            scored.sort(key=lambda x: -x[0])
            top = scored[:n_show]

            if not top:
                continue

            entries = []
            for combined, obj in top:
                seg = obj.segmented_object
                img = _load_image(seg.source_image)
                src_name = Path(seg.source_image.source_path or "?").name
                area = int(seg.mask.sum())
                title = f"{src_name}\ncombined={combined:.3f}  area={area}"
                entries.append((img, seg.mask, cid, title))

            reason_str = "\n".join(info["failed"])
            save_mask_panel(
                entries,
                out_dir / f"cluster_{cid}_filtered.png",
                suptitle=(
                    f"cluster_{cid}  [FILTERED]\n"
                    f"n={info['n_objects']}  "
                    f"freq={info['image_frequency']:.3f}  "
                    f"lconf={info['avg_labeling_confidence']:.3f}  "
                    f"sam={info['avg_sam_confidence']}\n"
                    f"Reason: {reason_str}"
                ),
                n_cols=min(n_show, 6),
            )

        logger.info(
            f"Filtered cluster panels ({len(filtered)}) -> filtered_clusters/"
        )

    def _write_summary_json(
        self,
        good_clusters,
        labeled_by_image_clustering,
        labeled_by_image_propagated,
        prototypes,
        k,
        cluster_to_obj_id,
        alpha,
        num_images,
    ):
        total_images = len(labeled_by_image_clustering)

        # Per-cluster frequency from clustering output
        freq_by_cluster: dict[int, float] = {}
        for cid in good_clusters:
            imgs_with_cid = sum(
                1 for objs in labeled_by_image_clustering.values()
                if any(o.organ_id == cid and not o.is_noise for o in objs)
            )
            freq_by_cluster[cid] = round(imgs_with_cid / total_images, 4) if total_images else 0.0

        # Propagated masks per cluster
        propagated_per_cluster: dict[int, int] = {cid: 0 for cid in good_clusters}
        for objs in labeled_by_image_propagated.values():
            for obj in objs:
                if obj.organ_id in propagated_per_cluster:
                    propagated_per_cluster[obj.organ_id] += 1

        # Prototype metadata
        proto_info: dict[str, list] = {}
        for cid, proto_list in prototypes.items():
            proto_info[f"cluster_{cid}"] = []
            for obj in proto_list:
                seg = obj.segmented_object
                sam = seg.confidence or 0.0
                combined = alpha * sam + (1.0 - alpha) * obj.labeling_confidence
                ys, xs = np.where(seg.mask)
                proto_info[f"cluster_{cid}"].append({
                    "source": Path(seg.source_image.source_path or "?").name,
                    "sam_score": round(float(sam), 4),
                    "labeling_confidence": round(float(obj.labeling_confidence), 4),
                    "combined_score": round(float(combined), 4),
                    "area": int(seg.mask.sum()),
                    "centroid": [
                        round(float(xs.mean()), 1) if len(xs) else 0.0,
                        round(float(ys.mean()), 1) if len(ys) else 0.0,
                    ],
                    "obj_id": cluster_to_obj_id.get(cid),
                })

        summary = {
            "num_images": num_images,
            "references_per_cluster_k": k,
            "good_clusters": sorted(good_clusters),
            "cluster_to_obj_id": {str(k_): v for k_, v in cluster_to_obj_id.items()},
            "image_frequency": {f"cluster_{c}": freq_by_cluster[c] for c in sorted(good_clusters)},
            "propagated_masks_per_cluster": {
                f"cluster_{c}": propagated_per_cluster[c] for c in sorted(good_clusters)
            },
            "prototypes": proto_info,
        }

        out = self.phase2_dir / "summary.json"
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary JSON -> {out.name}")

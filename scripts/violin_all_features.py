"""
Violin plots of ALL moment features (16) by cluster, like the Phase-2 artifact.

The Phase-2 pipeline only logs violins for the features used for clustering
(the 12 selected in the YAML). This script reproduces that artifact but plots
every feature the extractor computes (16), including the unused ones
(ecc, hu1, hu2, orientation).

Fast path: it does NOT re-segment. It reuses
  - the cached Phase-1 segmentation (results/_segmentation/...): full 16-feature
    vectors + masks keyed by object UUID, and
  - the saved clustering features CSV to confirm we reproduce the exact run.
Cluster labels are reproduced by running the same ClusteringLabeler with the
resolved config on the cached objects (deterministic -> identical labels).

Features are standardized (z-score) so the panels match the style of the
Phase-2 artifact. Set --raw to plot natural units instead.

Usage:
    python -m scripts.violin_all_features
    python -m scripts.violin_all_features --raw
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from project.pipeline.artifacts import load_segmented
from project.labeling.clustering import ClusteringConfig, ClusteringLabeler
from project.feature_extraction.feature_names import FEATURE_NAMES
from project.evaluation.cluster_vis import save_feature_violin

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("violin_all_features")

REPO = Path(__file__).resolve().parents[1]

# dataset -> (experiment results dir, seg cache dir)
DATASETS = {
    "XRay": (
        REPO / "results/moments/XRay/unsup_hdbscan_propagation_iter",
        REPO / "results/_segmentation/XRay/unsupervised_grid6_st0.50_iou0.50_n911__18298beb",
    ),
    "Sunnybrook": (
        REPO / "results/moments/Sunnybrook/unsup_hdbscan_propagation_iter",
        REPO / "results/_segmentation/Sunnybrook/unsupervised_grid6_st0.50_iou0.50_n860__b1595645",
    ),
}

OUT_ROOT = REPO / "figuras_out" / "violin_all_features"


def build_labels(seg_dir: Path, resolved_cfg: dict):
    """Load cached objects and reproduce cluster labels with the resolved config."""
    all_objects, objects_by_image = load_segmented(seg_dir)

    config = ClusteringConfig(**resolved_cfg["labeler"])
    labeler = ClusteringLabeler(config)
    labeler.resolve_adaptive_params(num_images=len(objects_by_image))
    labeler.fit(all_objects)
    labeled = labeler.label(all_objects)

    cluster_id_by_object_id = {
        lo.segmented_object.id: int(lo.organ_id) for lo in labeled
    }
    return all_objects, cluster_id_by_object_id


def write_full_feature_csv(all_objects, out_csv: Path, standardize: bool) -> None:
    """Write object_id + all 16 features (optionally z-scored) to CSV."""
    X = np.stack([obj.features for obj in all_objects]).astype(np.float64)  # (N, 16)
    if standardize:
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)
    df = pd.DataFrame(X, columns=list(FEATURE_NAMES))
    df.insert(0, "object_id", [obj.id for obj in all_objects])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true",
                    help="Plot raw feature values instead of z-scored.")
    args = ap.parse_args()
    standardize = not args.raw

    for name, (exp_dir, seg_dir) in DATASETS.items():
        logger.info(f"=== {name} ===")
        if not seg_dir.exists():
            logger.error(f"Seg cache missing: {seg_dir} — skipping")
            continue

        resolved_cfg = json.loads((exp_dir / "resolved_config.json").read_text())
        all_objects, cluster_id_by_object_id = build_labels(seg_dir, resolved_cfg)

        n_clusters = len({c for c in cluster_id_by_object_id.values() if c != -1})
        n_noise = sum(1 for c in cluster_id_by_object_id.values() if c == -1)
        logger.info(f"{name}: {len(all_objects)} objects, "
                    f"{n_clusters} clusters, {n_noise} noise")

        out_dir = OUT_ROOT / name
        out_csv = out_dir / "all_features.csv"
        write_full_feature_csv(all_objects, out_csv, standardize)

        label = f"{name} moments — all 16 features ({'z-score' if standardize else 'raw'})"
        save_feature_violin(
            cluster_id_by_object_id, out_csv, out_dir, method_label=label,
        )
        logger.info(f"{name}: violin -> {out_dir / 'feature_distributions_violin.png'}")


if __name__ == "__main__":
    main()

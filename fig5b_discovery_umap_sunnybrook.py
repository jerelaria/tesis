#!/usr/bin/env python3
"""
fig5b_discovery_umap_sunnybrook.py
----------------------------------
Figura 5b — UMAP de descubrimiento en Sunnybrook, resaltando el VD.

El VD no está en el GT de Sunnybrook, así que no se puede colorear por órgano.
Se resalta por ID de cluster (el 4, que el cluster_map mapea a rv_cavity) y al
lado se muestra el mismo UMAP coloreado por GT, donde esos puntos caen bajo
"none". El contraste es el descubrimiento: un grupo denso y separable que el GT
llama fondo. La confirmación de que es VD viene de la transferencia a ACDC.

Reproduce *exactamente* las etiquetas de cluster de la fase 2 reconstruyendo el
ClusteringLabeler con el config resuelto del run (HDBSCAN determinista → los IDs
0,4,5,6,7 son idénticos a los logueados). NO se re-clusteriza con otros params.

Datos por defecto: run results/moments/Sunnybrook/unsup_hdbscan_propagation.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from project.labeling.clustering import ClusteringLabeler, ClusteringConfig
from project.pipeline.artifacts import load_segmented

# Reutiliza helpers del pipeline de fase 2 (mismas funciones que generan el
# UMAP logueado), para no duplicar lógica.
from phase2_log import (
    _NOISE_LABEL,
    _assign_gt_organs,
    _project_2d,
    _scatter_by_cluster,
    _scatter_by_organ,
)


def reconstruct_labels(run_dir: Path, seg_cache_dir: Path, n_images: int):
    """Re-fit determinista del labeler → {object_id: cluster_id}, igual que fase 2."""
    cfg = json.loads((run_dir / "resolved_config.json").read_text())
    labeler = ClusteringLabeler(ClusteringConfig(**cfg["labeler"]))
    labeler.resolve_adaptive_params(n_images)

    all_objects, _ = load_segmented(seg_cache_dir)
    labeler.fit(all_objects)
    return labeler, all_objects


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path,
                    default=Path("results/moments/Sunnybrook/unsup_hdbscan_propagation"))
    ap.add_argument("--seg-cache", type=Path,
                    default=Path("results/_segmentation/Sunnybrook/"
                                 "unsupervised_grid6_st0.50_iou0.50_n860__b1595645"))
    ap.add_argument("--gt-dir", type=Path,
                    default=Path("data/processed/Sunnybrook/masks"))
    ap.add_argument("--highlight-cluster", type=int, default=4,
                    help="ID de cluster a resaltar (cluster_map: 4 → rv_cavity).")
    ap.add_argument("--umap-neighbors", type=int, default=30)
    ap.add_argument("--output", type=Path,
                    default=Path("figuras_out/fig5b_discovery_umap_sunnybrook.png"))
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # --- Confirma el cluster_map del run para el subtítulo / sanity check ---
    meta_path = args.run_dir.parent.parent.parent / "acdc_transfer" / "transfer_meta.json"
    cluster_map = {}
    if not meta_path.exists():
        meta_path = Path("results/acdc_transfer/transfer_meta.json")
    if meta_path.exists():
        cluster_map = json.loads(meta_path.read_text()).get("cluster_map", {})
    mapped = cluster_map.get(str(args.highlight_cluster), "rv_cavity")

    # --- Features logueadas en fase 2 (mismo X que usó el UMAP del run) ---
    csv_path = args.run_dir / "phase2_clustering" / "clustering_features.csv"
    df = pd.read_csv(csv_path)
    object_ids = df["object_id"].tolist()
    X = df.drop(columns=["object_id"]).values.astype(np.float32)
    n_images = json.loads(
        (args.run_dir / "resolved_config.json").read_text()
    ).get("num_images", len(object_ids))

    print("Reconstruyendo etiquetas de fase 2 (re-fit determinista) ...")
    labeler, all_objects = reconstruct_labels(args.run_dir, args.seg_cache, n_images)
    clusters = np.array(
        [labeler._id_to_cluster.get(oid, _NOISE_LABEL) for oid in object_ids]
    )

    present = sorted({int(c) for c in clusters if c != _NOISE_LABEL})
    print(f"Clusters presentes: {present}")
    if args.highlight_cluster not in present:
        raise SystemExit(
            f"El cluster {args.highlight_cluster} no aparece tras reconstruir "
            f"(presentes: {present}). Revisa --seg-cache / --run-dir."
        )
    n_hl = int((clusters == args.highlight_cluster).sum())
    print(f"Cluster {args.highlight_cluster} → '{mapped}': {n_hl} objetos resaltados")

    # --- Órganos GT por objeto (el VD caerá en 'none') ---
    print("Asignando órganos GT ...")
    obj_by_id = {obj.id: obj for obj in all_objects}
    gt_organs = _assign_gt_organs(object_ids, obj_by_id, args.gt_dir)

    print("Proyectando a 2D (UMAP) ...")
    proj = _project_2d(X, args.umap_neighbors)
    xy = proj["umap"] if proj.get("umap") is not None else proj["pca"]
    method = "UMAP" if proj.get("umap") is not None else "PCA"

    fig, (ax_cl, ax_gt) = plt.subplots(1, 2, figsize=(13, 5.6))

    _scatter_by_cluster(
        ax_cl, xy, clusters, {args.highlight_cluster},
        f"{method} — por cluster (C{args.highlight_cluster} = VD descubierto)",
    )
    # En el panel por GT no resaltamos ningún órgano (el VD no existe en el GT):
    # los puntos del VD aparecen bajo 'none'.
    _scatter_by_organ(ax_gt, xy, gt_organs, None, f"{method} — por órgano GT")

    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    pdf = args.output.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {args.output}")
    print(f"Guardado: {pdf}")


if __name__ == "__main__":
    main()

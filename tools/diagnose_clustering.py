#!/usr/bin/env python3
"""
diagnose_clustering.py
----------------------
CPU-only diagnostic to understand WHY unsupervised clustering produces several
clusters for the same organ (over-segmentation), and whether it can be fixed by
changing hyperparameters, merging clusters, or changing the feature space.

The script does NOT touch the pipeline or any existing result file. It reads:
  - segmented.json + segmented.npz : per-object raw mask (for GT matching),
    raw 16-d moment vector, raw 256-d SAM2 embedding, source image path.
  - the GT masks tree                : data/processed/<Dataset>/masks/<stem>/<organ>_N.png
  - optionally clustering_features.csv: the exact standardized matrix the pipeline
    clustered (the 'as_clustered' feature set).

Symmetric per-feature-set treatment
-----------------------------------
Every feature set is first-class. For each one the script runs its OWN HDBSCAN
(with the parameters given for that set) and produces the full diagnostic:
projection (colored by that set's native clustering), contingency, fragmentation,
quality vs GT, and merge test. There is no privileged 'primary' set.

Clustering parameters are NOT read from a config; they are passed on the CLI.
A 'default' block applies to all sets; per-set blocks override it, so you can
give one setting for embeddings, another for moments, and so on.

Feature sets
------------
  moments_all    : 16 moment features (z-scored here).
  moments_geom   : reduced geometric subset (z-scored here).
  embeddings_raw : 256-d SAM2 embeddings (z-scored here), if cached.
  as_clustered   : the exact matrix the pipeline clustered (only if --features-csv
                   is given); already standardized.

Outputs (under --out-dir):
  projection_<set>.png   : UMAP + PCA, left colored by that set's native cluster,
                           right by GT organ.
  contingency_<set>.png/.csv : cluster x GT-organ counts for that set.
  frontier_sweep.csv     : best achievable V-measure per set over a parameter grid.
  diagnosis_report.json  : params, quality, fragmentation and merge test per set.

Reading guide:
  * One compact blob per organ in the GT panel, split into several colors in the
    cluster panel -> homogeneity high, completeness low -> over-segmentation is a
    clustering artifact; a merge step (or eom / epsilon) fixes it. Confirmed if the
    merge test recovers completeness and the sweep finds a high-V config on the
    SAME features.
  * One organ shown as 2+ separated blobs in the GT panel -> genuine feature-space
    multimodality; merge only helps if same-organ blobs are mutually closest.
  * Organs overlapping in the GT panel -> features do not separate organs
    (homogeneity also low) -> no hyperparameter fixes it; change features.

Parameter syntax
----------------
  --params "SET:key=val,key=val"   (repeatable)
  SET is one of: default, moments_all, moments_geom, embeddings_raw, as_clustered
  keys: min_cluster_size (int), min_samples (int), method (eom|leaf),
        epsilon (float), metric (str). Missing keys inherit from 'default',
        then from the built-in baseline (method=leaf, min_samples=5, epsilon=0.0).

Example
-------
  python tools/diagnose_clustering.py \
      --seg-cache    results/_segmentation/XRay/unsupervised_grid6_st0.50_iou0.50_n911__18298beb \
      --gt-dir       data/processed/XRay/masks \
      --features-csv results_leaf/embeddings_raw/XRay/unsup_hdbscan_propagation/phase2_clustering/clustering_features.csv \
      --out-dir      diagnostics/xray_hdbscan \
      --gt-iou-min   0.5 \
      --params       "default:min_cluster_size=136,min_samples=27,method=leaf" \
      --params       "embeddings_raw:min_cluster_size=272,method=eom" \
      --k-organs 2 --sweep
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    homogeneity_completeness_v_measure,
    adjusted_rand_score,
    adjusted_mutual_info_score,
)
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("diagnose")

# 16-d moment vector layout (project/feature_extraction/feature_names.py).
FEATURE_NAMES = [
    "V", "Cx", "Cy", "Dx", "Dy", "L",
    "ecc", "solidity", "extent", "compact",
    "hu0", "hu1", "hu2",
    "intensity_mean", "intensity_std",
    "orientation",
]
FEATURE_INDEX = {n: i for i, n in enumerate(FEATURE_NAMES)}

# The 12 features actually used by the pipeline (excludes ecc, hu1, hu2, orientation).
PIPELINE_SUBSET = ["V", "Cx", "Cy", "Dx", "Dy", "L",
                   "solidity", "extent", "compact", "hu0",
                   "intensity_mean", "intensity_std"]

# Reduced geometric subset recommended for position/shape-only clustering.
GEOM_SUBSET = ["V", "Cx", "Cy", "hu0", "solidity", "compact",
               "intensity_mean", "intensity_std"]

NOISE_LABEL = -1

VALID_SET_NAMES = {"moments_all", "moments_geom", "embeddings_raw", "as_clustered"}

# Baseline HDBSCAN parameters; overridden by --params default:... then by per-set.
BASELINE_PARAMS = {
    "min_cluster_size": None,   # required; falls back to N // 50 with a warning
    "min_samples": 5,
    "method": "leaf",
    "epsilon": 0.0,
    "metric": "euclidean",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_objects(seg_cache: Path) -> dict:
    """
    Parse segmented.json + segmented.npz from the Phase 1 cache.

    Returns a dict keyed by object_id with:
        mask (bool HxW), moments (16,), embedding (256,) or None,
        stem (image stem), confidence (float), label (str|None).
    """
    with open(seg_cache / "segmented.json") as f:
        meta = json.load(f)
    arrays = np.load(seg_cache / "segmented.npz", allow_pickle=False)

    objects: dict = {}
    for img in meta["images"]:
        stem = Path(img["source_path"]).stem
        for obj in img["objects"]:
            emb_key = obj.get("embedding_key")
            objects[obj["id"]] = {
                "mask": arrays[obj["mask_key"]].astype(bool),
                "moments": arrays[obj["features_key"]].astype(np.float64),
                "embedding": arrays[emb_key].astype(np.float64) if emb_key else None,
                "stem": stem,
                "confidence": float(obj.get("confidence") or 0.0),
                "label": obj.get("label"),
            }
    logger.info(f"Loaded {len(objects)} objects from {seg_cache}")
    return objects


def load_gt_for_stem(gt_dir: Path, stem: str) -> dict:
    """Load GT organ masks for one image: {organ_name: bool mask}."""
    d = gt_dir / stem
    if not d.is_dir():
        return {}
    out = {}
    for f in sorted(d.iterdir()):
        if f.suffix.lower() != ".png" or f.name.lower() == "image.png":
            continue
        organ = _strip_instance(f.stem)
        m = np.array(Image.open(f).convert("L")) > 127
        out.setdefault(organ, []).append(m)
    # Merge instances of the same organ into one mask (union).
    return {organ: np.logical_or.reduce(masks) for organ, masks in out.items()}


def _strip_instance(stem: str) -> str:
    """left_lung_1 -> left_lung ; obj_002 -> obj ; cluster_0_1 -> cluster_0."""
    import re
    m = re.match(r"^(.+)_(\d+)$", stem)
    return m.group(1) if m else stem


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def assign_gt_organ(objects: dict, gt_dir: Path, iou_min: float) -> None:
    """
    Add objects[oid]['gt_organ'] and ['gt_iou'] by best-IoU match to GT.

    GT masks are resized (nearest) to the object-mask shape when needed.
    Objects with best IoU below iou_min are labeled 'none' (spurious/background).
    Note: structures present in the images but absent from the GT (e.g. rv_cavity
    in Sunnybrook) fall into 'none' and count as impurity, not as a class.
    """
    gt_cache: dict = {}
    for oid, obj in objects.items():
        stem = obj["stem"]
        if stem not in gt_cache:
            gt_cache[stem] = load_gt_for_stem(gt_dir, stem)
        gt_masks = gt_cache[stem]
        if not gt_masks:
            obj["gt_organ"], obj["gt_iou"] = "none", 0.0
            continue
        h, w = obj["mask"].shape
        best_organ, best_iou = "none", 0.0
        for organ, gm in gt_masks.items():
            if gm.shape != (h, w):
                gm = np.array(
                    Image.fromarray(gm.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
                ) > 127
            v = _iou(obj["mask"], gm)
            if v > best_iou:
                best_organ, best_iou = organ, v
        obj["gt_organ"] = best_organ if best_iou >= iou_min else "none"
        obj["gt_iou"] = best_iou
    n_named = sum(1 for o in objects.values() if o["gt_organ"] != "none")
    logger.info(f"GT match: {n_named}/{len(objects)} objects mapped to an organ "
                f"(iou_min={iou_min})")


# ---------------------------------------------------------------------------
# Feature matrices
# ---------------------------------------------------------------------------

def build_feature_sets(objects: dict, ids: list, features_csv: Path | None,
                       standardize: bool = True) -> dict:
    """Build feature matrices aligned to `ids`. Returns {set_name: X}.

    standardize=False skips z-scoring for moments_all and embeddings_raw
    (useful for ablation). as_clustered is never re-standardized because
    the pipeline already outputs it standardized.
    """
    scale = _zscore if standardize else (lambda x: x)
    sets: dict = {}

    moments = np.stack([objects[i]["moments"] for i in ids])
    sets["moments_all"] = scale(moments)
    pipeline_idx = [FEATURE_INDEX[f] for f in PIPELINE_SUBSET]
    sets["moments_12"] = scale(moments[:, pipeline_idx])

    if all(objects[i]["embedding"] is not None for i in ids):
        emb = np.stack([objects[i]["embedding"] for i in ids])
        sets["embeddings_raw"] = scale(emb)
    else:
        logger.info("Embeddings missing for some objects -> skipping embeddings_raw")

    if features_csv is not None:
        # Hard guard: a missing CSV must not silently disappear.
        if not features_csv.exists():
            raise FileNotFoundError(
                f"--features-csv was passed but does not exist: {features_csv}. "
                "Fix the path (it usually shares the root of the run directory)."
            )
        df = pd.read_csv(features_csv).set_index("object_id")
        cols = [c for c in df.columns if c != "object_id"]
        sets["as_clustered"] = df.loc[ids, cols].to_numpy(dtype=np.float64)
    return sets


def _zscore(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)


# ---------------------------------------------------------------------------
# Parameter parsing (per feature set)
# ---------------------------------------------------------------------------

def _cast_param(key: str, value: str):
    if key in ("min_cluster_size", "min_samples"):
        return int(value)
    if key == "epsilon":
        return float(value)
    if key in ("method", "metric"):
        return value
    raise ValueError(f"Unknown clustering parameter '{key}'. "
                     f"Valid: min_cluster_size, min_samples, method, epsilon, metric")


def parse_params(param_specs: list[str] | None) -> tuple[dict, dict]:
    """
    Parse --params specs into (default_overrides, per_set_overrides).

    Each spec is 'SET:key=val,key=val'. SET='default' fills the default block;
    other SETs override it for that feature set only.
    """
    default_over: dict = {}
    per_set: dict = {}
    for spec in param_specs or []:
        if ":" not in spec:
            raise ValueError(f"Bad --params '{spec}'. Expected 'SET:key=val,...'")
        setname, kvs = spec.split(":", 1)
        setname = setname.strip()
        if setname != "default" and setname not in VALID_SET_NAMES:
            logger.warning(f"--params targets unknown set '{setname}' "
                           f"(valid: default, {', '.join(sorted(VALID_SET_NAMES))})")
        parsed = {}
        for kv in kvs.split(","):
            kv = kv.strip()
            if not kv:
                continue
            if "=" not in kv:
                raise ValueError(f"Bad key=value in --params: '{kv}'")
            k, v = kv.split("=", 1)
            parsed[k.strip()] = _cast_param(k.strip(), v.strip())
        if setname == "default":
            default_over.update(parsed)
        else:
            per_set.setdefault(setname, {}).update(parsed)
    return default_over, per_set


def resolve_set_params(name: str, default_over: dict, per_set: dict,
                       n_objects: int) -> dict:
    """Merge baseline < default block < per-set block; supply a mcs fallback."""
    p = {**BASELINE_PARAMS, **default_over, **per_set.get(name, {})}
    if p["min_cluster_size"] is None:
        p["min_cluster_size"] = max(2, n_objects // 50)
        logger.warning(f"[{name}] no min_cluster_size given -> fallback "
                       f"{p['min_cluster_size']} (= n_objects // 50)")
    return p


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_labels(X: np.ndarray, params: dict, epsilon: float | None = None) -> np.ndarray:
    eps = params.get("epsilon", 0.0) if epsilon is None else epsilon
    model = HDBSCAN(
        min_cluster_size=params["min_cluster_size"],
        min_samples=params["min_samples"],
        metric=params["metric"],
        cluster_selection_method=params["method"],
        cluster_selection_epsilon=eps,
        copy=True,  # silence the sklearn FutureWarning and avoid mutating X
    )
    return model.fit_predict(X).astype(int)


# ---------------------------------------------------------------------------
# Supervised diagnostics
# ---------------------------------------------------------------------------

def quality_vs_gt(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Homogeneity / completeness / V-measure / ARI / AMI of pred vs GT labels."""
    h, c, v = homogeneity_completeness_v_measure(gt, pred)
    return {
        "homogeneity": round(h, 4),
        "completeness": round(c, 4),
        "v_measure": round(v, 4),
        "ari": round(adjusted_rand_score(gt, pred), 4),
        "ami": round(adjusted_mutual_info_score(gt, pred), 4),
        "n_clusters": int(len({c for c in pred if c != NOISE_LABEL})),
        "noise_frac": round(float((pred == NOISE_LABEL).mean()), 4),
    }


def fragmentation_report(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Per organ: how many clusters claim it as their dominant organ (the direct
    measure of over-segmentation). Per cluster: dominant organ + purity + size.
    """
    by_cluster = defaultdict(list)
    for c, g in zip(pred, gt):
        by_cluster[int(c)].append(g)

    cluster_rows = {}
    organ_to_clusters = defaultdict(list)
    for c, organs in by_cluster.items():
        vals, counts = np.unique(organs, return_counts=True)
        dom = vals[counts.argmax()]
        purity = float(counts.max() / counts.sum())
        cluster_rows[str(c)] = {"dominant_organ": str(dom),
                                "purity": round(purity, 3),
                                "size": int(len(organs))}
        if c != NOISE_LABEL:
            organ_to_clusters[str(dom)].append(int(c))

    organ_rows = {
        organ: {"n_clusters_claiming": len(cs), "cluster_ids": sorted(cs)}
        for organ, cs in organ_to_clusters.items()
    }
    return {"per_cluster": cluster_rows, "per_organ": organ_rows}


def contingency(pred: np.ndarray, gt: np.ndarray, out_dir: Path, set_name: str) -> pd.DataFrame:
    """cluster x organ count table -> CSV + heatmap, named per feature set."""
    df = pd.DataFrame({"cluster": pred, "organ": gt})
    ct = pd.crosstab(df["cluster"], df["organ"])
    ct.to_csv(out_dir / f"contingency_{set_name}.csv")

    fig, ax = plt.subplots(figsize=(1.4 * len(ct.columns) + 3, 0.45 * len(ct) + 3))
    im = ax.imshow(ct.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(ct.columns)))
    ax.set_xticklabels(ct.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(ct.index)))
    ax.set_yticklabels([("noise" if c == NOISE_LABEL else f"c{c}") for c in ct.index])
    ax.set_xlabel("GT organ")
    ax.set_ylabel("cluster")
    ax.set_title(f"{set_name}: cluster x GT-organ counts")
    for i in range(len(ct.index)):
        for j in range(len(ct.columns)):
            ax.text(j, i, ct.values[i, j], ha="center", va="center",
                    color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out_dir / f"contingency_{set_name}.png", dpi=130)
    plt.close(fig)
    return ct


def merge_test(X: np.ndarray, pred: np.ndarray, gt: np.ndarray, k_organs: int) -> dict:
    """
    Over-cluster -> merge: agglomerate cluster centroids (in this set's space)
    down to k_organs groups, remap, recompute V-measure. If completeness rises
    while homogeneity holds, a simple centroid merge fixes the over-segmentation.
    """
    real = sorted({c for c in pred if c != NOISE_LABEL})
    if len(real) <= k_organs:
        return {"skipped": f"only {len(real)} clusters <= k_organs={k_organs}"}
    centroids = np.stack([X[pred == c].mean(axis=0) for c in real])
    merged_of = AgglomerativeClustering(n_clusters=k_organs).fit_predict(centroids)
    remap = {c: int(merged_of[i]) for i, c in enumerate(real)}
    merged_pred = np.array([remap.get(int(c), NOISE_LABEL) for c in pred])
    return {"before": quality_vs_gt(pred, gt),
            "after_merge_to_k": quality_vs_gt(merged_pred, gt)}


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------

def project_2d(X: np.ndarray, umap_neighbors: int) -> dict:
    """Return {'pca': (N,2), 'umap': (N,2)}; umap may be None if unavailable."""
    out = {"pca": PCA(n_components=2, random_state=42).fit_transform(X)}
    try:
        import umap
        out["umap"] = umap.UMAP(
            n_components=2, n_neighbors=umap_neighbors, min_dist=0.1, random_state=42
        ).fit_transform(X)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"UMAP unavailable ({e}); using PCA only. "
                       f"Install with: pip install umap-learn")
        out["umap"] = None
    return out


def _scatter(ax, xy, labels, title):
    cats = sorted(set(labels), key=lambda x: (str(x) == "none", str(x)))
    cmap = plt.cm.tab10
    for k, cat in enumerate(cats):
        m = np.array([l == cat for l in labels])
        is_bg = (cat == NOISE_LABEL) or (cat == "none")
        ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.55,
                   color=("#bbbbbb" if is_bg else cmap(k % 10)),
                   label=("noise/none" if is_bg else str(cat)))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=7, loc="best", framealpha=0.6)


def save_projection(name, proj, clusters, organs, out_dir, has_gt):
    """Left panel: this set's native clustering. Right panel: GT organ."""
    methods = [m for m in ("umap", "pca") if proj.get(m) is not None]
    ncol = 2 if has_gt else 1
    fig, axes = plt.subplots(len(methods), ncol,
                             figsize=(6.5 * ncol, 5.5 * len(methods)), squeeze=False)
    for r, m in enumerate(methods):
        _scatter(axes[r][0], proj[m], clusters, f"{name} | {m.upper()} | colored by CLUSTER")
        if has_gt:
            _scatter(axes[r][1], proj[m], organs, f"{name} | {m.upper()} | colored by GT ORGAN")
    fig.tight_layout()
    fig.savefig(out_dir / f"projection_{name}.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Frontier sweep (per feature set, using each set's params as the grid base)
# ---------------------------------------------------------------------------

def frontier_sweep(feature_sets: dict, set_params: dict, gt: np.ndarray, out_dir: Path) -> pd.DataFrame:
    """
    For each feature set, sweep HDBSCAN over method x min_cluster_size x epsilon
    (min_samples and metric taken from that set's params) and record the
    achievable homogeneity/completeness/V-measure. The best row per set is the
    ceiling on those features.
    """
    rows = []
    for set_name, X in feature_sets.items():
        base = set_params[set_name]["min_cluster_size"]
        grid_mcs = sorted({max(2, int(base * f)) for f in (0.5, 1.0, 2.0)})
        for method in ("eom", "leaf"):
            for mcs in grid_mcs:
                for eps in (0.0, 0.25, 0.5, 1.0):
                    params = {**set_params[set_name], "method": method,
                              "min_cluster_size": mcs}
                    pred = cluster_labels(X, params, epsilon=eps)
                    q = quality_vs_gt(pred, gt)
                    rows.append({"feature_set": set_name, "method": method,
                                 "min_cluster_size": mcs, "epsilon": eps, **q})
    df = pd.DataFrame(rows).sort_values(["feature_set", "v_measure"],
                                        ascending=[True, False])
    df.to_csv(out_dir / "frontier_sweep.csv", index=False)
    best = df.loc[df.groupby("feature_set")["v_measure"].idxmax()]
    logger.info("\nBest V-measure per feature set (ceiling on those features):")
    logger.info(best[["feature_set", "method", "min_cluster_size", "epsilon",
                      "homogeneity", "completeness", "v_measure", "n_clusters",
                      "noise_frac"]].to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg-cache", required=True, type=Path)
    ap.add_argument("--gt-dir", type=Path, default=None,
                    help="GT masks root (per-image subfolders). Enables supervised metrics.")
    ap.add_argument("--features-csv", type=Path, default=None,
                    help="clustering_features.csv (adds the as_clustered feature set).")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--gt-iou-min", type=float, default=0.10)
    ap.add_argument("--umap-neighbors", type=int, default=30)
    ap.add_argument("--params", action="append", default=None,
                    help="HDBSCAN params per set: 'SET:key=val,...' (repeatable). "
                         "SET in {default, moments_all, moments_geom, embeddings_raw, "
                         "as_clustered}.")
    ap.add_argument("--k-organs", type=int, default=None,
                    help="If set (and GT available), run the merge-to-k test for each "
                         "feature set. Use the number of distinct organs that actually "
                         "appear among the non-noise clusters (read it from the "
                         "contingency), not the nominal dataset organ count.")
    ap.add_argument("--sweep", action="store_true",
                    help="Run the HDBSCAN parameter frontier sweep per feature set.")
    ap.add_argument("--no-standardize", action="store_true",
                    help="Skip z-score normalization for moments_all and embeddings_raw. "
                         "Useful to ablate the effect of standardization. "
                         "as_clustered is never re-standardized regardless of this flag.")
    ap.add_argument("--dataset-name", default=None,
                    help="Dataset name stored in the report metadata. "
                         "Defaults to the gt-dir parent folder name if gt-dir is given.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    objects = load_objects(args.seg_cache)
    ids = list(objects.keys())

    has_gt = args.gt_dir is not None
    if has_gt:
        assign_gt_organ(objects, args.gt_dir, args.gt_iou_min)
        gt = np.array([objects[i]["gt_organ"] for i in ids])
    else:
        gt = None

    feature_sets = build_feature_sets(objects, ids, args.features_csv,
                                      standardize=not args.no_standardize)

    default_over, per_set = parse_params(args.params)
    set_params = {name: resolve_set_params(name, default_over, per_set, len(ids))
                  for name in feature_sets}

    dataset_name = (args.dataset_name
                    or (args.gt_dir.parent.name if args.gt_dir else None)
                    or "unknown")
    report: dict = {
        "meta": {
            "dataset": dataset_name,
            "standardized": not args.no_standardize,
        },
        "per_feature_set": {},
    }

    for name, X in feature_sets.items():
        params = set_params[name]
        labels = cluster_labels(X, params)
        n_clusters = len({c for c in labels if c != NOISE_LABEL})
        logger.info(f"\n=== {name} === params={params}")
        logger.info(f"  {n_clusters} clusters, noise={float((labels == NOISE_LABEL).mean()):.2%}")

        proj = project_2d(X, args.umap_neighbors)
        save_projection(name, proj, labels, gt if has_gt else labels, args.out_dir, has_gt)

        entry: dict = {"params": params, "n_clusters": n_clusters}
        if has_gt:
            entry["quality"] = quality_vs_gt(labels, gt)
            entry["fragmentation"] = fragmentation_report(labels, gt)
            ct = contingency(labels, gt, args.out_dir, name)
            logger.info("  contingency:\n" + ct.to_string().replace("\n", "\n  "))
            logger.info("  quality: " + json.dumps(entry["quality"]))
            if args.k_organs:
                entry["merge_test"] = merge_test(X, labels, gt, args.k_organs)
                logger.info("  merge_test: " + json.dumps(entry["merge_test"]))
        report["per_feature_set"][name] = entry

    logger.info(f"\nSaved projections and contingencies -> {args.out_dir}")

    if has_gt and args.sweep:
        frontier_sweep(feature_sets, set_params, gt, args.out_dir)

    with open(args.out_dir / "diagnosis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nReport -> {args.out_dir / 'diagnosis_report.json'}")


if __name__ == "__main__":
    main()
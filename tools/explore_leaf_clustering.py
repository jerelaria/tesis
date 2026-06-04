#!/usr/bin/env python3
"""
Standalone diagnostic for HDBSCAN leaf clustering.

Reads pre-computed (already-standardized) features from clustering_features.csv
and masks from the Phase 1 segmentation cache (NPZ+JSON). Runs HDBSCAN with a
configurable cluster_selection_method (default: 'leaf') so that the result can be
compared against the pipeline's 'eom' run without touching any existing files.

No GPU required. All computation is CPU-only.

Outputs (under phase2_clustering/):
  leaf_clustering_preview/   - one overlay PNG per sampled image
  leaf_clustering_references/ - top-K reference PNG per good cluster
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import HDBSCAN

# Shared visualization utilities (also used by the pipeline)
from project.evaluation.cluster_vis import (
    _get_cluster_color,
    save_mask_panel,
    save_feature_violin,
)

# ---------------------------------------------------------------------------
# Default paths (overridden by --run-dir at runtime)
# ---------------------------------------------------------------------------

_DEFAULT_RUN_DIR = Path(
    "/media/apoloml/DATOS_2/Tesis_Cosegmentacion/results/momentos"
    "/Sunnybrook/unsup_hdbscan_propagation"
)

SEG_CACHE_BASE = Path(
    "/media/apoloml/DATOS_2/Tesis_Cosegmentacion/results/_segmentation"
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def resolve_hdbscan_params(config: dict) -> tuple[int, int, str]:
    """Return (min_cluster_size, min_samples, metric) from the run config."""
    num_images = config["num_images"]
    hcfg = config["labeler"]["hdbscan"]

    mcs = hcfg.get("min_cluster_size")
    ms = hcfg.get("min_samples")
    mcs_frac = hcfg.get("min_cluster_size_fraction")
    ms_frac = hcfg.get("min_samples_fraction")
    metric = hcfg.get("metric", "euclidean")

    if mcs is None and mcs_frac is not None:
        mcs = max(2, int(mcs_frac * num_images))
    if mcs is None:
        mcs = 2

    if ms is None and ms_frac is not None:
        ms = max(1, int(ms_frac * num_images))
    if ms is None:
        ms = 1

    return mcs, ms, metric


# ---------------------------------------------------------------------------
# Segmentation cache index
# ---------------------------------------------------------------------------

def build_object_index(config: dict, dataset: str) -> tuple[dict, Path]:
    """
    Parse segmented.json from the Phase 1 cache.

    Parameters
    ----------
    config : dict
        Parsed resolved_config.json.
    dataset : str
        Dataset folder name inside SEG_CACHE_BASE (e.g. "Sunnybrook", "XRay").
        Inferred from run_dir.parent.name by the caller.

    Returns
    -------
    index : dict
        object_id -> {source_path, stem, confidence, mask_key}
    seg_dir : Path
        Directory that contains segmented.npz
    """
    fp = config["seg_fingerprint"]
    seg_dir = SEG_CACHE_BASE / dataset / fp

    with open(seg_dir / "segmented.json") as f:
        data = json.load(f)

    index: dict[str, dict] = {}
    for img in data["images"]:
        src = img["source_path"]
        stem = img["stem"]
        for obj in img["objects"]:
            index[obj["id"]] = {
                "source_path": src,
                "stem": stem,
                "confidence": obj["confidence"],  # SAM score
                "mask_key": obj["mask_key"],
            }

    return index, seg_dir


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _build_feature_matrix(
    features_csv: Path,
    config: dict,
    obj_index: dict,
    sam_score_weight: float = 0.0,
    exclude_features: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Load the pre-standardized feature matrix and optionally append a
    standardized SAM score column.

    sam_score_weight > 0 enables SAM score as an extra feature.  The SAM
    scores are z-scored and then scaled by sam_score_weight so their influence
    relative to the shape features can be tuned.

    exclude_features : list of feature names to drop before clustering.

    Returns (raw_df, X, active_feature_cols).
    """
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(features_csv)
    feature_cols = config["labeler"]["features"]

    excluded = set(exclude_features or [])
    unknown  = excluded - set(feature_cols)
    if unknown:
        raise ValueError(f"Unknown features to exclude: {unknown}. "
                         f"Valid: {feature_cols}")
    active_cols = [f for f in feature_cols if f not in excluded]
    if excluded:
        print(f"Excluding features: {sorted(excluded)}  "
              f"({len(active_cols)}/{len(feature_cols)} kept)")

    X = df[active_cols].values.copy()

    if sam_score_weight > 0.0:
        sam_scores = np.array([
            float(obj_index.get(oid, {}).get("confidence") or 0.0)
            for oid in df["object_id"]
        ])
        sam_z = StandardScaler().fit_transform(sam_scores.reshape(-1, 1)).ravel()
        X = np.hstack([X, (sam_z * sam_score_weight).reshape(-1, 1)])

    return df, X, active_cols


def run_clustering(
    features_csv: Path,
    config: dict,
    method: str,
    min_samples_override: int | None = None,
    min_cluster_size_override: int | None = None,
    obj_index: dict | None = None,
    sam_score_weight: float = 0.0,
    exclude_features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fit HDBSCAN on the (optionally SAM-augmented, optionally feature-pruned)
    pre-standardized feature matrix.

    The CSV features are already z-scored by the pipeline — no re-scaling needed.
    When sam_score_weight > 0, the SAM score is z-scored and appended as an
    extra feature weighted by sam_score_weight.
    exclude_features drops named columns before clustering.

    Returns a DataFrame with columns [object_id, cluster_id].
    """
    _idx = obj_index or {}
    df, X, _ = _build_feature_matrix(
        features_csv, config, _idx, sam_score_weight, exclude_features
    )

    mcs, ms, metric = resolve_hdbscan_params(config)
    if min_samples_override is not None:
        ms = min_samples_override
    if min_cluster_size_override is not None:
        mcs = min_cluster_size_override

    sam_note = f"  sam_weight={sam_score_weight}" if sam_score_weight > 0 else ""
    print(f"\nHDBSCAN  min_cluster_size={mcs}  min_samples={ms}  "
          f"metric={metric}  method='{method}'{sam_note}")
    print(f"Feature matrix: {X.shape}")

    model = HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms,
        metric=metric,
        cluster_selection_method=method,
    )
    labels = model.fit_predict(X)

    return pd.DataFrame({"object_id": df["object_id"], "cluster_id": labels})


# ---------------------------------------------------------------------------
# Cross-analysis: EOM cluster_0 → leaf assignment
# ---------------------------------------------------------------------------

def cross_analyze(
    features_csv: Path,
    config: dict,
    method: str,
    obj_index: dict,
    output_dir: Path,
    min_samples_override: int | None = None,
    min_cluster_size_override: int | None = None,
    sam_score_weight: float = 0.0,
    exclude_features: list[str] | None = None,
) -> None:
    """
    Run EOM internally (no files written), then compare each EOM cluster's
    objects against the already-computed leaf (or other method) assignments.

    Outputs:
      - Console cross-tabulation
      - cross_eom_vs_{method}.png: feature distributions for objects that were
        EOM cluster_0 split by their leaf assignment (noise vs clusters)
    """
    print(f"\nCross-analysis: EOM vs {method} ...")

    eom_df = run_clustering(
        features_csv, config, "eom",
        min_samples_override=min_samples_override,
        min_cluster_size_override=min_cluster_size_override,
        obj_index=obj_index,
        sam_score_weight=sam_score_weight,
        exclude_features=exclude_features,
    )
    target_df = run_clustering(
        features_csv, config, method,
        min_samples_override=min_samples_override,
        min_cluster_size_override=min_cluster_size_override,
        obj_index=obj_index,
        sam_score_weight=sam_score_weight,
        exclude_features=exclude_features,
    )

    merged = eom_df.rename(columns={"cluster_id": "eom_cluster"}).merge(
        target_df.rename(columns={"cluster_id": f"{method}_cluster"}),
        on="object_id",
    )
    target_col = f"{method}_cluster"

    # Cross-tabulation
    print(f"\n  Cross-tabulation (rows=EOM, cols={method}):")
    ctab = pd.crosstab(
        merged["eom_cluster"].map(lambda c: "noise" if c == -1 else f"eom_cluster_{c}"),
        merged[target_col].map(lambda c: "noise" if c == -1 else f"{method}_cluster_{c}"),
    )
    print(ctab.to_string())

    # Feature distributions for EOM cluster_0 objects split by leaf assignment
    feat_df = pd.read_csv(features_csv)
    feature_cols = config["labeler"]["features"]

    eom_c0_ids = set(merged.loc[merged["eom_cluster"] == 0, "object_id"])
    subset = merged[merged["object_id"].isin(eom_c0_ids)].merge(
        feat_df, on="object_id", how="inner"
    )

    # Add SAM scores for inspection
    subset["sam_score"] = subset["object_id"].map(
        lambda oid: float(obj_index.get(oid, {}).get("confidence") or 0.0)
    )

    leaf_groups = sorted(subset[target_col].unique())
    group_labels = {c: ("noise" if c == -1 else f"{method}_{c}") for c in leaf_groups}

    print(f"\n  EOM cluster_0 objects ({len(subset)}) → {method} breakdown:")
    for c in leaf_groups:
        sub = subset[subset[target_col] == c]
        print(f"    {group_labels[c]:25s}  n={len(sub):4d}  "
              f"avg_sam={sub['sam_score'].mean():.3f}  "
              f"V={sub['V'].mean():+.3f}  ecc={sub['ecc'].mean():+.3f}  "
              f"solidity={sub['solidity'].mean():+.3f}")

    # Plot: for each feature, violin split by leaf assignment within EOM cluster_0
    key_features = ["V", "ecc", "solidity", "extent", "compact",
                    "intensity_mean", "intensity_std", "Cx", "Cy", "L"]
    key_features = [f for f in key_features if f in feature_cols]

    n_cols_p = 5
    n_rows_p = (len(key_features) + n_cols_p - 1) // n_cols_p
    fig, axes = plt.subplots(n_rows_p, n_cols_p,
                             figsize=(4 * n_cols_p, 3.5 * n_rows_p))
    axes_flat = axes.flatten()

    label_order = [c for c in leaf_groups]
    tick_labels = [group_labels[c] for c in label_order]

    for i, feat in enumerate(key_features):
        ax = axes_flat[i]
        groups = [subset.loc[subset[target_col] == c, feat].values
                  for c in label_order]
        parts = ax.violinplot(groups, positions=range(len(label_order)),
                              showmedians=True, showextrema=True)
        for pc, c in zip(parts["bodies"], label_order):
            color = "#888888" if c == -1 else "#{:02x}{:02x}{:02x}".format(
                *[int(255 * v) for v in plt.cm.tab10(c % plt.cm.tab10.N)[:3]]
            )
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        ax.set_xticks(range(len(label_order)))
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=20, ha="right")
        ax.set_title(feat, fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    # Also plot SAM score
    if len(key_features) < len(axes_flat):
        ax = axes_flat[len(key_features)]
        groups = [subset.loc[subset[target_col] == c, "sam_score"].values
                  for c in label_order]
        parts = ax.violinplot(groups, positions=range(len(label_order)),
                              showmedians=True)
        ax.set_xticks(range(len(label_order)))
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=20, ha="right")
        ax.set_title("sam_score", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    for ax in axes_flat[len(key_features) + 1:]:
        ax.set_visible(False)

    fig.suptitle(
        f"EOM cluster_0 objects — feature split by {method} assignment\n"
        f"(n={len(subset)}, noise=gray, clusters=colored)",
        fontsize=11,
    )
    fig.tight_layout()
    out = output_dir / f"cross_eom_vs_{method}.png"
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"\n  Saved -> {out.name}")


# ---------------------------------------------------------------------------
# Per-object stats (SAM score, labeling_confidence, stem mapping)
# ---------------------------------------------------------------------------

def enrich_with_metadata(result_df: pd.DataFrame, obj_index: dict) -> pd.DataFrame:
    """
    Attach source_path, stem, SAM score, labeling_confidence, and mask_key
    to each row in result_df.

    labeling_confidence follows the same rule as the pipeline: 1.0 for every
    non-noise object, 0.0 for noise (cluster_id == -1).
    """
    rows = []
    for _, row in result_df.iterrows():
        oid = row["object_id"]
        cid = int(row["cluster_id"])
        meta = obj_index.get(oid, {})
        rows.append({
            "object_id": oid,
            "cluster_id": cid,
            "source_path": meta.get("source_path", ""),
            "stem": meta.get("stem", ""),
            "sam_score": float(meta.get("confidence") or 0.0),
            "labeling_confidence": 0.0 if cid == -1 else 1.0,
            "mask_key": meta.get("mask_key", ""),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Good-cluster detection (same criteria as ClusterFilter)
# ---------------------------------------------------------------------------

def get_good_cluster_ids(stats: pd.DataFrame, config: dict) -> list[int]:
    """
    Return cluster IDs that pass all three quality thresholds.

    Supports both old layout (config["cluster_filter"]) and new layout
    (config["propagation"]["quality"]).
    """
    # New layout (propagation pipeline)
    cf = (
        config.get("cluster_filter")
        or config.get("propagation", {}).get("quality")
        or {}
    )
    min_freq = cf.get("min_image_frequency", 0.3)
    min_lconf = cf.get("min_avg_labeling_confidence", 0.5)
    min_sam = cf.get("min_avg_sam_confidence", 0.6)
    num_images = config["num_images"]

    good = []
    for cid in sorted(stats["cluster_id"].unique()):
        if cid == -1:
            continue
        sub = stats[stats["cluster_id"] == cid]
        n_imgs = sub["stem"].nunique()
        image_freq = n_imgs / num_images
        avg_lconf = sub["labeling_confidence"].mean()
        avg_sam = sub["sam_score"].mean()

        if (
            image_freq >= min_freq
            and avg_lconf >= min_lconf
            and avg_sam >= min_sam
        ):
            good.append(cid)

    return good


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(stats: pd.DataFrame, config: dict, good_ids: list[int]) -> None:
    num_images = config["num_images"]
    cf = config["cluster_filter"]

    print("\n" + "=" * 64)
    print("LEAF CLUSTERING SUMMARY")
    print("=" * 64)

    unique_ids = sorted(stats["cluster_id"].unique())
    n_real = sum(1 for c in unique_ids if c != -1)
    noise_n = int((stats["cluster_id"] == -1).sum())

    print(f"Total objects  : {len(stats)}")
    print(f"Noise objects  : {noise_n}")
    print(f"Real clusters  : {n_real}")
    print()

    for cid in unique_ids:
        sub = stats[stats["cluster_id"] == cid]
        n_imgs = sub["stem"].nunique()
        freq = n_imgs / num_images
        avg_sam = sub["sam_score"].mean()
        avg_lc = sub["labeling_confidence"].mean()
        tag = "[GOOD]" if cid in good_ids else "[filtered]"
        lbl = "noise (-1)" if cid == -1 else f"cluster_{cid}"
        print(f"  {lbl:20s}  n={len(sub):5d}  images={n_imgs:4d}  "
              f"freq={freq:.2f}  avg_sam={avg_sam:.2f}  "
              f"avg_lconf={avg_lc:.2f}  {tag}")

    print()
    print(f"Good clusters  : {[f'cluster_{c}' for c in good_ids]}")
    print(f"Thresholds     : image_freq>={cf['min_image_frequency']}  "
          f"avg_lconf>={cf['min_avg_labeling_confidence']}  "
          f"avg_sam>={cf['min_avg_sam_confidence']}")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Output 1: preview overlays  (uses shared save_mask_panel)
# ---------------------------------------------------------------------------

def generate_preview(
    stem: str,
    stem_df: pd.DataFrame,
    npz,
    output_path: Path,
    method: str,
) -> None:
    """Save one PNG: MRI with all masks overlaid and colored by cluster_id."""
    if stem_df.empty:
        return

    source_path = stem_df.iloc[0]["source_path"]
    image = np.array(plt.imread(source_path))

    entries = []
    for _, row in stem_df.iterrows():
        cid = int(row["cluster_id"])
        mask = npz[row["mask_key"]].astype(bool)
        label_text = "-" if cid == -1 else str(cid)
        entries.append((image, mask, cid, label_text))

    save_mask_panel(
        entries,
        output_path,
        suptitle=f"{stem}  [{method} clustering]",
        n_cols=len(entries),
    )


# ---------------------------------------------------------------------------
# Output 2: reference panels  (uses shared save_mask_panel)
# ---------------------------------------------------------------------------

def generate_references(
    cluster_id: int,
    stats: pd.DataFrame,
    npz,
    config: dict,
    output_path: Path,
    method: str,
    num_references_override: int | None = None,
) -> None:
    """Save one PNG with the top-K reference masks for a good cluster."""
    # Support both old (refinement) and new (propagation) config layouts
    msc = (
        config.get("refinement", {}).get("mask_selection")
        or config.get("propagation", {}).get("mask_selection")
        or {}
    )
    alpha = msc.get("sam_score_weight", 0.5)
    min_comb = msc.get("min_combined_score", 0.5)
    k = num_references_override if num_references_override is not None \
        else msc.get("num_reference_frames", msc.get("references_per_cluster", 5))

    sub = stats[stats["cluster_id"] == cluster_id].copy()
    sub["combined_score"] = (
        alpha * sub["sam_score"] + (1 - alpha) * sub["labeling_confidence"]
    )
    sub = sub[sub["combined_score"] >= min_comb]
    sub = sub.sort_values("combined_score", ascending=False).head(k)

    if sub.empty:
        print(f"    cluster_{cluster_id}: no candidates above "
              f"min_combined_score={min_comb}")
        return

    entries = []
    for _, row in sub.iterrows():
        img = np.array(plt.imread(row["source_path"]))
        mask = npz[row["mask_key"]].astype(bool)
        ys, xs = np.where(mask)
        area = int(mask.sum())
        cx = float(xs.mean()) if len(xs) else 0.0
        cy = float(ys.mean()) if len(ys) else 0.0
        src_name = Path(row["source_path"]).name
        title = (
            f"{src_name}\n"
            f"sam={row['sam_score']:.3f}  lconf={row['labeling_confidence']:.3f}\n"
            f"combined={row['combined_score']:.3f}  area={area}\n"
            f"centroid=({cx:.1f},{cy:.1f})"
        )
        entries.append((img, mask, cluster_id, title))

    save_mask_panel(
        entries,
        output_path,
        suptitle=f"cluster_{cluster_id} — top-{len(entries)} references [{method}]",
        n_cols=len(entries),
    )


# ---------------------------------------------------------------------------
# Output 3: feature distributions  (uses shared save_feature_violin)
# ---------------------------------------------------------------------------

def plot_feature_distributions(
    stats: pd.DataFrame,
    features_csv: Path,
    config: dict,
    output_dir: Path,
    method: str,
) -> None:
    """Violin + scatter per feature, split by cluster_id."""
    cluster_id_by_object_id = dict(zip(
        stats["object_id"].astype(str),
        stats["cluster_id"].astype(int),
    ))
    save_feature_violin(cluster_id_by_object_id, features_csv, output_dir,
                        method_label=method)

    # Console: mean ± std per feature per cluster
    feat_df = pd.read_csv(features_csv)
    merged = feat_df.merge(stats[["object_id", "cluster_id"]], on="object_id", how="inner")
    feature_cols = [c for c in feat_df.columns if c not in ("object_id",)]
    unique_ids = sorted(merged["cluster_id"].unique())
    labels_map = {cid: ("noise" if cid == -1 else f"cluster_{cid}") for cid in unique_ids}
    label_order = ["noise"] + [f"cluster_{c}" for c in unique_ids if c != -1]
    merged["label"] = merged["cluster_id"].map(labels_map)

    print(f"\n  {'Feature':<16}", end="")
    for lbl in label_order:
        print(f"  {lbl:>18}", end="")
    print()
    for feat in feature_cols:
        if feat == "object_id":
            continue
        print(f"  {feat:<16}", end="")
        for lbl in label_order:
            vals = merged.loc[merged["label"] == lbl, feat]
            print(f"  {vals.mean():+7.3f} ± {vals.std():.3f}", end="")
        print()


# ---------------------------------------------------------------------------
# LV cavity inspection: identify candidate mask and find nearest neighbors
# ---------------------------------------------------------------------------

def inspect_candidate_mask(
    stem: str,
    obj_index: dict,
    features_csv: Path,
    npz,
    output_dir: Path,
    n_neighbors: int = 10,
    exclude_features: list[str] | None = None,
) -> None:
    """
    For a given image stem, identify the most likely LV cavity mask using
    purely geometric heuristics (largest area + most circular + most central),
    then find its n nearest neighbors in the standardized feature space and
    save all masks as PNGs for visual inspection.

    Identification heuristic (applied to the raw mask pixels, not to the
    already-standardized features in the CSV):
      score = area * circularity / (1 + dist_to_center_normalized)
      where circularity = 4π * area / perimeter²

    The mask with the highest score is taken as the LV cavity candidate.
    """
    from scipy.ndimage import label as scipy_label

    # --- Collect all objects for this stem ---
    stem_objects = [(oid, meta) for oid, meta in obj_index.items()
                    if meta["stem"] == stem]
    if not stem_objects:
        print(f"  No objects found for stem '{stem}'")
        return

    print(f"\n  Image: {stem}  ({len(stem_objects)} objects)")

    # --- Score each mask geometrically ---
    candidates = []
    for oid, meta in stem_objects:
        mask = npz[meta["mask_key"]].astype(bool)
        H, W = mask.shape
        area = int(mask.sum())
        if area == 0:
            continue

        ys, xs = np.where(mask)
        cy, cx = ys.mean(), xs.mean()

        # Distance of centroid from image center, normalized to [0,1]
        dist_center = np.sqrt(((cy - H / 2) / H) ** 2 + ((cx - W / 2) / W) ** 2)

        # Perimeter via contour pixel count (4-connectivity)
        padded = np.pad(mask, 1)
        shifted = (
            np.roll(padded, 1, axis=0) |
            np.roll(padded, -1, axis=0) |
            np.roll(padded, 1, axis=1) |
            np.roll(padded, -1, axis=1)
        )
        perimeter = int((padded & ~shifted)[1:-1, 1:-1].sum()) + 1  # avoid /0

        # Isoperimetric circularity: 1.0 = perfect circle, <1 = irregular
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        circularity = min(circularity, 1.0)

        score = area * circularity / (1.0 + dist_center)
        candidates.append({
            "object_id": oid,
            "mask_key": meta["mask_key"],
            "source_path": meta["source_path"],
            "sam_score": float(meta.get("confidence") or 0.0),
            "area": area,
            "circularity": circularity,
            "centroid": (cx, cy),
            "dist_center": dist_center,
            "score": score,
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]

    print(f"  Candidate LV cavity: object_id={best['object_id']}")
    print(f"    area={best['area']}  circularity={best['circularity']:.3f}  "
          f"dist_center={best['dist_center']:.3f}  sam={best['sam_score']:.3f}")
    print(f"    centroid=({best['centroid'][0]:.1f}, {best['centroid'][1]:.1f})")

    # --- Find candidate's feature vector in the CSV ---
    feat_df = pd.read_csv(features_csv)
    row = feat_df[feat_df["object_id"] == best["object_id"]]
    if row.empty:
        print("  ERROR: candidate object_id not found in features CSV")
        return

    excluded     = set(exclude_features or [])
    feature_cols = [c for c in feat_df.columns if c != "object_id" and c not in excluded]
    candidate_vec = row[feature_cols].values[0]

    print(f"  Feature vector (standardized):")
    for name, val in zip(feature_cols, candidate_vec):
        print(f"    {name:<16} {val:+.4f}")

    # --- Nearest neighbors in standardized feature space ---
    all_ids = feat_df["object_id"].values
    all_X = feat_df[feature_cols].values
    dists = np.linalg.norm(all_X - candidate_vec, axis=1)

    # Exclude the candidate itself
    sorted_idx = np.argsort(dists)
    neighbor_idx = [i for i in sorted_idx if all_ids[i] != best["object_id"]][:n_neighbors]

    print(f"\n  Top-{n_neighbors} nearest neighbors (euclidean in standardized space):")

    # --- Save candidate + neighbor masks as PNG grid ---
    n_total = 1 + n_neighbors
    n_cols_g = 4
    n_rows_g = (n_total + n_cols_g - 1) // n_cols_g
    fig, axes = plt.subplots(n_rows_g, n_cols_g,
                             figsize=(4 * n_cols_g, 4 * n_rows_g))
    axes_flat = axes.flatten()

    # Panel 0: candidate
    ax = axes_flat[0]
    cand_img = np.array(plt.imread(best["source_path"]))
    cand_mask = npz[best["mask_key"]].astype(bool)
    ax.imshow(cand_img, cmap="gray")
    overlay = np.zeros((*cand_mask.shape, 4), dtype=np.float32)
    overlay[cand_mask] = (1.0, 0.2, 0.2, 0.5)   # red = candidate
    ax.imshow(overlay)
    ax.contour(cand_mask, levels=[0.5], colors=["red"], linewidths=2.0)
    ax.set_title(
        f"CANDIDATE\n{Path(best['source_path']).name}\n"
        f"area={best['area']}  circ={best['circularity']:.2f}  "
        f"sam={best['sam_score']:.2f}",
        fontsize=7,
    )
    ax.axis("off")

    # Panels 1..n_neighbors: neighbors
    for panel_i, idx in enumerate(neighbor_idx, start=1):
        nb_id = all_ids[idx]
        nb_meta = obj_index.get(nb_id, {})
        dist_val = dists[idx]

        nb_mask = npz[nb_meta["mask_key"]].astype(bool)
        nb_img = np.array(plt.imread(nb_meta["source_path"]))

        ys, xs = np.where(nb_mask)
        nb_area = int(nb_mask.sum())
        nb_cx = float(xs.mean()) if len(xs) else 0.0
        nb_cy = float(ys.mean()) if len(ys) else 0.0
        nb_sam = float(nb_meta.get("confidence") or 0.0)

        ax = axes_flat[panel_i]
        ax.imshow(nb_img, cmap="gray")
        overlay = np.zeros((*nb_mask.shape, 4), dtype=np.float32)
        overlay[nb_mask] = (0.2, 0.6, 1.0, 0.5)  # blue = neighbor
        ax.imshow(overlay)
        ax.contour(nb_mask, levels=[0.5], colors=["#1166ff"], linewidths=1.5)
        ax.set_title(
            f"neighbor #{panel_i}  dist={dist_val:.3f}\n"
            f"{Path(nb_meta.get('source_path', '')).name}\n"
            f"area={nb_area}  sam={nb_sam:.2f}\n"
            f"centroid=({nb_cx:.0f},{nb_cy:.0f})",
            fontsize=7,
        )
        ax.axis("off")

        print(f"    #{panel_i:2d}  dist={dist_val:.4f}  {nb_meta.get('stem','')}  "
              f"area={nb_area}  sam={nb_sam:.3f}")

    for ax in axes_flat[n_total:]:
        ax.set_visible(False)

    fig.suptitle(
        f"LV cavity candidate + {n_neighbors} nearest neighbors\n"
        f"({stem})  red=candidate  blue=neighbors",
        fontsize=10,
    )
    fig.tight_layout()
    out = output_dir / f"lv_candidate_neighbors_{stem}.png"
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"\n  Saved -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostic for HDBSCAN leaf clustering on pre-computed features."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=_DEFAULT_RUN_DIR,
        help=(
            "Root directory of the experiment run "
            "(must contain resolved_config.json and phase2_clustering/). "
            f"Default: {_DEFAULT_RUN_DIR}"
        ),
    )
    parser.add_argument(
        "--method",
        default="leaf",
        choices=["leaf", "eom"],
        help="HDBSCAN cluster_selection_method (default: leaf)",
    )
    parser.add_argument(
        "--min-image-frequency",
        type=float,
        default=None,
        help="Override min_image_frequency from resolved_config.json (e.g. 0.1)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help=(
            "Override HDBSCAN min_samples (density threshold per point). "
            "Config default: min_samples_fraction × num_images = 25."
        ),
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=None,
        help=(
            "Override HDBSCAN min_cluster_size (minimum objects to form a cluster). "
            "Config default: min_cluster_size_fraction × num_images = 129."
        ),
    )
    parser.add_argument(
        "--inspect-image",
        type=str,
        default=None,
        metavar="STEM",
        help=(
            "Identify the most likely LV cavity mask in this image stem, "
            "find its 10 nearest neighbors in feature space, and save masks. "
            "Example: SCD0000901_IM_0003_0028"
        ),
    )
    parser.add_argument(
        "--num-references",
        type=int,
        default=None,
        help=(
            "Number of top reference masks to save per cluster. "
            "Overrides num_reference_frames from resolved_config.json (default: 3)."
        ),
    )
    parser.add_argument(
        "--exclude-features",
        nargs="+",
        default=None,
        metavar="FEATURE",
        help=(
            "Feature names to drop before clustering (space-separated). "
            "Example: --exclude-features orientation ecc"
        ),
    )
    parser.add_argument(
        "--sam-score-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of SAM score as an extra clustering feature (0 = disabled). "
            "The SAM score is z-scored and then scaled by this value before "
            "being appended to the 16 shape features."
        ),
    )
    parser.add_argument(
        "--cross-analyze",
        action="store_true",
        help=(
            "Run EOM clustering internally and cross-tabulate against --method "
            "to show where EOM cluster_0 objects land in the target method."
        ),
    )
    parser.add_argument(
        "--n-preview",
        type=int,
        default=10,
        help="Number of images to include in the preview output (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for image sampling (default: 42)",
    )
    args = parser.parse_args()

    method = args.method
    run_dir = args.run_dir.resolve()
    phase2_dir = run_dir / "phase2_clustering"
    features_csv = phase2_dir / "clustering_features.csv"
    resolved_config_path = run_dir / "resolved_config.json"

    # Dataset name is the parent folder of the run dir (e.g. "Sunnybrook", "XRay")
    dataset = run_dir.parent.name

    for p in (features_csv, resolved_config_path):
        if not p.exists():
            print(f"ERROR: required file not found: {p}", file=sys.stderr)
            sys.exit(1)

    # --- Load config ---------------------------------------------------------
    config = load_config(resolved_config_path)
    if args.min_image_frequency is not None:
        config["cluster_filter"]["min_image_frequency"] = args.min_image_frequency
        print(f"[override] min_image_frequency = {args.min_image_frequency}")
    if args.min_samples is not None:
        print(f"[override] min_samples = {args.min_samples}")
    if args.min_cluster_size is not None:
        print(f"[override] min_cluster_size = {args.min_cluster_size}")
    num_images = config["num_images"]
    print(f"Run    : {run_dir}")
    print(f"Dataset: {dataset}  ({num_images} images)")

    # --- Build object index from Phase 1 cache -------------------------------
    print("Loading segmentation cache index...")
    obj_index, seg_dir = build_object_index(config, dataset)
    print(f"  Objects indexed: {len(obj_index)}")

    # --- Run clustering on pre-standardized features -------------------------
    result_df = run_clustering(
        features_csv, config, method,
        min_samples_override=args.min_samples,
        min_cluster_size_override=args.min_cluster_size,
        obj_index=obj_index,
        sam_score_weight=args.sam_score_weight,
        exclude_features=args.exclude_features,
    )

    # --- Enrich with metadata ------------------------------------------------
    stats = enrich_with_metadata(result_df, obj_index)

    # --- Good-cluster detection ----------------------------------------------
    good_ids = get_good_cluster_ids(stats, config)

    # --- Console summary -----------------------------------------------------
    print_summary(stats, config, good_ids)

    # --- Load NPZ (lazy — stays as mmap object) ------------------------------
    npz_path = seg_dir / "segmented.npz"
    print(f"\nLoading mask cache: {npz_path.name}")
    npz = np.load(str(npz_path), allow_pickle=False)

    # --- Output 1: preview overlays ------------------------------------------
    preview_dir = phase2_dir / f"{method}_clustering_preview"
    preview_dir.mkdir(exist_ok=True)

    all_stems = sorted(stats["stem"].unique())
    rng = np.random.default_rng(args.seed)
    n_sample = min(args.n_preview, len(all_stems))
    sampled_stems = sorted(rng.choice(all_stems, size=n_sample, replace=False))

    print(f"\nGenerating {n_sample} preview images -> {preview_dir.name}/")
    for stem in sampled_stems:
        stem_df = stats[stats["stem"] == stem]
        out = preview_dir / f"{stem}_clustered.png"
        generate_preview(stem, stem_df, npz, out, method)
        print(f"  {out.name}")

    # --- Output 2: reference panels ------------------------------------------
    refs_dir = phase2_dir / f"{method}_clustering_references"
    refs_dir.mkdir(exist_ok=True)

    print(f"\nGenerating reference panels -> {refs_dir.name}/")
    if not good_ids:
        print("  No good clusters found — no reference panels generated.")
    for cid in good_ids:
        out = refs_dir / f"cluster_{cid}_references.png"
        generate_references(cid, stats, npz, config, out, method,
                            num_references_override=args.num_references)
        print(f"  {out.name}")

    # --- Output 3: feature distribution analysis ----------------------------
    analysis_dir = phase2_dir / f"{method}_feature_analysis"
    analysis_dir.mkdir(exist_ok=True)
    print(f"\nGenerating feature analysis -> {analysis_dir.name}/")
    plot_feature_distributions(stats, features_csv, config, analysis_dir, method)

    # --- Output 4: cross-analysis EOM vs target method ----------------------
    if args.cross_analyze and method != "eom":
        cross_analyze(
            features_csv, config, method, obj_index,
            output_dir=analysis_dir,
            min_samples_override=args.min_samples,
            min_cluster_size_override=args.min_cluster_size,
            sam_score_weight=args.sam_score_weight,
            exclude_features=args.exclude_features,
        )

    # --- Output 5: LV cavity candidate inspection ---------------------------
    if args.inspect_image:
        inspect_candidate_mask(
            stem=args.inspect_image,
            obj_index=obj_index,
            features_csv=features_csv,
            npz=npz,
            output_dir=analysis_dir,
            exclude_features=args.exclude_features,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
cluster_stability.py
--------------------
Bootstrap cluster stability analysis for HDBSCAN partitions.

Implements the clusterwise Jaccard bootstrap of Hennig (2007):
  1. Fit HDBSCAN on the full feature matrix -> original partition.
  2. Repeat B times: resample n objects with replacement (same n), refit
     HDBSCAN with identical hyperparameters, and compute the max Jaccard
     between each original cluster and any bootstrap cluster.
  3. Report mean ± std of max Jaccard across B bootstraps per cluster.

Jaccard is computed over sets of object_ids (not pixels):
  J(A, B) = |A ∩ B| / |A ∪ B|

Note on noise objects (label -1)
---------------------------------
Noise objects are excluded from all cluster sets before Jaccard is computed.
High noise fractions do NOT invalidate the metric: the stability score
measures reproducibility of the cluster *core* — the objects HDBSCAN was
confident enough to assign — not coverage or completeness. A tight,
well-separated cluster surrounded by a large noise halo can still score
near 1.0.

In-bag Jaccard (0–1 scale)
--------------------------
A with-replacement resample of size n contains only ≈ n·(1 − 1/e) ≈ 63% of the
unique objects. If the original cluster were compared as-is, the ≈37% that were
never drawn would deflate the Jaccard even for a perfectly reproducible cluster,
capping the score near 0.63. That 37% is absence by sampling, not instability.

To avoid this, the original cluster is restricted to its in-bag members (the
object_ids actually drawn into the resample) before intersecting with the
bootstrap clusters. A perfectly reproducible cluster then scores Jaccard 1.0,
while merges and splits stay penalised. This recovers the genuine 0–1 scale on
which Hennig's thresholds (0.75 stable, 0.85 very stable) are calibrated.

Output
------
  <output>/stability_<method>.json  — {cluster_id: {jaccard_mean, jaccard_std, n_objects}}
  <output>/stability_<method>.png   — bar chart with error bars,
                                      dashed reference lines at 0.75 and 0.85

CPU-only. No GPU required.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN

# Reuse param-resolution helpers from the existing diagnostic tool so that
# the HDBSCAN hyperparameters stay in sync with the pipeline without duplication.
from tools.explore_leaf_clustering import load_config, resolve_hdbscan_params


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def load_features(
    features_csv: Path,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the pre-standardized feature matrix from clustering_features.csv.

    Returns (X, object_ids).  X has shape (n, d); features are already
    z-scored by the pipeline and are used as-is.
    """
    df = pd.read_csv(features_csv)
    feature_cols = config["labeler"]["features"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns missing from CSV: {missing}")
    X = df[feature_cols].values.astype(float)
    object_ids = df["object_id"].values
    return X, object_ids


def load_good_clusters(summary_path: Path) -> set[int] | None:
    """
    Read the set of *final* (non-filtered) cluster ids from the pipeline's
    phase2_clustering/summary.json.

    The pipeline drops low-quality clusters during phase 2 (e.g. too small or
    rejected by the labeler) and records the survivors under "good_clusters".
    Returns that set so reporting can be restricted to the clusters that
    actually make it into the final partition, or None if the file/key is
    unavailable (callers then fall back to reporting every cluster).
    """
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    good = summary.get("good_clusters")
    if good is None:
        return None
    return {int(c) for c in good}


# ---------------------------------------------------------------------------
# Bootstrap stability (Hennig 2007)
# ---------------------------------------------------------------------------

def _jaccard(a: set, b: set) -> float:
    union = len(a | b)
    return len(a & b) / union if union > 0 else 0.0


def bootstrap_stability(
    X: np.ndarray,
    object_ids: np.ndarray,
    original_labels: np.ndarray,
    hdbscan_kwargs: dict,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[int, list[float]]:
    """
    Compute per-bootstrap max Jaccard for each non-noise original cluster.

    For each bootstrap resample:
      - Draw n rows with replacement from (X, object_ids).
      - Fit HDBSCAN with hdbscan_kwargs.
      - For each original cluster, find the bootstrap cluster that maximises
        Jaccard(original_set, bootstrap_set) and record that value.

    Parameters
    ----------
    X : (n, d) pre-standardized feature matrix.
    object_ids : (n,) object identifiers (matched row-for-row to X).
    original_labels : (n,) cluster labels from the original fit (-1 = noise).
    hdbscan_kwargs : dict forwarded to HDBSCAN for every bootstrap run.
    n_bootstrap : number of bootstrap resamples.
    rng : NumPy Generator for reproducibility.

    Returns
    -------
    dict mapping each non-noise cluster_id to a list of B max-Jaccard values.
    """
    n = len(X)
    unique_clusters = sorted(c for c in np.unique(original_labels) if c != -1)

    orig_sets: dict[int, set] = {
        cid: set(object_ids[original_labels == cid])
        for cid in unique_clusters
    }
    jaccards: dict[int, list[float]] = {cid: [] for cid in unique_clusters}

    for b in range(n_bootstrap):
        if (b + 1) % 25 == 0:
            print(f"  bootstrap {b + 1}/{n_bootstrap} ...")

        idx = rng.integers(0, n, size=n)
        X_boot = X[idx]
        ids_boot = object_ids[idx]

        boot_labels = HDBSCAN(**hdbscan_kwargs).fit_predict(X_boot)

        # Unique object_ids that were actually drawn into this resample.
        # A with-replacement bootstrap contains only ≈63% of the unique
        # objects; the rest are absent purely by sampling, not instability.
        in_bag = set(ids_boot)

        # Bootstrap cluster sets: unique object_ids per cluster, noise excluded
        boot_sets: dict[int, set] = {}
        for cid in np.unique(boot_labels):
            if cid == -1:
                continue
            boot_sets[cid] = set(ids_boot[boot_labels == cid])

        for orig_cid, orig_set in orig_sets.items():
            # Restrict the original cluster to its in-bag members before
            # comparing. This removes the spurious "missing 37%" penalty so a
            # perfectly reproducible cluster scores Jaccard 1.0; merges/splits
            # are still penalised because foreign ids or missing in-bag
            # coverage appear inside the resample.
            restricted = orig_set & in_bag
            if not restricted:
                # None of this cluster's objects were drawn — the cluster is
                # simply absent from the resample, so Jaccard is undefined.
                # Skip rather than record a spurious 0.
                continue
            if not boot_sets:
                jaccards[orig_cid].append(0.0)
            else:
                max_j = max(_jaccard(restricted, bs) for bs in boot_sets.values())
                jaccards[orig_cid].append(max_j)

    return jaccards


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_results(results: dict) -> None:
    print(f"\n  {'Cluster':<12} {'n_objects':>10} {'jaccard_mean':>14} {'jaccard_std':>13}")
    print("  " + "-" * 52)
    for key in sorted(results, key=int):
        v = results[key]
        print(f"  cluster_{key:<4} {v['n_objects']:>10} "
              f"{v['jaccard_mean']:>14.3f} {v['jaccard_std']:>13.3f}")


def _plot_stability(
    results: dict,
    output_path: Path,
    method: str,
    n_bootstrap: int,
) -> None:
    """Bar chart: clusters on x, Jaccard stability on y, error bars = std."""
    cids = sorted(results.keys(), key=int)
    labels = [f"cluster_{c}" for c in cids]
    means  = [results[c]["jaccard_mean"] for c in cids]
    stds   = [results[c]["jaccard_std"]  for c in cids]

    x = np.arange(len(cids))
    fig, ax = plt.subplots(figsize=(max(5, len(cids) * 1.4), 4))

    ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.85,
           error_kw={"elinewidth": 1.5})
    ax.axhline(0.85, color="green",  linestyle="--", linewidth=1.2,
               label="0.85 — very stable")
    ax.axhline(0.75, color="orange", linestyle="--", linewidth=1.2,
               label="0.75 — stable")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Jaccard stability (mean ± std over bootstraps)")
    ax.set_title(
        f"HDBSCAN cluster stability  [{method} method, B={n_bootstrap} bootstraps]"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap cluster stability (Jaccard, Hennig 2007) for HDBSCAN. "
            "CPU-only, no GPU required."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Experiment run directory — must contain resolved_config.json "
             "and phase2_clustering/clustering_features.csv",
    )
    parser.add_argument(
        "--method", default="leaf", choices=["eom", "leaf"],
        help="HDBSCAN cluster_selection_method (default: leaf, matching the pipeline)",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=100,
        help="Number of bootstrap resamples",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for bootstrap resampling",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Directory to write stability_<method>.json and stability_<method>.png",
    )
    parser.add_argument(
        "--all-clusters", action="store_true",
        help="Report every HDBSCAN cluster. By default the print and plot are "
             "restricted to the final clusters (good_clusters in "
             "phase2_clustering/summary.json), hiding clusters the pipeline filtered out.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    features_csv = run_dir / "phase2_clustering" / "clustering_features.csv"
    summary_path = run_dir / "phase2_clustering" / "summary.json"
    config_path  = run_dir / "resolved_config.json"

    for p in (features_csv, config_path):
        if not p.exists():
            print(f"ERROR: required file not found: {p}", file=sys.stderr)
            sys.exit(1)

    config = load_config(config_path)
    mcs, ms, metric = resolve_hdbscan_params(config)
    hdbscan_kwargs = dict(
        min_cluster_size=mcs,
        min_samples=ms,
        metric=metric,
        cluster_selection_method=args.method,
    )

    print(f"Run dir   : {run_dir}")
    print(f"Method    : {args.method}")
    print(f"HDBSCAN   : min_cluster_size={mcs}  min_samples={ms}  metric={metric}")
    print(f"Bootstrap : B={args.n_bootstrap}  seed={args.seed}")

    X, object_ids = load_features(features_csv, config)
    print(f"Feature matrix: {X.shape}  ({X.shape[0]} objects, {X.shape[1]} features)")

    # Original clustering
    print("\nFitting original HDBSCAN ...")
    original_labels = HDBSCAN(**hdbscan_kwargs).fit_predict(X)

    unique_labels = np.unique(original_labels)
    n_noise    = int((original_labels == -1).sum())
    n_clusters = sum(1 for c in unique_labels if c != -1)
    print(f"  {n_clusters} clusters  |  "
          f"{n_noise} noise objects ({100 * n_noise / len(X):.1f}%)")

    if n_clusters == 0:
        print(
            "ERROR: original clustering produced no clusters. "
            "Adjust HDBSCAN parameters.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Bootstrap stability
    print(f"\nRunning {args.n_bootstrap} bootstrap resamples ...")
    rng = np.random.default_rng(args.seed)
    jaccards = bootstrap_stability(
        X, object_ids, original_labels, hdbscan_kwargs, args.n_bootstrap, rng
    )

    # Aggregate into per-cluster stats
    results: dict[str, dict] = {}
    for cid in sorted(jaccards):
        arr = np.array(jaccards[cid])
        results[str(cid)] = {
            "jaccard_mean": float(arr.mean()),
            "jaccard_std":  float(arr.std()),
            "n_objects":    int((original_labels == cid).sum()),
        }

    # Restrict the printed table and the bar chart to the final clusters that
    # survived phase-2 filtering. The full JSON below still records every
    # cluster so nothing is lost. Pass --all-clusters to disable the filter.
    display_results = results
    if not args.all_clusters:
        good = load_good_clusters(summary_path)
        if good is None:
            print("\n  [warn] good_clusters not found in summary.json; "
                  "showing all clusters. Use --all-clusters to silence this.")
        else:
            display_results = {k: v for k, v in results.items() if int(k) in good}
            filtered_out = sorted((int(k) for k in results if int(k) not in good))
            print(f"\n  Final clusters: {sorted(int(k) for k in display_results)}  "
                  f"|  filtered out (hidden): {filtered_out}")

    _print_results(display_results)

    args.output.mkdir(parents=True, exist_ok=True)

    json_path = args.output / f"stability_{args.method}.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "clusters": results,
                "config": {
                    "method":            args.method,
                    "n_bootstrap":       args.n_bootstrap,
                    "seed":              args.seed,
                    "min_cluster_size":  mcs,
                    "min_samples":       ms,
                    "metric":            metric,
                },
            },
            f,
            indent=2,
        )
    print(f"\n  Saved: {json_path}")

    _plot_stability(display_results, args.output / f"stability_{args.method}.png",
                    args.method, args.n_bootstrap)

    print("\nDone.")


if __name__ == "__main__":
    main()

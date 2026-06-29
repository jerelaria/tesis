"""
reevaluate_all.py
-----------------
Re-run evaluation against existing masks/ directories.

Regenerates summary.json and metrics.csv without re-running the prediction
pipeline. Use this after updating the evaluation code (e.g., adding
pr_curve_per_threshold, pct_real_scores, match_threshold, or changing the
default IoU threshold grid) to backfill new keys into existing results.

The masks/ directories are left untouched; only summary.json and metrics.csv
are overwritten.

Usage
-----
    # Re-evaluate all experiments with default settings:
    python reevaluate_all.py

    # Stricter quality gate, custom IoU grid:
    python reevaluate_all.py --match-threshold 0.75 \\
        --iou-thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.9

    # Only specific versions and experiments:
    python reevaluate_all.py \\
        --versions v1_baseline v3_extended \\
        --experiments unsup_hdbscan unsup_hdbscan_refine fs_iter_refine_1ref

    # Preview what would be re-evaluated without writing files:
    python reevaluate_all.py --dry-run

    # Custom GT directories:
    python reevaluate_all.py \\
        --dataset-gt XRay:data/processed/XRay/masks \\
                     Sunnybrook:data/processed/Sunnybrook/masks

Dataset-to-GT mapping
---------------------
The script maps each dataset directory name to a GT masks directory. Defaults
are provided for JSRT (XRay) and Sunnybrook (Sunnybrook).
Override with --dataset-gt.

Matching strategy
-----------------
Auto-detected from the experiment name prefix:
    unsup_*  ->  greedy
    all else ->  semantic
Override with --matching.
"""

import argparse
from pathlib import Path

from project.evaluation.runner import evaluate
from project.evaluation.io import save_results, print_summary
from project.evaluation.cluster_map import load_cluster_map


# Default GT directory mapping: dataset_dir_name -> relative GT masks path
_DATASET_GT_DEFAULTS: dict[str, str] = {
    "XRay":       "data/processed/XRay/masks",
    "Sunnybrook": "data/processed/Sunnybrook/masks",
    "ACDC":       "data/processed/ACDC/masks",
}

# 1.0 excluded: pixel-perfect threshold is degenerate for real-world masks
_DEFAULT_IOU_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]


def _auto_matching(exp_name: str) -> str:
    """Infer matching strategy from experiment name prefix."""
    return "greedy" if exp_name.startswith("unsup_") else "semantic"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate existing predicted masks and regenerate summary.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Root results directory (version/dataset/experiment/)",
    )
    parser.add_argument(
        "--dataset-gt", nargs="*",
        metavar="DATASET:GT_DIR",
        help="Dataset name to GT masks dir mapping, e.g. "
             "XRay:data/processed/XRay/masks. "
             f"Defaults: {_DATASET_GT_DEFAULTS}",
    )
    parser.add_argument(
        "--versions", nargs="*",
        help="Filter to these version directories (default: all)",
    )
    parser.add_argument(
        "--datasets", nargs="*",
        metavar="DATASET",
        help="Filter to these dataset directory names (default: all with a GT mapping)",
    )
    parser.add_argument(
        "--experiments", nargs="*",
        help="Filter to these experiment names (default: all)",
    )
    parser.add_argument(
        "--match-threshold", type=float, default=0.5,
        help="IoU gate for quality matching: pairs below this threshold are "
             "demoted to missing (pred_name=None, Dice=0). Saved in summary.json.",
    )
    parser.add_argument(
        "--iou-thresholds", nargs="+", type=float,
        default=_DEFAULT_IOU_THRESHOLDS,
        help="IoU thresholds at which to report P/R/F1 coverage metrics.",
    )
    parser.add_argument(
        "--matching", default=None, choices=["semantic", "greedy"],
        help="Override matching strategy for all experiments. Default: "
             "auto-detect from experiment name (unsup_* -> greedy, "
             "all others -> semantic).",
    )
    parser.add_argument(
        "--cluster-map-filename", default="cluster_map.json",
        help="Per-experiment cluster-map file looked up in each experiment "
             "directory. When present it relabels predictions and forces "
             "semantic matching.",
    )
    parser.add_argument(
        "--ignore-cluster-maps", action="store_true",
        help="Ignore all per-experiment cluster_map.json files and fall back to "
             "auto-detect / --matching (useful for debug and the pure baseline).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be re-evaluated without writing any files.",
    )
    args = parser.parse_args()

    # Build dataset → GT path mapping
    dataset_gt: dict[str, Path] = {
        k: Path(v) for k, v in _DATASET_GT_DEFAULTS.items()
    }
    if args.dataset_gt:
        for item in args.dataset_gt:
            if ":" not in item:
                parser.error(f"--dataset-gt expects DATASET:GT_DIR, got: {item!r}")
            ds, gt_path = item.split(":", 1)
            dataset_gt[ds.strip()] = Path(gt_path.strip())

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Collect all (version, dataset, experiment) triples that have a masks/ dir
    triples = []
    for version_dir in sorted(results_dir.iterdir()):
        if (not version_dir.is_dir()
                or version_dir.name.startswith(".")
                or version_dir.name == "comparison"):
            continue
        if args.versions and version_dir.name not in args.versions:
            continue

        for dataset_dir in sorted(version_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            if args.datasets and dataset_dir.name not in args.datasets:
                continue
            gt_dir = dataset_gt.get(dataset_dir.name)
            if gt_dir is None:
                print(f"  [SKIP] No GT mapping for dataset: {dataset_dir.name}")
                continue
            if not gt_dir.is_dir():
                print(f"  [SKIP] GT dir not found: {gt_dir}")
                continue

            for exp_dir in sorted(dataset_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                masks_dir = exp_dir / "masks"
                if not masks_dir.is_dir():
                    continue
                if args.experiments and exp_dir.name not in args.experiments:
                    continue
                triples.append((
                    version_dir.name,
                    dataset_dir.name,
                    exp_dir.name,
                    masks_dir,       # pred_dir
                    gt_dir,
                    exp_dir,         # output_dir (overwrites existing summary)
                ))

    print(f"Found {len(triples)} experiment(s) to re-evaluate.")

    if args.dry_run:
        for version, dataset, exp, *_ in triples:
            print(f"  Would re-evaluate: {version}/{dataset}/{exp}")
        return

    n_ok = 0
    n_err = 0
    for version, dataset, exp, pred_dir, gt_dir, output_dir in triples:
        # A per-experiment cluster_map.json (next to masks/) relabels predictions
        # and forces semantic matching; the baseline simply omits the file.
        cluster_map = None
        cmap_path = output_dir / args.cluster_map_filename
        if not args.ignore_cluster_maps and cmap_path.exists():
            cluster_map = load_cluster_map(cmap_path)
            matching = "semantic"
            if args.matching == "greedy":
                print(f"  [WARN] experiment {exp}: --matching greedy ignored; "
                      f"cluster_map forces semantic")
            print(f"  experiment {exp}: applying cluster_map (semantic)")
        else:
            matching = args.matching or _auto_matching(exp)
        print(f"\nRe-evaluating {version}/{dataset}/{exp}"
              f" (matching={matching}, match_threshold={args.match_threshold})")
        try:
            all_results, summary = evaluate(
                gt_dir=gt_dir,
                pred_dir=pred_dir,
                matching=matching,
                iou_thresholds=args.iou_thresholds,
                compute_map_metric=True,
                match_threshold=args.match_threshold,
                cluster_map=cluster_map,
            )
            save_results(all_results, summary, output_dir)
            n_ok += 1
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            n_err += 1

    print(f"\nDone: {n_ok} succeeded, {n_err} failed.")


if __name__ == "__main__":
    main()

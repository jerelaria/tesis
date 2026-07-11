#!/usr/bin/env python3
"""Consolidate over-clustered organ labels in an existing results directory.

Collapses several clusters that consistently refer to the same organ (high
mask IoU across co-occurring images) into a single canonical label per organ,
using only the masks/scores already produced by save_predicted_masks. Runs
fully offline: no GPU, no re-segmentation, no re-propagation.

Usage:
    python scripts/consolidate_results.py --results-dir results/some_run \\
        --iou-threshold 0.5 --min-overlap-images 1
"""

import argparse
import json
import sys
from pathlib import Path

from project.consolidation.core import consolidate
from project.consolidation.io import read_records_from_results, write_consolidated_results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", required=True, type=Path,
        help="Results directory containing a masks/ subfolder.",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.5,
        help="Median IoU threshold to merge two labels (default: 0.5).",
    )
    parser.add_argument(
        "--min-overlap-images", type=int, default=1,
        help="Minimum number of co-occurring images required to consider merging (default: 1).",
    )
    parser.add_argument(
        "--out-name", default="masks_consolidated",
        help="Subdirectory name (under --results-dir) for the consolidated output.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    records_by_image = read_records_from_results(args.results_dir)
    consolidated_by_image, report = consolidate(
        records_by_image,
        iou_threshold=args.iou_threshold,
        min_overlap_images=args.min_overlap_images,
    )

    out_dir = args.results_dir / args.out_name
    n_written = write_consolidated_results(consolidated_by_image, out_dir)

    report_path = out_dir / "consolidation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("Consolidation summary")
    print(f"  Labels before:  {report['n_labels_before']}")
    print(f"  Organs after:   {report['n_organs_after']}")
    if report["groups"]:
        print("  Merged groups:")
        for group in report["groups"]:
            print(f"    {group['canonical']} <- {group['discarded']}")
    else:
        print("  Merged groups:  none")
    print(f"  Masks dropped by dedup: {report['n_masks_dropped_dedup']}")
    print(f"  Masks written: {n_written}")
    print(f"  Report: {report_path}")
    print(f"\nPoint evaluation at: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

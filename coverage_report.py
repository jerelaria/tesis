"""
coverage_report.py
------------------
Diagnostic: how many GT organs each method covers, name-agnostic.

For each (image, GT organ), compute the max IoU over all predicted masks
in that image. A GT organ is "covered" by a method if max IoU >= threshold.

This is independent of prediction names: a baseline that names predictions
'obj_000', 'obj_001', etc., is judged purely on geometric overlap with GT.
This decouples the coverage question (do my predictions exist where the GT
says they should?) from the labeling question (are they named correctly?).

Useful to verify:
- Whether the baseline already has high coverage (small delta is expected).
- Whether refinement actually recovers missing organs.
- Whether pipeline predictions are mostly noise or anatomically meaningful.

Usage:
    python coverage_report.py \
        --gt data/processed/XRayNicoSent/masks \
        --methods \
            "baseline:results/v0_baseline/XRayNicoSent/unsup_baseline/masks" \
            "kmeans+r:results/v1_baseline/XRayNicoSent/unsup_kmeans_refine/masks" \
        --threshold 0.5 \
        --output coverage_report.json
"""

import argparse
import json
import re
import numpy as np
from pathlib import Path
from PIL import Image


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over Union between two binary masks."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def parse_organ_name(stem: str) -> str:
    """Strip trailing _N (instance number) from a mask filename stem."""
    match = re.match(r"^(.+)_(\d+)$", stem)
    return match.group(1) if match else stem


def load_masks(d: Path) -> dict[str, np.ndarray]:
    """Load all .png masks from a directory as binary arrays."""
    masks = {}
    if not d.is_dir():
        return masks
    for f in sorted(d.iterdir()):
        if f.suffix.lower() != ".png" or f.name == "image.png":
            continue
        masks[f.stem] = np.array(Image.open(f).convert("L")) > 127
    return masks


# ---------------------------------------------------------------------------
# Per-method evaluation
# ---------------------------------------------------------------------------

def evaluate_method(
    label: str,
    pred_dir: Path,
    gt_data: dict[str, dict[str, np.ndarray]],
    threshold: float,
) -> dict:
    """
    Compute per-organ coverage and prediction-side relevance for one method.

    Returns a dict with:
      - per_organ:        {organ: {total, covered, coverage_pct,
                                   mean_max_iou, std_max_iou}}
      - total_gt:         total GT organ instances across all images
      - total_covered:    total covered (max IoU >= threshold)
      - coverage_pct:     100 * total_covered / total_gt
      - n_pred_total:     total number of predicted masks across all images
      - n_pred_relevant:  predictions with IoU >= threshold vs any GT organ
      - n_pred_extra:     predictions with IoU < threshold vs every GT organ
                          (potential anatomical discoveries OR noise)
    """
    per_organ_iou: dict[str, list[float]] = {}
    per_organ_covered: dict[str, int] = {}
    per_organ_total: dict[str, int] = {}
    n_pred_total = 0
    n_pred_relevant = 0

    for stem, gt_masks in gt_data.items():
        pred_masks = load_masks(pred_dir / stem)
        n_pred_total += len(pred_masks)

        gt_arrays = list(gt_masks.values())
        pred_arrays = list(pred_masks.values())

        # Pred-side: count predictions overlapping any GT organ above threshold
        if gt_arrays:
            for pred_mask in pred_arrays:
                max_iou_vs_gt = max(
                    (iou(pred_mask, g) for g in gt_arrays), default=0.0
                )
                if max_iou_vs_gt >= threshold:
                    n_pred_relevant += 1

        # GT-side: max IoU per GT organ
        for gt_name, gt_mask in gt_masks.items():
            organ = parse_organ_name(gt_name)
            best = max(
                (iou(p, gt_mask) for p in pred_arrays), default=0.0
            )
            per_organ_iou.setdefault(organ, []).append(best)
            per_organ_total[organ] = per_organ_total.get(organ, 0) + 1
            if best >= threshold:
                per_organ_covered[organ] = (
                    per_organ_covered.get(organ, 0) + 1
                )

    # Aggregate per-organ
    organ_summary: dict[str, dict] = {}
    total_gt = 0
    total_covered = 0
    for organ in sorted(per_organ_total):
        ious_list = per_organ_iou[organ]
        cov = per_organ_covered.get(organ, 0)
        tot = per_organ_total[organ]
        organ_summary[organ] = {
            "total": tot,
            "covered": cov,
            "coverage_pct": 100.0 * cov / tot,
            "mean_max_iou": float(np.mean(ious_list)),
            "std_max_iou": float(np.std(ious_list)),
        }
        total_gt += tot
        total_covered += cov

    return {
        "label": label,
        "pred_dir": str(pred_dir),
        "per_organ": organ_summary,
        "total_gt": total_gt,
        "total_covered": total_covered,
        "coverage_pct": (100.0 * total_covered / total_gt) if total_gt else 0.0,
        "n_pred_total": n_pred_total,
        "n_pred_relevant": n_pred_relevant,
        "n_pred_extra": n_pred_total - n_pred_relevant,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: list[dict], threshold: float) -> None:
    if not results:
        print("No methods to report.")
        return

    organs = sorted({o for r in results for o in r["per_organ"]})
    label_w = max(len(r["label"]) for r in results) + 2
    label_w = max(label_w, 12)

    # ------------------------------------------------------------------
    # Coverage table
    # ------------------------------------------------------------------
    print(f"\nCoverage  (a GT organ is 'covered' if max IoU >= {threshold})")
    print("=" * 100)
    header = f"{'Method':<{label_w}}"
    for o in organs:
        header += f"{o:>16}"
    header += f"{'TOTAL':>18}"
    print(header)
    print("-" * len(header))

    for r in results:
        line = f"{r['label']:<{label_w}}"
        for o in organs:
            stat = r["per_organ"].get(o)
            if stat:
                cell = f"{stat['covered']}/{stat['total']} ({stat['coverage_pct']:.0f}%)"
                line += f"{cell:>16}"
            else:
                line += f"{'-':>16}"
        cell = f"{r['total_covered']}/{r['total_gt']} ({r['coverage_pct']:.0f}%)"
        line += f"{cell:>18}"
        print(line)

    # ------------------------------------------------------------------
    # Quality: mean max IoU per GT organ
    # ------------------------------------------------------------------
    print(f"\nMean max IoU per GT organ  (quality of best matching prediction, including 0s for missing)")
    print("=" * 100)
    header = f"{'Method':<{label_w}}"
    for o in organs:
        header += f"{o:>16}"
    print(header)
    print("-" * len(header))

    for r in results:
        line = f"{r['label']:<{label_w}}"
        for o in organs:
            stat = r["per_organ"].get(o)
            if stat:
                cell = f"{stat['mean_max_iou']:.3f}±{stat['std_max_iou']:.2f}"
                line += f"{cell:>16}"
            else:
                line += f"{'-':>16}"
        print(line)

    # ------------------------------------------------------------------
    # Pred-side relevance (FP / discovery analysis)
    # ------------------------------------------------------------------
    print(f"\nPrediction relevance  (preds with IoU >= {threshold} vs any GT organ)")
    print("=" * 100)
    print(f"{'Method':<{label_w}}{'Total preds':>14}{'Relevant':>12}"
          f"{'Extra':>10}{'Extra %':>10}")
    print("-" * 100)
    for r in results:
        extra_pct = (
            100.0 * r["n_pred_extra"] / r["n_pred_total"]
            if r["n_pred_total"] else 0.0
        )
        print(f"{r['label']:<{label_w}}"
              f"{r['n_pred_total']:>14}"
              f"{r['n_pred_relevant']:>12}"
              f"{r['n_pred_extra']:>10}"
              f"{extra_pct:>9.0f}%")

    # ------------------------------------------------------------------
    # Delta vs first method
    # ------------------------------------------------------------------
    if len(results) > 1:
        baseline = results[0]
        print(f"\nDelta coverage vs '{baseline['label']}'  "
              f"(positive = recovered organs vs baseline)")
        print("=" * 100)
        header = f"{'Method':<{label_w}}"
        for o in organs:
            header += f"{o:>16}"
        header += f"{'TOTAL':>18}"
        print(header)
        print("-" * len(header))
        for r in results[1:]:
            line = f"{r['label']:<{label_w}}"
            for o in organs:
                base_stat = baseline["per_organ"].get(o)
                cur_stat = r["per_organ"].get(o)
                if base_stat and cur_stat:
                    delta = cur_stat["covered"] - base_stat["covered"]
                    sign = "+" if delta >= 0 else ""
                    cell = f"{sign}{delta}"
                    line += f"{cell:>16}"
                else:
                    line += f"{'-':>16}"
            total_delta = r["total_covered"] - baseline["total_covered"]
            sign = "+" if total_delta >= 0 else ""
            line += f"{sign}{total_delta:>17}"
            print(line)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic coverage report comparing methods against GT."
    )
    parser.add_argument("--gt", required=True,
                        help="GT masks directory (per-image subfolders)")
    parser.add_argument("--methods", nargs="+", required=True,
                        help="One or more 'label:pred_dir' pairs. "
                             "The first method is used as the delta baseline.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="IoU threshold for 'covered' (default: 0.5)")
    parser.add_argument("--output", default=None,
                        help="Optional JSON output path")
    args = parser.parse_args()

    gt_dir = Path(args.gt)
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT dir not found: {gt_dir}")

    # Parse "label:path" pairs
    methods = []
    for m in args.methods:
        if ":" not in m:
            raise ValueError(f"Method must be 'label:pred_dir', got: {m}")
        label, pred_dir = m.split(":", 1)
        methods.append((label, Path(pred_dir)))

    # Collect stems that have predictions in any method directory.
    # This restricts evaluation to images that were actually processed,
    # avoiding denominator inflation from unprocessed GT images.
    pred_stems: set[str] = set()
    for _, pred_dir in methods:
        if pred_dir.is_dir():
            pred_stems.update(d.name for d in pred_dir.iterdir() if d.is_dir())

    # Load GT restricted to images present in at least one pred dir
    print(f"Loading GT from {gt_dir}...")
    gt_data_full: dict[str, dict[str, np.ndarray]] = {}
    for d in sorted(gt_dir.iterdir()):
        if d.is_dir():
            gt_data_full[d.name] = load_masks(d)

    n_gt_total = len(gt_data_full)
    gt_data = {stem: masks for stem, masks in gt_data_full.items()
               if stem in pred_stems}
    n_images = len(gt_data)
    n_organs = sum(len(m) for m in gt_data.values())
    print(f"  GT total: {n_gt_total} images")
    print(f"  Restricted to {n_images} images present in predictions "
          f"({n_gt_total - n_images} images excluded)")
    print(f"  {n_organs} GT organ instances in evaluated set")

    # Evaluate each method
    results = []
    for label, pred_dir in methods:
        print(f"\nEvaluating: {label}  ({pred_dir})")
        if not pred_dir.is_dir():
            print(f"  [WARN] pred dir does not exist; counts will be zero")
        r = evaluate_method(label, pred_dir, gt_data, args.threshold)
        results.append(r)

    print_report(results, args.threshold)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "threshold": args.threshold,
                "gt_dir": str(gt_dir),
                "n_images_gt_total": n_gt_total,
                "n_images_evaluated": n_images,
                "n_organs": n_organs,
                "methods": results,
            }, f, indent=2)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
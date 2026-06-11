#!/usr/bin/env python3
"""
rescue_report.py
----------------
Quantify how many GT instances are rescued by the final method over the baseline.

For each GT organ in each image:
  baseline_valid  = any baseline candidate with sam_score > sam_threshold
                    AND IoU(candidate, GT) > 0.5
  final_valid     = any final prediction with IoU(pred, GT) > 0.5
  rescued         = GT present AND NOT baseline_valid AND final_valid
  lost            = GT present AND baseline_valid AND NOT final_valid
  net             = rescued - lost

Both gates (SAM score AND IoU) are required for baseline_valid: a high-scoring
but misplaced prediction does not count as a detection.

Aggregates rescued/lost/net per organ and globally, and writes:
  <output>/rescue_report.json
  <output>/rescue_bars.png

With --dump-examples N --example-organ ORGAN: saves up to N rescued example
figures (image / GT mask / final prediction) for the specified organ under
<output>/examples/.

Usage
-----
    python scripts/rescue_report.py \\
        --baseline-dir results/baseline/Sunnybrook/unsup_baseline/masks \\
        --final-dir    results/embeddings/Sunnybrook/unsup_hdbscan/masks \\
        --gt-dir       data/processed/Sunnybrook/masks \\
        --output       results/rescue_sunnybrook

    # With example dumps for right-ventricle:
    python scripts/rescue_report.py ... \\
        --dump-examples 5 --example-organ rv_cavity \\
        --images-dir data/raw/Sunnybrook/images
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from project.evaluation.io import load_masks_from_dir
from project.evaluation.matching import parse_organ_name
from project.evaluation.metrics_paired import iou_score


_IOU_THRESHOLD = 0.5


def _load_sam_scores(mask_dir: Path) -> dict[str, float]:
    """
    Load per-prediction SAM scores from scores.json in mask_dir.

    Raises FileNotFoundError if scores.json is absent. The baseline
    (unsup_baseline / grid-prompt SAM2) always writes scores.json; its
    absence signals that the wrong directory was passed.

    Returns a dict mapping mask stem to the 'sam' field.
    """
    scores_path = mask_dir / "scores.json"
    if not scores_path.exists():
        raise FileNotFoundError(
            f"scores.json not found in baseline image dir: {mask_dir}\n"
            "The baseline experiment must include SAM scores."
        )
    with open(scores_path) as f:
        raw = json.load(f)
    return {
        Path(filename).stem: float(entry["sam"])
        for filename, entry in raw.items()
    }


def _baseline_valid(
    gt_mask: np.ndarray,
    baseline_masks: dict[str, np.ndarray],
    sam_scores: dict[str, float],
    sam_threshold: float,
) -> bool:
    """
    True if any baseline candidate passes both the SAM score and IoU gates.

    Requires sam_score > sam_threshold AND IoU(candidate, GT) > _IOU_THRESHOLD.
    A high-scoring but poorly-localised prediction does not count.
    """
    for name, mask in baseline_masks.items():
        if sam_scores.get(name, 0.0) > sam_threshold:
            if iou_score(mask, gt_mask) > _IOU_THRESHOLD:
                return True
    return False


def _final_valid(
    gt_mask: np.ndarray,
    final_masks: dict[str, np.ndarray],
) -> bool:
    """True if any final prediction has IoU(pred, GT) > _IOU_THRESHOLD."""
    for mask in final_masks.values():
        if iou_score(mask, gt_mask) > _IOU_THRESHOLD:
            return True
    return False


def compute_rescue_stats(
    baseline_dir: Path,
    final_dir: Path,
    gt_dir: Path,
    sam_threshold: float,
) -> tuple[dict, list[dict]]:
    """
    Compute per-image, per-GT-organ rescue/loss counts.

    Parameters
    ----------
    baseline_dir, final_dir, gt_dir :
        Masks directories with one subdirectory per image stem.
    sam_threshold :
        Minimum SAM score (exclusive) for a baseline candidate to be considered.

    Returns
    -------
    per_organ : dict[str, dict]
        {organ: {'rescued': int, 'lost': int}}
    rescued_examples : list[dict]
        One entry per rescued (image, gt_organ) pair, with keys:
        'image', 'organ', 'gt_name', 'final_pred_name' (stem of best IoU match).
    """
    gt_stems = {d.name for d in gt_dir.iterdir() if d.is_dir()}
    baseline_stems = {d.name for d in baseline_dir.iterdir() if d.is_dir()}
    final_stems = {d.name for d in final_dir.iterdir() if d.is_dir()}

    common = sorted(gt_stems & baseline_stems & final_stems)
    if not common:
        raise ValueError(
            "No common image stems found across GT, baseline, and final directories.\n"
            f"  GT:       {gt_dir}\n"
            f"  Baseline: {baseline_dir}\n"
            f"  Final:    {final_dir}"
        )

    gt_only = gt_stems - baseline_stems - final_stems
    if gt_only:
        print(f"  Warning: {len(gt_only)} GT images have no matching predictions, skipped")

    per_organ: dict[str, dict] = {}
    rescued_examples: list[dict] = []

    for stem in common:
        gt_masks = load_masks_from_dir(gt_dir / stem)
        baseline_masks = load_masks_from_dir(baseline_dir / stem)
        final_masks = load_masks_from_dir(final_dir / stem)
        sam_scores = _load_sam_scores(baseline_dir / stem)

        for gt_name, gt_mask in gt_masks.items():
            organ = parse_organ_name(gt_name)
            bv = _baseline_valid(gt_mask, baseline_masks, sam_scores, sam_threshold)
            fv = _final_valid(gt_mask, final_masks)

            counts = per_organ.setdefault(organ, {"rescued": 0, "lost": 0})

            if not bv and fv:
                counts["rescued"] += 1
                best_pred = max(
                    final_masks.keys(),
                    key=lambda n: iou_score(final_masks[n], gt_mask),
                )
                rescued_examples.append({
                    "image": stem,
                    "organ": organ,
                    "gt_name": gt_name,
                    "final_pred_name": best_pred,
                })
            elif bv and not fv:
                counts["lost"] += 1

    return per_organ, rescued_examples


def _print_table(per_organ: dict) -> None:
    """Print rescued / lost / net per organ to stdout."""
    organs = sorted(per_organ)
    total_rescued = sum(v["rescued"] for v in per_organ.values())
    total_lost = sum(v["lost"] for v in per_organ.values())
    total_net = total_rescued - total_lost

    print(f"\n  {'Organ':<22} {'Rescued':>9} {'Lost':>7} {'Net':>7}")
    print("  " + "-" * 48)
    for organ in organs:
        r = per_organ[organ]["rescued"]
        l = per_organ[organ]["lost"]
        n = r - l
        print(f"  {organ:<22} {r:>9} {l:>7} {n:>+7}")
    print("  " + "-" * 48)
    print(f"  {'TOTAL':<22} {total_rescued:>9} {total_lost:>7} {total_net:>+7}")


def _plot_bars(per_organ: dict, output_path: Path) -> None:
    """Save a grouped bar chart (rescued / lost / net) per organ + global."""
    organs = sorted(per_organ)
    all_labels = organs + ["TOTAL"]

    def _vals(key: str) -> list[int]:
        vals = [per_organ[o][key] for o in organs]
        vals.append(sum(vals))
        return vals

    rescued = _vals("rescued")
    lost    = _vals("lost")
    net     = [r - l for r, l in zip(rescued, lost)]

    x = np.arange(len(all_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, len(all_labels) * 1.4), 4))
    ax.bar(x - width, rescued, width, label="rescued", color="steelblue")
    ax.bar(x,         lost,    width, label="lost",    color="salmon")
    ax.bar(x + width, net,     width, label="net",     color="seagreen")

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=30, ha="right")
    ax.set_ylabel("Instance count")
    ax.set_title("Rescue report: instances rescued / lost vs. baseline")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _dump_examples(
    rescued_examples: list[dict],
    organ: str,
    n: int,
    final_dir: Path,
    gt_dir: Path,
    images_dir: Path | None,
    output_dir: Path,
) -> None:
    """
    Save up to N rescued examples for organ as composite PNG figures.

    Each figure shows [raw image (optional), GT mask, final prediction].
    Figures are saved as output_dir/example_{organ}_{i:02d}.png.

    If images_dir is None or the image file is not found, the raw image panel
    is omitted and only GT + final prediction are shown.
    """
    cases = [e for e in rescued_examples if e["organ"] == organ]
    if not cases:
        print(f"  No rescued examples found for organ '{organ}'")
        return

    cases = cases[:n]
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, case in enumerate(cases):
        stem = case["image"]
        gt_name = case["gt_name"]
        pred_name = case["final_pred_name"]

        gt_mask = load_masks_from_dir(gt_dir / stem)[gt_name]
        pred_mask = load_masks_from_dir(final_dir / stem)[pred_name]

        panels: list[np.ndarray] = []
        titles: list[str] = []

        if images_dir is not None:
            img_path = images_dir / f"{stem}.png"
            if img_path.exists():
                from PIL import Image as _Image
                panels.append(np.array(_Image.open(img_path).convert("L")))
                titles.append("image")
            else:
                print(f"  Warning: raw image not found: {img_path}")

        panels.append(gt_mask.astype(np.uint8) * 255)
        titles.append(f"GT: {gt_name}")
        panels.append(pred_mask.astype(np.uint8) * 255)
        titles.append(f"final: {pred_name}")

        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
        if len(panels) == 1:
            axes = [axes]
        for ax, img, title in zip(axes, panels, titles):
            ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"rescued  |  {stem}  |  {organ}", fontsize=10)
        fig.tight_layout()

        out_path = output_dir / f"example_{organ}_{i:02d}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved example: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rescue report: GT instances recovered by the final method over baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline-dir", required=True,
                        help="masks/ directory of the baseline experiment")
    parser.add_argument("--final-dir", required=True,
                        help="masks/ directory of the final experiment")
    parser.add_argument("--gt-dir", required=True,
                        help="GT masks directory (one subdir per image stem)")
    parser.add_argument("--sam-threshold", type=float, default=0.5,
                        help="Minimum SAM score (exclusive) for a baseline candidate "
                             "to count as a valid detection")
    parser.add_argument("--output", required=True,
                        help="Directory to save rescue_report.json and rescue_bars.png")
    parser.add_argument("--dump-examples", type=int, default=None, metavar="N",
                        help="Save N rescued example figures for the organ given "
                             "by --example-organ")
    parser.add_argument("--example-organ", default=None,
                        help="Organ name to use with --dump-examples")
    parser.add_argument("--images-dir", default=None,
                        help="Directory of raw images ({stem}.png) for example figures. "
                             "Optional: omitted panels are dropped silently.")
    args = parser.parse_args()

    if args.dump_examples is not None and args.example_organ is None:
        parser.error("--dump-examples requires --example-organ")

    baseline_dir = Path(args.baseline_dir)
    final_dir    = Path(args.final_dir)
    gt_dir       = Path(args.gt_dir)
    output_dir   = Path(args.output)

    for label, path in [("baseline-dir", baseline_dir),
                        ("final-dir",    final_dir),
                        ("gt-dir",       gt_dir)]:
        if not path.is_dir():
            raise FileNotFoundError(f"--{label} not found: {path}")

    print(f"Baseline: {baseline_dir}")
    print(f"Final:    {final_dir}")
    print(f"GT:       {gt_dir}")
    print(f"SAM threshold: {args.sam_threshold}  (IoU threshold: {_IOU_THRESHOLD})")

    per_organ, rescued_examples = compute_rescue_stats(
        baseline_dir, final_dir, gt_dir, args.sam_threshold
    )

    _print_table(per_organ)

    output_dir.mkdir(parents=True, exist_ok=True)

    total_rescued = sum(v["rescued"] for v in per_organ.values())
    total_lost    = sum(v["lost"]    for v in per_organ.values())
    report = {
        "per_organ": {
            organ: {**v, "net": v["rescued"] - v["lost"]}
            for organ, v in sorted(per_organ.items())
        },
        "global": {
            "rescued": total_rescued,
            "lost":    total_lost,
            "net":     total_rescued - total_lost,
        },
        "config": {
            "baseline_dir":  str(baseline_dir),
            "final_dir":     str(final_dir),
            "gt_dir":        str(gt_dir),
            "sam_threshold": args.sam_threshold,
            "iou_threshold": _IOU_THRESHOLD,
        },
    }

    json_path = output_dir / "rescue_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {json_path}")

    _plot_bars(per_organ, output_dir / "rescue_bars.png")

    if args.dump_examples is not None:
        _dump_examples(
            rescued_examples,
            organ=args.example_organ,
            n=args.dump_examples,
            final_dir=final_dir,
            gt_dir=gt_dir,
            images_dir=Path(args.images_dir) if args.images_dir else None,
            output_dir=output_dir / "examples",
        )


if __name__ == "__main__":
    main()

"""
compare_versions.py
-------------------
Cross-approach, cross-dataset comparison plots.

Scans  results_dir/{version}/{dataset}/{experiment}/summary.json  and produces
a lean set of figures designed to make the differences *between approaches*
immediately readable.

Mental model
------------
Each top-level directory ("version") is an APPROACH, not an orthogonal feature
tweak applied uniformly to every experiment:

    baseline            -> raw grid prompting (feature-independent reference)
    moments             -> unsupervised HDBSCAN on moment features (+ refine)
    embeddings          -> unsupervised HDBSCAN on SAM2 embeddings        (future)
    embeddings_moments  -> unsupervised HDBSCAN on hybrid features        (future)
    fs_propagation      -> few-shot SAM2-Video propagation with K refs

Each version may contain a different set of experiments, and the raw grid run is
often duplicated into several versions for reference. Every figure first
collapses each version to a single REPRESENTATIVE experiment (best F1@0.5,
ignoring runs that also exist in the baseline version). New versions are
auto-discovered, so adding `embeddings`/`embeddings_moments` needs no code edit.

Two threshold notions (kept distinct on purpose)
------------------------------------------------
* IoU thresholds for DETECTION metrics (recall@T / precision@T / f1@T / mAP) are
  *scanned*; they report how detection degrades as the overlap requirement
  tightens. plot_pr_f1_vs_iou visualises this scan and is the tool for choosing
  a per-dataset match cutoff (look for the precision/F1 knee).
* match_threshold is the IoU gate of the QUALITY matching: pairs below it are
  demoted to missing (Dice=0). Re-running reevaluate_all.py --match-threshold T
  recomputes Dice/IoU/HD95 under that gate. Picking T from the detection curves
  and feeding it to the quality re-eval is non-circular because the runner
  scans detection metrics independently of match_threshold.

Generated figures (core set, one per dataset unless noted)
----------------------------------------------------------
1.  approach_comparison_{dataset}.png
        Grouped bars over approaches: Precision@0.5, Recall@0.5, F1@0.5 (solid
        detection metrics) and Dice (solid = incl. missing, translucent
        extension up to detected-only ceiling). Delta-vs-baseline on F1.

2.  pr_f1_vs_iou_{dataset}.png
        P/R/F1 as a function of the IoU threshold, one panel per approach.
        Vertical guides at the candidate match cutoffs (annotated with P and
        F1). This is the figure used to choose the per-dataset match_threshold.

3.  recall_per_organ_{dataset}.png   (+ precision_per_organ if data has it)
        Grouped bars: x=organ, groups=approaches. Per-organ detection success.

4.  dice_gap_{dataset}.png
        Per approach, Dice detected-only vs incl. missing, gap annotated.

5.  map_summary_{dataset}.png
        Compact heatmap: approaches x [mAP@[.5:.95], AP@0.5, AP@0.75].

6.  feature_ablation_{dataset}.png
        The shared clustering experiment swept across the feature versions
        (moments / embeddings / hybrid), baseline grid drawn as a reference line.

7.  all_versions_summary.csv

Optional
--------
8.  pr_overlay_{dataset}.png            (--pr-overlay)
        COCO-style PR curves (sweep prediction SCORE) overlaid per approach.
9.  refinement_impact_{dataset}.png     (--refinement)

Usage
-----
    python compare_versions.py --results_dir results_leaf --output results_leaf/comparison

    # Candidate match cutoffs to mark on the vs-IoU plot:
    python compare_versions.py --cutoffs 0.5 0.75

    # Pin a specific experiment per version:
    python compare_versions.py --pin moments:unsup_hdbscan_refine
"""

import argparse
import csv
import json
import math
from itertools import cycle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from project.evaluation.display_names import get_display_name
from project.evaluation.plot_utils import (
    _compat_get_global,
    _sort_experiments,
)


# ---------------------------------------------------------------------------
# Presentation constants
# ---------------------------------------------------------------------------

DEFAULT_VERSION_ORDER = [
    "baseline",
    "moments",
    "embeddings",
    "embeddings_moments",
    "embeddings+moments",
    "emb_moments",
    "hybrid",
    "fs_propagation",
]

VERSION_LABELS = {
    "baseline":           "Baseline (grid)",
    "moments":            "Moments",
    "embeddings":         "Embeddings",
    "embeddings_moments": "Emb. + Moments",
    "embeddings+moments": "Emb. + Moments",
    "emb_moments":        "Emb. + Moments",
    "hybrid":             "Emb. + Moments",
    "fs_propagation":     "Few-shot prop.",
}

APPROACH_COLORS = {
    "baseline":           "#9AA0A6",
    "moments":            "#5B8DB8",
    "embeddings":         "#E8A838",
    "embeddings_moments": "#9C5BB8",
    "embeddings+moments": "#9C5BB8",
    "emb_moments":        "#9C5BB8",
    "hybrid":             "#9C5BB8",
    "fs_propagation":     "#6BBF6B",
}
_FALLBACK_PALETTE = ["#4FB0AE", "#C98BBA", "#B57F50", "#7E57C2", "#26A69A"]

DEFAULT_DATASET_LABELS = {
    "XRayNicoSent":       "JSRT",
    "SunnybrookNicoSent": "Sunnybrook",
    "Sunnybrook_test":    "Sunnybrook",
    "JSRT_test":          "JSRT",
}

# Headline metrics. Detection metrics are drawn as solid bars; the quality
# metric is drawn as a solid bar (incl. missing) with a translucent extension
# up to the detected-only ceiling, so the gap = cost of non-detection.
HEADLINE_DETECTION = [
    ("precision@0.5", "Precision@0.5", "#1A6FB0"),
    ("recall@0.5",    "Recall@0.5",    "#67A9CF"),
    ("f1@0.5",        "F1@0.5",        "#2E7D32"),
]
HEADLINE_QUALITY = [
    ("dice", "Dice", "#B2182B"),
]

PRIMARY_METRIC = "f1@0.5"

# Default IoU cutoffs to mark on the vs-IoU plot (candidate match thresholds).
DEFAULT_CUTOFFS = [0.5, 0.75]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(
    results_dir: Path,
    versions: list[str] | None = None,
    datasets: list[str] | None = None,
    experiments: list[str] | None = None,
) -> dict:
    """
    Scan results_dir/{version}/{dataset}/{experiment}/summary.json.

    Returns {version: {dataset: {experiment: summary_dict}}}. Filters are
    inclusion lists; None means "include all".
    """
    data: dict = {}

    if versions:
        version_dirs = [results_dir / v for v in versions]
    else:
        version_dirs = sorted(
            d for d in results_dir.iterdir()
            if d.is_dir()
            and not d.name.startswith(".")
            and d.name != "comparison"
        )

    for version_dir in version_dirs:
        if not version_dir.is_dir():
            print(f"  [SKIP] Not a directory: {version_dir}")
            continue

        version = version_dir.name
        data[version] = {}

        for dataset_dir in sorted(version_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            if datasets and dataset not in datasets:
                continue
            data[version][dataset] = {}

            for exp_dir in sorted(dataset_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                if experiments and exp_dir.name not in experiments:
                    continue
                summary_path = exp_dir / "summary.json"
                if not summary_path.exists():
                    continue
                with open(summary_path) as f:
                    data[version][dataset][exp_dir.name] = json.load(f)

    for version in data:
        for dataset in data[version]:
            n = len(data[version][dataset])
            print(f"  {version}/{dataset}: {n} experiments")

    return data


# ---------------------------------------------------------------------------
# Small accessors
# ---------------------------------------------------------------------------

def _g(summary: dict, key: str, legacy: str | None = None):
    """Global metric with legacy-flat-key fallback. None if absent."""
    val = _compat_get_global(summary, key)
    if val is None and legacy:
        val = summary.get("global", {}).get(legacy)
    return val


def _organ_stat(summary: dict, organ: str, key: str, legacy: str | None = None):
    """Per-organ metric with optional legacy fallback. None if absent."""
    od = summary.get("per_organ", {}).get(organ, {})
    val = od.get(key)
    if val is None and legacy:
        val = od.get(legacy)
    return val


def _nan(val) -> float:
    return float("nan") if val is None else float(val)


def _quality_pair(summary: dict, base: str):
    """Return (with_missing, detected_only) for a quality metric base name."""
    legacy = "dice_mean" if base == "dice" else None
    wm = _g(summary, f"{base}_mean_with_missing", legacy)
    do = _g(summary, f"{base}_mean_detected_only")
    return wm, do


def _get_pr_curve(summary: dict, iou: float) -> dict | None:
    curves = summary.get("pr_curve_per_threshold") or {}
    for key in (f"{iou:.2f}", f"{iou:.1f}", str(iou)):
        if key in curves and curves[key]:
            return curves[key]
    return None


def _interp_at(xs, ys, x0):
    """Linear interpolation of y at x0 over the valid (non-NaN) points."""
    pts = [(x, y) for x, y in zip(xs, ys)
           if y is not None and not (isinstance(y, float) and math.isnan(y))]
    if len(pts) < 2:
        return None
    xa = [p[0] for p in pts]
    ya = [p[1] for p in pts]
    if x0 < xa[0] or x0 > xa[-1]:
        return None
    return float(np.interp(x0, xa, ya))


# ---------------------------------------------------------------------------
# Approach resolution
# ---------------------------------------------------------------------------

def _order_versions(versions: list[str]) -> list[str]:
    rank = {name: i for i, name in enumerate(DEFAULT_VERSION_ORDER)}
    return sorted(versions, key=lambda v: (rank.get(v, 10_000), v))


def _exp_score(summary: dict) -> tuple[float, float]:
    f1 = _g(summary, "f1@0.5")
    dice = _g(summary, "dice_mean_detected_only", "dice_mean")
    return (f1 if f1 is not None else -1.0,
            dice if dice is not None else -1.0)


def representative_experiment(
    data: dict, version: str, dataset: str,
    baseline_version: str, pins: dict[str, str],
) -> str | None:
    """
    Pick the experiment that best represents `version` on `dataset`:
      1. honour an explicit pin if present;
      2. drop experiments that also exist in the baseline version (duplicated
         feature-independent reference runs) when feature-dependent ones remain;
      3. take the best F1@0.5.
    """
    exps = dict(data.get(version, {}).get(dataset, {}))
    if not exps:
        return None

    pinned = pins.get(version)
    if pinned and pinned in exps:
        return pinned

    if version != baseline_version:
        base_exps = set(data.get(baseline_version, {}).get(dataset, {}).keys())
        feature_only = {e: s for e, s in exps.items() if e not in base_exps}
        if feature_only:
            exps = feature_only

    return max(exps, key=lambda e: _exp_score(exps[e]))


def _approach_color(version: str, fallback: cycle) -> str:
    return APPROACH_COLORS.get(version, next(fallback))


def build_approaches(
    data: dict, dataset: str, version_order: list[str],
    baseline_version: str, pins: dict[str, str],
) -> list[dict]:
    fallback = cycle(_FALLBACK_PALETTE)
    approaches: list[dict] = []
    for version in version_order:
        exp = representative_experiment(
            data, version, dataset, baseline_version, pins
        )
        if exp is None:
            continue
        approaches.append({
            "version":    version,
            "experiment": exp,
            "label":      VERSION_LABELS.get(version, version),
            "color":      _approach_color(version, fallback),
            "summary":    data[version][dataset][exp],
        })
    return approaches


def _print_selection(data, version_order, datasets, baseline_version, pins):
    print("\nRepresentative experiment per (version, dataset):")
    for version in version_order:
        for dataset in datasets:
            exp = representative_experiment(
                data, version, dataset, baseline_version, pins
            )
            if exp is not None:
                print(f"  {version:<20} {dataset:<18} -> {exp}")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _bar_value_label(ax, x, height, text, color="#222", dy=0.02, fs=7):
    if height is None or (isinstance(height, float) and math.isnan(height)):
        return
    ax.text(x, height + dy, text, ha="center", va="bottom",
            fontsize=fs, color=color)


def _dataset_label(dataset: str, labels: dict[str, str]) -> str:
    return labels.get(dataset, dataset)


def _all_datasets(data: dict) -> list[str]:
    seen: set[str] = set()
    for v in data.values():
        seen.update(v.keys())
    return sorted(seen)


# ---------------------------------------------------------------------------
# Figure 1: approach comparison (headline) with detection + Dice gap
# ---------------------------------------------------------------------------

def plot_approach_comparison(data, output_dir, version_order, baseline_version,
                             pins, dataset_labels):
    """
    Grouped bars over approaches. Detection metrics (Precision/Recall/F1 @0.5)
    are solid bars. The quality metric (Dice) is drawn as a solid bar at the
    incl.-missing value with a translucent extension up to the detected-only
    ceiling; the gap is the cost of non-detection. The primary metric is
    annotated with its delta against the baseline approach.
    """
    metrics = (
        [("det", k, lbl, col) for k, lbl, col in HEADLINE_DETECTION]
        + [("qual", b, lbl, col) for b, lbl, col in HEADLINE_QUALITY]
    )

    for dataset in _all_datasets(data):
        approaches = build_approaches(
            data, dataset, version_order, baseline_version, pins
        )
        if not approaches:
            continue

        n_app = len(approaches)
        n_met = len(metrics)
        x = np.arange(n_app)
        width = 0.8 / n_met

        fig, ax = plt.subplots(figsize=(max(8, n_app * 2.0), 5.6))

        base = next((a for a in approaches
                     if a["version"] == baseline_version), None)
        base_primary = _g(base["summary"], PRIMARY_METRIC) if base else None

        for mi, (kind, key_or_base, label, color) in enumerate(metrics):
            offset = (mi - n_met / 2 + 0.5) * width
            for ai, a in enumerate(approaches):
                bx = ai + offset
                s = a["summary"]

                if kind == "det":
                    v = _g(s, key_or_base)
                    if v is None:
                        continue
                    ax.bar(bx, v, width, color=color, alpha=0.9,
                           edgecolor="white", linewidth=0.5,
                           label=label if ai == 0 else None)
                    if key_or_base == PRIMARY_METRIC and base_primary is not None \
                            and a["version"] != baseline_version:
                        delta = v - base_primary
                        sign = "+" if delta >= 0 else ""
                        _bar_value_label(
                            ax, bx, v, f"{v:.3f}\n({sign}{delta:.3f})",
                            color="#2E7D32" if delta >= 0 else "#C62828", fs=7,
                        )
                    else:
                        _bar_value_label(ax, bx, v, f"{v:.3f}", fs=7)

                else:  # quality metric: solid (with missing) + translucent gap
                    wm, do = _quality_pair(s, key_or_base)
                    if wm is None and do is None:
                        continue
                    base_h = wm if wm is not None else do
                    ax.bar(bx, base_h, width, color=color, alpha=0.92,
                           edgecolor="white", linewidth=0.5,
                           label=label if ai == 0 else None)
                    _bar_value_label(ax, bx, base_h, f"{base_h:.3f}", fs=7)
                    if do is not None and wm is not None and do > wm + 1e-6:
                        ax.bar(bx, do - wm, width, bottom=wm, color=color,
                               alpha=0.30, edgecolor="none")
                        ax.text(bx, do + 0.015,
                                f"{do:.3f}  (gap {do - wm:.3f})",
                                ha="center", va="bottom", fontsize=6.5,
                                color="#555")

        ax.set_xticks(x)
        ax.set_xticklabels([a["label"] for a in approaches],
                           fontsize=10, rotation=15, ha="right")
        ax.set_ylim(0, 1.22)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(
            f"Approach comparison — {_dataset_label(dataset, dataset_labels)}\n"
            "(Dice: solid = incl. missing, translucent = detected-only ceiling; "
            "delta vs baseline on F1)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=9, loc="lower right", ncol=2)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        out = output_dir / f"approach_comparison_{dataset}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2: P/R/F1 vs IoU threshold (match-cutoff selection)
# ---------------------------------------------------------------------------

def plot_pr_f1_vs_iou(data, output_dir, version_order, baseline_version, pins,
                      dataset_labels, cutoffs):
    """
    For each dataset, one panel per approach plotting Precision/Recall/F1 as a
    function of the IoU detection threshold. Vertical guides mark the candidate
    match cutoffs and annotate Precision/F1 there, so the panel doubles as the
    tool to choose the per-dataset match_threshold (look for the precision/F1
    knee). Uses summary['iou_thresholds'] and global recall@/precision@/f1@.
    """
    line_spec = [
        ("precision", "Precision", "#1A6FB0", "o"),
        ("recall",    "Recall",    "#2E7D32", "s"),
        ("f1",        "F1",        "#C62828", "^"),
    ]

    for dataset in _all_datasets(data):
        approaches = build_approaches(
            data, dataset, version_order, baseline_version, pins
        )
        if not approaches:
            continue

        n = len(approaches)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 4.2, nrows * 3.8),
            squeeze=False, constrained_layout=True,
        )

        for ai, a in enumerate(approaches):
            ax = axes[ai // ncols][ai % ncols]
            s = a["summary"]
            iou_thrs = sorted(float(t) for t in s.get("iou_thresholds", []))
            if not iou_thrs:
                ax.text(0.5, 0.5, "no thresholds", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#aaa")
                ax.set_title(a["label"], fontsize=10, fontweight="bold")
                continue

            series = {}
            for prefix, label, color, marker in line_spec:
                ys = [_nan(_g(s, f"{prefix}@{t}")) for t in iou_thrs]
                series[prefix] = ys
                ax.plot(iou_thrs, ys, marker=marker, color=color,
                        linewidth=1.6, markersize=4, label=label)

            # Candidate match cutoffs.
            for c in cutoffs:
                if iou_thrs[0] <= c <= iou_thrs[-1]:
                    ax.axvline(c, color="#888", linestyle="--",
                               linewidth=1.0, alpha=0.7)
                    p_at = _interp_at(iou_thrs, series["precision"], c)
                    f_at = _interp_at(iou_thrs, series["f1"], c)
                    txt = []
                    if p_at is not None:
                        txt.append(f"P={p_at:.2f}")
                    if f_at is not None:
                        txt.append(f"F1={f_at:.2f}")
                    if txt:
                        ax.text(c, 1.02, "  ".join(txt), rotation=0,
                                ha="center", va="bottom", fontsize=6.5,
                                color="#555")

            ax.set_xlim(iou_thrs[0] - 0.03, iou_thrs[-1] + 0.03)
            ax.set_ylim(-0.05, 1.12)
            ax.set_xlabel("IoU threshold", fontsize=9)
            ax.set_ylabel("Score", fontsize=9)
            ax.set_title(a["label"], fontsize=10, fontweight="bold")
            ax.legend(fontsize=7, loc="lower left")
            ax.grid(alpha=0.3)

        for k in range(n, nrows * ncols):
            axes[k // ncols][k % ncols].set_visible(False)

        fig.suptitle(
            f"P/R/F1 vs IoU threshold — "
            f"{_dataset_label(dataset, dataset_labels)}\n"
            "(dashed = candidate match cutoffs)",
            fontsize=13, fontweight="bold",
        )
        out = output_dir / f"pr_f1_vs_iou_{dataset}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3: per-organ metric across approaches (recall always; precision if any)
# ---------------------------------------------------------------------------

ORGAN_COLORS = {
    "heart":       "#D45B5B",
    "left_lung":   "#5B8DB8",
    "right_lung":  "#6BBF6B",
    "lung":        "#5BA8D4",
    "lv_cavity":   "#9C27B0",
    "myocardium":  "#FF9800",
    "rv_cavity":   "#26A69A",
}


def plot_per_organ(data, output_dir, version_order, baseline_version, pins,
                   dataset_labels, metric_key, metric_label, file_stem):
    """
    Grouped bars per dataset: x=organ, one bar per approach, height = the
    requested per-organ metric (e.g. recall@0.5). Skips the figure entirely for
    a dataset if no organ on any approach has the metric (so a precision-per-
    organ call is a no-op until coverage.py stores per-organ precision).
    """
    for dataset in _all_datasets(data):
        approaches = build_approaches(
            data, dataset, version_order, baseline_version, pins
        )
        if not approaches:
            continue

        organs: set[str] = set()
        for a in approaches:
            for organ, stats in a["summary"].get("per_organ", {}).items():
                if stats.get(metric_key) is not None:
                    organs.add(organ)
        organs = sorted(organs)
        if not organs:
            print(f"  [SKIP] {file_stem} ({dataset}): no per-organ "
                  f"'{metric_key}' present.")
            continue

        n_app = len(approaches)
        x = np.arange(len(organs))
        width = 0.8 / n_app

        fig, ax = plt.subplots(figsize=(max(7, len(organs) * 1.6), 5.2))

        for ai, a in enumerate(approaches):
            vals = [_nan(_organ_stat(a["summary"], organ, metric_key))
                    for organ in organs]
            offset = (ai - n_app / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=a["label"],
                          color=a["color"], alpha=0.9,
                          edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if not math.isnan(v):
                    _bar_value_label(
                        ax, bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                        fs=6,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([o.replace("_", " ").title() for o in organs],
                           fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(
            f"Per-organ {metric_label} — "
            f"{_dataset_label(dataset, dataset_labels)}",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        out = output_dir / f"{file_stem}_{dataset}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 4: Dice gap per approach
# ---------------------------------------------------------------------------

def plot_dice_gap(data, output_dir, version_order, baseline_version, pins,
                  dataset_labels):
    """Per dataset, two bars per approach: Dice detected-only vs incl. missing,
    the gap annotated (global-Dice cost of undetected organs)."""
    for dataset in _all_datasets(data):
        approaches = build_approaches(
            data, dataset, version_order, baseline_version, pins
        )
        if not approaches:
            continue

        n_app = len(approaches)
        x = np.arange(n_app)
        width = 0.38

        wm_list, do_list = [], []
        for a in approaches:
            wm, do = _quality_pair(a["summary"], "dice")
            wm_list.append(_nan(wm))
            do_list.append(_nan(do))

        fig, ax = plt.subplots(figsize=(max(7, n_app * 1.7), 5.2))
        ax.bar(x - width / 2, do_list, width, label="Dice (detected only)",
               color="#67A9CF", alpha=0.9, edgecolor="white", linewidth=0.5)
        ax.bar(x + width / 2, wm_list, width, label="Dice (incl. missing)",
               color="#2166AC", alpha=0.9, edgecolor="white", linewidth=0.5)

        for xi, (d, w) in enumerate(zip(do_list, wm_list)):
            if not math.isnan(d):
                _bar_value_label(ax, xi - width / 2, d, f"{d:.3f}", fs=7)
            if not math.isnan(w):
                _bar_value_label(ax, xi + width / 2, w, f"{w:.3f}", fs=7)
            if not (math.isnan(d) or math.isnan(w)):
                gap = d - w
                ax.text(xi, max(d, w) + 0.08, f"gap {gap:.3f}",
                        ha="center", fontsize=8, fontweight="bold",
                        color="#C62828" if gap > 0.01 else "#777")

        ax.set_xticks(x)
        ax.set_xticklabels([a["label"] for a in approaches],
                           fontsize=10, rotation=15, ha="right")
        ax.set_ylim(0, 1.22)
        ax.set_ylabel("Dice", fontsize=11)
        ax.set_title(
            f"Detection cost on Dice — "
            f"{_dataset_label(dataset, dataset_labels)}\n"
            "(gap = drop caused by undetected organs)",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        out = output_dir / f"dice_gap_{dataset}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 5: mAP summary heatmap
# ---------------------------------------------------------------------------

def _annotate_cell(ax, col, row, value, vmin, vmax):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        ax.text(col, row, "—", ha="center", va="center", fontsize=8, color="#bbb")
        return
    norm = (value - vmin) / (vmax - vmin + 1e-12)
    color = "white" if norm < 0.35 or norm > 0.85 else "#222"
    ax.text(col, row, f"{value:.3f}", ha="center", va="center",
            fontsize=9, fontweight="bold", color=color)


def plot_map_summary(data, output_dir, version_order, baseline_version, pins,
                     dataset_labels):
    col_keys = ["map", "map_50", "map_75"]
    col_labels = ["mAP@[.5:.95]", "AP@0.5", "AP@0.75"]

    for dataset in _all_datasets(data):
        approaches = build_approaches(
            data, dataset, version_order, baseline_version, pins
        )
        if not approaches:
            continue

        n_app = len(approaches)
        n_col = len(col_keys)
        matrix = np.full((n_app, n_col), np.nan)
        for ai, a in enumerate(approaches):
            for ci, key in enumerate(col_keys):
                val = _g(a["summary"], key)
                if val is not None:
                    matrix[ai, ci] = val

        fig, ax = plt.subplots(
            figsize=(max(4.5, n_col * 2.0), max(3.0, n_app * 0.7 + 1.2))
        )
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)
        for ai in range(n_app):
            for ci in range(n_col):
                _annotate_cell(ax, ci, ai, matrix[ai, ci], 0.0, 1.0)

        ax.set_xticks(range(n_col))
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(n_app))
        ax.set_yticklabels([a["label"] for a in approaches], fontsize=9)
        ax.set_title(f"mAP summary — {_dataset_label(dataset, dataset_labels)}",
                     fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8, label="AP")
        fig.tight_layout()

        out = output_dir / f"map_summary_{dataset}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 6: feature-space ablation
# ---------------------------------------------------------------------------

def detect_ablation_experiment(data, datasets, baseline_version, override):
    if override:
        return override

    base_exps: set[str] = set()
    for dataset in datasets:
        base_exps.update(data.get(baseline_version, {}).get(dataset, {}).keys())

    version_count: dict[str, int] = {}
    for version, ds_map in data.items():
        if version == baseline_version:
            continue
        present: set[str] = set()
        for dataset in datasets:
            present.update(ds_map.get(dataset, {}).keys())
        for exp in present - base_exps:
            version_count[exp] = version_count.get(exp, 0) + 1

    candidates = {e: c for e, c in version_count.items() if c >= 2}
    if not candidates:
        return None
    ranked = _sort_experiments(list(candidates))
    return max(ranked, key=lambda e: candidates[e])


def plot_feature_ablation(data, output_dir, version_order, baseline_version,
                          pins, dataset_labels, ablation_experiment):
    if ablation_experiment is None:
        print("  [SKIP] feature ablation: no shared feature-dependent "
              "experiment across >=2 versions.")
        return

    metrics = [
        ("f1@0.5",                  "F1@0.5",          None, "#2166AC"),
        ("dice_mean_detected_only", "Dice (detected)", "dice_mean", "#67A9CF"),
    ]

    for dataset in _all_datasets(data):
        versions = [
            v for v in version_order
            if v != baseline_version
            and ablation_experiment in data.get(v, {}).get(dataset, {})
        ]
        if len(versions) < 2:
            continue

        n_ver = len(versions)
        n_met = len(metrics)
        x = np.arange(n_ver)
        width = 0.8 / n_met

        fig, ax = plt.subplots(figsize=(max(6, n_ver * 1.8), 5.2))

        base_exp = representative_experiment(
            data, baseline_version, dataset, baseline_version, pins
        )
        base_summary = (
            data.get(baseline_version, {}).get(dataset, {}).get(base_exp)
            if base_exp else None
        )

        for mi, (key, label, legacy, color) in enumerate(metrics):
            vals = [
                _nan(_g(data[v][dataset][ablation_experiment], key, legacy))
                for v in versions
            ]
            offset = (mi - n_met / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=label,
                          color=color, alpha=0.9,
                          edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if not math.isnan(v):
                    _bar_value_label(
                        ax, bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}",
                        fs=7,
                    )
            if base_summary is not None:
                bval = _g(base_summary, key, legacy)
                if bval is not None:
                    ax.axhline(bval, color=color, linestyle="--",
                               linewidth=1.1, alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([VERSION_LABELS.get(v, v) for v in versions],
                           fontsize=10, rotation=15, ha="right")
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(
            f"Feature-space ablation — "
            f"{_dataset_label(dataset, dataset_labels)}\n"
            f"(experiment: {get_display_name(ablation_experiment)}; "
            "dashed = baseline grid)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        out = output_dir / f"feature_ablation_{dataset}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Optional Figure: PR-curve overlay (COCO-style, sweeps prediction SCORE)
# ---------------------------------------------------------------------------

def plot_pr_overlay(data, output_dir, version_order, baseline_version, pins,
                    dataset_labels, iou_values=(0.5, 0.75)):
    """
    COCO-style PR curves overlaid per approach (one panel per IoU threshold).
    These sweep the prediction CONFIDENCE SCORE, not the IoU; AP is the area.
    When predictions lack real scores (low pct_real_scores) the curve degenerates
    toward a single operating point, so read this together with that field.
    """
    ap_key = {0.5: "map_50", 0.75: "map_75"}

    for dataset in _all_datasets(data):
        approaches = build_approaches(
            data, dataset, version_order, baseline_version, pins
        )
        if not approaches:
            continue

        n_panels = len(iou_values)
        fig, axes = plt.subplots(
            1, n_panels, figsize=(n_panels * 4.4, 4.6), squeeze=False
        )
        axes = axes[0]

        for pi, iou in enumerate(iou_values):
            ax = axes[pi]
            any_drawn = False
            for a in approaches:
                s = a["summary"]
                color = a["color"]
                ap = _g(s, ap_key.get(iou, ""))
                ap_txt = f" (AP={ap:.2f})" if ap is not None else ""
                label = a["label"] + ap_txt
                curve = _get_pr_curve(s, iou)
                if curve and curve.get("recall"):
                    r, p = curve["recall"], curve["precision"]
                    ax.plot(r, p, "-", color=color, linewidth=1.8,
                            alpha=0.95, label=label)
                    ax.plot(r[-1], p[-1], "o", color=color, markersize=5, zorder=6)
                    any_drawn = True
                else:
                    r = _g(s, f"recall@{iou}")
                    p = _g(s, f"precision@{iou}")
                    if r is not None and p is not None:
                        ax.scatter([r], [p], color=color, s=55, marker="o",
                                   zorder=6, label=label + " (point)")
                        any_drawn = True

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Recall", fontsize=10)
            if pi == 0:
                ax.set_ylabel("Precision", fontsize=10)
            ax.set_title(f"IoU ≥ {iou:g}", fontsize=11, fontweight="bold")
            ax.grid(alpha=0.25)
            if any_drawn:
                ax.legend(fontsize=8, loc="lower left")
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="#aaa")

        fig.suptitle(
            f"PR curves by approach (score sweep) — "
            f"{_dataset_label(dataset, dataset_labels)}",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()
        out = output_dir / f"pr_overlay_{dataset}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Optional Figure: refinement impact
# ---------------------------------------------------------------------------

def plot_refinement_impact(data, output_dir, dataset_labels):
    metrics = [
        ("f1@0.5",                 "F1@0.5",            None),
        ("dice_mean_with_missing", "Dice (incl. miss)", "dice_mean"),
    ]
    for dataset in _all_datasets(data):
        triples: list[tuple[str, str, str]] = []
        for version, ds_map in data.items():
            exps = ds_map.get(dataset, {})
            for base in sorted(exps):
                refine = base + "_refine"
                if refine in exps:
                    triples.append((version, base, refine))
        if not triples:
            continue

        x = np.arange(len(triples))
        width = 0.38
        fig, axes = plt.subplots(
            1, len(metrics),
            figsize=(len(metrics) * max(4.5, len(triples) * 1.4), 5.0),
            squeeze=False,
        )
        axes = axes[0]

        for mi, (key, label, legacy) in enumerate(metrics):
            ax = axes[mi]
            before = [_nan(_g(data[v][dataset][b], key, legacy))
                      for v, b, _ in triples]
            after = [_nan(_g(data[v][dataset][r], key, legacy))
                     for v, _, r in triples]
            ax.bar(x - width / 2, before, width, label="Without refine",
                   color="#5B8DB8", alpha=0.9, edgecolor="white", linewidth=0.5)
            ax.bar(x + width / 2, after, width, label="With refine",
                   color="#E8A838", alpha=0.9, edgecolor="white", linewidth=0.5)
            for xi, (b, a) in enumerate(zip(before, after)):
                if math.isnan(b) or math.isnan(a):
                    continue
                delta = a - b
                sign = "+" if delta >= 0 else ""
                ax.text(xi, max(b, a) + 0.03, f"{sign}{delta:.3f}",
                        ha="center", fontsize=8, fontweight="bold",
                        color="#2E7D32" if delta >= 0 else "#C62828")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{v}\n{get_display_name(b)}"
                                for v, b, _ in triples],
                               fontsize=7, rotation=20, ha="right")
            ax.set_ylim(0, 1.18)
            ax.set_ylabel(label, fontsize=10)
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"Refinement impact — {_dataset_label(dataset, dataset_labels)}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        out = output_dir / f"refinement_impact_{dataset}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Flat CSV export
# ---------------------------------------------------------------------------

def save_full_csv(data, output_dir, baseline_version, version_order, pins):
    rows: list[dict] = []
    rep: dict[tuple[str, str], str] = {}
    for version in version_order:
        for dataset in data.get(version, {}):
            exp = representative_experiment(
                data, version, dataset, baseline_version, pins
            )
            if exp is not None:
                rep[(version, dataset)] = exp

    thresholds_seen: set[float] = set()
    for v in data.values():
        for ds in v.values():
            for s in ds.values():
                thresholds_seen.update(float(t) for t in s.get("iou_thresholds", []))
    thresholds = sorted(thresholds_seen)

    for version in data:
        for dataset in data[version]:
            for exp, summary in data[version][dataset].items():
                g = summary.get("global", {})
                row: dict = {
                    "version":          version,
                    "dataset":          dataset,
                    "experiment":       exp,
                    "is_representative": rep.get((version, dataset)) == exp,
                    "n_images":         summary.get("n_images", 0),
                    "matching":         summary.get("matching", ""),
                    "match_threshold":  summary.get("match_threshold"),
                    "dice_detected_only":
                        g.get("dice_mean_detected_only") or g.get("dice_mean"),
                    "dice_with_missing":
                        g.get("dice_mean_with_missing") or g.get("dice_mean"),
                    "iou_detected_only": g.get("iou_mean_detected_only"),
                    "iou_with_missing":  g.get("iou_mean_with_missing"),
                    "map":    g.get("map"),
                    "map_50": g.get("map_50"),
                    "map_75": g.get("map_75"),
                    "pct_real_scores": g.get("pct_real_scores"),
                }
                for thr in thresholds:
                    row[f"recall@{thr}"]    = g.get(f"recall@{thr}")
                    row[f"precision@{thr}"] = g.get(f"precision@{thr}")
                    row[f"f1@{thr}"]        = g.get(f"f1@{thr}")
                for organ, stats in summary.get("per_organ", {}).items():
                    row[f"{organ}_recall@0.5"] = stats.get("recall@0.5")
                    row[f"{organ}_precision@0.5"] = stats.get("precision@0.5")
                    row[f"{organ}_dice_detected_only"] = \
                        stats.get("dice_mean_detected_only")
                rows.append(row)

    if not rows:
        return

    base_fields = [
        "version", "dataset", "experiment", "is_representative",
        "n_images", "matching", "match_threshold",
        "dice_detected_only", "dice_with_missing",
        "iou_detected_only", "iou_with_missing",
        "map", "map_50", "map_75", "pct_real_scores",
    ]
    for thr in thresholds:
        base_fields += [f"recall@{thr}", f"precision@{thr}", f"f1@{thr}"]
    extra = sorted({k for row in rows for k in row if k not in base_fields})
    fieldnames = base_fields + extra

    out = output_dir / "all_versions_summary.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_pins(items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items or []:
        if ":" not in item:
            raise ValueError(f"Expected VERSION:EXPERIMENT, got: {item!r}")
        version, exp = item.split(":", 1)
        result[version.strip()] = exp.strip()
    return result


def _parse_dataset_labels(items: list[str]) -> dict[str, str]:
    result = dict(DEFAULT_DATASET_LABELS)
    for item in items or []:
        if ":" not in item:
            raise ValueError(f"Expected DATASET:LABEL, got: {item!r}")
        ds, label = item.split(":", 1)
        result[ds.strip()] = label.strip()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare pipeline approaches across versions and datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_dir", default="results_leaf",
                        help="Root dir (version/dataset/experiment/summary.json).")
    parser.add_argument("--versions", nargs="*",
                        help="Versions to include and order (default: all).")
    parser.add_argument("--datasets", nargs="*",
                        help="Datasets to include (default: all).")
    parser.add_argument("--experiments", nargs="*",
                        help="Restrict loaded experiment names (default: all).")
    parser.add_argument("--output", default=None,
                        help="Output dir (default: <results_dir>/comparison).")
    parser.add_argument("--baseline-version", default="baseline",
                        help="Version treated as feature-independent reference.")
    parser.add_argument("--pin", nargs="*", metavar="VERSION:EXPERIMENT",
                        default=[], help="Pin representative experiment per version.")
    parser.add_argument("--cutoffs", nargs="*", type=float, default=DEFAULT_CUTOFFS,
                        help="Candidate match cutoffs marked on the vs-IoU plot.")
    parser.add_argument("--ablation-experiment", default=None,
                        help="Experiment held fixed in the feature ablation "
                             "(default: auto-detected).")
    parser.add_argument("--pr-overlay", action="store_true",
                        help="Also generate the COCO-style PR-curve overlay "
                             "(score sweep).")
    parser.add_argument("--refinement", action="store_true",
                        help="Also generate the refinement before/after figure.")
    parser.add_argument("--dataset-labels", nargs="*", metavar="DATASET:LABEL",
                        default=[], help="DATASET:LABEL overrides for titles.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output) if args.output else results_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    pins = _parse_pins(args.pin)
    dataset_labels = _parse_dataset_labels(args.dataset_labels)

    print(f"Loading results from {results_dir} ...")
    data = load_data(results_dir, args.versions, args.datasets, args.experiments)

    total = sum(len(ds) for v in data.values() for ds in v.values())
    if total == 0:
        print("No experiment summaries found. Nothing to plot.")
        return
    print(f"\nVersions: {list(data.keys())}")
    print(f"Total experiment results: {total}")

    version_order = (list(args.versions) if args.versions
                     else _order_versions(list(data.keys())))
    datasets = args.datasets if args.datasets else _all_datasets(data)

    _print_selection(data, version_order, datasets, args.baseline_version, pins)

    print("\n[1/6] Approach comparison (headline + Dice gap):")
    plot_approach_comparison(data, output_dir, version_order,
                             args.baseline_version, pins, dataset_labels)

    print("\n[2/6] P/R/F1 vs IoU threshold (match-cutoff selection):")
    plot_pr_f1_vs_iou(data, output_dir, version_order, args.baseline_version,
                      pins, dataset_labels, args.cutoffs)

    print("\n[3/6] Per-organ recall (and precision if available):")
    plot_per_organ(data, output_dir, version_order, args.baseline_version, pins,
                   dataset_labels, "recall@0.5", "Recall@0.5",
                   "recall_per_organ")
    plot_per_organ(data, output_dir, version_order, args.baseline_version, pins,
                   dataset_labels, "precision@0.5", "Precision@0.5",
                   "precision_per_organ")

    print("\n[4/6] Dice gap (detected vs incl. missing):")
    plot_dice_gap(data, output_dir, version_order, args.baseline_version, pins,
                  dataset_labels)

    print("\n[5/6] mAP summary heatmap:")
    plot_map_summary(data, output_dir, version_order, args.baseline_version,
                     pins, dataset_labels)

    print("\n[6/6] Feature-space ablation:")
    ablation_exp = detect_ablation_experiment(
        data, datasets, args.baseline_version, args.ablation_experiment
    )
    if ablation_exp:
        print(f"  Ablation experiment: {ablation_exp}")
    plot_feature_ablation(data, output_dir, version_order, args.baseline_version,
                          pins, dataset_labels, ablation_exp)

    if args.pr_overlay:
        print("\n[Optional] PR-curve overlay (score sweep):")
        plot_pr_overlay(data, output_dir, version_order, args.baseline_version,
                        pins, dataset_labels)

    if args.refinement:
        print("\n[Optional] Refinement impact:")
        plot_refinement_impact(data, output_dir, dataset_labels)

    print("\n[CSV] Flat export:")
    save_full_csv(data, output_dir, args.baseline_version, version_order, pins)

    print(f"\nAll comparison artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
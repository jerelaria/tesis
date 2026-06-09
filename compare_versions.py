"""
compare_versions.py
-------------------
Cross-approach, cross-dataset comparison plots for the cosegmentation thesis.

Scans  results_dir/{version}/{dataset}/{experiment}/summary.json  and produces
a lean set of figures designed to make the differences *between approaches*
immediately readable.

Data model: leaf = (version, experiment)
----------------------------------------
The unit of comparison is a LEAF: a single (version, experiment) pair, i.e. a
concrete result directory. Versions are NOT collapsed to one representative
experiment, so an approach that legitimately has several configs (e.g.
fs_propagation swept over K references) contributes several columns.

Each top-level directory ("version") is an APPROACH / feature space:

    baseline            -> raw grid prompting (feature-independent reference)
    moments             -> unsupervised HDBSCAN on moment features
    embeddings          -> unsupervised HDBSCAN on SAM2 embeddings
    embeddings_moments  -> unsupervised HDBSCAN on hybrid features
    fs_propagation      -> few-shot SAM2-Video propagation with K refs

Leaves are ordered into three GROUPS for every figure:

    baseline  ->  unsupervised approaches  ->  few-shot variants

New versions / configs are auto-discovered; adding a feature space needs no
code edit.

Two threshold notions (kept distinct on purpose)
------------------------------------------------
* IoU thresholds for DETECTION metrics (recall@T / precision@T / f1@T / mAP) are
  *scanned*; they report how detection degrades as the overlap requirement
  tightens. The pr_f1_vs_iou grid visualises this scan and is the tool for
  choosing a per-dataset match cutoff (look for the precision/F1 knee/elbow).
* match_threshold is the IoU gate of the QUALITY matching: pairs below it are
  demoted to missing (Dice=0, distance=penalty). Re-running
  reevaluate_all.py --match-threshold T recomputes Dice/IoU/HD95/ASSD under
  that gate. Picking T from the detection curves and feeding it to the quality
  re-eval is non-circular because detection metrics are scanned independently
  of match_threshold.

Workflow
--------
1. Run this script -> read the elbow off pr_f1_vs_iou_grid.png.
2. reevaluate_all.py --match-threshold T  (recompute quality under the gate).
3. Run this script again -> dice_*.png and quality_metrics_*.png now reflect T
   (the chosen match_threshold is annotated in their titles).

Generated figures
------------------
DETECTION (threshold-independent):
1.  pr_f1_vs_iou_grid.png
        THE grid: rows = datasets, columns = leaves (baseline -> unsup ->
        few-shot). Each cell: Precision/Recall/F1 vs IoU threshold, with
        dashed guides at the candidate match cutoffs. Used to pick T.
2.  approach_comparison_{dataset}.png
        Grouped bars over leaves: Precision@0.5, Recall@0.5, F1@0.5. Delta-vs-
        baseline annotated on F1. (Dice intentionally NOT here -> see dice_*.)
3.  recall_per_organ_{dataset}.png  (+ precision_per_organ if present)
4.  map_summary_{dataset}.png

QUALITY (depends on match_threshold; re-read after reevaluate_all.py):
5.  dice_{dataset}.png
        Dice on its own scale: solid = incl. missing, translucent extension up
        to the detected-only ceiling (gap = cost of non-detection).
6.  quality_metrics_{dataset}.png
        Panels for Dice / IoU / HD95 / ASSD (different scales), incl. missing
        vs detected-only. match_threshold annotated in the title.

7.  all_versions_summary.csv

Optional
--------
8.  pr_overlay_{dataset}.png   (--pr-overlay)   COCO-style PR (score sweep).

Usage
-----
    python compare_versions.py --results_dir results_leaf
    python compare_versions.py --results_dir results --cutoffs 0.5 0.75
    python compare_versions.py --datasets XRay_test Sunnybrook_test
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from project.evaluation.display_names import get_display_name, EXPERIMENT_ORDER
from project.evaluation.plot_utils import (
    _compat_get_global,
    _is_distance_metric,
    _metric_label,
)


# ---------------------------------------------------------------------------
# Presentation constants
# ---------------------------------------------------------------------------

# Order of versions WITHIN the unsupervised group (baseline / few-shot handled
# separately by group). Unknown versions are appended alphabetically.
UNSUP_VERSION_ORDER = [
    "moments",
    "momentos",
    "embeddings",
    "embeddings_moments",
    "embeddings+moments",
    "emb_moments",
    "emb_momentos",
    "hybrid",
]

VERSION_LABELS = {
    "baseline":           "Baseline",
    "moments":            "Moments",
    "momentos":           "Moments",
    "embeddings":         "Embeddings",
    "embeddings_moments": "Emb.+Mom.",
    "embeddings+moments": "Emb.+Mom.",
    "emb_moments":        "Emb.+Mom.",
    "emb_momentos":       "Emb.+Mom.",
    "hybrid":             "Emb.+Mom.",
    "fs_propagation":     "Few-shot",
}

DEFAULT_DATASET_LABELS = {
    "XRayNicoSent":       "JSRT",
    "SunnybrookNicoSent": "Sunnybrook",
    "Sunnybrook_test":    "Sunnybrook",
    "Sunnybrook":         "Sunnybrook",
    "XRay_test":          "JSRT",
    "XRay":               "JSRT",
    "JSRT_test":          "JSRT",
}

# Group ranking for ordering leaves left-to-right / top-to-bottom.
GROUP_BASELINE = 0
GROUP_UNSUP = 1
GROUP_FEWSHOT = 2
GROUP_OTHER = 3

# Per-group base colormaps for bar colouring (encodes the grouping visually).
_GROUP_CMAP = {
    GROUP_BASELINE: ("Greys", 0.55, 0.55),
    GROUP_UNSUP:    ("Blues", 0.45, 0.85),
    GROUP_FEWSHOT:  ("Greens", 0.45, 0.85),
    GROUP_OTHER:    ("Purples", 0.45, 0.85),
}

# Detection headline metrics (solid bars in approach_comparison).
HEADLINE_DETECTION = [
    ("precision@0.5", "Precision@0.5", "#1A6FB0"),
    ("recall@0.5",    "Recall@0.5",    "#67A9CF"),
    ("f1@0.5",        "F1@0.5",        "#2E7D32"),
]
PRIMARY_METRIC = "f1@0.5"

# Quality metrics (different scales) for the quality_metrics figure.
# (base_name, label, is_distance)
QUALITY_PANELS = [
    ("dice",         "Dice",      False),
    ("iou",          "IoU",       False),
    ("hausdorff_95", "HD95 (px)", True),
    ("assd",         "ASSD (px)", True),
]

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
            and not d.name.startswith("comparison")
            and not d.name.startswith("_")
        )

    for version_dir in version_dirs:
        if not version_dir.is_dir():
            print(f"  [SKIP] Not a directory: {version_dir}")
            continue

        version = version_dir.name
        data[version] = {}

        for dataset_dir in sorted(version_dir.iterdir()):
            if not dataset_dir.is_dir() or dataset_dir.name == "plots_dist":
                continue
            dataset = dataset_dir.name
            if datasets and dataset not in datasets:
                continue
            data[version][dataset] = {}

            for exp_dir in sorted(dataset_dir.iterdir()):
                if not exp_dir.is_dir() or exp_dir.name == "plots":
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


def _organ_stat(summary: dict, organ: str, key: str):
    return summary.get("per_organ", {}).get(organ, {}).get(key)


def _nan(val) -> float:
    return float("nan") if val is None else float(val)


def _quality_pair(summary: dict, base: str):
    """Return (with_missing, detected_only) for a quality metric base name."""
    legacy = f"{base}_mean"
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


def _match_threshold(data: dict) -> float | None:
    """Most common match_threshold across all loaded summaries (for titles)."""
    seen: dict[float, int] = {}
    for v in data.values():
        for ds in v.values():
            for s in ds.values():
                mt = s.get("match_threshold")
                if mt is not None:
                    seen[float(mt)] = seen.get(float(mt), 0) + 1
    if not seen:
        return None
    return max(seen, key=seen.get)


# ---------------------------------------------------------------------------
# Leaf model: (version, experiment) pairs ordered baseline -> unsup -> few-shot
# ---------------------------------------------------------------------------

_EXP_ORDER = {name: i for i, name in enumerate(EXPERIMENT_ORDER)}


def _leaf_group(version: str, exp: str, baseline_version: str) -> int:
    if version == baseline_version or exp.startswith("unsup_baseline") \
            or exp.startswith("fs_indep_baseline") or exp.startswith("fs_baseline"):
        return GROUP_BASELINE
    if exp.startswith("fs"):
        return GROUP_FEWSHOT
    if exp.startswith("unsup"):
        return GROUP_UNSUP
    return GROUP_OTHER


def _version_rank(version: str) -> int:
    try:
        return UNSUP_VERSION_ORDER.index(version)
    except ValueError:
        return 10_000


def _leaf_label(version: str, exp: str, group: int,
                multi_config_versions: set[str]) -> str:
    """Column label: version name, disambiguated by config when needed."""
    base = VERSION_LABELS.get(version, version)
    if group == GROUP_BASELINE:
        return base
    # When a version contributes several configs (e.g. few-shot K sweep),
    # append a short config tag so columns stay distinguishable.
    if version in multi_config_versions:
        k = _ref_count(exp)
        variant = "Iter." if "iter" in exp else ""
        tag = []
        if variant:
            tag.append(variant)
        if k is not None:
            tag.append(f"K={k}")
        if tag:
            return f"{base}\n({' '.join(tag)})"
        return f"{base}\n{get_display_name(exp)}"
    return base


def _ref_count(exp: str) -> int | None:
    import re
    m = re.search(r"(\d+)ref", exp)
    return int(m.group(1)) if m else None


def build_leaves(data: dict, dataset: str, baseline_version: str) -> list[dict]:
    """
    Build the ordered list of leaves present for `dataset`.

    Each leaf: {version, experiment, group, label, summary}. Ordered by
    (group, version_rank, experiment_order). Colours are assigned per group.
    """
    raw: list[dict] = []
    configs_per_version: dict[str, set[str]] = {}
    for version, ds_map in data.items():
        exps = ds_map.get(dataset, {})
        for exp, summary in exps.items():
            configs_per_version.setdefault(version, set()).add(exp)
            raw.append({
                "version": version,
                "experiment": exp,
                "group": _leaf_group(version, exp, baseline_version),
                "summary": summary,
            })

    multi = {v for v, cfgs in configs_per_version.items() if len(cfgs) > 1}

    raw.sort(key=lambda L: (
        L["group"],
        _version_rank(L["version"]),
        _EXP_ORDER.get(L["experiment"], 999),
        L["version"],
        L["experiment"],
    ))

    for L in raw:
        L["label"] = _leaf_label(
            L["version"], L["experiment"], L["group"], multi
        )

    _assign_leaf_colors(raw)
    return raw


def _assign_leaf_colors(leaves: list[dict]) -> None:
    """Colour each leaf from its group colormap (shaded within the group)."""
    by_group: dict[int, list[dict]] = {}
    for L in leaves:
        by_group.setdefault(L["group"], []).append(L)
    for group, members in by_group.items():
        cmap_name, lo, hi = _GROUP_CMAP.get(group, _GROUP_CMAP[GROUP_OTHER])
        cmap = plt.get_cmap(cmap_name)
        n = len(members)
        for i, L in enumerate(members):
            t = lo if n == 1 else lo + (hi - lo) * i / (n - 1)
            L["color"] = cmap(t)


def _all_datasets(data: dict) -> list[str]:
    """Datasets that have at least one experiment somewhere (empty dirs skipped)."""
    seen: set[str] = set()
    for v in data.values():
        for ds, exps in v.items():
            if exps:
                seen.add(ds)
    return sorted(seen)


def _union_leaf_keys(
    data: dict, datasets: list[str], baseline_version: str,
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], str]]:
    """Ordered union of (version, experiment) keys + their labels, across
    the given datasets."""
    order: dict[tuple[str, str], dict] = {}
    for dataset in datasets:
        for L in build_leaves(data, dataset, baseline_version):
            order[(L["version"], L["experiment"])] = L
    leaves = list(order.values())
    leaves.sort(key=lambda L: (
        L["group"],
        _version_rank(L["version"]),
        _EXP_ORDER.get(L["experiment"], 999),
        L["version"],
        L["experiment"],
    ))
    return [(L["version"], L["experiment"]) for L in leaves], \
           {(L["version"], L["experiment"]): L["label"] for L in leaves}


def _dataset_label(dataset: str, labels: dict[str, str]) -> str:
    return labels.get(dataset, dataset)


def _bar_label(ax, x, height, text, color="#222", dy=0.02, fs=7):
    if height is None or (isinstance(height, float) and math.isnan(height)):
        return
    ax.text(x, height + dy, text, ha="center", va="bottom",
            fontsize=fs, color=color)


# ---------------------------------------------------------------------------
# Figure 1: P/R/F1 vs IoU grid  (rows = datasets, cols = leaves)
# ---------------------------------------------------------------------------

def plot_pr_f1_vs_iou_grid(data, output_dir, datasets, baseline_version,
                           dataset_labels, cutoffs):
    """
    One figure: rows = datasets, columns = leaves (baseline -> unsup ->
    few-shot). Each cell plots Precision/Recall/F1 as a function of the IoU
    detection threshold, with dashed guides at the candidate match cutoffs
    (annotated with P and F1). This is the figure used to read the elbow and
    choose the per-dataset match_threshold.
    """
    leaf_keys, leaf_label = _union_leaf_keys(data, datasets, baseline_version)
    if not leaf_keys:
        print("  [SKIP] pr_f1_vs_iou_grid: no leaves.")
        return

    line_spec = [
        ("precision", "Precision", "#1A6FB0", "o"),
        ("recall",    "Recall",    "#2E7D32", "s"),
        ("f1",        "F1",        "#C62828", "^"),
    ]

    nrows = len(datasets)
    ncols = len(leaf_keys)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(max(3.2, ncols * 3.2), max(3.0, nrows * 3.2)),
        squeeze=False,
    )

    for ri, dataset in enumerate(datasets):
        for ci, (version, exp) in enumerate(leaf_keys):
            ax = axes[ri][ci]
            summary = data.get(version, {}).get(dataset, {}).get(exp)

            if ri == 0:
                ax.set_title(leaf_label[(version, exp)], fontsize=9,
                             fontweight="bold")
            if ci == 0:
                ax.set_ylabel(
                    f"{_dataset_label(dataset, dataset_labels)}\nScore",
                    fontsize=10, fontweight="bold",
                )

            if summary is None:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, fontsize=14, color="#ccc")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            iou_thrs = sorted(float(t) for t in summary.get("iou_thresholds", []))
            if not iou_thrs:
                ax.text(0.5, 0.5, "no thresholds", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="#aaa")
                continue

            series = {}
            for prefix, label, color, marker in line_spec:
                ys = [_nan(_g(summary, f"{prefix}@{t}")) for t in iou_thrs]
                series[prefix] = ys
                ax.plot(iou_thrs, ys, marker=marker, color=color,
                        linewidth=1.5, markersize=3.5, label=label)

            for c in cutoffs:
                if iou_thrs[0] <= c <= iou_thrs[-1]:
                    ax.axvline(c, color="#888", linestyle="--",
                               linewidth=1.0, alpha=0.7)
                    p_at = _interp_at(iou_thrs, series["precision"], c)
                    f_at = _interp_at(iou_thrs, series["f1"], c)
                    txt = []
                    if f_at is not None:
                        txt.append(f"F1={f_at:.2f}")
                    if p_at is not None:
                        txt.append(f"P={p_at:.2f}")
                    if txt:
                        ax.text(c, 1.03, "\n".join(txt), ha="center",
                                va="bottom", fontsize=6, color="#555")

            ax.set_xlim(iou_thrs[0] - 0.03, iou_thrs[-1] + 0.03)
            ax.set_ylim(-0.05, 1.18)
            if ri == nrows - 1:
                ax.set_xlabel("IoU threshold", fontsize=9)
            ax.grid(alpha=0.3)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=7, loc="lower left")

    fig.suptitle(
        "P/R/F1 vs IoU threshold — rows: datasets, cols: approaches\n"
        "(dashed = candidate match cutoffs; read the F1/precision elbow here)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = output_dir / "pr_f1_vs_iou_grid.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2: approach comparison (detection headline only — no Dice)
# ---------------------------------------------------------------------------

def plot_approach_comparison(data, output_dir, baseline_version, dataset_labels):
    """
    Grouped bars over leaves: Precision@0.5, Recall@0.5, F1@0.5 (detection).
    The primary metric (F1) is annotated with its delta against the baseline
    leaf. Dice is intentionally excluded (different scale) -> see plot_dice.
    """
    for dataset in _all_datasets(data):
        leaves = build_leaves(data, dataset, baseline_version)
        if not leaves:
            continue

        n_leaf = len(leaves)
        n_met = len(HEADLINE_DETECTION)
        x = np.arange(n_leaf)
        width = 0.8 / n_met

        fig, ax = plt.subplots(figsize=(max(8, n_leaf * 1.7), 5.4))

        base = next((L for L in leaves if L["group"] == GROUP_BASELINE), None)
        base_primary = _g(base["summary"], PRIMARY_METRIC) if base else None

        for mi, (key, label, color) in enumerate(HEADLINE_DETECTION):
            offset = (mi - n_met / 2 + 0.5) * width
            for li, L in enumerate(leaves):
                v = _g(L["summary"], key)
                if v is None:
                    continue
                bx = li + offset
                ax.bar(bx, v, width, color=color, alpha=0.9,
                       edgecolor="white", linewidth=0.5,
                       label=label if li == 0 else None)
                if key == PRIMARY_METRIC and base_primary is not None \
                        and L["group"] != GROUP_BASELINE:
                    delta = v - base_primary
                    sign = "+" if delta >= 0 else ""
                    _bar_label(ax, bx, v, f"{v:.3f}\n({sign}{delta:.3f})",
                               color="#2E7D32" if delta >= 0 else "#C62828",
                               fs=6.5)
                else:
                    _bar_label(ax, bx, v, f"{v:.3f}", fs=6.5)

        ax.set_xticks(x)
        ax.set_xticklabels([L["label"] for L in leaves],
                           fontsize=9, rotation=15, ha="right")
        ax.set_ylim(0, 1.22)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(
            f"Detection comparison @ IoU 0.5 — "
            f"{_dataset_label(dataset, dataset_labels)}\n"
            "(delta vs baseline on F1)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=9, loc="lower right", ncol=3)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        out = output_dir / f"approach_comparison_{dataset}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3: per-organ metric across leaves (recall always; precision if any)
# ---------------------------------------------------------------------------

def plot_per_organ(data, output_dir, baseline_version, dataset_labels,
                   metric_key, metric_label, file_stem):
    """
    Grouped bars per dataset: x = organ, one bar per leaf, height = the
    requested per-organ metric. Skips a dataset entirely if no organ on any
    leaf has the metric.
    """
    for dataset in _all_datasets(data):
        leaves = build_leaves(data, dataset, baseline_version)
        if not leaves:
            continue

        organs: set[str] = set()
        for L in leaves:
            for organ, stats in L["summary"].get("per_organ", {}).items():
                if stats.get(metric_key) is not None:
                    organs.add(organ)
        organs = sorted(organs)
        if not organs:
            print(f"  [SKIP] {file_stem} ({dataset}): no per-organ "
                  f"'{metric_key}'.")
            continue

        n_leaf = len(leaves)
        x = np.arange(len(organs))
        width = 0.8 / n_leaf

        fig, ax = plt.subplots(figsize=(max(7, len(organs) * 2.4), 5.2))

        for li, L in enumerate(leaves):
            vals = [_nan(_organ_stat(L["summary"], organ, metric_key))
                    for organ in organs]
            offset = (li - n_leaf / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=L["label"].replace("\n", " "),
                   color=L["color"], alpha=0.92,
                   edgecolor="white", linewidth=0.5)

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
        ax.legend(fontsize=8, loc="lower right", ncol=2)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        out = output_dir / f"{file_stem}_{dataset}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 4: mAP summary heatmap (leaves x [mAP, AP50, AP75])
# ---------------------------------------------------------------------------

def _annotate_cell(ax, col, row, value, vmin, vmax):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        ax.text(col, row, "—", ha="center", va="center", fontsize=8, color="#bbb")
        return
    norm = (value - vmin) / (vmax - vmin + 1e-12)
    color = "white" if norm < 0.35 or norm > 0.85 else "#222"
    ax.text(col, row, f"{value:.3f}", ha="center", va="center",
            fontsize=9, fontweight="bold", color=color)


def plot_map_summary(data, output_dir, baseline_version, dataset_labels):
    col_keys = ["map", "map_50", "map_75"]
    col_labels = ["mAP@[.5:.95]", "AP@0.5", "AP@0.75"]

    for dataset in _all_datasets(data):
        leaves = build_leaves(data, dataset, baseline_version)
        if not leaves:
            continue

        n_leaf = len(leaves)
        n_col = len(col_keys)
        matrix = np.full((n_leaf, n_col), np.nan)
        for li, L in enumerate(leaves):
            for ci, key in enumerate(col_keys):
                val = _g(L["summary"], key)
                if val is not None:
                    matrix[li, ci] = val

        fig, ax = plt.subplots(
            figsize=(max(4.5, n_col * 2.0), max(3.0, n_leaf * 0.7 + 1.2))
        )
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)
        for li in range(n_leaf):
            for ci in range(n_col):
                _annotate_cell(ax, ci, li, matrix[li, ci], 0.0, 1.0)

        ax.set_xticks(range(n_col))
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(n_leaf))
        ax.set_yticklabels([L["label"].replace("\n", " ") for L in leaves],
                           fontsize=9)
        ax.set_title(f"mAP summary — {_dataset_label(dataset, dataset_labels)}",
                     fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8, label="AP")
        fig.tight_layout()

        out = output_dir / f"map_summary_{dataset}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 5: Dice on its own scale (solid = incl. missing, gap to detected-only)
# ---------------------------------------------------------------------------

def plot_dice(data, output_dir, baseline_version, dataset_labels, match_thr):
    """
    Per dataset, one bar per leaf: solid = Dice incl. missing, translucent
    extension up to the detected-only ceiling (the gap = cost of undetected
    organs). On its own figure because its scale differs from detection and
    distance metrics. Reflects the current match_threshold.
    """
    mt_txt = f" (match IoU ≥ {match_thr:g})" if match_thr is not None else ""
    for dataset in _all_datasets(data):
        leaves = build_leaves(data, dataset, baseline_version)
        if not leaves:
            continue

        n_leaf = len(leaves)
        x = np.arange(n_leaf)
        fig, ax = plt.subplots(figsize=(max(7, n_leaf * 1.5), 5.2))

        for li, L in enumerate(leaves):
            wm, do = _quality_pair(L["summary"], "dice")
            if wm is None and do is None:
                continue
            base_h = wm if wm is not None else do
            ax.bar(li, base_h, 0.62, color=L["color"], alpha=0.95,
                   edgecolor="white", linewidth=0.5)
            _bar_label(ax, li, base_h, f"{base_h:.3f}", fs=7)
            if do is not None and wm is not None and do > wm + 1e-6:
                ax.bar(li, do - wm, 0.62, bottom=wm, color=L["color"],
                       alpha=0.32, edgecolor="none")
                ax.text(li, do + 0.015, f"{do:.3f}\n(gap {do - wm:.3f})",
                        ha="center", va="bottom", fontsize=6.5, color="#555")

        ax.set_xticks(x)
        ax.set_xticklabels([L["label"] for L in leaves],
                           fontsize=9, rotation=15, ha="right")
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Dice", fontsize=11)
        ax.set_title(
            f"Dice — {_dataset_label(dataset, dataset_labels)}{mt_txt}\n"
            "(solid = incl. missing, translucent = detected-only ceiling)",
            fontsize=12, fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        out = output_dir / f"dice_{dataset}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 6: quality metrics panels (Dice / IoU / HD95 / ASSD; different scales)
# ---------------------------------------------------------------------------

def plot_quality_metrics(data, output_dir, baseline_version, dataset_labels,
                         match_thr):
    """
    Per dataset, one panel per quality metric (Dice, IoU, HD95, ASSD). Each
    panel: two bars per leaf — incl. missing vs detected-only. Distance panels
    (HD95/ASSD, lower=better) are autoscaled; overlap panels (Dice/IoU) use
    [0,1]. This is the figure to regenerate after reevaluate_all.py with the
    chosen match_threshold (annotated in the suptitle).
    """
    mt_txt = f"match IoU ≥ {match_thr:g}" if match_thr is not None \
        else "match threshold unknown"
    for dataset in _all_datasets(data):
        leaves = build_leaves(data, dataset, baseline_version)
        if not leaves:
            continue

        n_leaf = len(leaves)
        x = np.arange(n_leaf)
        width = 0.38

        fig, axes = plt.subplots(2, 2, figsize=(max(11, n_leaf * 2.0), 9),
                                 squeeze=False)

        for pi, (base, label, is_dist) in enumerate(QUALITY_PANELS):
            ax = axes[pi // 2][pi % 2]
            wm_list, do_list = [], []
            for L in leaves:
                wm, do = _quality_pair(L["summary"], base)
                wm_list.append(_nan(wm))
                do_list.append(_nan(do))

            ax.bar(x - width / 2, do_list, width, label="detected only",
                   color="#67A9CF", alpha=0.92, edgecolor="white", linewidth=0.5)
            ax.bar(x + width / 2, wm_list, width, label="incl. missing",
                   color="#2166AC", alpha=0.92, edgecolor="white", linewidth=0.5)

            fmt = "{:.2f}" if is_dist else "{:.3f}"
            top = max([v for v in do_list + wm_list if not math.isnan(v)],
                      default=1.0)
            dy = top * 0.02 if is_dist else 0.015
            for xi, (d, w) in enumerate(zip(do_list, wm_list)):
                _bar_label(ax, xi - width / 2, d, fmt.format(d), dy=dy, fs=6.5)
                _bar_label(ax, xi + width / 2, w, fmt.format(w), dy=dy, fs=6.5)

            ax.set_xticks(x)
            ax.set_xticklabels([L["label"] for L in leaves],
                               fontsize=8, rotation=20, ha="right")
            arrow = "↓ better" if is_dist else "↑ better"
            ax.set_ylabel(f"{label}  ({arrow})", fontsize=10)
            if not is_dist:
                ax.set_ylim(0, 1.12)
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.legend(fontsize=8, loc="best")
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(
            f"Quality metrics — {_dataset_label(dataset, dataset_labels)} "
            f"({mt_txt})",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out = output_dir / f"quality_metrics_{dataset}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Optional: COCO-style PR overlay (sweeps prediction SCORE)
# ---------------------------------------------------------------------------

def plot_pr_overlay(data, output_dir, baseline_version, dataset_labels,
                    iou_values=(0.5, 0.75)):
    ap_key = {0.5: "map_50", 0.75: "map_75"}
    for dataset in _all_datasets(data):
        leaves = build_leaves(data, dataset, baseline_version)
        if not leaves:
            continue

        n_panels = len(iou_values)
        fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 4.6, 4.8),
                                 squeeze=False)
        axes = axes[0]

        for pi, iou in enumerate(iou_values):
            ax = axes[pi]
            any_drawn = False
            for L in leaves:
                s = L["summary"]
                ap = _g(s, ap_key.get(iou, ""))
                ap_txt = f" (AP={ap:.2f})" if ap is not None else ""
                label = L["label"].replace("\n", " ") + ap_txt
                curve = _get_pr_curve(s, iou)
                if curve and curve.get("recall"):
                    ax.plot(curve["recall"], curve["precision"], "-",
                            color=L["color"], linewidth=1.8, alpha=0.95,
                            label=label)
                    any_drawn = True
                else:
                    r = _g(s, f"recall@{iou}")
                    p = _g(s, f"precision@{iou}")
                    if r is not None and p is not None:
                        ax.scatter([r], [p], color=L["color"], s=55, marker="o",
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
                ax.legend(fontsize=7, loc="lower left")
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
# Flat CSV export
# ---------------------------------------------------------------------------

def save_full_csv(data, output_dir, baseline_version):
    rows: list[dict] = []

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
                group = _leaf_group(version, exp, baseline_version)
                row: dict = {
                    "version":         version,
                    "dataset":         dataset,
                    "experiment":      exp,
                    "group":           ["baseline", "unsup", "fewshot",
                                        "other"][group],
                    "n_images":        summary.get("n_images", 0),
                    "matching":        summary.get("matching", ""),
                    "match_threshold": summary.get("match_threshold"),
                    "dice_detected_only":
                        g.get("dice_mean_detected_only") or g.get("dice_mean"),
                    "dice_with_missing":
                        g.get("dice_mean_with_missing") or g.get("dice_mean"),
                    "iou_detected_only": g.get("iou_mean_detected_only"),
                    "iou_with_missing":  g.get("iou_mean_with_missing"),
                    "hd95_detected_only":
                        g.get("hausdorff_95_mean_detected_only"),
                    "hd95_with_missing":
                        g.get("hausdorff_95_mean_with_missing")
                        or g.get("hausdorff_95_mean"),
                    "assd_detected_only": g.get("assd_mean_detected_only"),
                    "assd_with_missing":
                        g.get("assd_mean_with_missing") or g.get("assd_mean"),
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
        "version", "dataset", "experiment", "group",
        "n_images", "matching", "match_threshold",
        "dice_detected_only", "dice_with_missing",
        "iou_detected_only", "iou_with_missing",
        "hd95_detected_only", "hd95_with_missing",
        "assd_detected_only", "assd_with_missing",
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
# CLI
# ---------------------------------------------------------------------------

def _parse_dataset_labels(items: list[str]) -> dict[str, str]:
    result = dict(DEFAULT_DATASET_LABELS)
    for item in items or []:
        if ":" not in item:
            raise ValueError(f"Expected DATASET:LABEL, got: {item!r}")
        ds, label = item.split(":", 1)
        result[ds.strip()] = label.strip()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compare pipeline approaches across versions and datasets "
                    "(leaf = version x experiment).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_dir", default="results_leaf",
                        help="Root dir (version/dataset/experiment/summary.json).")
    parser.add_argument("--versions", nargs="*",
                        help="Versions to include (default: all).")
    parser.add_argument("--datasets", nargs="*",
                        help="Datasets to include and order (default: all).")
    parser.add_argument("--experiments", nargs="*",
                        help="Restrict loaded experiment names (default: all).")
    parser.add_argument("--output", default=None,
                        help="Output dir (default: <results_dir>/comparison).")
    parser.add_argument("--baseline-version", default="baseline",
                        help="Version treated as feature-independent reference.")
    parser.add_argument("--cutoffs", nargs="*", type=float, default=DEFAULT_CUTOFFS,
                        help="Candidate match cutoffs marked on the vs-IoU grid.")
    parser.add_argument("--pr-overlay", action="store_true",
                        help="Also generate the COCO-style PR overlay (score sweep).")
    parser.add_argument("--dataset-labels", nargs="*", metavar="DATASET:LABEL",
                        default=[], help="DATASET:LABEL overrides for titles.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output) if args.output else results_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_labels = _parse_dataset_labels(args.dataset_labels)

    print(f"Loading results from {results_dir} ...")
    data = load_data(results_dir, args.versions, args.datasets, args.experiments)

    total = sum(len(ds) for v in data.values() for ds in v.values())
    if total == 0:
        print("No experiment summaries found. Nothing to plot.")
        return
    print(f"\nVersions: {list(data.keys())}")
    print(f"Total experiment results (leaves): {total}")

    datasets = args.datasets if args.datasets else _all_datasets(data)
    match_thr = _match_threshold(data)
    if match_thr is not None:
        print(f"Detected match_threshold (mode across summaries): {match_thr:g}")

    print("\n[1/7] P/R/F1 vs IoU grid (rows=datasets, cols=approaches):")
    plot_pr_f1_vs_iou_grid(data, output_dir, datasets, args.baseline_version,
                           dataset_labels, args.cutoffs)

    print("\n[2/7] Detection comparison (P/R/F1 @0.5):")
    plot_approach_comparison(data, output_dir, args.baseline_version,
                             dataset_labels)

    print("\n[3/7] Per-organ recall (and precision if available):")
    plot_per_organ(data, output_dir, args.baseline_version, dataset_labels,
                   "recall@0.5", "Recall@0.5", "recall_per_organ")
    plot_per_organ(data, output_dir, args.baseline_version, dataset_labels,
                   "precision@0.5", "Precision@0.5", "precision_per_organ")

    print("\n[4/7] mAP summary heatmap:")
    plot_map_summary(data, output_dir, args.baseline_version, dataset_labels)

    print("\n[5/7] Dice (own scale):")
    plot_dice(data, output_dir, args.baseline_version, dataset_labels, match_thr)

    print("\n[6/7] Quality metrics panels (Dice/IoU/HD95/ASSD):")
    plot_quality_metrics(data, output_dir, args.baseline_version,
                         dataset_labels, match_thr)

    if args.pr_overlay:
        print("\n[Optional] PR-curve overlay (score sweep):")
        plot_pr_overlay(data, output_dir, args.baseline_version, dataset_labels)

    print("\n[7/7] Flat CSV export:")
    save_full_csv(data, output_dir, args.baseline_version)

    print(f"\nAll comparison artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()

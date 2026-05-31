"""
compare_versions.py
-------------------
Generate cross-version, cross-dataset comparison plots.

Scans results/{version}/{dataset}/{experiment}/summary.json and produces
a lean set of figures designed for the paper.

Generated figures (core set)
-----------------------------
1.  recall_per_organ_{dataset}.png
        Grouped bars: x=experiments, groups=organs. Direct view of per-organ
        detection success (Recall@0.5).

2.  refinement_impact_{dataset}.png
        Cross-version refinement impact (with vs without _refine suffix).
        Supports the claim that the Refiner is the isolated contributor.

3.  dice_gap_{dataset}.png
        Dice gap: solid bar=with_missing; translucent extension to
        detected_only. Reveals the cost of non-detection per organ.

4.  pr_grid_{config}.png
        COCO-style PR curve grid (rows=datasets, cols=feature versions).
        Annotates AP@0.5 / AP@0.75 / mAP per cell.

5.  pr_f1_vs_iou_{dataset}.png
        P/R/F1 curves vs IoU threshold (NOT a PR curve; shows detection
        degradation as overlap requirement tightens).

6.  map_summary_{dataset}.png
        Compact mAP heatmap (configs x [mAP, AP@0.5, AP@0.75]).

7.  all_versions_summary.csv
        Flat CSV with every (version, dataset, experiment, metric).

Appendix figures (--appendix, off by default)
----------------------------------------------
A1. box_plot_dice_{dataset}_{version}_{slug}.png
        Side-by-side Dice box plots comparing baseline vs pipeline variants.

A2. pr_trajectory_{dataset}.png
        P-R trajectory parametrised by IoU threshold (NOT a PR curve;
        no area is computed).

Optional (--skip-recovery to skip)
-----------------------------------
O1. organ_recovery_{dataset}_{version}_{slug}.png
        Baseline-miss → pipeline-hit recovery counts.

Usage
-----
    python compare_versions.py --results_dir results/ --output results/comparison/

    python compare_versions.py \\
        --versions v1_baseline v3_extended v5_ext_emb_red \\
        --experiments unsup_hdbscan_refine fs_iter_refine_1ref \\
        --configs unsup_hdbscan unsup_hdbscan_refine fs_iter_refine_1ref \\
        --feature-versions v1_baseline:"Baseline (6 mom.)" v5_ext_emb_red:Hybrid \\
        --output results/comparison_focused/

    python compare_versions.py --appendix --skip-recovery
"""

import argparse
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from project.evaluation.plot_utils import (
    ALL_V0_EQUIVALENCES,
    DEFAULT_METRICS,
    METRIC_META,
    _compat_get_global,
    _compat_get_per_organ,
    _annotate_heatmap_cell,
    _get_experiment_group,
    _heatmap_vrange,
    _is_distance_metric,
    _metric_label,
    _metric_cmap,
    _parse_threshold_dict,
    _sort_experiments,
    get_display_name,
)
from project.evaluation.plots_extra import (
    plot_pr_curve_grid,
    plot_pr_f1_vs_iou,
    plot_map_summary,
    plot_pr_space_trajectory,
)

_ALL_BASELINE_EXPERIMENTS: set[str] = set(ALL_V0_EQUIVALENCES.keys())

# ---------------------------------------------------------------------------
# Default grid parameters (overridable via CLI)
# ---------------------------------------------------------------------------

_DEFAULT_CONFIGS = [
    "unsup_hdbscan",
    "unsup_hdbscan_refine",
    "fs_iter_refine_1ref",
]

_DEFAULT_FEATURE_VERSIONS = [
    ("v1_baseline", "Baseline (6 mom.)"),
    ("v3_extended", "Moments (16)"),
    ("v4_emb_only", "Embeddings"),
    ("v5_ext_emb_red", "Hybrid"),
]

_DEFAULT_DATASET_LABELS = {
    "XRayNicoSent":       "JSRT",
    "SunnybrookNicoSent": "Sunnybrook",
}


# ---------------------------------------------------------------------------
# Small helpers (local to compare_versions, not shared with plots_extra)
# ---------------------------------------------------------------------------

def _common_prefix(strings: list[str]) -> str:
    if not strings:
        return ""
    s_min, s_max = min(strings), max(strings)
    for i, ch in enumerate(s_min):
        if ch != s_max[i]:
            return s_min[:i]
    return s_min


def _short_dataset_labels(datasets: list[str]) -> list[str]:
    if len(datasets) < 2:
        return list(datasets)
    prefix = _common_prefix(datasets)
    last_sep = max(prefix.rfind("_"), prefix.rfind("-"), prefix.rfind("/"))
    if last_sep < 0:
        return list(datasets)
    cut = last_sep + 1
    return [d[cut:] for d in datasets]


def _draw_group_separators(ax, experiments: list[str], axis: str = "y"):
    prev_group = None
    for idx, exp in enumerate(experiments):
        group = _get_experiment_group(exp)
        if prev_group is not None and group != prev_group:
            pos = idx - 0.5
            if axis == "y":
                ax.axhline(y=pos, color="grey", linewidth=1.2,
                           linestyle="--", alpha=0.5)
            else:
                ax.axvline(x=pos, color="grey", linewidth=1.2,
                           linestyle="--", alpha=0.5)
        prev_group = group


# ---------------------------------------------------------------------------
# Quality heatmap (experiments × versions, one per metric per dataset)
# ---------------------------------------------------------------------------

def plot_metric_heatmap_per_dataset(data: dict, output_dir: Path, metric: str):
    """
    For each dataset, heatmap with experiments on y-axis and versions on x-axis
    showing the global value of `metric`. Skips when fewer than 2 versions are
    present (single-version runs still produce a 1-column heatmap).
    """
    versions = list(data.keys())
    label = _metric_label(metric)
    cmap  = _metric_cmap(metric)

    all_datasets: set[str] = set()
    for v in data.values():
        all_datasets.update(v.keys())

    for dataset in sorted(all_datasets):
        all_exps: set[str] = set()
        for v in versions:
            all_exps.update(data.get(v, {}).get(dataset, {}).keys())
        experiments = _sort_experiments(list(all_exps))
        if not experiments:
            continue

        matrix = np.full((len(experiments), len(versions)), np.nan)
        for j, v in enumerate(versions):
            for i, exp in enumerate(experiments):
                summary = data.get(v, {}).get(dataset, {}).get(exp)
                if summary:
                    val = _compat_get_global(summary, metric)
                    if val is not None:
                        matrix[i, j] = val

        vmin, vmax = _heatmap_vrange(metric, matrix)
        fig, ax = plt.subplots(
            figsize=(max(4, len(versions) * 2),
                     max(6, len(experiments) * 0.55)),
        )
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        for i in range(len(experiments)):
            for j in range(len(versions)):
                _annotate_heatmap_cell(ax, j, i, matrix[i, j], vmin, vmax)

        _draw_group_separators(ax, experiments, axis="y")
        ax.set_xticks(range(len(versions)))
        ax.set_xticklabels(versions, fontsize=10, rotation=30, ha="right")
        ax.set_yticks(range(len(experiments)))
        ax.set_yticklabels(
            [get_display_name(e) for e in experiments], fontsize=9,
        )
        ax.set_title(f"Global {label} — {dataset}",
                     fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8, label=label)
        fig.tight_layout()

        out_path = (
            output_dir
            / f"metric_heatmap_{metric.replace('@', '_at_')}_{dataset}.png"
        )
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _extend_filter_with_baselines(experiments_filter: list[str]) -> list[str]:
    filter_set = set(experiments_filter)
    extras: list[str] = []
    for ref_exp, targets in ALL_V0_EQUIVALENCES.items():
        if ref_exp in filter_set:
            continue
        targets_list = targets if isinstance(targets, list) else [targets]
        if any(t in filter_set for t in targets_list):
            extras.append(ref_exp)
    if extras:
        print(f"  [auto-filter] Adding baseline experiments: {extras}")
    return list(filter_set | set(extras))


def load_all_results(
    results_dir: Path,
    versions: list[str] | None = None,
    experiments_filter: list[str] | None = None,
) -> dict[str, dict[str, dict[str, dict]]]:
    """
    Recursively load summary.json files.

    Returns {version: {dataset: {experiment: summary_dict}}}
    """
    if experiments_filter:
        experiments_filter = _extend_filter_with_baselines(experiments_filter)

    data: dict = {}

    if versions:
        version_dirs = [results_dir / v for v in versions]
    else:
        version_dirs = sorted([
            d for d in results_dir.iterdir()
            if d.is_dir()
            and not d.name.startswith(".")
            and d.name != "comparison"
        ])

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
            data[version][dataset] = {}

            for exp_dir in sorted(dataset_dir.iterdir()):
                if experiments_filter and exp_dir.name not in experiments_filter:
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


def _filter_datasets(data: dict, datasets_filter: list[str]) -> dict:
    keep = list(datasets_filter)
    keep_set = set(keep)
    pruned: dict = {}
    for v, ds_map in data.items():
        pruned[v] = {d: ds_map[d] for d in keep if d in ds_map}
        missing = keep_set - set(ds_map.keys())
        if missing:
            print(f"  [WARN] {v}: datasets not found and skipped: "
                  f"{sorted(missing)}")
    return pruned


def _load_metrics_csv(csv_path: Path) -> list[dict]:
    rows = []
    if not csv_path.exists():
        return rows
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Plot 1: Recall@0.5 per organ
# ---------------------------------------------------------------------------

def plot_recall_per_organ(data: dict, output_dir: Path):
    """
    For each dataset, grouped bars: x=experiments, one bar group per organ.

    Direct expression of detection success per organ class (Recall@0.5).
    Aggregates across versions by using the last version with data per
    experiment.
    """
    all_datasets: set[str] = set()
    for v in data.values():
        all_datasets.update(v.keys())

    versions = list(data.keys())

    for dataset in sorted(all_datasets):
        all_exps: set[str] = set()
        all_organs: set[str] = set()
        for v in versions:
            for exp_name, summary in data.get(v, {}).get(dataset, {}).items():
                all_exps.add(exp_name)
                all_organs.update(summary.get("per_organ", {}).keys())

        experiments = _sort_experiments(list(all_exps))
        organs = sorted(all_organs)

        if not experiments or not organs:
            continue

        n_exp = len(experiments)
        n_org = len(organs)
        x = np.arange(n_exp)
        width = 0.8 / max(n_org, 1)

        organ_colors = {
            "heart":      "#D45B5B",
            "left_lung":  "#5B8DB8",
            "right_lung": "#6BBF6B",
            "lv_cavity":  "#9C27B0",
            "myocardium": "#FF9800",
            "lung":       "#5BA8D4",
        }

        fig, ax = plt.subplots(figsize=(max(8, n_exp * 1.2), 5))

        for oi, organ in enumerate(organs):
            recalls = []
            for exp in experiments:
                val = None
                for v in reversed(versions):
                    summary = data.get(v, {}).get(dataset, {}).get(exp)
                    if summary is None:
                        continue
                    od = summary.get("per_organ", {}).get(organ, {})
                    val = od.get("recall@0.5")
                    if val is None:
                        n_tot = od.get("n_total", od.get("count", 0))
                        n_cov = od.get("n_covered@0.5", 0)
                        if n_tot and n_tot > 0:
                            val = n_cov / n_tot
                    if val is not None:
                        break
                recalls.append(val if val is not None else 0.0)

            color = organ_colors.get(organ, f"C{oi}")
            offset = (oi - n_org / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, recalls, width,
                label=organ.replace("_", " ").title(),
                color=color, alpha=0.82,
                edgecolor="white", linewidth=0.5,
            )
            for bar, v in zip(bars, recalls):
                if 0.0 < v < 1.0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        v + 0.02, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6,
                    )

        _draw_group_separators(ax, experiments, axis="x")
        ax.set_ylabel("Recall @ IoU ≥ 0.5", fontsize=11)
        ax.set_title(f"Recall@0.5 per Organ — {dataset}", fontsize=13,
                     fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [get_display_name(e) for e in experiments],
            fontsize=8, rotation=35, ha="right",
        )
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        out_path = output_dir / f"recall_per_organ_{dataset}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Refinement impact cross-version
# ---------------------------------------------------------------------------

def plot_refinement_impact_cross_version(data: dict, output_dir: Path):
    """
    For each dataset, detect (_base, _base_refine) experiment pairs by suffix.

    For each pair, grouped bars per version showing Dice (incl. missing) and
    F1@0.5 with and without refinement, with delta annotation.
    Supports the paper claim that the Refiner is the isolated contributor.
    """
    metrics_to_plot = ["dice_mean_with_missing", "f1@0.5"]
    m_labels = [_metric_label(m) for m in metrics_to_plot]
    colors_base   = ["#5B8DB8", "#6BBF6B"]
    colors_refine = ["#E8A838", "#D45B5B"]

    versions = list(data.keys())
    all_datasets: set[str] = set()
    for v in data.values():
        all_datasets.update(v.keys())

    for dataset in sorted(all_datasets):
        all_exps_seen: set[str] = set()
        for v in versions:
            all_exps_seen.update(data.get(v, {}).get(dataset, {}).keys())

        pairs = [
            (base, base + "_refine")
            for base in sorted(all_exps_seen)
            if base + "_refine" in all_exps_seen
        ]

        if not pairs:
            continue

        n_pairs = len(pairs)
        n_metrics = len(metrics_to_plot)
        n_versions = len(versions)

        fig, axes = plt.subplots(
            n_pairs, n_metrics,
            figsize=(n_metrics * max(4, n_versions * 1.2), n_pairs * 3.5),
            squeeze=False,
        )

        for pi, (base, refine) in enumerate(pairs):
            for mi, metric in enumerate(metrics_to_plot):
                ax = axes[pi][mi]

                base_vals, refine_vals = [], []
                present_vers = []
                for v in versions:
                    b_sum = data.get(v, {}).get(dataset, {}).get(base)
                    r_sum = data.get(v, {}).get(dataset, {}).get(refine)
                    if b_sum is None and r_sum is None:
                        continue
                    present_vers.append(v)
                    bv = _compat_get_global(b_sum or {}, metric)
                    rv = _compat_get_global(r_sum or {}, metric)
                    base_vals.append(bv if bv is not None else np.nan)
                    refine_vals.append(rv if rv is not None else np.nan)

                if not present_vers:
                    ax.set_visible(False)
                    continue

                x = np.arange(len(present_vers))
                width = 0.35
                ax.bar(x - width / 2, base_vals, width,
                       label="Without refine",
                       color=colors_base[mi], alpha=0.88,
                       edgecolor="white", linewidth=0.5)
                ax.bar(x + width / 2, refine_vals, width,
                       label="With refine",
                       color=colors_refine[mi], alpha=0.88,
                       edgecolor="white", linewidth=0.5)

                for i, (bv, rv) in enumerate(zip(base_vals, refine_vals)):
                    if np.isnan(bv) or np.isnan(rv):
                        continue
                    delta = rv - bv
                    sign = "+" if delta >= 0 else ""
                    ax.text(
                        i, max(bv, rv) + 0.03,
                        f"{sign}{delta:.3f}",
                        ha="center", fontsize=8, fontweight="bold",
                        color="#2E7D32" if delta >= 0 else "#C62828",
                    )

                ax.set_ylim(0, 1.18)
                ax.set_ylabel(m_labels[mi], fontsize=9)
                ax.set_xticks(x)
                ax.set_xticklabels(present_vers, fontsize=8, rotation=25, ha="right")
                ax.legend(fontsize=7)
                ax.grid(axis="y", alpha=0.3)
                if pi == 0:
                    ax.set_title(m_labels[mi], fontsize=10, fontweight="bold")

            axes[pi][0].set_ylabel(
                f"{get_display_name(base)}\nvs refine",
                fontsize=8,
            )

        fig.suptitle(
            f"Refinement Impact across Versions — {dataset}",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()
        out_path = output_dir / f"refinement_impact_{dataset}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: Dice gap (with_missing vs detected_only)
# ---------------------------------------------------------------------------

def plot_dice_gap(data: dict, output_dir: Path):
    """
    For each dataset, one figure showing the Dice gap between with_missing and
    detected_only variants, faceted by organ + global panel.

    Solid bar = dice_mean_with_missing.
    Translucent extension from with_missing to detected_only = the gap caused
    by missing detections (detected_only >= with_missing always).
    If detected_only is None (organ never detected), only the solid bar at 0
    is drawn — communicating "never found".
    """
    all_datasets: set[str] = set()
    for v in data.values():
        all_datasets.update(v.keys())

    for dataset in sorted(all_datasets):
        all_exps: set[str] = set()
        all_organs: set[str] = set()
        for v in data:
            for exp_name, summary in data.get(v, {}).get(dataset, {}).items():
                all_exps.add(exp_name)
                all_organs.update(summary.get("per_organ", {}).keys())

        experiments = _sort_experiments(list(all_exps))
        organs = sorted(all_organs)

        if not experiments or not organs:
            continue

        versions = list(data.keys())

        def _get_wm_do(exp: str, organ: str | None) -> tuple[float | None, float | None]:
            wm, do = None, None
            for v in reversed(versions):
                summary = data.get(v, {}).get(dataset, {}).get(exp)
                if summary is None:
                    continue
                if organ is None:
                    wm = _compat_get_global(summary, "dice_mean_with_missing")
                    do = summary.get("global", {}).get("dice_mean_detected_only")
                else:
                    wm = _compat_get_per_organ(summary, organ, "dice_mean_with_missing")
                    do = summary.get("per_organ", {}).get(organ, {}).get(
                        "dice_mean_detected_only"
                    )
                if wm is not None:
                    break
            return wm, do

        panels = organs + [None]
        n_panels = len(panels)
        n_exp = len(experiments)

        ncols = min(n_panels, 4)
        nrows = (n_panels + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * max(3.5, n_exp * 0.55 + 1), nrows * 4),
            squeeze=False,
        )

        colors = plt.get_cmap("tab10").colors
        x = np.arange(n_exp)

        for panel_idx, panel in enumerate(panels):
            row = panel_idx // ncols
            col = panel_idx % ncols
            ax = axes[row][col]

            wm_vals, do_vals = [], []
            for exp in experiments:
                wm, do = _get_wm_do(exp, panel)
                wm_vals.append(wm if wm is not None else 0.0)
                do_vals.append(do)

            panel_title = panel.replace("_", " ").title() if panel else "Global"
            bar_colors = [
                colors[_sort_experiments(list(all_exps)).index(e) % len(colors)]
                for e in experiments
            ]

            ax.bar(x, wm_vals, color=bar_colors, alpha=0.85,
                   edgecolor="white", linewidth=0.5)

            for xi, (wm_v, do_v) in enumerate(zip(wm_vals, do_vals)):
                if do_v is not None and do_v > wm_v:
                    ax.bar(xi, do_v - wm_v, bottom=wm_v,
                           color=bar_colors[xi], alpha=0.28, edgecolor="none")
                    ax.text(xi, do_v + 0.02, f"{do_v:.2f}",
                            ha="center", va="bottom", fontsize=6, color="#555")

            for xi, v in enumerate(wm_vals):
                if v > 0.01:
                    ax.text(xi, v / 2, f"{v:.2f}",
                            ha="center", va="center", fontsize=6,
                            color="white", fontweight="bold")

            _draw_group_separators(ax, experiments, axis="x")

            ax.set_title(panel_title, fontsize=10, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [get_display_name(e) for e in experiments],
                fontsize=7, rotation=35, ha="right",
            )
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("Dice", fontsize=9)
            ax.grid(axis="y", alpha=0.3)

        for panel_idx in range(n_panels, nrows * ncols):
            axes[panel_idx // ncols][panel_idx % ncols].set_visible(False)

        fig.suptitle(
            f"Dice: incl. missing (solid) vs detected-only (+ translucent gap)"
            f" — {dataset}",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()
        out_path = output_dir / f"dice_gap_{dataset}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Optional: Organ recovery analysis (reads metrics.csv)
# ---------------------------------------------------------------------------

RECOVERY_PAIRS_UNSUP = {
    "unsup_baseline": [
        "unsup_kmeans", "unsup_kmeans_refine",
        "unsup_hdbscan", "unsup_hdbscan_refine",
    ],
}

RECOVERY_PAIRS_FS = {
    "fs_indep_baseline_1ref": [
        "fs_indep_1ref", "fs_indep_refine_1ref",
        "fs_iter_1ref",  "fs_iter_refine_1ref",
    ],
    "fs_indep_baseline_3ref": [
        "fs_indep_3ref", "fs_indep_refine_3ref",
        "fs_iter_3ref",  "fs_iter_refine_3ref",
    ],
}


def _build_organ_status(rows: list[dict]) -> dict[tuple[str, str], dict]:
    status = {}
    for row in rows:
        key = (row["image"], row["organ"])
        pred_name = row.get("pred_name", "")
        is_missing = (
            pred_name == "" or pred_name == "None" or pred_name is None
        )
        status[key] = {
            "pred_name": None if is_missing else pred_name,
            "dice": float(row.get("dice", 0)),
        }
    return status


def _analyze_recovery(baseline_rows: list[dict], pipeline_rows: list[dict]) -> dict:
    b_status = _build_organ_status(baseline_rows)
    p_status = _build_organ_status(pipeline_rows)
    all_keys = set(b_status.keys()) | set(p_status.keys())

    recovered, lost, both_hit, both_miss = [], [], [], []

    for key in sorted(all_keys):
        image, organ = key
        b = b_status.get(key, {"pred_name": None, "dice": 0})
        p = p_status.get(key, {"pred_name": None, "dice": 0})
        entry = {
            "image": image, "organ": organ,
            "baseline_dice": b["dice"], "pipeline_dice": p["dice"],
        }
        b_present = b["pred_name"] is not None
        p_present = p["pred_name"] is not None

        if not b_present and p_present:
            recovered.append(entry)
        elif b_present and not p_present:
            lost.append(entry)
        elif b_present and p_present:
            both_hit.append(entry)
        else:
            both_miss.append(entry)

    recovery_by_organ: dict[str, list] = defaultdict(list)
    for r in recovered:
        recovery_by_organ[r["organ"]].append(r)

    return {
        "n_recovered": len(recovered),
        "n_lost": len(lost),
        "n_both_hit": len(both_hit),
        "n_both_miss": len(both_miss),
        "total": len(all_keys),
        "recovered": recovered,
        "lost": lost,
        "recovery_by_organ": dict(recovery_by_organ),
        "recovered_dice_mean": (
            float(np.mean([r["pipeline_dice"] for r in recovered]))
            if recovered else None
        ),
    }


def plot_organ_recovery(
    results_dir: Path,
    data: dict,
    output_dir: Path,
    recovery_pairs: dict | None = None,
):
    """
    For each (version, dataset, baseline_exp, pipeline_exp), plot recovered vs
    lost organ instance counts. Reads existing metrics.csv files; does not
    re-run the pipeline.
    """
    if recovery_pairs is None:
        recovery_pairs = {**RECOVERY_PAIRS_UNSUP, **RECOVERY_PAIRS_FS}

    for version in data:
        for dataset in data[version]:
            for baseline_exp, pipeline_exps in recovery_pairs.items():
                baseline_csv = (
                    results_dir / version / dataset
                    / baseline_exp / "metrics.csv"
                )
                if not baseline_csv.exists():
                    continue

                baseline_rows = _load_metrics_csv(baseline_csv)
                if not baseline_rows:
                    continue

                all_results: dict[str, dict] = {}

                for pipe_exp in pipeline_exps:
                    pipe_csv = (
                        results_dir / version / dataset
                        / pipe_exp / "metrics.csv"
                    )
                    if not pipe_csv.exists():
                        continue
                    pipe_rows = _load_metrics_csv(pipe_csv)
                    result = _analyze_recovery(baseline_rows, pipe_rows)
                    all_results[pipe_exp] = result
                    print(
                        f"  {version}/{dataset}: {pipe_exp} vs {baseline_exp}"
                        f" — recovered={result['n_recovered']}"
                        f", lost={result['n_lost']}"
                    )

                if not all_results:
                    continue

                labels = list(all_results.keys())
                n_rec  = [r["n_recovered"] for r in all_results.values()]
                n_lost = [-r["n_lost"] for r in all_results.values()]

                x = np.arange(len(labels))
                width = 0.35
                fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 6))

                bars_r = ax.bar(x - width / 2, n_rec, width,
                                label="Recovered", color="#4CAF50", alpha=0.85)
                bars_l = ax.bar(x + width / 2, n_lost, width,
                                label="Lost", color="#F44336", alpha=0.85)

                for bar, v in zip(bars_r, n_rec):
                    if v > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.5, str(v),
                                ha="center", va="bottom",
                                fontsize=10, fontweight="bold")
                for bar, v in zip(bars_l, n_lost):
                    if v < 0:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                v - 0.5, str(abs(v)),
                                ha="center", va="top",
                                fontsize=10, fontweight="bold", color="#B71C1C")

                ax.axhline(y=0, color="black", linewidth=0.8)
                ax.set_ylabel("Organ instances", fontsize=12)
                ax.set_title(
                    f"Organ Recovery vs {get_display_name(baseline_exp)}"
                    f" — {dataset} / {version}",
                    fontsize=13, fontweight="bold",
                )
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [get_display_name(l) for l in labels],
                    fontsize=9, rotation=20, ha="right",
                )
                ax.legend(fontsize=10)
                ax.grid(axis="y", alpha=0.3)
                fig.tight_layout()

                slug = baseline_exp.replace("fs_indep_baseline_", "fs_")
                out_path = (
                    output_dir
                    / f"organ_recovery_{dataset}_{version}_{slug}.png"
                )
                fig.savefig(out_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved: {out_path}")

                # Per-organ breakdown for best pipeline
                best_exp = max(
                    all_results, key=lambda k: all_results[k]["n_recovered"]
                )
                best = all_results[best_exp]

                if best["recovery_by_organ"]:
                    organs = sorted(best["recovery_by_organ"].keys())
                    counts = [len(best["recovery_by_organ"][o]) for o in organs]
                    mean_dices = [
                        np.mean([e["pipeline_dice"]
                                 for e in best["recovery_by_organ"][o]])
                        for o in organs
                    ]

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    ax1.barh(organs, counts, color="#4CAF50", alpha=0.85)
                    for i, v in enumerate(counts):
                        ax1.text(v + 0.3, i, str(v), va="center",
                                 fontweight="bold")
                    ax1.set_xlabel("Recovered instances")
                    ax1.set_title("Recovery count")

                    ax2.barh(organs, mean_dices, color="#2196F3", alpha=0.85)
                    for i, v in enumerate(mean_dices):
                        ax2.text(v + 0.01, i, f"{v:.3f}", va="center",
                                 fontweight="bold")
                    ax2.set_xlabel("Mean Dice of recovered organs")
                    ax2.set_xlim(0, 1)
                    ax2.set_title("Recovery quality")

                    fig.suptitle(
                        f"Recovery by Organ — {get_display_name(best_exp)}"
                        f" — {dataset} / {version}",
                        fontsize=13, fontweight="bold",
                    )
                    fig.tight_layout()
                    out_path = (
                        output_dir
                        / f"organ_recovery_detail_{dataset}_{version}_{slug}.png"
                    )
                    fig.savefig(out_path, dpi=200, bbox_inches="tight")
                    plt.close(fig)
                    print(f"  Saved: {out_path}")

                # CSV per-case details
                csv_rows = []
                for pipe_exp, result in all_results.items():
                    for r in result["recovered"]:
                        csv_rows.append({
                            "version": version, "dataset": dataset,
                            "baseline": baseline_exp, "pipeline": pipe_exp,
                            "status": "recovered", **r,
                        })
                    for r in result["lost"]:
                        csv_rows.append({
                            "version": version, "dataset": dataset,
                            "baseline": baseline_exp, "pipeline": pipe_exp,
                            "status": "lost", **r,
                        })

                if csv_rows:
                    csv_path = (
                        output_dir
                        / f"organ_recovery_{dataset}_{version}_{slug}.csv"
                    )
                    fieldnames = [
                        "version", "dataset", "baseline", "pipeline",
                        "status", "image", "organ",
                        "baseline_dice", "pipeline_dice",
                    ]
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(
                            f, fieldnames=fieldnames, extrasaction="ignore",
                        )
                        writer.writeheader()
                        writer.writerows(csv_rows)
                    print(f"  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Appendix: Box plots consolidated
# ---------------------------------------------------------------------------

BOX_PLOT_PAIRS_UNSUP = {
    "unsup_baseline": [
        "unsup_hdbscan", "unsup_hdbscan_refine",
        "unsup_kmeans_refine",
    ],
}

BOX_PLOT_PAIRS_FS = {
    "fs_indep_baseline_1ref": ["fs_iter_1ref", "fs_iter_refine_1ref"],
    "fs_indep_baseline_3ref": ["fs_iter_3ref", "fs_iter_refine_3ref"],
}


def plot_box_plots_consolidated(
    results_dir: Path,
    data: dict,
    output_dir: Path,
    box_plot_pairs: dict | None = None,
):
    """
    For each (version, dataset, baseline_exp), one figure with subplots=organs.

    Side-by-side box plots comparing baseline vs all pipeline variants.
    The mean is marked with a diamond. (Appendix figure.)
    """
    if box_plot_pairs is None:
        box_plot_pairs = {**BOX_PLOT_PAIRS_UNSUP, **BOX_PLOT_PAIRS_FS}

    for version in data:
        for dataset in data[version]:
            for baseline_exp, pipeline_exps in box_plot_pairs.items():
                baseline_csv = (
                    results_dir / version / dataset
                    / baseline_exp / "metrics.csv"
                )
                if not baseline_csv.exists():
                    continue

                baseline_rows = _load_metrics_csv(baseline_csv)
                if not baseline_rows:
                    continue

                dice_by_exp: dict[str, dict[str, list[float]]] = {}
                dice_by_exp[baseline_exp] = defaultdict(list)
                for row in baseline_rows:
                    dice_by_exp[baseline_exp][row["organ"]].append(
                        float(row.get("dice", 0))
                    )

                present_exps = [baseline_exp]
                for pipe_exp in pipeline_exps:
                    pipe_csv = (
                        results_dir / version / dataset
                        / pipe_exp / "metrics.csv"
                    )
                    if not pipe_csv.exists():
                        continue
                    pipe_rows = _load_metrics_csv(pipe_csv)
                    dice_by_exp[pipe_exp] = defaultdict(list)
                    for row in pipe_rows:
                        dice_by_exp[pipe_exp][row["organ"]].append(
                            float(row.get("dice", 0))
                        )
                    present_exps.append(pipe_exp)

                if len(present_exps) < 2:
                    continue

                all_organs: set[str] = set()
                for exp_data in dice_by_exp.values():
                    all_organs.update(exp_data.keys())
                organs = sorted(all_organs)

                if not organs:
                    continue

                n_organs = len(organs)
                ncols = min(n_organs, 3)
                nrows = (n_organs + ncols - 1) // ncols
                fig, axes = plt.subplots(
                    nrows, ncols,
                    figsize=(ncols * max(4, len(present_exps) * 1.0),
                             nrows * 4),
                    squeeze=False,
                )
                colors = [
                    "#BDBDBD", "#4CAF50", "#2196F3", "#FF9800", "#9C27B0"
                ]

                for oi, organ in enumerate(organs):
                    row_idx = oi // ncols
                    col_idx = oi % ncols
                    ax = axes[row_idx][col_idx]

                    box_data, box_labels, box_colors = [], [], []
                    for idx, exp in enumerate(present_exps):
                        values = dice_by_exp.get(exp, {}).get(organ, [])
                        if values:
                            box_data.append(values)
                            box_labels.append(get_display_name(exp))
                            box_colors.append(colors[idx % len(colors)])

                    if len(box_data) < 1:
                        ax.set_visible(False)
                        continue

                    bp = ax.boxplot(
                        box_data,
                        labels=box_labels,
                        patch_artist=True,
                        widths=0.6,
                        showmeans=True,
                        meanprops=dict(
                            marker="D", markerfacecolor="white",
                            markeredgecolor="black", markersize=6,
                        ),
                    )
                    for patch, color in zip(bp["boxes"], box_colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)

                    for i, vals in enumerate(box_data):
                        mean_v = np.mean(vals)
                        ax.text(i + 1, mean_v + 0.02, f"μ={mean_v:.3f}",
                                ha="center", fontsize=7, color="#333")

                    ax.set_title(organ.replace("_", " ").title(),
                                 fontsize=10, fontweight="bold")
                    ax.set_ylabel("Dice", fontsize=9)
                    ax.set_ylim(-0.05, 1.10)
                    ax.grid(axis="y", alpha=0.3)
                    ax.tick_params(axis="x", rotation=20, labelsize=8)

                for oi in range(n_organs, nrows * ncols):
                    axes[oi // ncols][oi % ncols].set_visible(False)

                slug = baseline_exp.replace("fs_indep_baseline_", "fs_")
                fig.suptitle(
                    f"Dice Distribution — {dataset} / {version}"
                    f" (vs {get_display_name(baseline_exp)})",
                    fontsize=12, fontweight="bold",
                )
                fig.tight_layout()
                out_path = (
                    output_dir
                    / f"box_plot_dice_{dataset}_{version}_{slug}.png"
                )
                fig.savefig(out_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Flat CSV export
# ---------------------------------------------------------------------------

def save_full_csv(data: dict, output_dir: Path):
    """
    Flatten all results into a single CSV.

    Includes global quality metrics (both split variants), P/R/F1 at each
    available IoU threshold, per-organ metrics, and P/R/F1 from
    detection_per_threshold if present.
    """
    rows: list[dict] = []

    thresholds_seen: set[float] = set()
    for v in data.values():
        for ds in v.values():
            for s in ds.values():
                for thr in s.get("iou_thresholds", []):
                    thresholds_seen.add(float(thr))
    thresholds = sorted(thresholds_seen)

    for version in data:
        for dataset in data[version]:
            for exp, summary in data[version][dataset].items():
                g = summary.get("global", {})
                row: dict = {
                    "version":    version,
                    "dataset":    dataset,
                    "experiment": exp,
                    "n_images":   summary.get("n_images", 0),
                    "matching":   summary.get("matching", ""),
                    "match_threshold": summary.get("match_threshold"),
                    "dice_mean_with_missing":   g.get("dice_mean_with_missing"),
                    "dice_std_with_missing":    g.get("dice_std_with_missing"),
                    "dice_mean_detected_only":  g.get("dice_mean_detected_only"),
                    "dice_std_detected_only":   g.get("dice_std_detected_only"),
                    "iou_mean_with_missing":    g.get("iou_mean_with_missing"),
                    "iou_std_with_missing":     g.get("iou_std_with_missing"),
                    "iou_mean_detected_only":   g.get("iou_mean_detected_only"),
                    "iou_std_detected_only":    g.get("iou_std_detected_only"),
                    # Legacy flat keys (may be None for new-format summaries)
                    "dice_mean":  g.get("dice_mean"),
                    "dice_std":   g.get("dice_std"),
                    "iou_mean":   g.get("iou_mean"),
                    "iou_std":    g.get("iou_std"),
                    "hd95_mean_with_missing":
                        g.get("hausdorff_95_mean_with_missing")
                        or g.get("hausdorff_95_mean"),
                    "hd95_std_with_missing":
                        g.get("hausdorff_95_std_with_missing")
                        or g.get("hausdorff_95_std"),
                    "hd95_mean_detected_only":
                        g.get("hausdorff_95_mean_detected_only"),
                    "assd_mean_with_missing":
                        g.get("assd_mean_with_missing") or g.get("assd_mean"),
                    "assd_mean_detected_only":
                        g.get("assd_mean_detected_only"),
                    "n_gt_total":        g.get("n_gt_total"),
                    "n_pred_total":      g.get("n_pred_total"),
                    "pct_real_scores":   g.get("pct_real_scores"),
                    "map":               g.get("map"),
                    "map_50":            g.get("map_50"),
                    "map_75":            g.get("map_75"),
                }

                for thr in thresholds:
                    row[f"recall@{thr}"]    = g.get(f"recall@{thr}")
                    row[f"precision@{thr}"] = g.get(f"precision@{thr}")
                    row[f"f1@{thr}"]        = g.get(f"f1@{thr}")
                    row[f"n_gt_covered@{thr}"]    = g.get(f"n_gt_covered@{thr}")
                    row[f"n_pred_relevant@{thr}"] = g.get(f"n_pred_relevant@{thr}")

                det_thr = summary.get("detection_per_threshold")
                if det_thr:
                    for thr_f, d in _parse_threshold_dict(det_thr):
                        t_str = f"{thr_f:.2f}"
                        row[f"det_recall@{t_str}"]    = d.get("recall")
                        row[f"det_precision@{t_str}"] = d.get("precision")
                        row[f"det_f1@{t_str}"]        = d.get("f1")

                for organ, stats in summary.get("per_organ", {}).items():
                    row[f"{organ}_dice_with_missing"] = _compat_get_per_organ(
                        summary, organ, "dice_mean_with_missing"
                    )
                    row[f"{organ}_dice_detected_only"] = stats.get(
                        "dice_mean_detected_only"
                    )
                    row[f"{organ}_missing"] = stats.get("missing", 0)
                    for thr in thresholds:
                        row[f"{organ}_recall@{thr}"] = stats.get(f"recall@{thr}")

                rows.append(row)

    if not rows:
        return

    fieldnames = [
        "version", "dataset", "experiment", "n_images", "matching",
        "match_threshold",
        "dice_mean_with_missing", "dice_std_with_missing",
        "dice_mean_detected_only", "dice_std_detected_only",
        "iou_mean_with_missing", "iou_std_with_missing",
        "iou_mean_detected_only", "iou_std_detected_only",
        "dice_mean", "dice_std", "iou_mean", "iou_std",
        "hd95_mean_with_missing", "hd95_std_with_missing",
        "hd95_mean_detected_only",
        "assd_mean_with_missing", "assd_mean_detected_only",
        "n_gt_total", "n_pred_total", "pct_real_scores",
        "map", "map_50", "map_75",
    ]
    for thr in thresholds:
        fieldnames += [
            f"recall@{thr}", f"precision@{thr}", f"f1@{thr}",
            f"n_gt_covered@{thr}", f"n_pred_relevant@{thr}",
        ]
    extra = sorted({k for row in rows for k in row if k not in fieldnames})
    fieldnames.extend(extra)

    csv_path = output_dir / "all_versions_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_version_label_pairs(items: list[str]) -> list[tuple[str, str]]:
    """Parse 'VERSION:LABEL' strings into (version, label) tuples."""
    result = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"Expected VERSION:LABEL, got: {item!r}")
        v, label = item.split(":", 1)
        result.append((v.strip(), label.strip()))
    return result


def _parse_dataset_label_map(items: list[str]) -> dict[str, str]:
    """Parse 'DATASET:LABEL' strings into {dataset: label} dict."""
    result = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Expected DATASET:LABEL, got: {item!r}")
        ds, label = item.split(":", 1)
        result[ds.strip()] = label.strip()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compare pipeline results across versions and datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir", default="results",
        help="Root results dir (version/dataset/experiment/summary.json)",
    )
    parser.add_argument(
        "--versions", nargs="*",
        help="Versions to load (default: all subdirs of results_dir)",
    )
    parser.add_argument(
        "--datasets", nargs="*",
        help="Datasets to include (default: all). Order is preserved.",
    )
    parser.add_argument(
        "--experiments", nargs="*",
        help="Filter to these experiment names only (default: all). "
             "Baseline experiments whose targets are in the filter are "
             "added automatically.",
    )
    parser.add_argument(
        "--output", default="results/comparison",
        help="Output directory for plots and CSV.",
    )
    # Grid parametrisation
    parser.add_argument(
        "--configs", nargs="*",
        default=_DEFAULT_CONFIGS,
        help="Experiment names used as rows/subplots in the new grid plots.",
    )
    parser.add_argument(
        "--feature-versions", nargs="*",
        metavar="VERSION:LABEL",
        default=[f"{v}:{l}" for v, l in _DEFAULT_FEATURE_VERSIONS],
        help="VERSION:LABEL pairs defining the columns of the PR curve grid.",
    )
    parser.add_argument(
        "--dataset-labels", nargs="*",
        metavar="DATASET:LABEL",
        default=[f"{k}:{v}" for k, v in _DEFAULT_DATASET_LABELS.items()],
        help="DATASET:LABEL pairs for display in grid row headers.",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=DEFAULT_METRICS,
        help="Metrics to include in quality heatmaps (keys in summary[global]).",
    )
    # Mode flags
    parser.add_argument(
        "--appendix", action="store_true",
        help="Also generate appendix figures (box plots, P-R trajectory).",
    )
    parser.add_argument(
        "--skip-recovery", action="store_true",
        help="Skip organ recovery analysis (needs metrics.csv).",
    )
    parser.add_argument(
        "--skip-dice-gap", action="store_true",
        help="Skip Dice gap (with_missing vs detected_only).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse grid parameters
    feature_versions = _parse_version_label_pairs(args.feature_versions)
    dataset_labels   = _parse_dataset_label_map(args.dataset_labels)

    print("Loading results...")
    data = load_all_results(
        results_dir,
        versions=args.versions,
        experiments_filter=args.experiments,
    )

    if args.datasets:
        data = _filter_datasets(data, args.datasets)
        print(f"\nFiltered to {len(args.datasets)} dataset(s): "
              f"{list(args.datasets)}")

    total = sum(len(exps) for v in data.values() for exps in v.values())
    if total == 0:
        print("No results found.")
        return

    print(f"\nVersions: {list(data.keys())}")
    print(f"Total experiment results: {total}")

    # Determine dataset order for plots
    dataset_order = (
        args.datasets
        if args.datasets
        else sorted({d for v in data.values() for d in v})
    )

    # ---------- Core figures ----------

    print("\n[1/8] Quality metric heatmaps (Dice, IoU, Recall, F1):")
    for metric in args.metrics:
        plot_metric_heatmap_per_dataset(data, output_dir, metric=metric)

    print("\n[2/8] Recall@0.5 per organ:")
    plot_recall_per_organ(data, output_dir)

    print("\n[3/8] Refinement impact cross-version:")
    plot_refinement_impact_cross_version(data, output_dir)

    if args.skip_dice_gap:
        print("\n[4/8] Dice gap: SKIPPED (--skip-dice-gap)")
    else:
        print("\n[4/8] Dice gap (with_missing vs detected_only):")
        plot_dice_gap(data, output_dir)

    print("\n[5/8] COCO-style PR curve grids:")
    plot_pr_curve_grid(
        data, output_dir,
        configs=args.configs,
        feature_versions=feature_versions,
        dataset_order=dataset_order,
        dataset_labels=dataset_labels,
    )

    print("\n[6/8] P/R/F1 vs IoU threshold:")
    plot_pr_f1_vs_iou(
        data, output_dir,
        configs=args.configs,
        dataset_order=dataset_order,
    )

    print("\n[7/8] mAP summary heatmap:")
    plot_map_summary(
        data, output_dir,
        dataset_order=dataset_order,
        # No configs filter: show all available experiments so baseline is visible
    )

    print("\n[8/8] Flat CSV export:")
    save_full_csv(data, output_dir)

    # ---------- Optional: organ recovery ----------

    if args.skip_recovery:
        print("\n[Optional] Organ recovery: SKIPPED (--skip-recovery)")
    else:
        print("\n[Optional] Organ recovery analysis:")
        plot_organ_recovery(results_dir, data, output_dir)

    # ---------- Appendix ----------

    if args.appendix:
        print("\n[Appendix A1] Per-organ box plots (consolidated):")
        plot_box_plots_consolidated(results_dir, data, output_dir)

        print("\n[Appendix A2] P-R space trajectory:")
        plot_pr_space_trajectory(
            data, output_dir,
            configs=args.configs,
            dataset_order=dataset_order,
        )

    print(f"\nAll comparison artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()

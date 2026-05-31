"""
Extra comparison plots for the evaluation pipeline.

    plot_pr_curve_grid      COCO-style PR curve grids (rows=datasets, cols=versions)
    plot_pr_f1_vs_iou       P/R/F1 as functions of the IoU threshold (NOT a PR curve)
    plot_map_summary        Compact mAP heatmap per dataset
    plot_pr_space_trajectory  (appendix) P-R trajectory parametrised by IoU gate

All functions share the same data dict structure used by compare_versions.py:
    data = {version: {dataset: {experiment: summary_dict}}}
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from project.evaluation.plot_utils import (
    _compat_get_global,
    _parse_threshold_dict,
    _sort_experiments,
    METRIC_META,
    _annotate_heatmap_cell,
    get_display_name,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _last_summary(data: dict, dataset: str, config: str) -> dict | None:
    """Return the summary from the most-recent version that has (dataset, config)."""
    for v in reversed(list(data.keys())):
        s = data.get(v, {}).get(dataset, {}).get(config)
        if s is not None:
            return s
    return None


def _safe_list(lst: list) -> list:
    return [v if v is not None else float("nan") for v in lst]


# ---------------------------------------------------------------------------
# T4.1 — COCO-style PR curve grid
# ---------------------------------------------------------------------------

def plot_pr_curve_grid(
    data: dict,
    output_dir: Path,
    configs: list[str],
    feature_versions: list[tuple[str, str]],
    dataset_order: list[str],
    dataset_labels: dict[str, str],
    iou_display: float = 0.5,
    iou_overlay: float = 0.75,
    fallback_score_pct_threshold: float = 5.0,
) -> None:
    """
    One figure per config. Rows = datasets, cols = feature versions.

    Each cell draws the COCO-style PR curve at iou_display (solid blue) and
    iou_overlay (dashed red), annotating AP_50 / AP_75 / mAP in a lower-left
    corner box. Square aspect, x=Recall[0,1], y=Precision[0,1].

    Column labels appear only on the top row; dataset labels only on the left
    column. File: pr_grid_{config}.png at dpi=200.

    Legacy summaries (no pr_curve_per_threshold) fall back to a single scatter
    point at (recall@iou_display, precision@iou_display) plus a "legacy / no
    curve" annotation. If pct_real_scores < fallback_score_pct_threshold, an
    additional warning is drawn in the cell.

    Parameters
    ----------
    feature_versions : list of (version_dir_name, display_label)
        Defines the column order and labels.
    dataset_labels : dict
        Maps dataset directory name to a display label (e.g. "XRayNicoSent" ->
        "JSRT").
    """
    iou_display_key = f"{iou_display:.2f}"
    iou_overlay_key = f"{iou_overlay:.2f}"

    for config in configs:
        n_datasets = len(dataset_order)
        n_versions = len(feature_versions)
        if n_datasets == 0 or n_versions == 0:
            continue

        cell = 3.2
        fig, axes = plt.subplots(
            n_datasets, n_versions,
            figsize=(n_versions * cell, n_datasets * cell),
            squeeze=False,
            constrained_layout=True,
        )

        for ri, dataset in enumerate(dataset_order):
            ds_label = dataset_labels.get(dataset, dataset)
            for ci, (version, ver_label) in enumerate(feature_versions):
                ax = axes[ri][ci]
                summary = data.get(version, {}).get(dataset, {}).get(config)

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.grid(alpha=0.20)
                ax.tick_params(labelsize=7)

                # Column labels only on top row
                if ri == 0:
                    ax.set_title(ver_label, fontsize=9, fontweight="bold")
                # Dataset labels only on left column
                if ci == 0:
                    ax.set_ylabel(ds_label, fontsize=9, fontweight="bold")
                else:
                    ax.set_yticklabels([])

                if ri < n_datasets - 1:
                    ax.set_xticklabels([])

                if summary is None:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8, color="#aaa")
                    continue

                g = summary.get("global", {})
                curves = summary.get("pr_curve_per_threshold") or {}
                pct_real = g.get("pct_real_scores", 100.0)
                ap50 = g.get("map_50")
                ap75 = g.get("map_75")
                map_val = g.get("map")

                # Build annotation
                ann_lines = []
                if ap50 is not None:
                    ann_lines.append(f"AP@0.5={ap50:.3f}")
                if ap75 is not None:
                    ann_lines.append(f"AP@0.75={ap75:.3f}")
                if map_val is not None:
                    ann_lines.append(f"mAP={map_val:.3f}")
                if pct_real < fallback_score_pct_threshold:
                    ann_lines.append("scores mostly fallback\ncurve ≈ point")

                def _draw_pr_curve(curve_data, fallback_r, fallback_p,
                                   color, linestyle, marker, label):
                    """Draw curve if available, else scatter a single fallback point."""
                    r_list = (curve_data or {}).get("recall", [])
                    p_list = (curve_data or {}).get("precision", [])
                    if r_list:
                        ax.plot(r_list, p_list, linestyle, color=color,
                                linewidth=1.8 if linestyle == "-" else 1.4,
                                alpha=1.0 if linestyle == "-" else 0.85,
                                label=label)
                        # Endpoint dot makes small curves visible (e.g. low-recall cases)
                        ax.plot(r_list[-1], p_list[-1], marker, color=color,
                                markersize=6, zorder=6)
                    else:
                        if fallback_r is not None and fallback_p is not None:
                            ax.scatter([fallback_r], [fallback_p], color=color,
                                       s=60, marker=marker, zorder=5, label=label)
                        ax.text(0.5, 0.06, "legacy / no curve",
                                ha="center", va="bottom",
                                transform=ax.transAxes, fontsize=6.5, color="#999")

                _draw_pr_curve(
                    curves.get(iou_display_key),
                    g.get(f"recall@{iou_display}"), g.get(f"precision@{iou_display}"),
                    color="#2166ac", linestyle="-", marker="o",
                    label=f"IoU≥{iou_display:.2f}",
                )
                _draw_pr_curve(
                    curves.get(iou_overlay_key),
                    g.get(f"recall@{iou_overlay}"), g.get(f"precision@{iou_overlay}"),
                    color="#d6604d", linestyle="--", marker="s",
                    label=f"IoU≥{iou_overlay:.2f}",
                )

                # Annotation box (lower-left)
                if ann_lines:
                    ax.text(
                        0.03, 0.03, "\n".join(ann_lines),
                        ha="left", va="bottom",
                        transform=ax.transAxes,
                        fontsize=6.5, color="#222",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  alpha=0.85, ec="#ccc", linewidth=0.8),
                    )

        slug = config.replace("_", "-")
        fig.suptitle(
            f"COCO-style PR curves — {get_display_name(config)}",
            fontsize=11, fontweight="bold",
        )
        out_path = output_dir / f"pr_grid_{slug}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# T4.2 — P/R/F1 vs IoU threshold
# ---------------------------------------------------------------------------

def plot_pr_f1_vs_iou(
    data: dict,
    output_dir: Path,
    configs: list[str],
    dataset_order: list[str],
) -> None:
    """
    For each dataset, one figure with subplots per config.

    x = IoU threshold (sourced from summary["iou_thresholds"]), y = score.
    Three curves: Precision, Recall, F1, each as a function of the IoU gate.
    A vertical dotted line marks summary["match_threshold"] (the operating
    point used during evaluation).

    NOTE: this is NOT a precision-recall curve. It shows how detection
    degrades as the overlap requirement tightens. The title and axis labels
    explicitly say "vs IoU threshold", never "PR curve".

    Uses the last version that has data for each (dataset, config) pair.
    File: pr_f1_vs_iou_{dataset}.png at dpi=200.
    """
    for dataset in dataset_order:
        n_configs = len(configs)
        if n_configs == 0:
            continue

        ncols = min(n_configs, 3)
        nrows = (n_configs + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 4.2, nrows * 3.5),
            squeeze=False,
        )

        for ei, config in enumerate(configs):
            row = ei // ncols
            col = ei % ncols
            ax = axes[row][col]

            summary = _last_summary(data, dataset, config)

            if summary is None:
                ax.set_visible(False)
                continue

            g = summary.get("global", {})
            iou_thrs = sorted(summary.get("iou_thresholds", []))
            match_thr = summary.get("match_threshold", 0.5)

            if not iou_thrs:
                ax.text(0.5, 0.5, "no iou_thresholds",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="#aaa")
                ax.set_title(get_display_name(config), fontsize=9)
                continue

            precisions = _safe_list([g.get(f"precision@{t}") for t in iou_thrs])
            recalls    = _safe_list([g.get(f"recall@{t}")    for t in iou_thrs])
            f1s        = _safe_list([g.get(f"f1@{t}")        for t in iou_thrs])

            ax.plot(iou_thrs, precisions, "b-o", markersize=4,
                    label="Precision", linewidth=1.6)
            ax.plot(iou_thrs, recalls,    "g-s", markersize=4,
                    label="Recall",    linewidth=1.6)
            ax.plot(iou_thrs, f1s,        "r-^", markersize=4,
                    label="F1",        linewidth=1.6)

            # Reference lines at the two standard thresholds
            for ref_t, ref_color in [(0.5, "#4477aa"), (0.7, "#aa7744")]:
                if ref_t in iou_thrs:
                    ax.axvline(x=ref_t, color=ref_color, linestyle="--",
                               linewidth=0.9, alpha=0.7)
                    # Annotate F1 value at top of the line
                    idx = iou_thrs.index(ref_t)
                    f1_val = f1s[idx]
                    if not np.isnan(f1_val):
                        ax.text(ref_t + 0.01, 1.04, f"F1={f1_val:.2f}",
                                fontsize=6, color=ref_color, va="top")

            lo = min(iou_thrs) - 0.05
            hi = max(iou_thrs) + 0.05
            ax.set_xlim(lo, hi)
            ax.set_ylim(-0.05, 1.10)
            ax.set_xlabel("IoU threshold", fontsize=8)
            ax.set_ylabel("Score", fontsize=8)
            ax.set_title(get_display_name(config), fontsize=9, fontweight="bold")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(alpha=0.3)

        for ei in range(n_configs, nrows * ncols):
            axes[ei // ncols][ei % ncols].set_visible(False)

        fig.suptitle(
            f"P/R/F1 vs IoU threshold — {dataset}",
            fontsize=11, fontweight="bold",
        )
        fig.tight_layout()
        out_path = output_dir / f"pr_f1_vs_iou_{dataset}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# T4.3 — mAP summary heatmap
# ---------------------------------------------------------------------------

def plot_map_summary(
    data: dict,
    output_dir: Path,
    dataset_order: list[str],
    configs: list[str] | None = None,
) -> None:
    """
    One compact heatmap per dataset.

    Rows = config (display names), cols = [mAP@[.5:.95], AP@0.5, AP@0.75].
    Colormap RdYlGn, fixed range (0, 1), annotated cells.
    Uses the last version that has data for each (dataset, config) pair.
    File: map_summary_{dataset}.png at dpi=200.
    """
    col_metrics = ["map", "map_50", "map_75"]
    col_labels  = ["mAP@[.5:.95]", "AP@0.5", "AP@0.75"]

    for dataset in dataset_order:
        all_exps: set[str] = set()
        for v_data in data.values():
            all_exps.update(v_data.get(dataset, {}).keys())

        experiments = _sort_experiments(list(all_exps))
        if configs is not None:
            # Preserve caller-specified order
            experiments = [e for e in configs if e in all_exps]

        if not experiments:
            continue

        n_exp = len(experiments)
        n_col = len(col_metrics)

        matrix = np.full((n_exp, n_col), np.nan)
        for ei, exp in enumerate(experiments):
            summary = _last_summary(data, dataset, exp)
            if summary is None:
                continue
            for ci, metric in enumerate(col_metrics):
                val = summary.get("global", {}).get(metric)
                if val is not None:
                    matrix[ei, ci] = val

        fig, ax = plt.subplots(
            figsize=(max(4.5, n_col * 2.0), max(3.0, n_exp * 0.65 + 1.2))
        )
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)

        for ei in range(n_exp):
            for ci in range(n_col):
                _annotate_heatmap_cell(ax, ci, ei, matrix[ei, ci], 0.0, 1.0)

        ax.set_xticks(range(n_col))
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(n_exp))
        ax.set_yticklabels([get_display_name(e) for e in experiments], fontsize=8)
        ax.set_title(f"mAP Summary — {dataset}", fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8, label="AP")
        fig.tight_layout()

        out_path = output_dir / f"map_summary_{dataset}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# T4.4 — (APPENDIX) P-R trajectory parametrised by IoU threshold
# ---------------------------------------------------------------------------

def plot_pr_space_trajectory(
    data: dict,
    output_dir: Path,
    configs: list[str],
    dataset_order: list[str],
) -> None:
    """
    Plot (recall@t, precision@t) points connected across IoU thresholds,
    annotating the IoU value at each point.

    EXPLICIT label: 'P-R trajectory across IoU thresholds (NOT a PR curve;
    no area is computed)'. No area is computed or reported under this curve.

    Uses the last version that has data for each (dataset, config) pair.
    File: pr_trajectory_{dataset}.png at dpi=200.
    """
    for dataset in dataset_order:
        n_configs = len(configs)
        if n_configs == 0:
            continue

        ncols = min(n_configs, 3)
        nrows = (n_configs + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 3.5, nrows * 3.5),
            squeeze=False,
        )

        for ei, config in enumerate(configs):
            row = ei // ncols
            col = ei % ncols
            ax = axes[row][col]

            summary = _last_summary(data, dataset, config)

            if summary is None:
                ax.set_visible(False)
                continue

            g = summary.get("global", {})
            iou_thrs = sorted(summary.get("iou_thresholds", []))

            if not iou_thrs:
                ax.set_visible(False)
                continue

            recalls    = [g.get(f"recall@{t}",    float("nan")) for t in iou_thrs]
            precisions = [g.get(f"precision@{t}", float("nan")) for t in iou_thrs]

            ax.plot(recalls, precisions, "o-", color="#2166ac",
                    linewidth=1.5, markersize=5)

            for t, r, p in zip(iou_thrs, recalls, precisions):
                if not (np.isnan(r) or np.isnan(p)):
                    ax.annotate(
                        f"{t:.1f}", (r, p),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=6, color="#444",
                    )

            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Recall", fontsize=8)
            ax.set_ylabel("Precision", fontsize=8)
            ax.set_aspect("equal")
            ax.set_title(get_display_name(config), fontsize=9, fontweight="bold")
            ax.grid(alpha=0.3)

        for ei in range(n_configs, nrows * ncols):
            axes[ei // ncols][ei % ncols].set_visible(False)

        fig.suptitle(
            f"P-R trajectory across IoU thresholds — {dataset}\n"
            "(NOT a PR curve; no area is computed)",
            fontsize=11, fontweight="bold",
        )
        fig.tight_layout()
        out_path = output_dir / f"pr_trajectory_{dataset}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  Saved: {out_path}")

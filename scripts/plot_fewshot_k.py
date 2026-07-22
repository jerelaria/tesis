#!/usr/bin/env python3
"""
plot_fewshot_k.py
-----------------
Plot Dice vs K (number of reference frames) for few-shot experiments.

Experiment naming convention
----------------------------
Each experiment directory under --results-dir must follow the pattern:

    fs_{mode}[_{suffix}]_{K}ref

  mode   : 'indep'  (independent propagation)
          | 'iter'  (iterative propagation)
  suffix : 'refine' | 'baseline'  (optional; absent for the plain variant)
  K      : integer number of reference frames (1, 3, 5, …)

Examples:
    fs_indep_1ref          -> independent, no suffix, K=1
    fs_indep_refine_3ref   -> independent, refine,    K=3
    fs_iter_5ref           -> iterative,  no suffix,  K=5
    fs_iter_refine_1ref    -> iterative,  refine,     K=1
    fs_indep_baseline_3ref -> independent, baseline,  K=3

Any directory whose name starts with 'fs_' but does NOT match the pattern
raises ValueError with the offending name.  Non-'fs_' directories (e.g.
unsup_baseline) are silently skipped.

Figure layout
-------------
  columns : one per propagation mode present  (independent | iterative)
  rows    : one per suffix present  (plain | refine | baseline)

Each subplot:
  x-axis : K
  y-axis : Dice (variant controlled by --variant)
  lines  : one per organ

A single-K subplot falls back to a bar chart.

Reference lines
---------------
--reference-lines organ=value  (repeatable, space-separated list)

Draws a horizontal dashed line per organ, coloured to match the organ's curve.
Intended for external context (e.g. Ouyang et al. cardiac Dice) — not as a
strict threshold.  Default: no reference lines.

Usage
-----
    python scripts/plot_fewshot_k.py \\
        --results-dir results/embeddings/Sunnybrook \\
        --variant detected_only \\
        --output plots/fewshot_sunnybrook.png \\
        --reference-lines lv_cavity=0.89 myocardium=0.75
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Experiment name parsing
# ---------------------------------------------------------------------------

# Pattern: fs_{mode}[_{suffix}]_{K}ref
_FS_PATTERN = re.compile(r"^fs_(indep|iter)(?:_(refine|baseline))?_(\d+)ref$")

_MODE_LABELS   = {"indep": "Independent", "iter": "Iterative"}
_SUFFIX_LABELS = {None: "", "refine": " + refine", "baseline": " (baseline)"}
# Canonical display order for rows
_SUFFIX_ORDER  = [None, "refine", "baseline"]

# ---------------------------------------------------------------------------
# i18n
# ---------------------------------------------------------------------------

_MODE_LABELS_ES   = {"indep": "Independiente", "iter": "Iterativo"}
_SUFFIX_LABELS_ES = {None: "", "refine": " + refinamiento", "baseline": " (baseline)"}

_ORGAN_LABELS_ES = {
    "heart": "corazón",
    "left_lung": "pulmón izquierdo",
    "right_lung": "pulmón derecho",
    "lv_cavity": "cavidad VI",
    "myocardium": "miocardio",
    "rv_cavity": "cavidad VD",
}

_STRINGS = {
    "en": {
        "title": "Few-shot Dice vs K  |  {dataset_part}  |  {variant}",
        "xlabel": "K (number of reference frames)",
        "ylabel_detected": "Dice (detected only)",
        "ylabel_missing": "Dice (with missing)",
        "ref": "ref {val:.2f}",
        "ref_legend": "{organ} ref = {val:.2f}",
        "mode_labels": _MODE_LABELS,
        "suffix_labels": _SUFFIX_LABELS,
        "organ_labels": {},
        "variant_labels": {},
    },
    "es": {
        "title": "Dice few-shot vs K  |  {dataset_part}  |  {variant}",
        "xlabel": "K (número de frames de referencia)",
        "ylabel_detected": "Dice (solo detectados)",
        "ylabel_missing": "Dice (con faltantes)",
        "ref": "ref {val:.2f}",
        "ref_legend": "{organ} ref = {val:.2f}",
        "mode_labels": _MODE_LABELS_ES,
        "suffix_labels": _SUFFIX_LABELS_ES,
        "organ_labels": _ORGAN_LABELS_ES,
        "variant_labels": {
            "detected_only": "solo detectadas",
            "with_missing": "con faltantes",
        },
    },
}


def _organ_label(organ: str, strings: dict) -> str:
    return strings["organ_labels"].get(organ, organ)


def parse_experiment_name(name: str) -> tuple[str, str | None, int]:
    """
    Parse an 'fs_*' experiment name into (mode, suffix, K).

    Raises ValueError with the offending name if the pattern does not match.
    Caller is responsible for only passing names that start with 'fs_'.
    """
    m = _FS_PATTERN.match(name)
    if m is None:
        raise ValueError(
            f"Experiment name {name!r} starts with 'fs_' but does not match "
            "expected pattern 'fs_(indep|iter)[_(refine|baseline)]_{K}ref'. "
            "Update _FS_PATTERN or rename the directory."
        )
    return m.group(1), m.group(2), int(m.group(3))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_fewshot_data(
    results_dir: Path,
    variant: str,
) -> dict[tuple[str, str | None], dict[int, dict[str, float]]]:
    """
    Scan results_dir for 'fs_*' experiment directories and collect per-organ Dice.

    Returns
    -------
    data : {(mode, suffix): {K: {organ: dice}}}
        Only entries where the metric key is present in summary.json are kept.
    """
    metric_key = f"dice_mean_{variant}"
    data: dict = {}

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        if not name.startswith("fs_"):
            continue  # non-few-shot experiments are out of scope

        summary_path = exp_dir / "summary.json"
        if not summary_path.exists():
            print(f"  Warning: {name}/ has no summary.json — skipped", file=sys.stderr)
            continue

        mode, suffix, k = parse_experiment_name(name)  # raises on bad names

        with open(summary_path) as f:
            summary = json.load(f)

        organ_dice: dict[str, float] = {
            organ: organ_data[metric_key]
            for organ, organ_data in summary.get("per_organ", {}).items()
            if metric_key in organ_data and organ_data[metric_key] is not None
        }

        data.setdefault((mode, suffix), {})[k] = organ_dice

    return data


# ---------------------------------------------------------------------------
# Reference line parsing
# ---------------------------------------------------------------------------

def parse_reference_lines(refs: list[str] | None) -> dict[str, float]:
    """
    Parse ['organ=value', ...] into {organ: float}.

    Raises ValueError if any entry is not in 'organ=value' form.
    """
    if not refs:
        return {}
    result: dict[str, float] = {}
    for entry in refs:
        if "=" not in entry:
            raise ValueError(
                f"--reference-lines entry {entry!r} must be 'organ=value', "
                f"e.g. lv_cavity=0.89"
            )
        organ, _, raw = entry.partition("=")
        result[organ.strip()] = float(raw.strip())
    return result


# ---------------------------------------------------------------------------
# Subplot helpers
# ---------------------------------------------------------------------------

def _organ_colors(organs: list[str]) -> dict[str, tuple]:
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(organs), 1))
    return {organ: cmap(i) for i, organ in enumerate(sorted(organs))}


def _ref_hlines(
    ax: plt.Axes,
    present_organs: list[str],
    ref_lines: dict[str, float],
    colors: dict[str, tuple],
    strings: dict,
) -> None:
    """Draw horizontal dashed reference lines for organs present in the subplot."""
    for organ, val in ref_lines.items():
        if organ not in present_organs:
            continue
        ax.axhline(val, linestyle="--", linewidth=1.1, alpha=0.65,
                   color=colors.get(organ, "gray"))
        # Right-edge annotation
        ax.annotate(
            strings["ref"].format(val=val),
            xy=(1, val), xycoords=("axes fraction", "data"),
            fontsize=7, va="bottom", ha="right", alpha=0.75,
            color=colors.get(organ, "gray"),
        )


def _draw_lines_subplot(
    ax: plt.Axes,
    all_k: list[int],
    k_data: dict[int, dict[str, float]],
    all_organs: list[str],
    colors: dict[str, tuple],
    ref_lines: dict[str, float],
    ylabel: str,
    ylim: tuple[float, float],
    strings: dict,
) -> None:
    for organ in all_organs:
        pairs = [(k, k_data[k][organ]) for k in all_k if organ in k_data.get(k, {})]
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=5,
                label=_organ_label(organ, strings), color=colors[organ])
    ax.set_xticks(all_k)
    ax.set_xlabel(strings["xlabel"])
    _ref_hlines(ax, all_organs, ref_lines, colors, strings)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)


def _draw_bars_subplot(
    ax: plt.Axes,
    organ_dice: dict[str, float],
    all_organs: list[str],
    colors: dict[str, tuple],
    ref_lines: dict[str, float],
    ylabel: str,
    k: int,
    ylim: tuple[float, float],
    strings: dict,
) -> None:
    present = [o for o in sorted(all_organs) if o in organ_dice]
    x = np.arange(len(present))
    ax.bar(x, [organ_dice[o] for o in present],
           color=[colors[o] for o in present], alpha=0.85, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([_organ_label(o, strings) for o in present],
                        rotation=20, ha="right", fontsize=9)
    ax.set_xlabel(f"K = {k}")
    _ref_hlines(ax, present, ref_lines, colors, strings)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)


def _fill_subplot(
    ax: plt.Axes,
    k_data: dict[int, dict[str, float]],
    all_organs: list[str],
    colors: dict[str, tuple],
    ref_lines: dict[str, float],
    ylabel: str,
    ylim: tuple[float, float],
    strings: dict,
) -> None:
    if not k_data:
        ax.set_visible(False)
        return
    all_k = sorted(k_data)
    if len(all_k) == 1:
        _draw_bars_subplot(ax, k_data[all_k[0]], all_organs, colors,
                           ref_lines, ylabel, all_k[0], ylim, strings)
    else:
        _draw_lines_subplot(ax, all_k, k_data, all_organs, colors,
                            ref_lines, ylabel, ylim, strings)


def _compute_ylim(
    data: dict[tuple[str, str | None], dict[int, dict[str, float]]],
    ref_lines: dict[str, float],
    pad: float = 0.05,
) -> tuple[float, float]:
    """Tight shared y-limits from all plotted Dice values (+ reference lines)."""
    vals = [
        v
        for series_data in data.values()
        for kd in series_data.values()
        for v in kd.values()
    ]
    vals.extend(ref_lines.values())
    if not vals:
        return (0.0, 1.05)
    lo, hi = min(vals), max(vals)
    span = hi - lo
    margin = pad if span == 0 else pad * span
    lo = max(0.0, lo - margin)
    hi = min(1.05, hi + margin)
    if hi - lo < 1e-3:          # degenerate: give it some breathing room
        lo, hi = max(0.0, lo - 0.05), min(1.05, hi + 0.05)
    return (lo, hi)


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def build_figure(
    data: dict[tuple[str, str | None], dict[int, dict[str, float]]],
    variant: str,
    ref_lines: dict[str, float],
    title: str,
    ylim: tuple[float, float] | None = None,
    lang: str = "en",
) -> plt.Figure:
    """
    Assemble the full figure from loaded data.

    Layout: columns = modes present (indep before iter), rows = suffixes present
    (None → refine → baseline, in that order).
    """
    strings = _STRINGS[lang]
    modes   = [m for m in ("indep", "iter") if any(k[0] == m for k in data)]
    suffixes = [s for s in _SUFFIX_ORDER if any(k[1] == s for k in data)]

    if not modes or not suffixes:
        raise ValueError("No plottable data found.")

    all_organs = sorted({
        o
        for series_data in data.values()
        for kd in series_data.values()
        for o in kd
    })
    colors = _organ_colors(all_organs)
    ylabel = (strings["ylabel_detected"]
              if variant == "detected_only" else strings["ylabel_missing"])
    if ylim is None:
        ylim = _compute_ylim(data, ref_lines)

    n_rows, n_cols = len(suffixes), len(modes)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 4 * n_rows),
                             squeeze=False)

    for row, suffix in enumerate(suffixes):
        for col, mode in enumerate(modes):
            ax = axes[row][col]
            label = strings["mode_labels"][mode] + strings["suffix_labels"].get(suffix, f" ({suffix})")
            ax.set_title(label, fontsize=10, fontweight="bold")
            _fill_subplot(ax, data.get((mode, suffix), {}),
                          all_organs, colors, ref_lines, ylabel, ylim, strings)

    # Figure-level legend: organ lines (solid) + reference lines (dashed)
    handles = [
        mlines.Line2D([], [], color=colors[o], marker="o",
                      linewidth=1.8, markersize=5, label=_organ_label(o, strings))
        for o in all_organs
    ]
    for organ, val in ref_lines.items():
        if organ in colors:
            handles.append(
                mlines.Line2D([], [], color=colors[organ], linestyle="--",
                              linewidth=1.1,
                              label=strings["ref_legend"].format(
                                  organ=_organ_label(organ, strings), val=val))
            )

    if handles:
        fig.legend(handles=handles, loc="lower center",
                   ncol=min(len(handles), 6),
                   bbox_to_anchor=(0.5, 0.0), fontsize=9, framealpha=0.85)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.06 if handles else 0, 1, 1])
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Dice vs K for few-shot experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="Path to the version/dataset directory containing fs_* subdirs",
    )
    parser.add_argument(
        "--variant", default="detected_only",
        choices=["detected_only", "with_missing"],
        help="Which Dice variant to plot",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output figure path (e.g. plots/fewshot_k.png)",
    )
    parser.add_argument(
        "--reference-lines", nargs="+", default=None, metavar="ORGAN=VALUE",
        help=(
            "Draw horizontal dashed reference lines, e.g.: "
            "lv_cavity=0.89 myocardium=0.75 "
            "(annotated as external context, not a strict threshold)"
        ),
    )
    parser.add_argument(
        "--ylim", nargs=2, type=float, default=None, metavar=("MIN", "MAX"),
        help=(
            "Fixed y-axis range, e.g. --ylim 0.5 1.0. "
            "If omitted, limits are auto-fitted tightly to the data."
        ),
    )
    parser.add_argument(
        "--lang", default="en", choices=["en", "es"],
        help="Language for all figure text (titles, axes, legend).",
    )
    parser.add_argument(
        "--dataset-label", default=None,
        help="Display name for the dataset in the title (default: results-dir folder name).",
    )
    parser.add_argument(
        "--hide-version", action="store_true",
        help="Omit the 'version/' prefix (results-dir parent folder name) from the title.",
    )
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        raise FileNotFoundError(f"--results-dir not found: {args.results_dir}")

    data = load_fewshot_data(args.results_dir, args.variant)
    if not data:
        print(f"No few-shot experiments found under {args.results_dir}",
              file=sys.stderr)
        sys.exit(1)

    ref_lines = parse_reference_lines(args.reference_lines)

    dataset  = args.dataset_label or args.results_dir.name
    dataset_part = dataset if args.hide_version else f"{args.results_dir.parent.name}/{dataset}"
    variant_label = _STRINGS[args.lang]["variant_labels"].get(args.variant, args.variant)
    title    = _STRINGS[args.lang]["title"].format(
        dataset_part=dataset_part, variant=variant_label)

    ylim = tuple(args.ylim) if args.ylim is not None else None
    fig = build_figure(data, args.variant, ref_lines, title, ylim, lang=args.lang)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

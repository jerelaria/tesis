#!/usr/bin/env python3
"""
build_results_table.py
----------------------
Assemble the main per-organ x method results table for the unsupervised
results section (6.1) from per-experiment summary.json files.

Layout: methods are columns, (organ x metric) are grouped rows. This makes the
method comparison immediate (read across a row) and lets the best method per
metric be bold-faced. One table per dataset, plus a compact global panoptic
table (PQ / SQ / RQ).

All quality numbers come from the two-variant aggregation written by
project/evaluation/aggregation.py:
  <key>_mean_detected_only : mask quality conditional on detection
  <key>_mean_with_missing  : end-to-end quality, missing organs penalised
SQ is the detected-only mean IoU. RQ is the F1 at the chosen matching IoU
threshold. PQ = SQ * RQ. RQ and PQ are recomputed here at --match-iou so the
table stays consistent with the relaxed-threshold regime, regardless of the
(currently hard-coded 0.5) value used inside add_panoptic_quality().

Per-organ RQ/PQ are intentionally not reported: per-organ precision is
undefined under greedy matching (synthetic prediction names). Only recall and
SQ are shown per organ; PQ/SQ/RQ appear in the global block.

Outputs (per dataset):
  <output-dir>/table_<dataset>.tex   LaTeX (requires \\usepackage{booktabs,multirow})
  <output-dir>/table_<dataset>.csv   tidy long-format CSV

Structural problems (missing summary.json) raise immediately. Missing metric
keys inside a present summary degrade to "--" and are reported as warnings.

CPU-only, standard library only.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Editable labels (Spanish, for the thesis). Adjust freely.
# ---------------------------------------------------------------------------

DEFAULT_ORGAN_LABELS = {
    "left_lung": "Pulmón izq.",
    "right_lung": "Pulmón der.",
    "heart": "Corazón",
    "lv_cavity": "Cavidad VI",
    "myocardium": "Miocardio",
    "rv_cavity": "Cavidad VD",
}

# Preferred display order; organs not listed are appended alphabetically.
ORGAN_ORDER = list(DEFAULT_ORGAN_LABELS.keys())


# Per-organ metric rows.
#   label, base_key, is_distance, higher_is_better, variant_sensitive
# variant_sensitive=True means the key takes a _detected_only / _with_missing
# suffix; recall and SQ are handled specially and are not variant-sensitive.
PER_ORGAN_METRICS = [
    ("Recall", "__recall__",            False, True,  False),
    ("Dice",   "dice_mean",             False, True,  True),
    ("HD95",   "hausdorff_95_mean",     True,  False, True),
    ("ASSD",   "assd_mean",             True,  False, True),
    ("SQ",     "iou_mean_detected_only", False, True, False),
]

EPS = 1e-9


# ---------------------------------------------------------------------------
# Key resolution and loading
# ---------------------------------------------------------------------------

def find_threshold_key(d: dict, prefix: str, thr: float) -> str | None:
    """
    Find the key '<prefix>@<num>' in d whose numeric threshold is closest to
    thr. Returns None if no such key exists. Handles 0.1 vs 0.10 vs 0.5 etc.
    """
    pat = re.compile(rf"^{re.escape(prefix)}@([0-9]*\.?[0-9]+)")
    best_key, best_diff = None, None
    for k in d:
        m = pat.match(k)
        if m:
            diff = abs(float(m.group(1)) - thr)
            if best_diff is None or diff < best_diff:
                best_key, best_diff = k, diff
    return best_key


def load_summary(path: Path) -> dict:
    """Load a summary.json. Raises FileNotFoundError if absent."""
    if not path.exists():
        raise FileNotFoundError(
            f"summary.json not found: {path}\n"
            f"Run evaluate / reevaluate_all for this experiment first."
        )
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Value extraction
# ---------------------------------------------------------------------------

def organ_value(summary: dict, organ: str, base_key: str, variant: str,
                thr: float, warnings: list[str]) -> float | None:
    """
    Resolve a single (organ, metric) value from a summary.

    variant is 'detected_only' or 'with_missing' and is applied only to
    variant-sensitive keys (those ending in _mean here).
    """
    per_organ = summary.get("per_organ", {})
    if organ not in per_organ:
        return None
    d = per_organ[organ]

    if base_key == "__recall__":
        rk = find_threshold_key(d, "recall", thr)
        if rk is None:
            warnings.append(f"recall@~{thr} missing for organ '{organ}'")
            return None
        return d.get(rk)

    if base_key.endswith("_mean"):
        key = f"{base_key}_{variant}"
    else:
        key = base_key  # e.g. iou_mean_detected_only (already fully qualified)

    if key not in d:
        warnings.append(f"key '{key}' missing for organ '{organ}'")
        return None
    return d[key]


def global_panoptic(summary: dict, thr: float,
                    warnings: list[str]) -> dict[str, float | None]:
    """
    Recompute global PQ / SQ / RQ at the chosen matching threshold, plus
    global detected-only Dice and recall for context.
    """
    g = summary.get("global", {})
    sq = g.get("iou_mean_detected_only")
    f1_key = find_threshold_key(g, "f1", thr)
    rq = g.get(f1_key) if f1_key else None
    if f1_key is None:
        warnings.append(f"f1@~{thr} missing in global block")
    pq = sq * rq if (sq is not None and rq is not None) else None

    rec_key = find_threshold_key(g, "recall", thr)
    recall = g.get(rec_key) if rec_key else None

    return {
        "PQ": pq,
        "SQ": sq,
        "RQ": rq,
        "Dice": g.get("dice_mean_detected_only"),
        "Recall": recall,
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt(value: float | None, is_distance: bool, is_best: bool) -> str:
    if value is None:
        return "--"
    s = f"{value:.2f}" if is_distance else f"{value:.3f}"
    return f"\\textbf{{{s}}}" if is_best else s


def best_indices(values: list[float | None], higher_is_better: bool) -> set[int]:
    present = [(i, v) for i, v in enumerate(values) if v is not None]
    if not present:
        return set()
    best_val = (max if higher_is_better else min)(v for _, v in present)
    return {i for i, v in present if abs(v - best_val) <= EPS}


def organ_label(organ: str) -> str:
    return DEFAULT_ORGAN_LABELS.get(organ, organ.replace("_", " ").capitalize())


def ordered_organs(summaries: list[dict]) -> list[str]:
    present = set()
    for s in summaries:
        present.update(s.get("per_organ", {}).keys())
    ordered = [o for o in ORGAN_ORDER if o in present]
    extras = sorted(o for o in present if o not in ORGAN_ORDER)
    return ordered + extras


# ---------------------------------------------------------------------------
# Row expansion (variant handling)
# ---------------------------------------------------------------------------

def expand_metric_rows(variant_mode: str) -> list[tuple]:
    """
    Expand PER_ORGAN_METRICS into concrete rows given the variant mode.

    variant_mode:
      'detected_only' -> each variant-sensitive metric uses detected_only
      'with_missing'  -> each variant-sensitive metric uses with_missing
      'both'          -> variant-sensitive metrics become two rows (det / full)

    Returns list of (row_label, base_key, variant, is_distance, higher).
    """
    rows = []
    for label, base_key, is_dist, higher, vsens in PER_ORGAN_METRICS:
        if not vsens:
            rows.append((label, base_key, "detected_only", is_dist, higher))
            continue
        if variant_mode == "both":
            rows.append((f"{label} (det)", base_key, "detected_only", is_dist, higher))
            rows.append((f"{label} (full)", base_key, "with_missing", is_dist, higher))
        else:
            rows.append((label, base_key, variant_mode, is_dist, higher))
    return rows


# ---------------------------------------------------------------------------
# LaTeX builders
# ---------------------------------------------------------------------------

def build_organ_table(dataset: str, organs: list[str],
                      methods: list[tuple[str, dict]], variant_mode: str,
                      thr: float, warnings: list[str]) -> str:
    metric_rows = expand_metric_rows(variant_mode)
    n_methods = len(methods)
    col_spec = "ll" + "r" * n_methods
    head = " & ".join(["Órgano", "Métrica"] + [m[0] for m in methods]) + r" \\"

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{Resultados del modo no supervisado en {dataset}. "
        rf"Recall y SQ mayor es mejor; HD95 y ASSD menor es mejor. "
        rf"Emparejamiento con IoU $\geq$ {thr:g}. En negrita el mejor método por fila.}}",
        rf"\label{{tab:nosup_{dataset.lower()}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        head,
        r"\midrule",
    ]

    for oi, organ in enumerate(organs):
        first = True
        for (row_label, base_key, variant, is_dist, higher) in metric_rows:
            values = [
                organ_value(s, organ, base_key, variant, thr, warnings)
                for _, s in methods
            ]
            best = best_indices(values, higher)
            cells = [fmt(v, is_dist, i in best) for i, v in enumerate(values)]

            org_cell = (
                rf"\multirow{{{len(metric_rows)}}}{{*}}{{{organ_label(organ)}}}"
                if first else ""
            )
            lines.append(" & ".join([org_cell, row_label] + cells) + r" \\")
            first = False

        if oi < len(organs) - 1:
            lines.append(rf"\cmidrule(lr){{1-{2 + n_methods}}}")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def build_global_table(dataset: str, methods: list[tuple[str, dict]],
                      thr: float, warnings: list[str]) -> str:
    panoptic = [(label, global_panoptic(s, thr, warnings)) for label, s in methods]
    n_methods = len(methods)
    col_spec = "l" + "r" * n_methods
    head = " & ".join(["Métrica"] + [m[0] for m in methods]) + r" \\"

    # rows: PQ, SQ, RQ (panoptic), then global Dice and Recall for context
    metric_keys = [
        ("PQ", True), ("SQ", True), ("RQ", True),
        ("Dice", True), ("Recall", True),
    ]

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{Métricas globales en {dataset} (modo no supervisado). "
        rf"PQ $=$ SQ $\times$ RQ, con RQ $=$ F1 e IoU $\geq$ {thr:g}. "
        rf"Mayor es mejor; en negrita el mejor método por fila.}}",
        rf"\label{{tab:nosup_global_{dataset.lower()}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        head,
        r"\midrule",
    ]

    for metric, higher in metric_keys:
        values = [p[metric] for _, p in panoptic]
        best = best_indices(values, higher)
        cells = [fmt(v, False, i in best) for i, v in enumerate(values)]
        lines.append(" & ".join([metric] + cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tidy CSV
# ---------------------------------------------------------------------------

def write_csv(path: Path, dataset: str, organs: list[str],
              methods: list[tuple[str, dict]], variant_mode: str,
              thr: float, warnings: list[str]) -> None:
    metric_rows = expand_metric_rows(variant_mode)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "organ", "metric", "variant", "method", "value"])
        for organ in organs:
            for (row_label, base_key, variant, _is_dist, _higher) in metric_rows:
                for label, s in methods:
                    v = organ_value(s, organ, base_key, variant, thr, warnings)
                    w.writerow([dataset, organ, row_label, variant, label,
                                "" if v is None else f"{v:.6f}"])
        for label, s in methods:
            p = global_panoptic(s, thr, warnings)
            for metric, v in p.items():
                w.writerow([dataset, "GLOBAL", metric, "global", label,
                            "" if v is None else f"{v:.6f}"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def default_methods(results_dir: Path, dataset: str, version: str,
                    baseline_version: str) -> list[tuple[str, Path]]:
    return [
        ("Baseline",
         results_dir / baseline_version / dataset / "unsup_baseline" / "summary.json"),
        ("Prop. indep.",
         results_dir / version / dataset / "unsup_hdbscan_propagation" / "summary.json"),
        ("Prop. iter.",
         results_dir / version / dataset / "unsup_hdbscan_propagation_iter" / "summary.json"),
    ]


def parse_method_overrides(raw: list[str]) -> list[tuple[str, Path]]:
    out = []
    for token in raw:
        if "=" not in token:
            raise ValueError(f"--method expects LABEL=PATH, got: {token!r}")
        label, path = token.split("=", 1)
        out.append((label.strip(), Path(path.strip())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the per-organ x method results table (section 6.1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--results-dir", type=Path, default=Path("results_leaf"))
    ap.add_argument("--dataset", required=True,
                    help="Dataset short name, e.g. SunnybrookNicoSent or XRayNicoSent.")
    ap.add_argument("--version", default="moments",
                    help="Feature version dir for the propagation experiments.")
    ap.add_argument("--baseline-version", default="baseline",
                    help="Version dir holding unsup_baseline.")
    ap.add_argument("--match-iou", type=float, default=0.1,
                    help="Matching IoU threshold for recall and RQ.")
    ap.add_argument("--variant", choices=["detected_only", "with_missing", "both"],
                    default="detected_only",
                    help="Which quality variant(s) to show for Dice/HD95/ASSD.")
    ap.add_argument("--method", action="append", default=[],
                    help="Override a method as LABEL=path/to/summary.json. "
                         "Repeatable; replaces the default three if given.")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    if args.method:
        method_paths = parse_method_overrides(args.method)
    else:
        method_paths = default_methods(
            args.results_dir, args.dataset, args.version, args.baseline_version
        )

    methods: list[tuple[str, dict]] = []
    for label, path in method_paths:
        methods.append((label, load_summary(path)))

    organs = ordered_organs([s for _, s in methods])
    if not organs:
        print("ERROR: no organs found across summaries.", file=sys.stderr)
        sys.exit(1)

    warnings: list[str] = []

    organ_tex = build_organ_table(
        args.dataset, organs, methods, args.variant, args.match_iou, warnings
    )
    global_tex = build_global_table(
        args.dataset, methods, args.match_iou, warnings
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = args.output_dir / f"table_{args.dataset}.tex"
    csv_path = args.output_dir / f"table_{args.dataset}.csv"

    with open(tex_path, "w") as f:
        f.write("% Requires \\usepackage{booktabs,multirow}\n\n")
        f.write(organ_tex)
        f.write("\n")
        f.write(global_tex)

    write_csv(csv_path, args.dataset, organs, methods,
              args.variant, args.match_iou, warnings)

    print(f"Wrote: {tex_path}")
    print(f"Wrote: {csv_path}")
    print(f"Organs: {organs}")
    print(f"Methods: {[m[0] for m in methods]}")
    print(f"Match IoU: {args.match_iou:g}   Variant: {args.variant}")

    if warnings:
        print(f"\n{len(warnings)} warning(s) (cells left as '--'):")
        for w in dict.fromkeys(warnings):  # de-duplicated, order-preserving
            print(f"  - {w}")


if __name__ == "__main__":
    main()
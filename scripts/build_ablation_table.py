#!/usr/bin/env python3
"""
scripts/build_ablation_table.py
--------------------------------
Genera tablas LaTeX de ablación de estandarización a partir de los
diagnosis_report.json producidos por tools/diagnose_clustering.py.

Genera una tabla por dataset Y una tabla combinada con los 3 datasets
agrupados por bloques (ablation_std_all.tex).

Uso:
    python scripts/build_ablation_table.py \
        --ablation-dir results/ablation \
        --output-dir results/tables/ablation
"""
import argparse
import json
from pathlib import Path

# Etiquetas para mostrar en la tabla.
FS_LABELS = {
    "moments_all": "16 momentos",
    "moments_12":  "12 momentos",
    "embeddings_raw": "Embeddings SAM2",
    "as_clustered":  "12 mom. (pipeline)",
}

# Orden de feature sets en la tabla.
FS_ORDER = ["moments_all", "moments_12", "embeddings_raw", "as_clustered"]

# Métricas a mostrar: (clave en quality dict, cabecera LaTeX, formato)
METRICS = [
    ("homogeneity", "Homog.",    "{:.3f}"),
    ("v_measure",   "V-measure", "{:.3f}"),
    ("ari",         "ARI",       "{:.3f}"),
    ("n_clusters",  "$k$",       "{:d}"),
    ("noise_frac",  "Ruido",     "{:.3f}"),
]


def load_report(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _fmt(value, fmt: str) -> str:
    if value is None:
        return "--"
    return fmt.format(value)


def _best(rows_values: list[float | None], higher_is_better: bool) -> set[int]:
    present = [(i, v) for i, v in enumerate(rows_values) if v is not None]
    if not present:
        return set()
    best = (max if higher_is_better else min)(v for _, v in present)
    return {i for i, v in present if abs(v - best) < 1e-9}


# metrics where lower is better
LOWER_BETTER = {"noise_frac"}
NO_BOLD = {"n_clusters"}


def _build_rows(std_report: dict, nostd_report: dict) -> list[tuple[str, str, dict]]:
    """Build (fs_label, estand_label, quality_dict) rows for one dataset."""
    std_sets   = std_report["per_feature_set"]
    nostd_sets = nostd_report["per_feature_set"]
    rows = []
    for fs in FS_ORDER:
        if fs == "as_clustered":
            continue
        label = FS_LABELS.get(fs, fs)
        if fs in nostd_sets:
            q = {**nostd_sets[fs]["quality"], "n_clusters": nostd_sets[fs]["n_clusters"]}
            rows.append((label, "No", q))
        if fs in std_sets:
            q = {**std_sets[fs]["quality"], "n_clusters": std_sets[fs]["n_clusters"]}
            rows.append((label, "Sí", q))
    return rows


def _best_per_metric(rows: list[tuple[str, str, dict]]) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for key, _, _ in METRICS:
        if key in NO_BOLD:
            result[key] = set()
            continue
        values = [float(q.get(key)) if q.get(key) is not None else None
                  for _, _, q in rows]
        result[key] = _best(values, key not in LOWER_BETTER)
    return result


def _render_block(rows: list[tuple[str, str, dict]],
                  best: dict[str, set[int]],
                  row_offset: int = 0,
                  dataset_label: str | None = None) -> list[str]:
    """Render data rows, applying bold based on pre-computed best sets offset by row_offset.

    If dataset_label is given, it is rendered as a leading column, shown
    (italicised) only on the first row of the block and left blank on the
    rest, so the dataset reads as a column instead of a separating row.
    """
    lines = []
    prev_label = None
    for local_idx, (fs_label, estand, q) in enumerate(rows):
        global_idx = row_offset + local_idx
        cells = []
        for key, _, fmt in METRICS:
            val = q.get(key)
            cell = _fmt(val, fmt)
            if global_idx in best[key]:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        if prev_label is not None and fs_label != prev_label:
            lines.append(r"\addlinespace[2pt]")
        label_cell = fs_label if fs_label != prev_label else ""
        row_cells = [label_cell, estand] + cells
        if dataset_label is not None:
            dataset_cell = rf"\textit{{{dataset_label}}}" if local_idx == 0 else ""
            row_cells = [dataset_cell] + row_cells
        lines.append(" & ".join(row_cells) + r" \\")
        prev_label = fs_label
    return lines


def build_table(dataset: str, std_report: dict, nostd_report: dict) -> str:
    rows = _build_rows(std_report, nostd_report)
    best = _best_per_metric(rows)

    col_spec = "ll" + "r" * len(METRICS)
    metric_headers = " & ".join(h for _, h, _ in METRICS)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{Ablación de estandarización en {dataset}. "
        rf"Mayor es mejor en Homog., V-measure y ARI; "
        rf"menor es mejor en Ruido. "
        rf"En negrita el mejor valor por columna.}}",
        rf"\label{{tab:ablation_std_{dataset.lower()}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Feature set & Estand. & {metric_headers} \\",
        r"\midrule",
    ]
    lines.extend(_render_block(rows, best))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def build_combined_table(dataset_reports: list[tuple[str, dict, dict]]) -> str:
    """Build a single table with the dataset as a leading column (not a
    separating block row)."""
    col_spec = "l" + "ll" + "r" * len(METRICS)
    metric_headers = " & ".join(h for _, h, _ in METRICS)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Dataset & Feature set & Estand. & {metric_headers} \\",
        r"\midrule",
    ]

    for i, (dataset, std_report, nostd_report) in enumerate(dataset_reports):
        rows = _build_rows(std_report, nostd_report)
        best = _best_per_metric(rows)
        if i > 0:
            lines.append(r"\midrule")
        lines.extend(_render_block(rows, best, dataset_label=dataset))

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Ablación de estandarización sobre los tres conjuntos de datos, "
        r"evaluada a nivel del agrupamiento intermedio. Se contrasta cada "
        r"representación con y sin estandarización. Mayor es mejor en "
        r"homogeneidad, V-measure y ARI; menor es mejor en la fracción de "
        r"ruido. La columna $k$ indica la cantidad de grupos formados y se "
        r"reporta a título descriptivo. En negrita, el mejor valor de cada "
        r"columna dentro de cada conjunto.}",
        r"\label{tab:ablation_std_all}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablation-dir", type=Path, default=Path("results/ablation"),
                    help="Directorio raíz con subdirs <dataset>/std/ y <dataset>/no_std/.")
    ap.add_argument("--output-dir", type=Path, default=Path("results/tables/ablation"))
    ap.add_argument("--datasets", nargs="+",
                    default=["Sunnybrook", "ACDC", "XRay"])
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaded = []
    for dataset in args.datasets:
        std_path   = args.ablation_dir / dataset / "std"   / "diagnosis_report.json"
        nostd_path = args.ablation_dir / dataset / "no_std" / "diagnosis_report.json"

        if not std_path.exists() or not nostd_path.exists():
            print(f"SKIP {dataset}: falta std o no_std report.")
            continue

        std_report   = load_report(std_path)
        nostd_report = load_report(nostd_path)
        loaded.append((dataset, std_report, nostd_report))

        tex = build_table(dataset, std_report, nostd_report)
        out = args.output_dir / f"ablation_std_{dataset}.tex"
        out.write_text("% Requiere \\usepackage{booktabs}\n\n" + tex)
        print(f"Wrote: {out}")

    if len(loaded) > 1:
        tex = build_combined_table(loaded)
        out = args.output_dir / "ablation_std_all.tex"
        out.write_text("% Requiere \\usepackage{booktabs}\n\n" + tex)
        print(f"Wrote: {out}")


if __name__ == "__main__":
    main()

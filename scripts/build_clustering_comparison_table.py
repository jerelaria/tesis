#!/usr/bin/env python3
"""
scripts/build_clustering_comparison_table.py
--------------------------------------------
Tabla de comparación de representaciones de features para clustering (sec. 6.5.3).

Lee el campo as_clustered de cada diagnosis_report.json y construye una sola
tabla LaTeX con los 3 datasets agrupados por bloques.

Uso:
    python scripts/build_clustering_comparison_table.py \
        --ablation-dir results/ablation \
        --output results/tables/ablation/clustering_repr_all.tex
"""
import argparse
import json
from pathlib import Path

EPS = 1e-9

VARIANTS = [
    ("12 momentos",       "std"),
    ("Emb. SAM2 raw",     "embeddings"),
    ("Emb. SAM2 PCA-16",  "embeddings_red"),
    ("Mom.~+~Emb.",       "emb_momentos"),
]

METRICS = [
    ("homogeneity", "Homog.",    "{:.3f}", True),
    ("v_measure",   "V-measure", "{:.3f}", True),
    ("ari",         "ARI",       "{:.3f}", True),
    ("n_clusters",  "$k$",       "{:d}",   None),   # None = no bold
    ("noise_frac",  "Ruido",     "{:.3f}", False),
]

def bold(s: str) -> str:
    return rf"\textbf{{{s}}}"


def best_idx(values: list, higher: bool) -> set[int]:
    present = [(i, v) for i, v in enumerate(values) if v is not None]
    if not present:
        return set()
    best = (max if higher else min)(v for _, v in present)
    return {i for i, v in present if abs(v - best) < EPS}


def load_rows(dataset: str, ablation_dir: Path) -> list[tuple[str, dict]]:
    rows = []
    for label, subdir in VARIANTS:
        path = ablation_dir / dataset / subdir / "diagnosis_report.json"
        if not path.exists():
            print(f"  MISS: {path}")
            continue
        d = json.load(open(path))
        ac = d["per_feature_set"].get("as_clustered")
        if ac is None:
            print(f"  No as_clustered in {path}")
            continue
        q = {**ac["quality"], "n_clusters": ac["n_clusters"]}
        rows.append((label, q))
    return rows


def render_block(rows: list[tuple[str, dict]], dataset_label: str) -> list[str]:
    """Render data rows for one dataset block, with per-block bold.

    The dataset is rendered as a leading column, shown (italicised) only on
    the first row of the block and left blank on the rest, so the dataset
    reads as a column instead of a separating row.
    """
    best_per_col: dict[str, set[int]] = {}
    for key, _, _, higher in METRICS:
        if higher is None:
            best_per_col[key] = set()
            continue
        vals = [float(v) if (v := q.get(key)) is not None else None
                for _, q in rows]
        best_per_col[key] = best_idx(vals, higher)

    lines = []
    for row_idx, (label, q) in enumerate(rows):
        cells = []
        for key, _, fmt, higher in METRICS:
            val = q.get(key)
            cell = fmt.format(val) if val is not None else "--"
            if higher is not None and row_idx in best_per_col[key]:
                cell = bold(cell)
            cells.append(cell)
        dataset_cell = rf"\textit{{{dataset_label}}}" if row_idx == 0 else ""
        lines.append(" & ".join([dataset_cell, label] + cells) + r" \\")
    return lines


def build_combined_table(datasets: list[str], ablation_dir: Path) -> str:
    col_spec = "l" + "l" + "r" * len(METRICS)
    metric_headers = " & ".join(h for _, h, _, _ in METRICS)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Dataset & Representación & {metric_headers} \\",
        r"\midrule",
    ]

    for i, dataset in enumerate(datasets):
        rows = load_rows(dataset, ablation_dir)
        if not rows:
            continue

        if i > 0:
            lines.append(r"\midrule")

        lines.extend(render_block(rows, dataset))

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Comparación de representaciones del vector de features para "
        r"el agrupamiento, en los tres conjuntos de datos, con todas las "
        r"representaciones estandarizadas. Mayor es mejor en homogeneidad, "
        r"V-measure y ARI; menor es mejor en la fracción de ruido. La columna "
        r"$k$ indica la cantidad de grupos formados y se reporta a título "
        r"descriptivo. En negrita, el mejor valor de cada columna dentro de "
        r"cada conjunto.}",
        r"\label{tab:clustering_repr_all}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablation-dir", type=Path, default=Path("results/ablation"))
    ap.add_argument("--output", type=Path,
                    default=Path("results/tables/ablation/clustering_repr_all.tex"))
    ap.add_argument("--datasets", nargs="+", default=["Sunnybrook", "ACDC", "XRay"])
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tex = build_combined_table(args.datasets, args.ablation_dir)
    args.output.write_text("% Requiere \\usepackage{booktabs}\n\n" + tex)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()

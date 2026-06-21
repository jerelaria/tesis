#!/usr/bin/env python3
"""
Figure 2 - Conceptual schema of the two study axes.

Rows are the feature versions; columns are the supervision/propagation
configurations. Cells that cross both axes (unsupervised clustering x feature
versions) are marked. Configurations that do not depend on the feature axis
(raw baseline and the few-shot regime) are shown as bands spanning all rows,
hatched, and labelled as out of the cross.

Outputs a vector PDF (fig2_ejes_estudio.pdf) ready to include in LaTeX.
Only matplotlib is required. All visible text is in Spanish (thesis language).
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = Path("figuras_out/fig2_ejes_estudio.pdf")

# Axes content.
VERSIONS = ["Momentos (16)", "Embeddings\ncompletos",
            "Embeddings\n(PCA)", "Momentos +\nEmbeddings"]

# (label, kind): "cross" -> crosses the feature axis; "indep" -> independent band.
COLUMNS = [
    ("No supervisado\nProp. independiente", "cross"),
    ("No supervisado\nProp. iterativa", "cross"),
    ("Baseline crudo", "indep"),
    ("Few-shot\nProp. independiente", "indep"),
    ("Few-shot\nProp. iterativa", "indep"),
]

# Visual constants.
NAME_W = 2.6      # width of the left version-name column
COL_W = 2.1       # width of each configuration column
ROW_H = 1.0       # height of each version row
HEAD_H = 1.5      # height of the top header band
GREEN = "#cfe8cf"
GREEN_EDGE = "#2e7d32"
GREY = "#e6e6e6"
GREY_EDGE = "#9e9e9e"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(VERSIONS)
    n_cols = len(COLUMNS)
    body_h = n_rows * ROW_H

    fig, ax = plt.subplots(figsize=(2.0 + 0.9 * n_cols, 1.6 + 0.8 * n_rows))
    ax.set_xlim(0, NAME_W + n_cols * COL_W)
    ax.set_ylim(-1.4, body_h + HEAD_H + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Top-left corner label (axes legend).
    ax.text(NAME_W / 2, body_h + HEAD_H / 2, "Versiones \\ Configuraciones",
            ha="center", va="center", fontsize=8, style="italic", color="#555")

    # Version row labels.
    for r, name in enumerate(VERSIONS):
        y = body_h - (r + 0.5) * ROW_H
        ax.text(NAME_W / 2, y, name, ha="center", va="center", fontsize=9)

    # Column headers + cells.
    for c, (label, kind) in enumerate(COLUMNS):
        x0 = NAME_W + c * COL_W
        # Header.
        ax.add_patch(Rectangle((x0, body_h), COL_W, HEAD_H,
                               facecolor="#f3f3f3", edgecolor="#bdbdbd"))
        ax.text(x0 + COL_W / 2, body_h + HEAD_H / 2, label,
                ha="center", va="center", fontsize=8.5)

        if kind == "cross":
            # One marked cell per version row.
            for r in range(n_rows):
                y0 = body_h - (r + 1) * ROW_H
                ax.add_patch(Rectangle((x0, y0), COL_W, ROW_H,
                                       facecolor=GREEN, edgecolor=GREEN_EDGE,
                                       linewidth=1.0))
                ax.text(x0 + COL_W / 2, y0 + ROW_H / 2, "\u2713",
                        ha="center", va="center", fontsize=13, color=GREEN_EDGE)
        else:
            # Single band spanning all rows.
            ax.add_patch(Rectangle((x0, 0), COL_W, body_h, facecolor=GREY,
                                   edgecolor=GREY_EDGE, hatch="////", linewidth=1.0))
            note = "Una corrida"
            if "Few-shot" in label:
                note = "K \u2208 {1, 3, 5}"
            ax.text(x0 + COL_W / 2, body_h / 2,
                    f"Independiente\ndel eje de features\n({note})",
                    ha="center", va="center", fontsize=8, rotation=0, color="#444")

    # Group brackets on top.
    cross_cols = [c for c, (_, k) in enumerate(COLUMNS) if k == "cross"]
    indep_cols = [c for c, (_, k) in enumerate(COLUMNS) if k == "indep"]
    _bracket(ax, NAME_W + min(cross_cols) * COL_W,
             NAME_W + (max(cross_cols) + 1) * COL_W,
             body_h + HEAD_H + 0.25, "Se cruzan con las versiones")
    _bracket(ax, NAME_W + min(indep_cols) * COL_W,
             NAME_W + (max(indep_cols) + 1) * COL_W,
             body_h + HEAD_H + 0.25, "Fuera del cruce")

    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT.resolve()}")


def _bracket(ax, x0, x1, y, label):
    """Horizontal bracket with a centered label above a group of columns."""
    ax.plot([x0 + 0.1, x1 - 0.1], [y, y], color="#333", linewidth=1.0)
    ax.plot([x0 + 0.1, x0 + 0.1], [y - 0.15, y], color="#333", linewidth=1.0)
    ax.plot([x1 - 0.1, x1 - 0.1], [y - 0.15, y], color="#333", linewidth=1.0)
    ax.text((x0 + x1) / 2, y + 0.18, label, ha="center", va="bottom",
            fontsize=8.5, fontweight="bold")


if __name__ == "__main__":
    main()
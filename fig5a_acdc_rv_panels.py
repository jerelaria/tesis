#!/usr/bin/env python3
"""
fig5a_acdc_rv_panels.py
-----------------------
Figura 5a — paneles cualitativos del VD (rv_cavity) propagado a ACDC.

Una grilla de N cortes de ACDC, 3 columnas por fila:
    original  |  contorno GT del VD  |  contorno de la predicción (Dice en el título)

Los cortes se eligen para cubrir la distribución de Dice de forma honesta:
2 cerca del p90, 2 cerca de la mediana, 1 cerca del p25 (por defecto).

Datos:
    predicciones : results/acdc_transfer/masks/<stem>/rv_cavity.png
    GT           : data/processed/ACDC/masks/<stem>/rv_cavity.png
    imágenes     : data/processed/ACDC/images/<stem>.png
    Dice/corte   : results/acdc_transfer/metrics.csv  (organ == rv_cavity)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

_GT_COLOR = "#21B32B"    # verde
_PRED_COLOR = "#E64C12"  # naranja-rojo


def _load_image(images_dir: Path, stem: str) -> np.ndarray | None:
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        p = images_dir / (stem + ext)
        if p.exists():
            return np.array(Image.open(p).convert("RGB"))
    return None


def _load_mask(path: Path, h: int, w: int) -> np.ndarray | None:
    if not path.exists():
        return None
    m = np.array(Image.open(path).convert("L")) > 127
    if m.shape != (h, w):
        m = np.array(
            Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
        ) > 127
    return m


def select_stems(df: pd.DataFrame, run_dir: Path) -> list[tuple[str, float, str]]:
    """
    Pick stems covering the Dice distribution: 2 ~p90, 2 ~p50, 1 ~p25.
    Returns list of (stem, dice, tag) sorted by Dice descending.
    """
    # Solo cortes con predicción de VD en disco y Dice válido.
    df = df[df["dice"].notna()].copy()
    have_pred = df["image"].map(
        lambda s: (run_dir / s / "rv_cavity.png").exists()
    )
    df = df[have_pred]
    if df.empty:
        raise RuntimeError("No hay cortes rv_cavity con predicción y Dice válido.")

    targets = [
        ("p90", np.percentile(df["dice"], 90), 2),
        ("mediana", np.percentile(df["dice"], 50), 2),
        ("p25", np.percentile(df["dice"], 25), 1),
    ]

    chosen: list[tuple[str, float, str]] = []
    used: set[str] = set()
    for tag, target, k in targets:
        cand = df[~df["image"].isin(used)].copy()
        cand["d"] = (cand["dice"] - target).abs()
        for _, row in cand.sort_values("d").head(k).iterrows():
            chosen.append((row["image"], float(row["dice"]), tag))
            used.add(row["image"])

    chosen.sort(key=lambda t: t[1], reverse=True)
    return chosen


def render_row(axes, stem: str, dice: float, tag: str,
               images_dir: Path, run_dir: Path, gt_dir: Path) -> None:
    ax_img, ax_gt, ax_pred = axes

    img = _load_image(images_dir, stem)
    gt = _load_mask(gt_dir / stem / "rv_cavity.png",
                    *(img.shape[:2] if img is not None else (256, 256)))
    if img is None:
        h, w = gt.shape if gt is not None else (256, 256)
        img = np.full((h, w, 3), 30, dtype=np.uint8)
    h, w = img.shape[:2]
    pred = _load_mask(run_dir / stem / "rv_cavity.png", h, w)

    for ax in axes:
        ax.imshow(img)
        ax.axis("off")

    # Etiqueta de stem + percentil en la columna original.
    ax_img.text(0.03, 0.97, f"{stem}\n[{tag}]", transform=ax_img.transAxes,
                fontsize=7, color="white", ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="#00000088", ec="none"))

    if gt is not None and gt.any():
        ax_gt.contour(gt.astype(float), levels=[0.5],
                      colors=[_GT_COLOR], linewidths=1.8)
    if pred is not None and pred.any():
        ax_pred.contour(pred.astype(float), levels=[0.5],
                        colors=[_PRED_COLOR], linewidths=1.8)

    ax_pred.set_title(f"Dice = {dice:.2f}", fontsize=10, fontweight="bold")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path,
                    default=Path("results/acdc_transfer/masks"))
    ap.add_argument("--metrics", type=Path,
                    default=Path("results/acdc_transfer/metrics.csv"))
    ap.add_argument("--gt-dir", type=Path,
                    default=Path("data/processed/ACDC/masks"))
    ap.add_argument("--images-dir", type=Path,
                    default=Path("data/processed/ACDC/images"))
    ap.add_argument("--organ", default="rv_cavity")
    ap.add_argument("--stems", nargs="*", default=None,
                    help="Stems explícitos (anula la selección por Dice).")
    ap.add_argument("--output", type=Path,
                    default=Path("figuras_out/fig5a_acdc_rv_panels.png"))
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metrics)
    df = df[df["organ"] == args.organ]

    if args.stems:
        dmap = dict(zip(df["image"], df["dice"]))
        rows = [(s, float(dmap.get(s, np.nan)), "manual") for s in args.stems]
    else:
        rows = select_stems(df, args.run_dir)

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(9.5, n * 3.15), squeeze=False)

    for col, hdr in enumerate(["Imagen ACDC", "GT (VD)", "Predicción (VD)"]):
        axes[0][col].set_title(hdr, fontsize=12, fontweight="bold", pad=22)

    for r, (stem, dice, tag) in enumerate(rows):
        render_row(axes[r], stem, dice, tag,
                   args.images_dir, args.run_dir, args.gt_dir)

    gt_line = mlines.Line2D([], [], color=_GT_COLOR, lw=2, label="GT VD")
    pred_line = mlines.Line2D([], [], color=_PRED_COLOR, lw=2, label="Predicción VD")
    fig.legend(handles=[gt_line, pred_line], fontsize=10, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, 0.005))

    dices = [d for _, d, _ in rows if not np.isnan(d)]
    mean_d = float(np.mean(dices)) if dices else 0.0
    fig.suptitle(
        f"VD propagado Sunnybrook → ACDC  (Dice medio mostrado = {mean_d:.2f})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.965])
    fig.savefig(args.output, dpi=190, bbox_inches="tight")
    pdf = args.output.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {args.output}")
    print(f"Guardado: {pdf}")
    print(f"Cortes ({n}):")
    for stem, d, tag in rows:
        print(f"  {stem:38s} Dice={d:.3f}  [{tag}]")


if __name__ == "__main__":
    main()

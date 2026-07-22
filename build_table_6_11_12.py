"""build_table_6_11_12.py

Regenerates results/tables/table_6_11.tex and table_6_12.tex.

No previous script for these tables was recoverable in the repo (checked
git history, bash history, all .py/.ipynb) -- these numbers were originally
computed ad-hoc. This script reconstructs the pipeline in three explicit
stages, matching how the tables' own methodology note describes them:

  1. "Mascara (metodo)" / "Mascara transferida": filter the already-computed
     project/evaluation metrics.csv (raw discovery/transfer masks vs GT) down
     to the SAME test-set image list used to evaluate the landmarks model,
     so mask and landmark numbers are comparable (same images, same GT).

  2. "Landmarks (metodo/GT/transfer/Sunny->ACDC)": rasterize MaskHybridGNet's
     predicted contours (5_Predict.py --landmarks JSON output) and compare
     against the same GT masks, using project/evaluation's own metric
     functions (medpy-real HD95/ASSD after the 2026-07-21 fix). Myocardium
     has no direct landmark group -- MHG predicts "LV epi" (outer wall) and
     "LV endo" (cavity) separately, so myocardium = filled(LV epi) AND NOT
     filled(LV endo), consistent with how the GT myocardium mask itself is
     defined (see table captions).

  3. Assemble both .tex tables in the existing format (no bold; multirow
     per dataset).

Fairness note (mask detection vs landmark prediction): the unsupervised
discovery method sometimes finds nothing for an organ in a given image
(up to ~16% of ACDC's RV cavity test images); MaskHybridGNet never does --
its decoder always emits a contour for every organ. Using "detected only"
for the mask side would let it skip its hardest failures while landmarks
never get that option. Both sides are aggregated as "with_missing" here:
missing detections are penalised with the image diagonal (project/evaluation's
own convention), not excluded, so mask and landmark columns are evaluated
over the identical 100% of the test set.

Usage:
    PYTHONPATH=. python build_table_6_11_12.py
    PYTHONPATH=. python build_table_6_11_12.py --dry-run   # print, don't write
"""
import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from project.evaluation.metrics_paired import compute_quality_metrics
from project.evaluation.aggregation import aggregate_quality

MHG = Path("mhg_workspace")

# ---------------------------------------------------------------------------
# Stage 1: mask metrics filtered to the landmarks test set
# ---------------------------------------------------------------------------

def load_test_ids(test_list_path: Path) -> set[str]:
    return {Path(l.strip()).stem for l in open(test_list_path) if l.strip()}


def aggregate_mask_metrics(metrics_csv: Path, gt_masks_root: Path,
                            test_ids: set[str], organ: str) -> dict:
    """Filter metrics.csv to `organ` + test_ids; aggregate BOTH variants:
    detected_only (missing images excluded) and with_missing (missing
    images penalised with the image diagonal). The mask column shows both
    side by side -- unlike MaskHybridGNet, the discovery method can fail to
    detect an organ at all, so a reader needs to see "quality when it
    works" (det) and "quality once failures are counted" (full/with_missing)
    separately, not just one or the other."""
    dices_det, hd95s_det, assds_det = [], [], []
    dices_full, hd95s_full, assds_full = [], [], []
    n_missing = 0
    for r in csv.DictReader(open(metrics_csv)):
        if r["organ"] != organ:
            continue
        image_id = Path(r["image"]).stem
        if image_id not in test_ids:
            continue
        dice = float(r["dice"])
        dices_full.append(dice)
        missing = not r["pred_name"]
        if missing or r["hausdorff_95"] == "inf" or r["assd"] == "inf":
            n_missing += 1
            gt_path = gt_masks_root / image_id / f"{organ}.png"
            h, w = np.array(Image.open(gt_path)).shape[:2]
            diag = math.sqrt(h**2 + w**2)
            hd95s_full.append(diag)
            assds_full.append(diag)
        else:
            dices_det.append(dice)
            hd95_v, assd_v = float(r["hausdorff_95"]), float(r["assd"])
            hd95s_det.append(hd95_v)
            assds_det.append(assd_v)
            hd95s_full.append(hd95_v)
            assds_full.append(assd_v)
    n = len(dices_full)
    if n == 0:
        return dict(n=0, n_missing=0, dice_det=None, hd95_det=None, assd_det=None,
                    dice_full=None, hd95_full=None, assd_full=None)
    return dict(
        n=n, n_missing=n_missing,
        dice_det=float(np.mean(dices_det)) if dices_det else None,
        hd95_det=float(np.mean(hd95s_det)) if hd95s_det else None,
        assd_det=float(np.mean(assds_det)) if assds_det else None,
        dice_full=float(np.mean(dices_full)),
        hd95_full=float(np.mean(hd95s_full)),
        assd_full=float(np.mean(assds_full)),
    )


# ---------------------------------------------------------------------------
# Stage 2: landmark metrics (rasterize predicted contours vs GT masks)
# ---------------------------------------------------------------------------

def rasterize(contour, shape) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def get_pred_mask(pred_landmarks: dict, organ_spec, shape):
    """organ_spec: a single JSON key, or ('outer_key', 'inner_key') for a
    derived ring organ (outer filled minus inner filled). Returns None if
    the required key(s) are absent from this image's prediction."""
    if isinstance(organ_spec, tuple):
        outer_key, inner_key = organ_spec
        if outer_key not in pred_landmarks or inner_key not in pred_landmarks:
            return None
        outer = rasterize(pred_landmarks[outer_key], shape)
        inner = rasterize(pred_landmarks[inner_key], shape)
        return outer & ~inner
    if organ_spec not in pred_landmarks:
        return None
    return rasterize(pred_landmarks[organ_spec], shape)


def aggregate_landmark_metrics(gt_masks_root: Path, pred_dir: Path,
                                test_ids: set[str], gt_organ: str,
                                organ_spec) -> dict:
    entries = []
    for image_id in sorted(test_ids):
        pred_path = pred_dir / f"{image_id}.json"
        gt_path = gt_masks_root / image_id / f"{gt_organ}.png"
        if not pred_path.exists() or not gt_path.exists():
            continue
        gt = np.array(Image.open(gt_path)) > 0
        pred_landmarks = json.load(open(pred_path))
        pred_mask = get_pred_mask(pred_landmarks, organ_spec, gt.shape)
        pred_name = gt_organ if pred_mask is not None else None
        if pred_mask is None:
            pred_mask = np.zeros_like(gt)
        m = compute_quality_metrics(pred_mask, gt)
        m["organ"] = gt_organ
        m["pred_name"] = pred_name
        entries.append(m)
    if not entries:
        return dict(dice=None, hd95=None, assd=None, n=0)
    summary = aggregate_quality(entries)["per_organ"].get(gt_organ, {})
    return dict(
        dice=summary.get("dice_mean_with_missing"),
        hd95=summary.get("hausdorff_95_mean_with_missing"),
        assd=summary.get("assd_mean_with_missing"),
        n=summary.get("count", 0),
    )


# ---------------------------------------------------------------------------
# Dataset / row configuration
# ---------------------------------------------------------------------------
# organ_spec follows MHG's predicted JSON keys: a plain string for a direct
# landmark group, or (outer, inner) for a derived ring (myocardium).

XRAY = dict(
    test_list=MHG / "Dataset/CXRAY/Landmarks_2_10_realGT_v2/test.txt",
    gt_masks=Path("data/processed/XRay/masks"),
    mask_metrics=Path("results/moments/XRay/unsup_hdbscan_propagation_iter/metrics.csv"),
    pred_metodo=MHG / "Results/xray_moments_iter_500k/pred_realGT_full",
    pred_gt=MHG / "Results/xray_gt/pred_realGT_full",
    rows=[
        ("Pulmón izq.", "left_lung", "Left Lung"),
        ("Pulmón der.", "right_lung", "Right Lung"),
    ],
)

SUNNYBROOK = dict(
    test_list=MHG / "Dataset/Sunnybrook/Landmarks_3_10_realGT/test.txt",
    gt_masks=Path("data/processed/Sunnybrook/masks"),
    mask_metrics=Path("results/moments/Sunnybrook/unsup_hdbscan_propagation_iter/metrics.csv"),
    pred_metodo=MHG / "Results/sunnybrook_moments_iter_500k/pred_realGT_full",
    pred_gt=MHG / "Results/sunnybrook_gt/pred_realGT_full",
    rows=[
        ("Cavidad VI", "lv_cavity", "LV endo"),
        ("Miocardio", "myocardium", ("LV epi", "LV endo")),
    ],
)

ACDC_NATIVE = dict(
    test_list=MHG / "Dataset/ACDC/Landmarks_3_10_realGT/test.txt",
    gt_masks=Path("data/processed/ACDC/masks"),
    mask_metrics=Path("results/moments/ACDC/unsup_hdbscan_propagation_iter/metrics.csv"),
    pred_metodo=MHG / "Results/acdc_moments_iter_500k/pred_realGT_full",
    pred_gt=MHG / "Results/acdc_gt/pred_realGT_full",
    rows=[
        ("Cavidad VI", "lv_cavity", "LV endo"),
        ("Miocardio", "myocardium", ("LV epi", "LV endo")),
        ("Cavidad VD", "rv_cavity", "RV cavity"),
    ],
)

ACDC_TRANSFER = dict(
    test_list=MHG / "Dataset/ACDC/Landmarks_3_10_realGT/test.txt",
    gt_masks=Path("data/processed/ACDC/masks"),
    mask_metrics=Path("results/acdc_transfer/metrics.csv"),
    pred_transfer=MHG / "Results/acdc_transfer_500k/pred_realGT_full",
    pred_sunny_to_acdc=MHG / "Results/sunnybrook_moments_iter_500k/pred_ACDC_crossdataset",
    rows=[
        ("Cavidad VI", "lv_cavity", "LV endo"),
        ("Miocardio", "myocardium", ("LV epi", "LV endo")),
        ("Cavidad VD", "rv_cavity", "RV cavity"),
    ],
)


def compute_dataset_rows(cfg: dict) -> list[dict]:
    test_ids = load_test_ids(cfg["test_list"])
    out = []
    for label, gt_organ, organ_spec in cfg["rows"]:
        mask = aggregate_mask_metrics(cfg["mask_metrics"], cfg["gt_masks"], test_ids, gt_organ)
        metodo = aggregate_landmark_metrics(cfg["gt_masks"], cfg["pred_metodo"], test_ids, gt_organ, organ_spec)
        gt = aggregate_landmark_metrics(cfg["gt_masks"], cfg["pred_gt"], test_ids, gt_organ, organ_spec)
        out.append(dict(label=label, mask=mask, metodo=metodo, gt=gt))
    return out


def compute_transfer_rows(cfg: dict, acdc_native_gt_rows: list[dict]) -> list[dict]:
    test_ids = load_test_ids(cfg["test_list"])
    out = []
    for i, (label, gt_organ, organ_spec) in enumerate(cfg["rows"]):
        mask = aggregate_mask_metrics(cfg["mask_metrics"], cfg["gt_masks"], test_ids, gt_organ)
        transfer = aggregate_landmark_metrics(cfg["gt_masks"], cfg["pred_transfer"], test_ids, gt_organ, organ_spec)
        sunny_acdc = aggregate_landmark_metrics(cfg["gt_masks"], cfg["pred_sunny_to_acdc"], test_ids, gt_organ, organ_spec)
        acdc_gt = acdc_native_gt_rows[i]["gt"]  # reuse, same column as table 6.11's ACDC "Landmarks (GT)"
        out.append(dict(label=label, mask=mask, transfer=transfer, sunny_acdc=sunny_acdc, acdc_gt=acdc_gt))
    return out


# ---------------------------------------------------------------------------
# Stage 3: table formatting
# ---------------------------------------------------------------------------

def fmt3(d: dict) -> str:
    if d["dice"] is None:
        return " -- & -- & --"
    return f"{d['dice']:.3f} & {d['hd95']:.2f} & {d['assd']:.2f}"


def fmt_mask6(d: dict) -> str:
    """Mask column: detected_only triplet + with_missing triplet, side by side."""
    def f(v, dec):
        return "--" if v is None else f"{v:.{dec}f}"
    return (f"{f(d['dice_det'], 3)} & {f(d['hd95_det'], 2)} & {f(d['assd_det'], 2)} & "
            f"{f(d['dice_full'], 3)} & {f(d['hd95_full'], 2)} & {f(d['assd_full'], 2)}")


def build_table_6_11(datasets: dict[str, list[dict]]) -> str:
    lines = [
        r"% Requiere \usepackage{booktabs,multirow}",
        r"% Tabla 6.11 -- Mascaras automaticas frente a anotacion humana (JSRT + Sunnybrook + ACDC nativo)",
        r"% Regenerado por build_table_6_11_12.py. Metricas de mascara (metodo) y de landmarks",
        r"% (metodo/GT) agregadas 'with_missing' sobre el mismo test set de landmarks en cada",
        r"% dataset -- las fallas de deteccion de mascara se penalizan con la diagonal de la",
        r"% imagen en vez de excluirse, para ser comparables contra MaskHybridGNet (que nunca",
        r"% tiene 'missing': su decoder siempre emite un contorno por organo).",
        r"% HD95 real (medpy.metric.binary.hd95), no el maximo. Miocardio = LV epi rellena",
        r"% menos LV endo rellena, tanto en landmarks predichos como en la mascara GT.",
        r"% ACDC aqui es el descubrimiento NATIVO (metodo corriendo directo sobre ACDC,",
        r"% sin transferencia desde Sunnybrook) -- el brazo de transferencia esta en la",
        r"% Tabla 6.12.",
        "",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Comparación de calidad de máscara y de landmarks entre el modelo entrenado",
        r"con las máscaras del método (momentos, iterativo) y el modelo entrenado directamente",
        r"sobre anotación humana (GT, cota superior), evaluados sobre el mismo conjunto de test",
        r"en cada dataset. En ACDC, ``método'' es el descubrimiento nativo (sin transferencia).",
        r"Dice mayor es mejor; HD95 y ASSD menor es mejor.}",
        r"\label{tab:mask_vs_landmarks}",
        r"\begin{tabular}{llrrrrrrrrrrrrrr}",
        r"\toprule",
        r" &  & \multicolumn{6}{c}{Máscara (método)} & \multicolumn{3}{c}{Landmarks (método)} & \multicolumn{3}{c}{Landmarks (GT)} \\",
        r"\cmidrule(lr){3-8} \cmidrule(lr){9-11} \cmidrule(lr){12-14}",
        r" &  & \multicolumn{3}{c}{Detectados} & \multicolumn{3}{c}{Con missing} &  &  &  &  &  &  \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}",
        r"Dataset & Órgano & Dice & HD95 & ASSD & Dice & HD95 & ASSD & Dice & HD95 & ASSD & Dice & HD95 & ASSD \\",
        r"\midrule",
    ]
    ds_names = list(datasets.keys())
    for di, ds_name in enumerate(ds_names):
        rows = datasets[ds_name]
        for i, row in enumerate(rows):
            prefix = f"\\multirow{{{len(rows)}}}{{*}}{{{ds_name}}}" if i == 0 else ""
            cells = f"{fmt_mask6(row['mask'])} & {fmt3(row['metodo'])} & {fmt3(row['gt'])}"
            lines.append(f" {prefix} & {row['label']} & {cells} \\\\")
        if di < len(ds_names) - 1:
            lines.append(r"\cmidrule(lr){1-14}")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def build_table_6_12(rows: list[dict]) -> str:
    lines = [
        r"% Requiere \usepackage{booktabs,multirow}",
        r"% Tabla 6.12 -- Transferibilidad del descubrimiento y generalizacion cross-dataset (ACDC)",
        r"% Regenerado por build_table_6_11_12.py. Misma convencion 'with_missing' y misma",
        r"% definicion de miocardio que la Tabla 6.11 (ver ese archivo). La columna",
        r"% Landmarks (ACDC-GT) se reusa sin recalcular: es la misma columna Landmarks (GT)",
        r"% del bloque ACDC de la Tabla 6.11.",
        r"% HD95 real (medpy.metric.binary.hd95), no el maximo.",
        r"% El descubrimiento nativo sobre ACDC (sin transferencia) queda en la Tabla 6.11,",
        r"% junto a JSRT y Sunnybrook.",
        "",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Transferibilidad del descubrimiento no supervisado hacia ACDC: máscara",
        r"transferida desde Sunnybrook, el modelo de landmarks entrenado sobre esa máscara",
        r"transferida, el modelo de Sunnybrook aplicado directamente sobre ACDC sin",
        r"reentrenar (corrimiento de dominio), y el modelo entrenado sobre GT de ACDC (cota",
        r"superior). Dice mayor es mejor; HD95 y ASSD menor es mejor.}",
        r"\label{tab:acdc_transfer}",
        r"\begin{tabular}{lrrrrrrrrrrrrrrr}",
        r"\toprule",
        r" & \multicolumn{6}{c}{Máscara transferida} & \multicolumn{3}{c}{Landmarks (transfer)} & \multicolumn{3}{c}{Landmarks (Sunny→ACDC)} & \multicolumn{3}{c}{Landmarks (ACDC-GT)} \\",
        r"\cmidrule(lr){2-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13} \cmidrule(lr){14-16}",
        r" & \multicolumn{3}{c}{Detectados} & \multicolumn{3}{c}{Con missing} &  &  &  &  &  &  &  &  &  \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"Órgano & Dice & HD95 & ASSD & Dice & HD95 & ASSD & Dice & HD95 & ASSD & Dice & HD95 & ASSD & Dice & HD95 & ASSD \\",
        r"\midrule",
    ]
    for row in rows:
        cells = f"{fmt_mask6(row['mask'])} & {fmt3(row['transfer'])} & {fmt3(row['sunny_acdc'])} & {fmt3(row['acdc_gt'])}"
        lines.append(f"{row['label']} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Print tables, don't write files")
    ap.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    args = ap.parse_args()

    print("Computing JSRT (XRay)...")
    jsrt_rows = compute_dataset_rows(XRAY)
    print("Computing Sunnybrook...")
    sunnybrook_rows = compute_dataset_rows(SUNNYBROOK)
    print("Computing ACDC (native)...")
    acdc_rows = compute_dataset_rows(ACDC_NATIVE)
    print("Computing ACDC transfer...")
    transfer_rows = compute_transfer_rows(ACDC_TRANSFER, acdc_rows)

    table_611 = build_table_6_11({"JSRT": jsrt_rows, "Sunnybrook": sunnybrook_rows, "ACDC": acdc_rows})
    table_612 = build_table_6_12(transfer_rows)

    print("\n" + "=" * 80)
    print(table_611)
    print("=" * 80)
    print(table_612)

    if not args.dry_run:
        (args.output_dir / "table_6_11.tex").write_text(table_611)
        (args.output_dir / "table_6_12.tex").write_text(table_612)
        print(f"\nWrote {args.output_dir}/table_6_11.tex and table_6_12.tex")


if __name__ == "__main__":
    main()

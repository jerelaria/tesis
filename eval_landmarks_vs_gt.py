"""eval_landmarks_vs_gt.py

Evalua predicciones de landmarks de MaskHybridGNet (JSON con contornos por
organo, salida de 5_Predict.py --landmarks) contra las mascaras GT reales
en data/processed/<Dataset>/masks/<id>/<organ>.png, restringido a una lista
de imagenes (el test set de landmarks). Usa las mismas funciones de metricas
que el resto del pipeline (project/evaluation/metrics_paired.py), asi que
el HD95 es el real (medpy-style, percentil 95), no el Hausdorff maximo.

Pensado para llenar las columnas "Landmarks (metodo)" / "Landmarks (GT)" de
la Tabla 6.11 (mask_vs_landmarks), con una fila por organo.

Uso:
    PYTHONPATH=. python eval_landmarks_vs_gt.py \\
        --gt-masks data/processed/XRay/masks \\
        --pred mhg_workspace/Results/xray_moments_iter_500k/pred_realGT_full \\
        --test-list mhg_workspace/Dataset/CXRAY/Landmarks_2_10_realGT_v2/test.txt \\
        --organ-map "Left Lung=left_lung" --organ-map "Right Lung=right_lung"
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from project.evaluation.metrics_paired import compute_quality_metrics
from project.evaluation.aggregation import aggregate_quality


def rasterize(contour, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-masks", required=True, help="data/processed/<Dataset>/masks root")
    ap.add_argument("--pred", required=True, help="5_Predict.py --landmarks output dir")
    ap.add_argument("--test-list", required=True, help="test.txt with one image filename per line")
    ap.add_argument("--organ-map", action="append", required=True,
                     help="PRED_KEY=gt_mask_stem, repeatable. E.g. 'Left Lung=left_lung'")
    args = ap.parse_args()

    organ_map = dict(tok.split("=", 1) for tok in args.organ_map)

    gt_root = Path(args.gt_masks)
    pred_dir = Path(args.pred)
    ids = [Path(l.strip()).stem for l in open(args.test_list) if l.strip()]

    all_results = []
    n_skipped = 0
    for image_id in ids:
        pred_path = pred_dir / f"{image_id}.json"
        if not pred_path.exists():
            n_skipped += 1
            continue
        pred_landmarks = json.load(open(pred_path))

        for pred_key, gt_stem in organ_map.items():
            gt_path = gt_root / image_id / f"{gt_stem}.png"
            if not gt_path.exists():
                continue
            gt_mask = np.array(Image.open(gt_path)) > 0

            if pred_key in pred_landmarks:
                pred_mask = rasterize(pred_landmarks[pred_key], gt_mask.shape)
                pred_name = gt_stem
            else:
                pred_mask = np.zeros_like(gt_mask)
                pred_name = None

            metrics = compute_quality_metrics(pred_mask, gt_mask)
            metrics["organ"] = gt_stem
            metrics["pred_name"] = pred_name
            metrics["image"] = image_id
            all_results.append(metrics)

    if n_skipped:
        print(f"[WARN] {n_skipped}/{len(ids)} test images had no prediction JSON, skipped")

    summary = aggregate_quality(all_results)
    for organ, s in summary["per_organ"].items():
        print(f"{organ}: n={s['count']}  "
              f"dice_det={s['dice_mean_detected_only']:.4f}  "
              f"hd95_det={s['hausdorff_95_mean_detected_only']:.3f}  "
              f"assd_det={s['assd_mean_detected_only']:.3f}")


if __name__ == "__main__":
    main()

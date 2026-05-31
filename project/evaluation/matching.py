"""Mask matching strategies: semantic (by organ name) and Hungarian (by IoU)."""

import re
import numpy as np
from scipy.optimize import linear_sum_assignment

from project.evaluation.metrics_paired import iou_score, compute_quality_metrics


def parse_organ_name(stem: str) -> str:
    """
    Strip a trailing _N (instance number) from a filename stem.

    'lung_1'      -> 'lung'
    'heart_1'     -> 'heart'
    'obj_002'     -> 'obj'
    'cluster_0_1' -> 'cluster_0'
    """
    match = re.match(r"^(.+)_(\d+)$", stem)
    return match.group(1) if match else stem


def match_semantic(
    pred_masks: dict[str, np.ndarray],
    gt_masks: dict[str, np.ndarray],
    match_threshold: float = 0.5,
) -> list[dict]:
    """
    Match predictions to GT by organ name.

    Groups masks by base organ name (e.g., 'lung'), then within each organ
    uses Hungarian matching on IoU to pair instances (lung_1 <-> lung_1).

    A matched pair whose IoU < match_threshold is demoted to missing:
    the GT entry gets pred_name=None and quality computed against an empty
    mask. The prediction that falls below threshold simply goes unused here;
    its false-positive status is captured by the coverage/cleanliness family.

    GT organs with no prediction at all get dice=0, iou=0, hd95=inf.
    """
    gt_by_organ: dict[str, list[tuple[str, np.ndarray]]] = {}
    for name, mask in gt_masks.items():
        gt_by_organ.setdefault(parse_organ_name(name), []).append((name, mask))

    pred_by_organ: dict[str, list[tuple[str, np.ndarray]]] = {}
    for name, mask in pred_masks.items():
        pred_by_organ.setdefault(parse_organ_name(name), []).append((name, mask))

    results: list[dict] = []

    for organ, gt_list in gt_by_organ.items():
        pred_list = pred_by_organ.get(organ, [])

        if not pred_list:
            for gt_name, gt_mask in gt_list:
                h, w = gt_mask.shape
                results.append({
                    "gt_name": gt_name, "pred_name": None, "organ": organ,
                    **compute_quality_metrics(np.zeros((h, w), dtype=bool), gt_mask),
                })
            continue

        cost_matrix = np.zeros((len(gt_list), len(pred_list)))
        for i, (_, gt_mask) in enumerate(gt_list):
            for j, (_, pred_mask) in enumerate(pred_list):
                cost_matrix[i, j] = -iou_score(pred_mask, gt_mask)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_gt: set[int] = set()

        for i, j in zip(row_ind, col_ind):
            gt_name, gt_mask = gt_list[i]
            pred_name, pred_mask = pred_list[j]
            pair_iou = -cost_matrix[i, j]

            if pair_iou >= match_threshold:
                results.append({
                    "gt_name": gt_name, "pred_name": pred_name, "organ": organ,
                    **compute_quality_metrics(pred_mask, gt_mask),
                })
            else:
                h, w = gt_mask.shape
                results.append({
                    "gt_name": gt_name, "pred_name": None, "organ": organ,
                    **compute_quality_metrics(np.zeros((h, w), dtype=bool), gt_mask),
                })
            matched_gt.add(i)

        for i, (gt_name, gt_mask) in enumerate(gt_list):
            if i not in matched_gt:
                h, w = gt_mask.shape
                results.append({
                    "gt_name": gt_name, "pred_name": None, "organ": organ,
                    **compute_quality_metrics(np.zeros((h, w), dtype=bool), gt_mask),
                })

    return results


def match_hungarian(
    pred_masks: dict[str, np.ndarray],
    gt_masks: dict[str, np.ndarray],
    match_threshold: float = 0.5,
) -> list[dict]:
    """
    Match predictions to GT using global Hungarian assignment on IoU.

    Used for unsupervised mode where prediction names (e.g., obj_NNN) carry
    no semantic meaning. Finds the best global GT-to-pred assignment.

    GT organs whose best-matched prediction has IoU < match_threshold are
    treated as missing: pred_name=None, quality computed against an empty mask.
    """
    gt_list = list(gt_masks.items())
    pred_list = list(pred_masks.items())

    if not pred_list:
        return [
            {
                "gt_name": gt_name, "pred_name": None,
                "organ": parse_organ_name(gt_name),
                **compute_quality_metrics(np.zeros(gt_mask.shape, dtype=bool), gt_mask),
            }
            for gt_name, gt_mask in gt_list
        ]

    cost_matrix = np.zeros((len(gt_list), len(pred_list)))
    for i, (_, gt_mask) in enumerate(gt_list):
        for j, (_, pred_mask) in enumerate(pred_list):
            cost_matrix[i, j] = -iou_score(pred_mask, gt_mask)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    results: list[dict] = []
    matched_gt: set[int] = set()

    for i, j in zip(row_ind, col_ind):
        gt_name, gt_mask = gt_list[i]
        pred_name, pred_mask = pred_list[j]
        organ = parse_organ_name(gt_name)

        if -cost_matrix[i, j] >= match_threshold:
            results.append({
                "gt_name": gt_name, "pred_name": pred_name, "organ": organ,
                **compute_quality_metrics(pred_mask, gt_mask),
            })
        else:
            results.append({
                "gt_name": gt_name, "pred_name": None, "organ": organ,
                **compute_quality_metrics(np.zeros(gt_mask.shape, dtype=bool), gt_mask),
            })
        matched_gt.add(i)

    for i, (gt_name, gt_mask) in enumerate(gt_list):
        if i not in matched_gt:
            results.append({
                "gt_name": gt_name, "pred_name": None,
                "organ": parse_organ_name(gt_name),
                **compute_quality_metrics(np.zeros(gt_mask.shape, dtype=bool), gt_mask),
            })

    return results

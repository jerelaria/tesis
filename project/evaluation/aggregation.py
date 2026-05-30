"""Aggregation of per-pair quality metrics across images and organs."""

import numpy as np


_METRICS_KEYS = ["dice", "iou", "hausdorff", "hausdorff_95", "assd"]
_INF_METRICS = {"hausdorff", "hausdorff_95", "assd"}


def aggregate_quality(all_results: list[dict]) -> dict:
    """
    Aggregate per-pair quality metrics per organ and globally.

    HD, HD95, and ASSD entries equal to inf are excluded from mean/std
    but counted toward the 'missing' field.
    """
    by_organ: dict[str, list[dict]] = {}
    for r in all_results:
        by_organ.setdefault(r["organ"], []).append(r)

    summary: dict = {"per_organ": {}, "global": {}}

    for organ, entries in sorted(by_organ.items()):
        organ_summary: dict = {"count": len(entries)}
        for key in _METRICS_KEYS:
            values = [
                e[key] for e in entries
                if not (key in _INF_METRICS and e[key] == float("inf"))
            ]
            if values:
                organ_summary[f"{key}_mean"] = float(np.mean(values))
                organ_summary[f"{key}_std"] = float(np.std(values))
            else:
                organ_summary[f"{key}_mean"] = None
                organ_summary[f"{key}_std"] = None
        organ_summary["missing"] = sum(
            1 for e in entries if e.get("pred_name") is None
        )
        summary["per_organ"][organ] = organ_summary

    for key in _METRICS_KEYS:
        values = [
            r[key] for r in all_results
            if not (key in _INF_METRICS and r[key] == float("inf"))
        ]
        if values:
            summary["global"][f"{key}_mean"] = float(np.mean(values))
            summary["global"][f"{key}_std"] = float(np.std(values))
        else:
            summary["global"][f"{key}_mean"] = None
            summary["global"][f"{key}_std"] = None

    return summary

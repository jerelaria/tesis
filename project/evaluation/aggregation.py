"""Aggregation of per-pair quality metrics across images and organs."""

import numpy as np


_METRICS_KEYS = ["dice", "iou", "hausdorff", "hausdorff_95", "assd"]
_INF_METRICS = {"hausdorff", "hausdorff_95", "assd"}


def _resolve_distance_value(entry: dict, key: str) -> float:
    """Replace an inf distance metric with the image diagonal.

    Missing organs produce inf in metrics_paired.py. At aggregation we
    replace that inf with the image diagonal (a finite worst-case distance)
    so that missing organs are penalised rather than excluded, matching the
    documented methodology. Falls back to inf only if image_diagonal is
    unavailable (older result rows that pre-date this field).
    """
    value = entry[key]
    if key in _INF_METRICS and value == float("inf"):
        return entry.get("image_diagonal", value)
    return value


def aggregate_quality(all_results: list[dict]) -> dict:
    """
    Aggregate per-pair quality metrics per organ and globally.

    For HD, HD95, and ASSD, entries equal to inf (missing organs) are
    replaced by the entry's image_diagonal (finite worst-case distance)
    before computing mean/std, so that missing organs are penalised rather
    than excluded. The 'missing' count is determined separately from
    pred_name and is unaffected by this substitution.
    """
    by_organ: dict[str, list[dict]] = {}
    for r in all_results:
        by_organ.setdefault(r["organ"], []).append(r)

    summary: dict = {"per_organ": {}, "global": {}}

    for organ, entries in sorted(by_organ.items()):
        organ_summary: dict = {"count": len(entries)}
        for key in _METRICS_KEYS:
            values = [_resolve_distance_value(e, key) for e in entries]
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
        values = [_resolve_distance_value(r, key) for r in all_results]
        if values:
            summary["global"][f"{key}_mean"] = float(np.mean(values))
            summary["global"][f"{key}_std"] = float(np.std(values))
        else:
            summary["global"][f"{key}_mean"] = None
            summary["global"][f"{key}_std"] = None

    return summary

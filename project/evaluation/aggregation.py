"""Aggregation of per-pair quality metrics across images and organs."""

import numpy as np


_METRICS_KEYS = ["dice", "iou", "hausdorff", "hausdorff_95", "assd"]
_INF_METRICS = {"hausdorff", "hausdorff_95", "assd"}


def _values_with_missing(entries: list[dict], key: str) -> list[float]:
    """
    Collect metric values for ALL entries, penalising missing organs.

    For distance metrics (hausdorff, hausdorff_95, assd), ``inf`` values
    (produced when an organ is not detected) are replaced by the entry's
    ``image_diagonal``, a finite worst-case distance. Falls back to ``inf``
    if the field is absent (legacy rows that pre-date this field). For
    dice/iou, values are used as-is since missing organs already contribute
    0.0.
    """
    result = []
    for e in entries:
        value = e[key]
        if key in _INF_METRICS and value == float("inf"):
            value = e.get("image_diagonal", value)
        result.append(value)
    return result


def _values_detected_only(entries: list[dict], key: str) -> list[float]:
    """
    Collect metric values for DETECTED entries only (pred_name is not None).

    Missing organs (pred_name is None) are excluded entirely. For distance
    metrics, any ``inf`` that somehow appears among detected entries is also
    skipped — detected predictions should never produce inf, but we guard
    defensively without inventing a diagonal substitution.
    """
    result = []
    for e in entries:
        if e.get("pred_name") is None:
            continue
        value = e[key]
        if key in _INF_METRICS and value == float("inf"):
            continue  # defensive guard; should not occur for detected entries
        result.append(value)
    return result


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    """Return (mean, std) or (None, None) if the value list is empty."""
    if not values:
        return None, None
    arr = np.array(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def aggregate_quality(all_results: list[dict]) -> dict:
    """
    Aggregate per-pair quality metrics per organ and globally.

    For each of the five quality metrics (dice, iou, hausdorff, hausdorff_95,
    assd) two variants are reported:

    ``<key>_mean_with_missing`` / ``<key>_std_with_missing``
        All entries are included. For distance metrics, missing-organ ``inf``
        values are replaced by the entry's ``image_diagonal`` (a finite
        worst-case distance), so that missing organs are penalised rather than
        excluded. Dice/IoU for missing organs already contribute 0.0. This is
        the honest end-to-end metric answering "how well does the full pipeline
        perform, including failure to detect?"

    ``<key>_mean_detected_only`` / ``<key>_std_detected_only``
        Only entries where the organ was actually detected (``pred_name`` is
        not None) are included. Missing organs are excluded entirely. This
        measures mask quality conditional on detection: "given that the model
        found the organ at all, how accurate are the boundaries?" Returns
        None/None for a fully-missing organ (e.g. heart 30/30 undetected).

    The ``missing`` count and coverage metrics are unaffected — ``missing``
    is still determined solely by ``pred_name is None``.
    """
    by_organ: dict[str, list[dict]] = {}
    for r in all_results:
        by_organ.setdefault(r["organ"], []).append(r)

    summary: dict = {"per_organ": {}, "global": {}}

    for organ, entries in sorted(by_organ.items()):
        organ_summary: dict = {"count": len(entries)}
        for key in _METRICS_KEYS:
            wm_vals = _values_with_missing(entries, key)
            do_vals = _values_detected_only(entries, key)
            mean_wm, std_wm = _mean_std(wm_vals)
            mean_do, std_do = _mean_std(do_vals)
            organ_summary[f"{key}_mean_with_missing"]  = mean_wm
            organ_summary[f"{key}_std_with_missing"]   = std_wm
            organ_summary[f"{key}_mean_detected_only"] = mean_do
            organ_summary[f"{key}_std_detected_only"]  = std_do
        organ_summary["missing"] = sum(
            1 for e in entries if e.get("pred_name") is None
        )
        summary["per_organ"][organ] = organ_summary

    for key in _METRICS_KEYS:
        wm_vals = _values_with_missing(all_results, key)
        do_vals = _values_detected_only(all_results, key)
        mean_wm, std_wm = _mean_std(wm_vals)
        mean_do, std_do = _mean_std(do_vals)
        summary["global"][f"{key}_mean_with_missing"]  = mean_wm
        summary["global"][f"{key}_std_with_missing"]   = std_wm
        summary["global"][f"{key}_mean_detected_only"] = mean_do
        summary["global"][f"{key}_std_detected_only"]  = std_do

    return summary


def add_panoptic_quality(summary: dict) -> None:
    """
    Materialise Panoptic Quality (PQ = SQ * RQ) in-place on a summary dict.

    SQ (Segmentation Quality) is taken from ``iou_mean_detected_only`` — the
    mean IoU over detected instances, which matches the panoptic-quality SQ
    definition.

    RQ (Recognition Quality) is taken from ``f1@0.5`` — the F1 score at the
    standard IoU threshold.

    Per-organ RQ and PQ are set to None because ``f1@0.5`` is not available at
    organ level: precision per organ is undefined in greedy matching mode
    (predicted names are synthetic, so no per-organ cleanliness can be
    computed). Only recall is available per organ.

    Modifies ``summary["global"]`` and each entry in ``summary["per_organ"]``
    in-place; adds keys ``"sq"``, ``"rq"``, ``"pq"``.
    """
    g = summary["global"]
    sq = g.get("iou_mean_detected_only")
    rq = g.get("f1@0.5")
    g["sq"] = sq
    g["rq"] = rq
    g["pq"] = sq * rq if (sq is not None and rq is not None) else None

    for organ_data in summary["per_organ"].values():
        organ_data["sq"] = organ_data.get("iou_mean_detected_only")
        organ_data["rq"] = None  # per-organ precision undefined in greedy mode
        organ_data["pq"] = None

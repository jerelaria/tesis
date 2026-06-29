"""
cluster_map.py
--------------
Shared cluster-to-organ mapping utilities for evaluation.

The unsupervised method produces anonymous cluster labels ("cluster_0",
"cluster_1", ...).  A cluster-map is an *evaluation decision* made by the
researcher: it translates anonymous integer cluster IDs into named organs so
that predictions saved on disk can be re-scored with semantic matching against
named GT masks.  It is not inferred by the method itself.

This module centralises:
  - parse_cluster_map:   inline "0=heart 1=left_lung" → {0: 'heart', ...}
  - load_cluster_map:    JSON {"0": "heart"} → {0: 'heart', ...}
  - relabel_predictions: rename / drop on-disk predictions for one image
                         according to a cluster-map.
"""

import json
import re
from pathlib import Path

import numpy as np

# A cluster mapped to this label is excluded from evaluation entirely:
# it does not enter quality, coverage, precision, or mAP.
DISCARD_LABEL = "__discard__"

# Predicted mask stems are saved as "cluster_N" (single instance) or
# "cluster_N_i" (multiple instances, i from 1); see save_predicted_masks.
_STEM_RE = re.compile(r"^cluster_(\d+)(?:_(\d+))?$")


def parse_cluster_map(raw: str) -> dict[int, str]:
    """Parse "0=rv_cavity 1=lv_cavity 2=myocardium" into {0: 'rv_cavity', ...}.

    Raises ValueError on malformed entries.
    """
    result: dict[int, str] = {}
    for token in raw.strip().split():
        if "=" not in token:
            raise ValueError(
                f"Malformed cluster-map token: {token!r}. "
                "Expected format: <int>=<organ_name>, e.g. '0=rv_cavity'."
            )
        lhs, rhs = token.split("=", 1)
        try:
            cid = int(lhs)
        except ValueError:
            raise ValueError(
                f"Cluster ID must be an integer, got: {lhs!r} in token {token!r}"
            )
        organ = rhs.strip()
        if not organ:
            raise ValueError(f"Empty organ name in token: {token!r}")
        result[cid] = organ
    if not result:
        raise ValueError("--cluster-map produced an empty mapping.")
    return result


def load_cluster_map(path: Path) -> dict[int, str]:
    """Load a cluster-map from a JSON file like {"0": "heart", "3": "__discard__"}.

    Keys are converted to int (ValueError if a key is not an integer) and values
    are validated as non-empty strings.  Returns {int: str}.
    """
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"cluster_map JSON must be an object, got: {type(raw).__name__}")

    result: dict[int, str] = {}
    for key, value in raw.items():
        try:
            cid = int(key)
        except (TypeError, ValueError):
            raise ValueError(
                f"cluster_map key must be an integer, got: {key!r} in {path}"
            )
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"cluster_map value for key {key!r} must be a non-empty string, "
                f"got: {value!r} in {path}"
            )
        result[cid] = value.strip()
    if not result:
        raise ValueError(f"cluster_map is empty: {path}")
    return result


def _allocate_indexed(organ: str, taken: set[str]) -> str:
    """Return the first free "organ_k" (k from 1) not already in `taken`."""
    k = 1
    while f"{organ}_{k}" in taken:
        k += 1
    return f"{organ}_{k}"


def relabel_predictions(
    pred_masks: dict[str, np.ndarray],
    pred_scores: dict[str, float],
    cluster_map: dict[int, str],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Relabel one image's predictions according to a cluster-map.

    Three categories per prediction stem:
      (a) cluster mapped to a real organ: rename "cluster_N[_i]" → "<organ>[_i]",
          preserving the instance suffix.
      (b) cluster mapped to DISCARD_LABEL: drop the prediction from both masks
          and scores (excluded from every metric).
      (c) cluster not present in the map, or a stem that does not match
          ^cluster_(\\d+)(?:_(\\d+))?$: keep the stem unchanged.  Under semantic
          matching it matches no GT organ and is a precision false positive.

    Collisions (two predictions resolving to the same organ, or a renamed organ
    colliding with an existing key) are reindexed deterministically to
    "organ_1", "organ_2", ... by (cluster_id, instance); no key is overwritten.

    pred_masks and pred_scores are kept in sync: the same rename is applied to
    both, and a score key without a corresponding mask raises ValueError.
    """
    matched: list[tuple[int, int | None, str, str]] = []  # (cid, inst, stem, organ)
    kept: list[str] = []                                   # category (c)
    discarded: set[str] = set()                            # category (b)

    for stem in pred_masks:
        m = _STEM_RE.match(stem)
        if m is None:
            kept.append(stem)            # category (c): non-cluster stem
            continue
        cid = int(m.group(1))
        inst = int(m.group(2)) if m.group(2) is not None else None
        if cid not in cluster_map:
            kept.append(stem)            # category (c): unmapped cluster
            continue
        organ = cluster_map[cid]
        if organ == DISCARD_LABEL:
            discarded.add(stem)          # category (b)
            continue
        matched.append((cid, inst, stem, organ))  # category (a)

    # rename map old_stem -> new_stem (kept stems map to themselves).
    renames: dict[str, str] = {s: s for s in kept}
    taken: set[str] = set(kept)

    # Group renamed clusters by target organ so multi-cluster organs reindex
    # cleanly to organ_1, organ_2, ... instead of mixing "organ" with "organ_1".
    by_organ: dict[str, list[tuple[int, int | None, str]]] = {}
    for cid, inst, stem, organ in matched:
        by_organ.setdefault(organ, []).append((cid, inst, stem))

    for organ in sorted(by_organ):
        # Stable order: cluster_id, then instance (None first).
        members = sorted(by_organ[organ], key=lambda t: (t[0], -1 if t[1] is None else t[1]))
        if len(members) == 1:
            cid, inst, stem = members[0]
            desired = organ if inst is None else f"{organ}_{inst}"
            new = desired if desired not in taken else _allocate_indexed(organ, taken)
        else:
            members_new = []
            for cid, inst, stem in members:
                new = _allocate_indexed(organ, taken)
                taken.add(new)
                members_new.append((stem, new))
            for stem, new in members_new:
                renames[stem] = new
            continue
        renames[stem] = new
        taken.add(new)

    out_masks: dict[str, np.ndarray] = {}
    for stem, mask in pred_masks.items():
        if stem in discarded:
            continue
        out_masks[renames[stem]] = mask

    out_scores: dict[str, float] = {}
    for stem, score in pred_scores.items():
        if stem in discarded:
            continue
        if stem not in renames:
            raise ValueError(
                f"score key {stem!r} has no corresponding predicted mask; "
                "pred_masks and pred_scores are out of sync."
            )
        out_scores[renames[stem]] = score

    return out_masks, out_scores

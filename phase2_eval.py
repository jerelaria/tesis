"""
phase2_eval.py — Quantitative cluster<->organ alignment for Phase 2 sweeps.

Loads Phase 1 cache, runs clustering for a given config (+overrides), assigns a
GT organ to every object, and prints:
  - cluster x organ confusion matrix (rows: clusters incl noise -1)
  - per-cluster dominant organ + purity
  - homogeneity / completeness / v-measure / ARI on the GT-labeled subset
  - which organs are "captured" by a GOOD cluster (quality-filtered)

Diagnostic only. Does not write artifacts. Use to compare parameter settings.

Example:
  python phase2_eval.py --config configs/experiments/unsup_hdbscan_propagation.yaml \
    --dataset Sunnybrook/images --gt-dir data/processed/Sunnybrook/masks \
    --tag emb_pca16 \
    --override labeler.features=null labeler.embedding.enabled=true \
               labeler.embedding.reduction=pca labeler.embedding.n_components=16
"""
import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from project.core.config_utils import apply_config_overrides
from project.core.logging_setup import setup_logging
from project.data_io.utils import load_image_paths
from project.labeling.clustering import ClusteringLabeler, ClusteringConfig
from project.pipeline.artifacts import load_segmented
from project.pipeline.propagator import PropagationConfig
from project.segmentation.quality import compute_cluster_quality, select_prototypes

# reuse GT helpers + cache locator from phase2_log
from phase2_log import _assign_gt_organs, _dataset_short, _locate_cache

logging.getLogger().setLevel(logging.WARNING)
log = logging.getLogger("eval")
log.setLevel(logging.INFO)
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(_h)
log.propagate = False

_NOISE = -1


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.override:
        apply_config_overrides(cfg, args.override)

    extensions = cfg.get("dataset", {}).get("extensions", ["png"])
    image_paths = load_image_paths(args.dataset, extensions)
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    ds = _dataset_short(args.dataset)
    cache_dir, dir_name = _locate_cache(cfg, image_paths, ds, args.seg_cache_dir)
    if not cache_dir.exists():
        raise SystemExit(f"cache not found: {cache_dir}")

    all_objects, objects_by_image = load_segmented(cache_dir)

    labeler = ClusteringLabeler(ClusteringConfig(**cfg["labeler"]))
    labeler.resolve_adaptive_params(len(image_paths))
    labeler.fit(all_objects)

    labeled_by_image = {p: labeler.label(o) for p, o in objects_by_image.items()}

    # quality filtering / prototypes
    prop_cfg = PropagationConfig(**cfg.get("propagation", {}))
    quality = compute_cluster_quality(labeled_by_image, prop_cfg.quality)
    good = {cid for cid, info in quality.items() if info["good"]}

    # GT organ per object (only on objects we clustered)
    ids = [o.id for o in all_objects]
    obj_by_id = {o.id: o for o in all_objects}
    organs = _assign_gt_organs(ids, obj_by_id, Path(args.gt_dir), iou_min=args.iou_min)
    clusters = np.array([labeler._id_to_cluster.get(i, _NOISE) for i in ids])

    organ_set = sorted(set(organs), key=lambda x: (x == "none", x))
    organ_set_named = [o for o in organ_set if o != "none"]

    # confusion: cluster -> Counter(organ)
    conf = defaultdict(Counter)
    for c, o in zip(clusters, organs):
        conf[c][o] += 1

    tag = args.tag or "run"
    print("\n" + "=" * 78)
    print(f"[{tag}]  config={Path(args.config).name}  overrides={args.override}")
    n_clusters = len({c for c in clusters if c != _NOISE})
    n_noise = int((clusters == _NOISE).sum())
    print(f"objects={len(ids)}  clusters={n_clusters}  noise={n_noise} "
          f"({100*n_noise/len(ids):.1f}%)  good_clusters={sorted(good)}")
    gt_named = sum(1 for o in organs if o != "none")
    print(f"GT-labeled objects: {gt_named}/{len(ids)}  organs={ {o: organs.count(o) for o in organ_set_named} }")

    # confusion table
    header = f"{'cluster':>8} {'n':>6} {'good':>5} " + " ".join(f"{o[:9]:>9}" for o in organ_set) + f"  {'dom':>10} {'purity*':>7}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for c in sorted(conf.keys()):
        counts = conf[c]
        n = sum(counts.values())
        row = f"{c:>8} {n:>6} {('Y' if c in good else ''):>5} "
        row += " ".join(f"{counts.get(o,0):>9}" for o in organ_set)
        # dominant *named* organ + purity over named objects in this cluster
        named = {o: counts.get(o, 0) for o in organ_set_named}
        tot_named = sum(named.values())
        if tot_named > 0:
            dom = max(named, key=named.get)
            pur = named[dom] / tot_named
            row += f"  {dom:>10} {pur:>7.2f}"
        else:
            row += f"  {'-':>10} {'-':>7}"
        print(row)

    # metrics on GT-labeled subset
    mask = np.array([o != "none" for o in organs])
    if mask.sum() > 1:
        from sklearn.metrics import (homogeneity_completeness_v_measure,
                                      adjusted_rand_score)
        y_true = np.array(organs)[mask]
        y_pred = clusters[mask]
        h, comp, v = homogeneity_completeness_v_measure(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)
        print("-" * len(header))
        print(f"On GT-labeled subset (n={mask.sum()}):  "
              f"homogeneity={h:.3f}  completeness={comp:.3f}  v={v:.3f}  ARI={ari:.3f}")

        # organ recall: fraction of each organ's objects that land in a GOOD cluster
        print("Organ -> best GOOD cluster recall (frac of organ objects in that cluster):")
        for org in organ_set_named:
            org_clusters = Counter(clusters[(np.array(organs) == org)])
            tot = sum(org_clusters.values())
            best_good = [(cid, n) for cid, n in org_clusters.items() if cid in good]
            if best_good:
                cid, n = max(best_good, key=lambda x: x[1])
                print(f"   {org:>12}: cluster_{cid} captures {n}/{tot} ({100*n/tot:.0f}%)")
            else:
                in_noise = org_clusters.get(_NOISE, 0)
                print(f"   {org:>12}: NO good cluster  (noise={in_noise}/{tot}={100*in_noise/tot:.0f}%)")
    print("=" * 78)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--gt-dir", required=True)
    p.add_argument("--seg-cache-dir", default="results/_segmentation")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--max-images", type=int)
    p.add_argument("--iou-min", type=float, default=0.10)
    p.add_argument("--tag", default=None)
    main(p.parse_args())

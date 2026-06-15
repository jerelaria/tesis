#!/bin/bash
# ============================================================================
# run_emb_diagnosis.sh
#
# Empirical test of whether SAM2 embeddings work as clustering features on
# Sunnybrook, and (if not) WHY: representation failure vs. density/rarity.
#
# Four groups, cheap -> expensive:
#   A  representation x standardization        (phase2_eval, CPU, seconds)
#   B  min_cluster_size density sweep          (phase2_eval, CPU, seconds)
#   C  V-measure ceiling + separability plots  (diagnose_clustering --sweep)
#   D  UMAP figures for the thesis             (phase2_log)
#
# Storage note:
#   - segmented.npz stores RAW (unstandardized) features/embeddings, so
#     phase2_eval / phase2_log re-cluster honestly with standardize=true|false.
#   - clustering_features.csv is ALREADY standardized; diagnose's as_clustered
#     set comes from it, and diagnose z-scores its own sets internally, so
#     there is no standardize=false knob inside diagnose.
#
# Run everything:        ./run_emb_diagnosis.sh
# Run one group:         ./run_emb_diagnosis.sh A      (or B, C, D)
# Override cache dir:     SEG_CACHE_DIR=results/_segmentation/Sunnybrook/<hash> ./run_emb_diagnosis.sh C
#
# Run this from the repo root, inside the same Python env where phase2_eval
# already works (the (base) conda env you used before).
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DS="Sunnybrook/images"
GT="data/processed/Sunnybrook/masks"
CFG="configs/experiments/unsup_hdbscan_propagation.yaml"
SEG_CACHE_ROOT="results/_segmentation"
FEATURES_CSV="results_leaf/embeddings_raw/Sunnybrook_test/unsup_hdbscan_propagation/phase2_clustering/clustering_features.csv"

OUT="diagnostics"
LOGS="${OUT}/logs"
mkdir -p "${LOGS}"

# Embedding variant carried into Groups B/C/D as "the best embedding candidate".
# After Group A finishes, swap these overrides for whichever variant won.
EMB_BEST_OVR=(labeler.features=null labeler.embedding.enabled=true \
              labeler.embedding.reduction=pca labeler.embedding.n_components=16 \
              labeler.standardize=false)

MOMENTS_OVR=(labeler.embedding.enabled=false labeler.standardize=true)

GROUP="${1:-all}"

# ---------------------------------------------------------------------------
# Helper: run phase2_eval with a tag, tee to its own log
# ---------------------------------------------------------------------------
run_eval() {
    local tag="$1"; shift
    local iou="$1"; shift
    echo ">>> phase2_eval [${tag}] (iou_min=${iou})"
    python phase2_eval.py \
        --config "${CFG}" --dataset "${DS}" --gt-dir "${GT}" \
        --iou-min "${iou}" --tag "${tag}" \
        --override "$@" \
        2>&1 | tee "${LOGS}/eval_${tag}.log"
}

# ===========================================================================
# Group A — representation x standardization
# ===========================================================================
group_A() {
    echo "########## GROUP A: representation x standardization ##########"

    run_eval moments        0.3 "${MOMENTS_OVR[@]}"

    run_eval emb_raw_std    0.3 labeler.features=null labeler.embedding.enabled=true \
                                labeler.embedding.reduction=none labeler.standardize=true

    run_eval emb_raw_nostd  0.3 labeler.features=null labeler.embedding.enabled=true \
                                labeler.embedding.reduction=none labeler.standardize=false

    run_eval emb_pca16_std  0.3 labeler.features=null labeler.embedding.enabled=true \
                                labeler.embedding.reduction=pca labeler.embedding.n_components=16 \
                                labeler.standardize=true

    run_eval emb_pca16_nostd 0.3 labeler.features=null labeler.embedding.enabled=true \
                                 labeler.embedding.reduction=pca labeler.embedding.n_components=16 \
                                 labeler.standardize=false

    run_eval emb_momentos   0.3 labeler.embedding.enabled=true \
                                labeler.embedding.reduction=pca labeler.embedding.n_components=16 \
                                labeler.standardize=true

    # Sensitivity at a looser IoU: lv_cavity is small, many partial hits fall below 0.3.
    run_eval emb_raw_nostd_iou01   0.1 labeler.features=null labeler.embedding.enabled=true \
                                       labeler.embedding.reduction=none labeler.standardize=false
    run_eval emb_pca16_nostd_iou01 0.1 labeler.features=null labeler.embedding.enabled=true \
                                       labeler.embedding.reduction=pca labeler.embedding.n_components=16 \
                                       labeler.standardize=false
}

# ===========================================================================
# Group B — min_cluster_size density sweep
# ===========================================================================
group_B() {
    echo "########## GROUP B: min_cluster_size density sweep ##########"
    for FRAC in 0.10 0.05 0.03 0.02; do
        run_eval "moments_mcs${FRAC}" 0.3 \
            "${MOMENTS_OVR[@]}" "labeler.hdbscan.min_cluster_size_fraction=${FRAC}"

        run_eval "embBEST_mcs${FRAC}" 0.3 \
            "${EMB_BEST_OVR[@]}" "labeler.hdbscan.min_cluster_size_fraction=${FRAC}"
    done
}

# ===========================================================================
# Group C — V-measure ceiling + separability plots (diagnose_clustering)
# ===========================================================================
resolve_cache_dir() {
    if [[ -n "${SEG_CACHE_DIR:-}" ]]; then
        echo "${SEG_CACHE_DIR}"; return
    fi
    local base="${SEG_CACHE_ROOT}/Sunnybrook"
    local dirs=()
    while IFS= read -r d; do dirs+=("$d"); done \
        < <(find "${base}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)

    if [[ "${#dirs[@]}" -eq 0 ]]; then
        echo "ERROR: no segmentation cache under ${base}" >&2; exit 1
    elif [[ "${#dirs[@]}" -gt 1 ]]; then
        echo "ERROR: multiple caches under ${base}; pick one and re-run with" >&2
        printf '       SEG_CACHE_DIR=%s ./run_emb_diagnosis.sh C\n' "${dirs[@]}" >&2
        exit 1
    fi
    echo "${dirs[0]}"
}

group_C() {
    echo "########## GROUP C: diagnose ceiling + separability ##########"
    local cache; cache="$(resolve_cache_dir)"
    echo ">>> using seg-cache: ${cache}"

    local csv_args=()
    if [[ -f "${FEATURES_CSV}" ]]; then
        csv_args=(--features-csv "${FEATURES_CSV}")
    else
        echo "WARNING: ${FEATURES_CSV} not found; running without as_clustered set." >&2
    fi

    python tools/diagnose_clustering.py \
        --seg-cache "${cache}" \
        --gt-dir    "${GT}" \
        "${csv_args[@]}" \
        --out-dir   "${OUT}/diagnose_sunnybrook" \
        --gt-iou-min 0.3 \
        --params    "default:min_samples=4,method=leaf" \
        --params    "embeddings_raw:min_cluster_size=20" \
        --params    "moments_geom:min_cluster_size=20" \
        --k-organs 2 --sweep \
        2>&1 | tee "${LOGS}/diagnose.log"
}

# ===========================================================================
# Group D — UMAP figures for the thesis (phase2_log)
# ===========================================================================
run_log() {
    local name="$1"; shift
    echo ">>> phase2_log [${name}]"
    python phase2_log.py \
        --config "${CFG}" --dataset "${DS}" --gt-dir "${GT}" \
        --output-dir "${OUT}/umap/${name}" \
        --highlight-organ lv_cavity \
        --override "$@" \
        2>&1 | tee "${LOGS}/umap_${name}.log"
}

group_D() {
    echo "########## GROUP D: UMAP figures ##########"
    run_log moments "${MOMENTS_OVR[@]}"

    run_log emb_raw_std labeler.features=null labeler.embedding.enabled=true \
                        labeler.embedding.reduction=none labeler.standardize=true

    run_log emb_best "${EMB_BEST_OVR[@]}"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "${GROUP}" in
    A) group_A ;;
    B) group_B ;;
    C) group_C ;;
    D) group_D ;;
    all) group_A; group_B; group_C; group_D ;;
    *) echo "Unknown group '${GROUP}' (use A, B, C, D, or all)" >&2; exit 1 ;;
esac

echo "Done. Logs under ${LOGS}/ ; figures under ${OUT}/"
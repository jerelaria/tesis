#!/bin/bash
# ============================================================================
# run.sh — full experiment suite (Phase 1 + clustering + propagation)
# ----------------------------------------------------------------------------
# Generates the complete results/ tree in one shot. Mirrors exactly what is
# under results/:
#
#   results/baseline/<dataset>/unsup_baseline                       (3)
#       raw MedSAM2 grid, no clustering/propagation. Feature-independent.
#
#   results/<modo>/<dataset>/unsup_hdbscan_propagation{,_iter}      (24)
#       modo  in {moments, embeddings, embeddings_red, emb_momentos}
#       prop  in {independent (""), iterative ("_iter")}
#
#   results/fs_propagation/<dataset>/fs_{indep,iter}_<K>ref         (12)
#       few-shot human references, K in {1,3,5}, indep + iter.
#       Sunnybrook + XRay only (ACDC has no in-dataset references).
#
#   Datasets: Sunnybrook, XRay, ACDC.
#   Total: 3 + 24 + 12 = 39 runs.
#
# The shared Phase 1 segmentation cache (results/_segmentation) is reused; the
# first run per dataset computes it (--compute-seg-if-missing), the rest hit
# the cache. Propagation / HDBSCAN / quality thresholds are baked into the base
# configs (unsup_hdbscan_propagation*.yaml), so per-modo overrides only touch
# the labeler features + embedding settings.
#
# Evaluation is NOT run here: each leaf only gets its masks/ (+ pipeline
# artifacts). Build cluster maps and evaluate separately afterwards.
#
# Usage:
#   ./run.sh                 # full datasets
#   ./run.sh --max 50        # smoke test on 50 images/dataset
# ============================================================================
set -euo pipefail

# ── Environment (tolerant: skip if the venv layout differs) ─────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${PROJECT_ROOT}/venv/bin/activate" ]]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
    export LD_LIBRARY_PATH="${PROJECT_ROOT}/venv/lib/python3.13/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
fi

# ── CLI ─────────────────────────────────────────────────────────────────────
MAX_IMAGES=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max) MAX_IMAGES="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done
MAX_ARGS=()
[[ -n "$MAX_IMAGES" ]] && MAX_ARGS=(--max-images "$MAX_IMAGES")

# ── Layout ──────────────────────────────────────────────────────────────────
RESULTS_ROOT="results"
SEG_CACHE="${RESULTS_ROOT}/_segmentation"
CONFIG_BASE="configs/experiments/unsup_hdbscan_propagation"
BASELINE_CONFIG="configs/experiments/unsup_baseline.yaml"

DATASETS=(Sunnybrook XRay ACDC)
PROP_SUFFIXES=("" "_iter")          # indep + iterative
MODOS=(moments embeddings embeddings_red emb_momentos)
FS_K=(1 3 5)

# 'moments' feature set: 12 descriptors (all 16 moments minus the
# rotation-sensitive/redundant ecc, hu1, hu2). Matches the base config default.
MOMENTS_FEATURES="V,Cx,Cy,Dx,Dy,L,solidity,extent,compact,hu0,intensity_mean,intensity_std"

# Segmenter params: already baked into the base configs. Kept here so the
# baseline run has a non-empty --override and matches the cache key.
COMMON_OVR=("segmenter.grid_side=6"
            "segmenter.score_threshold=0.5"
            "segmenter.iou_threshold=0.5")

# Few-shot reference image stems per dataset (verified to contain every
# annotated organ). The K-prefix of each list is used for K=1,3,5.
REFS_Sunnybrook=(SCD0000101_IM_0003_0059 SCD0000101_IM_0003_0079 SCD0000101_IM_0003_0099 SCD0000101_IM_0003_0119 SCD0000101_IM_0003_0139)
REFS_XRay=(100469495785351489872749036114751610212_rfyvv7 10155709300728342918543955138521808206_f7cj92 10287653421930576798556842610982533460_vpbhw6 103416378058309979932405295235813040436_1g2pmw 10383960670432673238945376919735423432_hd3moq)

set_modo_overrides() {  # populates the global OVR array
    case "$1" in
        moments)           # 12-moment shape descriptors, no embeddings
            OVR=("labeler.standardize=true"
                 "extractor.features=${MOMENTS_FEATURES}"
                 "labeler.features=${MOMENTS_FEATURES}"
                 "labeler.embedding.enabled=false") ;;
        embeddings)        # raw SAM2 embeddings, no reduction
            OVR=("labeler.standardize=true"
                 "labeler.features=null"
                 "labeler.embedding.enabled=true"
                 "labeler.embedding.reduction=none") ;;
        embeddings_red)    # embeddings reduced with PCA(16)
            OVR=("labeler.standardize=true"
                 "labeler.features=null"
                 "labeler.embedding.enabled=true"
                 "labeler.embedding.reduction=pca"
                 "labeler.embedding.n_components=16") ;;
        emb_momentos)      # moments + embeddings PCA(16)
            OVR=("labeler.standardize=true"
                 "extractor.features=${MOMENTS_FEATURES}"
                 "labeler.features=${MOMENTS_FEATURES}"
                 "labeler.embedding.enabled=true"
                 "labeler.embedding.reduction=pca"
                 "labeler.embedding.n_components=16") ;;
        *) echo "Unknown modo: $1" >&2; exit 1 ;;
    esac
}

# run_experiment <out_dir> <config> <ds> [extra main args...]
# Reads the global RUN_OVR array for the --override list. Evaluation is left to
# a separate post-hoc step (cluster maps + re-evaluation).
run_experiment() {
    local out="$1" config="$2" ds="$3"; shift 3
    local extra=("$@")
    mkdir -p "$(dirname "$out")"

    echo ""
    echo "── RUN  ${out}   ($(date '+%H:%M:%S'))"
    python -m main \
        --config "$config" \
        --dataset "${ds}/images" \
        --output-dir "$out" \
        --seg-cache-dir "$SEG_CACHE" \
        --compute-seg-if-missing \
        ${MAX_ARGS[@]+"${MAX_ARGS[@]}"} \
        ${extra[@]+"${extra[@]}"} \
        --override "${RUN_OVR[@]}"
}

# ── Run ─────────────────────────────────────────────────────────────────────
echo "############################################################"
echo "  Full experiment suite   max=${MAX_IMAGES:-all}"
echo "  Datasets: ${DATASETS[*]}"
echo "  Modos:    ${MODOS[*]}   (prop: indep + iter)"
echo "  Few-shot: K=${FS_K[*]}  (Sunnybrook, XRay)"
echo "  $(date)"
echo "############################################################"

for DS in "${DATASETS[@]}"; do
    # ── 1. Baseline (feature-independent, no propagation) ────────────────────
    RUN_OVR=("${COMMON_OVR[@]}")
    run_experiment "${RESULTS_ROOT}/baseline/${DS}/unsup_baseline" \
        "$BASELINE_CONFIG" "$DS"

    # ── 2. Unsupervised: feature versions x propagation modes ────────────────
    for MODO in "${MODOS[@]}"; do
        set_modo_overrides "$MODO"
        RUN_OVR=("${OVR[@]}" "${COMMON_OVR[@]}")
        for SFX in "${PROP_SUFFIXES[@]}"; do
            run_experiment \
                "${RESULTS_ROOT}/${MODO}/${DS}/unsup_hdbscan_propagation${SFX}" \
                "${CONFIG_BASE}${SFX}.yaml" "$DS"
        done
    done

    # ── 3. Few-shot: indep + iter, swept over K (datasets with refs only) ────
    refs_var="REFS_${DS}[@]"
    if [[ -n "${!refs_var:-}" ]]; then
        all_refs=("${!refs_var}")
        RUN_OVR=("${COMMON_OVR[@]}")
        for K in "${FS_K[@]}"; do
            refs=("${all_refs[@]:0:$K}")
            run_experiment "${RESULTS_ROOT}/fs_propagation/${DS}/fs_indep_${K}ref" \
                configs/experiments/fs_propagation.yaml "$DS" \
                --num-refs "$K" --ref-images "${refs[@]}"
            run_experiment "${RESULTS_ROOT}/fs_propagation/${DS}/fs_iter_${K}ref" \
                configs/experiments/fs_propagation_iter.yaml "$DS" \
                --num-refs "$K" --ref-images "${refs[@]}"
        done
    else
        echo "── SKIP few-shot for ${DS} (no references defined)"
    fi
done

# ── Evaluate the map-independent experiments ────────────────────────────────
# baseline (greedy) and few-shot (semantic) need no cluster_map, so evaluate
# them now. The unsupervised feature versions are left for a later pass once
# the per-experiment cluster_map.json files exist (run reevaluate_all.py then).
echo ""
echo "── EVAL  baseline + few-shot   ($(date '+%H:%M:%S'))"
python reevaluate_all.py --versions baseline fs_propagation

echo ""
echo "############################################################"
echo "  Done at $(date)"
echo "  Results under ${RESULTS_ROOT}/"
echo "  Unsupervised feature versions NOT evaluated — add cluster maps, then:"
echo "    python reevaluate_all.py"
echo "############################################################"

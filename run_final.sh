#!/bin/bash
# ============================================================================
# run_final.sh
#
# Final experiments: full dataset, four feature-representation versions.
#
# Directory layout:  results/{version}/{dataset}/{exp_name}/
# Segmentation cache: results/_segmentation/
#
# Versions:
#   baseline     — clustering disabled; feature-independent experiments only.
#   momentos     — 16 moments + standardize, no embeddings.
#   embeddings   — SAM2 embeddings reduced by PCA (16 components), no moments.
#   emb_momentos — 16 moments + SAM2 embeddings PCA (hybrid).
#
# Cache-key note: the fingerprint covers (dataset, image list, mode,
# propagation_mode, segmenter params, reference stems). Feature and embedding
# settings are NOT part of the key. Furthermore, segmentation_cache.py always
# calls phase1 with force_embeddings=True, so every cache entry already stores
# SAM2 embeddings. All four versions therefore share one segmentation cache
# built in Step 0 — no extra pre-gen is needed for the embedding versions.
#
# Experiment counts:
#   baseline     :  4 exp × 2 datasets =   8
#   momentos     : 14 exp × 2 datasets =  28
#   embeddings   : 14 exp × 2 datasets =  28
#   emb_momentos : 14 exp × 2 datasets =  28
#   ─────────────────────────────────────────
#   Total                              :  92
#
# Per clustering version, breakdown of 14 experiments per dataset:
#   unsup_hdbscan, unsup_hdbscan_refine                          (matching: hungarian)
#   fs_indep_{K}ref, fs_indep_refine_{K}ref  for K in {1, 3, 5} (matching: semantic)
#   fs_iter_{K}ref,  fs_iter_refine_{K}ref   for K in {1, 3, 5} (matching: semantic)
#
# Usage:
#   ./run_final.sh                      # full dataset
#   ./run_final.sh --max 30             # cap at 30 images per dataset (quick test)
#   ./run_final.sh --max 30 2>&1 | tee run_final_test.log
#   ./run_final.sh 2>&1 | tee run_final.log
# ============================================================================

set -euo pipefail

# ── Argument parsing ──────────────────────────────────────────────────────────
# --max N   cap images per dataset to N (quick-test mode); omit for full dataset

MAX_IMAGES=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max) MAX_IMAGES="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

MAX_ARGS=()
if [[ -n "${MAX_IMAGES}" ]]; then
    MAX_ARGS=(--max-images "${MAX_IMAGES}")
    echo "Quick-test mode: --max-images ${MAX_IMAGES}"
else
    echo "Full-dataset mode"
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROJECT_ROOT}/venv/bin/activate"
export LD_LIBRARY_PATH="${PROJECT_ROOT}/venv/lib/python3.13/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_ROOT="results"
SEG_CACHE="${RESULTS_ROOT}/_segmentation"
DATASETS=("XRay" "Sunnybrook")

# Reference images per dataset.
# XRay: first 5 alphabetically; all have heart + left_lung + right_lung.
# Sunnybrook: frames 0059/0079/0099/0119/0139 of SCD0000101_IM_0003;
#             all verified to have lv_cavity + myocardium.
declare -A REFS_1 REFS_3 REFS_5

REFS_1["XRay"]="100469495785351489872749036114751610212_rfyvv7"
REFS_3["XRay"]="100469495785351489872749036114751610212_rfyvv7 \
10155709300728342918543955138521808206_f7cj92 \
10287653421930576798556842610982533460_vpbhw6"
REFS_5["XRay"]="100469495785351489872749036114751610212_rfyvv7 \
10155709300728342918543955138521808206_f7cj92 \
10287653421930576798556842610982533460_vpbhw6 \
103416378058309979932405295235813040436_1g2pmw \
10383960670432673238945376919735423432_hd3moq"

REFS_1["Sunnybrook"]="SCD0000101_IM_0003_0059"
REFS_3["Sunnybrook"]="SCD0000101_IM_0003_0059 \
SCD0000101_IM_0003_0079 \
SCD0000101_IM_0003_0099"
REFS_5["Sunnybrook"]="SCD0000101_IM_0003_0059 \
SCD0000101_IM_0003_0079 \
SCD0000101_IM_0003_0099 \
SCD0000101_IM_0003_0119 \
SCD0000101_IM_0003_0139"

# 16-moment feature set (shared by momentos and emb_momentos)
ALL_MOMENTS="V,Cx,Cy,Dx,Dy,L,ecc,solidity,extent,compact,hu0,hu1,hu2,intensity_mean,intensity_std,orientation"

# Per-version override arrays prepended to every run_main call.
# baseline has no overrides: clustering is already disabled in those configs.
MOMENTOS_OVR=(
    "labeler.standardize=true"
    "extractor.features=${ALL_MOMENTS}"
    "labeler.features=${ALL_MOMENTS}"
    "labeler.embedding.enabled=false"
)
EMBEDDINGS_OVR=(
    "labeler.features=null"
    "labeler.embedding.enabled=true"
    "labeler.embedding.reduction=pca"
    "labeler.embedding.n_components=16"
)
EMB_MOMENTOS_OVR=(
    "labeler.standardize=true"
    "extractor.features=${ALL_MOMENTS}"
    "labeler.features=${ALL_MOMENTS}"
    "labeler.embedding.enabled=true"
    "labeler.embedding.reduction=pca"
    "labeler.embedding.n_components=16"
)

# HDBSCAN overrides — applied on top of version overrides for all hdbscan experiments
H_OVR=(
    "labeler.hdbscan.min_cluster_size_fraction=0.15"
    "labeler.hdbscan.min_samples_fraction=0.03"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

step() {
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  $*"
    echo "  $(date '+%H:%M:%S')"
    echo "════════════════════════════════════════════════════════"
}

done_() { echo "  ✓ done at $(date '+%H:%M:%S')"; }

run_main() {
    # run_main <config> <version> <dataset_short> <out_name> [overrides...] -- [extra_args...]
    #
    # All positional args before -- become --override values.
    # Args after -- are forwarded verbatim to main (e.g. --num-refs, --ref-images).
    local cfg="$1" version="$2" ds="$3" out_name="$4"
    shift 4

    local overrides=() extra_args=() past_sep=false
    for arg in "$@"; do
        if   [[ "$arg" == "--" ]];       then past_sep=true
        elif [[ "$past_sep" == false ]]; then overrides+=("$arg")
        else                                  extra_args+=("$arg")
        fi
    done

    local out_dir="${RESULTS_ROOT}/${version}/${ds}/${out_name}"
    mkdir -p "${RESULTS_ROOT}/${version}/${ds}"
    echo ""
    echo "  → ${version}/${ds}/${out_name}"

    local cmd=(
        python -m main
        --config        "${cfg}"
        --dataset       "${ds}/images"
        --output-dir    "${out_dir}"
        --seg-cache-dir "${SEG_CACHE}"
        --compute-seg-if-missing
        "${MAX_ARGS[@]}"
    )
    [[ ${#overrides[@]} -gt 0 ]] && cmd+=(--override "${overrides[@]}")
    cmd+=("${extra_args[@]}")
    "${cmd[@]}"
}

evaluate() {
    # evaluate <version> <dataset_short> <exp_name> <matching>
    local version="$1" ds="$2" exp="$3" matching="$4"
    local out_dir="${RESULTS_ROOT}/${version}/${ds}/${exp}"
    local gt_dir="data/processed/${ds}/masks"
    echo ""
    echo "  → evaluate: ${version}/${ds}/${exp}  matching=${matching}"
    python evaluate.py \
        --gt       "${gt_dir}" \
        --pred     "${out_dir}/masks" \
        --output   "${out_dir}" \
        --matching "${matching}"
}

# ── Step 0: pre-generate segmentation caches ─────────────────────────────────
#
# 7 cache groups per dataset (14 pre-gen runs total):
#   [1/7] unsup          — covers all unsup experiments in all four versions
#   [2/7] fs_indep K=1   — covers baseline/fs_indep_baseline + clustering/fs_indep + fs_indep_refine
#   [3/7] fs_iter  K=1   — covers clustering/fs_iter + fs_iter_refine
#   [4/7] fs_indep K=3
#   [5/7] fs_iter  K=3
#   [6/7] fs_indep K=5
#   [7/7] fs_iter  K=5

step "STEP 0 — Pre-generate segmentation caches (all datasets)"

for DS in "${DATASETS[@]}"; do
    step "Pre-gen segmentations: ${DS}"

    echo "  [1/7] unsup"
    python -m main \
        --config        configs/experiments/unsup_baseline.yaml \
        --dataset       "${DS}/images" \
        --output-dir    "${RESULTS_ROOT}/_preseg/${DS}_unsup" \
        --seg-cache-dir "${SEG_CACHE}" \
        --segmentation-only \
        "${MAX_ARGS[@]}"

    preseg_idx=2
    for K in 1 3 5; do
        REFS_VAR="REFS_${K}[${DS}]"
        REFS="${!REFS_VAR}"

        echo "  [${preseg_idx}/7] fs_indep K=${K}"
        python -m main \
            --config        configs/experiments/fs_indep_baseline.yaml \
            --dataset       "${DS}/images" \
            --output-dir    "${RESULTS_ROOT}/_preseg/${DS}_fs_indep_K${K}" \
            --seg-cache-dir "${SEG_CACHE}" \
            --segmentation-only \
            --num-refs "${K}" \
            --ref-images ${REFS} \
            "${MAX_ARGS[@]}"
        preseg_idx=$(( preseg_idx + 1 ))

        echo "  [${preseg_idx}/7] fs_iter K=${K}"
        python -m main \
            --config        configs/experiments/fs_iter.yaml \
            --dataset       "${DS}/images" \
            --output-dir    "${RESULTS_ROOT}/_preseg/${DS}_fs_iter_K${K}" \
            --seg-cache-dir "${SEG_CACHE}" \
            --segmentation-only \
            --num-refs "${K}" \
            --ref-images ${REFS} \
            "${MAX_ARGS[@]}"
        preseg_idx=$(( preseg_idx + 1 ))
    done

    done_
done

# ── Step 1: baseline version ──────────────────────────────────────────────────
#
# No feature/embedding overrides: clustering is disabled in these configs.
# Experiments per dataset (4):
#   unsup_baseline
#   fs_indep_baseline_1ref, fs_indep_baseline_3ref, fs_indep_baseline_5ref

step "STEP 1 — baseline version"

for DS in "${DATASETS[@]}"; do
    step "baseline: ${DS}"

    run_main configs/experiments/unsup_baseline.yaml    baseline "${DS}" unsup_baseline
    evaluate                                            baseline "${DS}" unsup_baseline hungarian

    for K in 1 3 5; do
        REFS_VAR="REFS_${K}[${DS}]"
        REFS="${!REFS_VAR}"
        run_main configs/experiments/fs_indep_baseline.yaml baseline "${DS}" "fs_indep_baseline_${K}ref" \
            -- --num-refs "${K}" --ref-images ${REFS}
        evaluate                                             baseline "${DS}" "fs_indep_baseline_${K}ref" semantic
    done

    done_
done

# ── Steps 2-4: clustering versions ───────────────────────────────────────────
#
# Experiments per version/dataset (14):
#   unsup_hdbscan, unsup_hdbscan_refine                             (hungarian)
#   fs_indep_{K}ref, fs_indep_refine_{K}ref  for K in {1, 3, 5}   (semantic)
#   fs_iter_{K}ref,  fs_iter_refine_{K}ref   for K in {1, 3, 5}   (semantic)

STEP_N=2
for VERSION in momentos embeddings emb_momentos; do
    case "${VERSION}" in
        momentos)     VER_OVR=("${MOMENTOS_OVR[@]}") ;;
        embeddings)   VER_OVR=("${EMBEDDINGS_OVR[@]}") ;;
        emb_momentos) VER_OVR=("${EMB_MOMENTOS_OVR[@]}") ;;
    esac

    step "STEP ${STEP_N} — ${VERSION} version"

    for DS in "${DATASETS[@]}"; do
        step "${VERSION}: ${DS}"

        # Unsupervised (2 experiments)
        run_main configs/experiments/unsup_hdbscan.yaml        "${VERSION}" "${DS}" unsup_hdbscan \
            "${VER_OVR[@]}" "${H_OVR[@]}"
        evaluate                                               "${VERSION}" "${DS}" unsup_hdbscan hungarian

        run_main configs/experiments/unsup_hdbscan_refine.yaml "${VERSION}" "${DS}" unsup_hdbscan_refine \
            "${VER_OVR[@]}" "${H_OVR[@]}"
        evaluate                                               "${VERSION}" "${DS}" unsup_hdbscan_refine hungarian

        # Few-shot K=1, 3, 5 (12 experiments)
        for K in 1 3 5; do
            REFS_VAR="REFS_${K}[${DS}]"
            REFS="${!REFS_VAR}"

            run_main configs/experiments/fs_indep.yaml        "${VERSION}" "${DS}" "fs_indep_${K}ref" \
                "${VER_OVR[@]}" -- --num-refs "${K}" --ref-images ${REFS}
            evaluate                                          "${VERSION}" "${DS}" "fs_indep_${K}ref" semantic

            run_main configs/experiments/fs_indep_refine.yaml "${VERSION}" "${DS}" "fs_indep_refine_${K}ref" \
                "${VER_OVR[@]}" -- --num-refs "${K}" --ref-images ${REFS}
            evaluate                                          "${VERSION}" "${DS}" "fs_indep_refine_${K}ref" semantic

            run_main configs/experiments/fs_iter.yaml         "${VERSION}" "${DS}" "fs_iter_${K}ref" \
                "${VER_OVR[@]}" -- --num-refs "${K}" --ref-images ${REFS}
            evaluate                                          "${VERSION}" "${DS}" "fs_iter_${K}ref" semantic

            run_main configs/experiments/fs_iter_refine.yaml  "${VERSION}" "${DS}" "fs_iter_refine_${K}ref" \
                "${VER_OVR[@]}" -- --num-refs "${K}" --ref-images ${REFS}
            evaluate                                          "${VERSION}" "${DS}" "fs_iter_refine_${K}ref" semantic
        done

        done_
    done

    STEP_N=$(( STEP_N + 1 ))
done

# ── Step 5: per-version quick-look plots ──────────────────────────────────────

step "STEP 5 — plot_results.py (per version, per dataset)"

for VERSION in baseline momentos embeddings emb_momentos; do
    for DS in "${DATASETS[@]}"; do
        echo "  → ${VERSION}/${DS}"
        python plot_results.py \
            --results_dir "${RESULTS_ROOT}/${VERSION}/${DS}" \
            --output      "${RESULTS_ROOT}/${VERSION}/${DS}/plots"
    done
done

done_

# ── Step 6: cross-version comparison ─────────────────────────────────────────
#
# Compares the three clustering versions against each other.
# baseline is omitted: its experiment names differ (unsup_baseline,
# fs_indep_baseline_*) and it is referenced separately in the paper.

step "STEP 6 — compare_versions.py"

python compare_versions.py \
    --results_dir    "${RESULTS_ROOT}" \
    --versions       momentos embeddings emb_momentos \
    --datasets       XRay Sunnybrook \
    --output         "${RESULTS_ROOT}/comparison_final" \
    --configs        unsup_hdbscan unsup_hdbscan_refine \
                     fs_indep_1ref fs_indep_refine_1ref \
                     fs_iter_1ref  fs_iter_refine_1ref  \
                     fs_indep_3ref fs_indep_refine_3ref \
                     fs_iter_3ref  fs_iter_refine_3ref  \
                     fs_indep_5ref fs_indep_refine_5ref \
                     fs_iter_5ref  fs_iter_refine_5ref  \
    --feature-versions \
        "momentos:Moments" \
        "embeddings:Embeddings" \
        "emb_momentos:Emb+Moments" \
    --dataset-labels XRay:JSRT Sunnybrook:Sunnybrook \
    --skip-recovery

done_

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ALL DONE  $(date)"
echo "  Results:    ${RESULTS_ROOT}/"
echo "  Comparison: ${RESULTS_ROOT}/comparison_final/"
echo "════════════════════════════════════════════════════════"

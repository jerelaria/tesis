#!/bin/bash
# ============================================================================
# run_tests.sh
#
# Smoke-test suite: 30 images × 2 datasets × 13 experiments.
# Version: v3_extended — all 16 extended moment features + standardization,
#          no SAM2 embeddings.
# Experiments: unsup_baseline, unsup_hdbscan, unsup_hdbscan_refine,
#              fs_indep_baseline (1/3 ref), fs_indep (1/3 ref),
#              fs_indep_refine (1/3 ref), fs_iter (1/3 ref),
#              fs_iter_refine (1/3 ref).
# HDBSCAN override: min_cluster_size_fraction=0.15, min_samples_fraction=0.03
#
# Usage:
#   ./run_tests.sh              (uses defaults below)
#   ./run_tests.sh 2>&1 | tee run_tests.log
#
# All results go under results_tests/v_test/.
# Segmentation caches go under results_tests/_segmentation/.
# ============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROJECT_ROOT}/venv/bin/activate"
export LD_LIBRARY_PATH="${PROJECT_ROOT}/venv/lib/python3.13/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

# ── Config ────────────────────────────────────────────────────────────────────

VERSION="v_test"
RESULTS_ROOT="results_tests"
RESULTS="${RESULTS_ROOT}/${VERSION}"
SEG_CACHE="${RESULTS_ROOT}/_segmentation"
MAX="50"
DATASETS=("XRayNicoSent" "SunnybrookNicoSent")

# Explicit reference images per dataset (dirs under data/few_shot/{dataset}/).
# Only refs that have ALL organ masks are used.
# XRay: all refs have heart+left_lung+right_lung → first 3 alphabetically.
# Sunnybrook: only odd-index frames have both lv_cavity+myocardium → pick 3.
declare -A REFS_1  # K=1: first ref only
declare -A REFS_3  # K=3: first 3 refs
REFS_1["XRayNicoSent"]="100469495785351489872749036114751610212_rfyvv7"
REFS_3["XRayNicoSent"]="100469495785351489872749036114751610212_rfyvv7 10155709300728342918543955138521808206_f7cj92 10287653421930576798556842610982533460_vpbhw6"
REFS_1["SunnybrookNicoSent"]="SCD0000101_IM_0003_0059"
REFS_3["SunnybrookNicoSent"]="SCD0000101_IM_0003_0059 SCD0000101_IM_0003_0079 SCD0000101_IM_0003_0099"

# v3_extended: all 16 moment features + standardization (no embeddings)
ALL_MOMENTS="V,Cx,Cy,Dx,Dy,L,ecc,solidity,extent,compact,hu0,hu1,hu2,intensity_mean,intensity_std,orientation"
V3_OVR=(
    "labeler.standardize=true"
    "extractor.features=${ALL_MOMENTS}"
    "labeler.features=${ALL_MOMENTS}"
)

# HDBSCAN-specific overrides (applied on top of V3_OVR for unsup_hdbscan* only)
H_OVR=(
    "labeler.hdbscan.min_cluster_size_fraction=0.15"
    "labeler.hdbscan.min_samples_fraction=0.03"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

step() { echo ""; echo "════════════════════════════════════════════════════════"; echo "  $*"; echo "  $(date '+%H:%M:%S')"; echo "════════════════════════════════════════════════════════"; }
done_() { echo "  ✓ done at $(date '+%H:%M:%S')"; }

run_main() {
    # run_main <config> <dataset_short> <output_name> [extra_overrides...] -- [extra main args...]
    # V3_OVR (extended moments + standardize) is always prepended to overrides.
    # Extra overrides (e.g. HDBSCAN params) are passed as positional args before --.
    # Anything after -- is forwarded verbatim (e.g. --num-refs N).
    local cfg="$1"
    local ds="$2"
    local out_name="$3"
    shift 3

    local extra_overrides=()
    local extra_args=()
    local past_sep=false
    for arg in "$@"; do
        if [[ "$arg" == "--" ]]; then
            past_sep=true
        elif [[ "$past_sep" == false ]]; then
            extra_overrides+=("$arg")
        else
            extra_args+=("$arg")
        fi
    done

    local all_overrides=("${V3_OVR[@]}" "${extra_overrides[@]}")
    local out_dir="${RESULTS}/${ds}/${out_name}"
    mkdir -p "${RESULTS}/${ds}"
    echo ""
    echo "  → main: ${out_name}  (dataset=${ds})"
    python -m main \
        --config "${cfg}" \
        --dataset "${ds}/images" \
        --output-dir "${out_dir}" \
        --seg-cache-dir "${SEG_CACHE}" \
        --compute-seg-if-missing \
        --max-images "${MAX}" \
        --override "${all_overrides[@]}" \
        "${extra_args[@]}"
}

evaluate() {
    # evaluate <dataset_short> <exp_name> <matching>
    local ds="$1"
    local exp="$2"
    local matching="$3"
    local out_dir="${RESULTS}/${ds}/${exp}"
    local gt_dir="data/processed/${ds}/masks"
    echo ""
    echo "  → evaluate: ${exp}  matching=${matching}"
    python evaluate.py \
        --gt  "${gt_dir}" \
        --pred "${out_dir}/masks" \
        --output "${out_dir}" \
        --matching "${matching}"
}

# ── Step 0: pre-generate segmentations ───────────────────────────────────────
#
# The cache key depends on (dataset, image list, mode, propagation_mode,
# segmenter params, reference stems).  Clustering overrides don't affect it.
# We need one pre-generation per distinct (dataset × segmentation group):
#   • unsup        → unsup_baseline.yaml  (covers all 3 unsup experiments)
#   • fs_indep K=1 → fs_indep_baseline.yaml --num-refs 1
#   • fs_indep K=3 → fs_indep_baseline.yaml --num-refs 3
#   • fs_iter  K=1 → fs_iter.yaml --num-refs 1
#   • fs_iter  K=3 → fs_iter.yaml --num-refs 3

step "STEP 0 — Pre-generate segmentation caches (all datasets)"

for DS in "${DATASETS[@]}"; do
    step "Pre-gen segmentations: ${DS}"

    echo "  [0/5] unsup"
    python -m main \
        --config configs/experiments/unsup_baseline.yaml \
        --dataset "${DS}/images" \
        --output-dir "${RESULTS}/${DS}/_preseg_unsup" \
        --seg-cache-dir "${SEG_CACHE}" \
        --segmentation-only \
        --max-images "${MAX}"

    echo "  [1/5] fs_indep K=1"
    python -m main \
        --config configs/experiments/fs_indep_baseline.yaml \
        --dataset "${DS}/images" \
        --output-dir "${RESULTS}/${DS}/_preseg_fs_indep_1ref" \
        --seg-cache-dir "${SEG_CACHE}" \
        --segmentation-only \
        --num-refs 1 \
        --ref-images ${REFS_1[$DS]} \
        --max-images "${MAX}"

    echo "  [2/5] fs_indep K=3"
    python -m main \
        --config configs/experiments/fs_indep_baseline.yaml \
        --dataset "${DS}/images" \
        --output-dir "${RESULTS}/${DS}/_preseg_fs_indep_3ref" \
        --seg-cache-dir "${SEG_CACHE}" \
        --segmentation-only \
        --num-refs 3 \
        --ref-images ${REFS_3[$DS]} \
        --max-images "${MAX}"

    echo "  [3/5] fs_iter K=1"
    python -m main \
        --config configs/experiments/fs_iter.yaml \
        --dataset "${DS}/images" \
        --output-dir "${RESULTS}/${DS}/_preseg_fs_iter_1ref" \
        --seg-cache-dir "${SEG_CACHE}" \
        --segmentation-only \
        --num-refs 1 \
        --ref-images ${REFS_1[$DS]} \
        --max-images "${MAX}"

    echo "  [4/5] fs_iter K=3"
    python -m main \
        --config configs/experiments/fs_iter.yaml \
        --dataset "${DS}/images" \
        --output-dir "${RESULTS}/${DS}/_preseg_fs_iter_3ref" \
        --seg-cache-dir "${SEG_CACHE}" \
        --segmentation-only \
        --num-refs 3 \
        --ref-images ${REFS_3[$DS]} \
        --max-images "${MAX}"

    done_
done

# ── Step 1-13: run experiments + evaluate ────────────────────────────────────

for DS in "${DATASETS[@]}"; do
    step "STEP 1-3 — Unsupervised experiments: ${DS}"

    # 1. unsup_baseline (raw grid output, no clustering; V3 overrides still applied)
    run_main configs/experiments/unsup_baseline.yaml      "${DS}" unsup_baseline
    evaluate "${DS}" unsup_baseline hungarian

    # 2. unsup_hdbscan  (+HDBSCAN overrides)
    run_main configs/experiments/unsup_hdbscan.yaml       "${DS}" unsup_hdbscan \
        "${H_OVR[@]}"
    evaluate "${DS}" unsup_hdbscan hungarian

    # 3. unsup_hdbscan_refine  (+HDBSCAN overrides)
    run_main configs/experiments/unsup_hdbscan_refine.yaml "${DS}" unsup_hdbscan_refine \
        "${H_OVR[@]}"
    evaluate "${DS}" unsup_hdbscan_refine hungarian

    done_

    step "STEP 4-5 — Few-shot baseline (independent, no pipeline): ${DS}"

    # 4. fs_indep_baseline 1ref
    run_main configs/experiments/fs_indep_baseline.yaml   "${DS}" fs_indep_baseline_1ref \
        -- --num-refs 1 --ref-images ${REFS_1[$DS]}
    evaluate "${DS}" fs_indep_baseline_1ref semantic

    # 5. fs_indep_baseline 3ref
    run_main configs/experiments/fs_indep_baseline.yaml   "${DS}" fs_indep_baseline_3ref \
        -- --num-refs 3 --ref-images ${REFS_3[$DS]}
    evaluate "${DS}" fs_indep_baseline_3ref semantic

    done_

    step "STEP 6-9 — Few-shot independent pipeline: ${DS}"

    # 6. fs_indep 1ref
    run_main configs/experiments/fs_indep.yaml            "${DS}" fs_indep_1ref \
        -- --num-refs 1 --ref-images ${REFS_1[$DS]}
    evaluate "${DS}" fs_indep_1ref semantic

    # 7. fs_indep 3ref
    run_main configs/experiments/fs_indep.yaml            "${DS}" fs_indep_3ref \
        -- --num-refs 3 --ref-images ${REFS_3[$DS]}
    evaluate "${DS}" fs_indep_3ref semantic

    # 8. fs_indep_refine 1ref
    run_main configs/experiments/fs_indep_refine.yaml     "${DS}" fs_indep_refine_1ref \
        -- --num-refs 1 --ref-images ${REFS_1[$DS]}
    evaluate "${DS}" fs_indep_refine_1ref semantic

    # 9. fs_indep_refine 3ref
    run_main configs/experiments/fs_indep_refine.yaml     "${DS}" fs_indep_refine_3ref \
        -- --num-refs 3 --ref-images ${REFS_3[$DS]}
    evaluate "${DS}" fs_indep_refine_3ref semantic

    done_

    step "STEP 10-13 — Few-shot iterative pipeline: ${DS}"

    # 10. fs_iter 1ref
    run_main configs/experiments/fs_iter.yaml             "${DS}" fs_iter_1ref \
        -- --num-refs 1 --ref-images ${REFS_1[$DS]}
    evaluate "${DS}" fs_iter_1ref semantic

    # 11. fs_iter 3ref
    run_main configs/experiments/fs_iter.yaml             "${DS}" fs_iter_3ref \
        -- --num-refs 3 --ref-images ${REFS_3[$DS]}
    evaluate "${DS}" fs_iter_3ref semantic

    # 12. fs_iter_refine 1ref
    run_main configs/experiments/fs_iter_refine.yaml      "${DS}" fs_iter_refine_1ref \
        -- --num-refs 1 --ref-images ${REFS_1[$DS]}
    evaluate "${DS}" fs_iter_refine_1ref semantic

    # 13. fs_iter_refine 3ref
    run_main configs/experiments/fs_iter_refine.yaml      "${DS}" fs_iter_refine_3ref \
        -- --num-refs 3 --ref-images ${REFS_3[$DS]}
    evaluate "${DS}" fs_iter_refine_3ref semantic

    done_
done

# ── Step 14: per-dataset quick-look plots ────────────────────────────────────

step "STEP 14 — plot_results.py (per-dataset quick-look)"

for DS in "${DATASETS[@]}"; do
    echo "  → ${DS}"
    python plot_results.py \
        --results_dir "${RESULTS}/${DS}" \
        --output      "${RESULTS}/${DS}/plots"
done
done_

# ── Step 15: compare_versions.py ─────────────────────────────────────────────

step "STEP 15 — compare_versions.py"

python compare_versions.py \
    --results_dir "${RESULTS_ROOT}" \
    --versions    "${VERSION}" \
    --datasets    XRayNicoSent SunnybrookNicoSent \
    --output      "${RESULTS_ROOT}/comparison_${VERSION}" \
    --skip-metric-story

done_

# ── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ALL DONE  $(date)"
echo "  Results:    ${RESULTS}/"
echo "  Comparison: ${RESULTS_ROOT}/comparison_${VERSION}/"
echo "════════════════════════════════════════════════════════"

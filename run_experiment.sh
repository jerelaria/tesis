#!/bin/bash
# ============================================================================
# run_experiments.sh
# ----------------------------------------------------------------------------
# Final experiment catalog, per dataset (15 runs):
#   - baseline                 : raw grid (no clustering). Feature-independent,
#                                run once.                                  (1)
#   - unsupervised             : {propagation independent, propagation iterative}
#                                x feature versions {moments, embeddings_red,
#                                embeddings_raw, emb_momentos}.             (8)
#   - few-shot                 : {independent, iterative} x K in {1,3,5}.
#                                Reference-based, feature-independent.      (6)
#
#   Two datasets (Sunnybrook, XRay) -> 30 runs.
#
# Pass --no-embeddings to drop every embedding-based version (keeps only
# 'moments'); baseline and few-shot are unaffected.
#
# Each run is evaluated inline (hungarian for unsup_*, semantic for fs_*).
# The segmentation cache is shared; the first run of each segmentation group
# computes it (--compute-seg-if-missing), the rest reuse it.
#
# Usage:
#   ./run_experiments.sh                       # full suite, full dataset
#   ./run_experiments.sh --no-embeddings       # skip embedding versions
#   ./run_experiments.sh --max 20              # smoke test on 20 images/dataset
# ============================================================================
set -euo pipefail

# ── Environment (tolerant: skip if the venv layout differs) ─────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${PROJECT_ROOT}/venv/bin/activate" ]]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
    export LD_LIBRARY_PATH="${PROJECT_ROOT}/venv/lib/python3.13/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
fi

# ── CLI ─────────────────────────────────────────────────────────────────────
WITH_EMBEDDINGS=true
MAX_IMAGES=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-embeddings) WITH_EMBEDDINGS=false; shift ;;
        --max)           MAX_IMAGES="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done
MAX_ARGS=()
[[ -n "$MAX_IMAGES" ]] && MAX_ARGS=(--max-images "$MAX_IMAGES")

# ── Layout ──────────────────────────────────────────────────────────────────
RESULTS_ROOT="results"
SEG_CACHE="${RESULTS_ROOT}/_segmentation"

# Where to write the exported Sunnybrook (moments) prototypes for transfer to ACDC.
EXPORT_REFS_DIR="/media/apoloml/DATOS_2/Tesis_Cosegmentacion/data/few_shot/prototypes_sunnybrook_to_acdc"

# Datasets: <path under data/raw & data/processed>  +  <output label>.
DS_PATHS=(Sunnybrook XRay ACDC)
DS_LABELS=(Sunnybrook XRay ACDC)

# Few-shot reference counts.
FS_K=(1 3 5)

# ── Feature sets ────────────────────────────────────────────────────────────
# 'moments' version: the 13-moment set = all 16 moments minus ecc, hu1, hu2
# (the rotation-sensitive redundant descriptors).
MOMENTS_FEATURES="V,Cx,Cy,Dx,Dy,L,solidity,extent,compact,hu0,intensity_mean,intensity_std"

# ── Shared overrides ────────────────────────────────────────────────────────
# Segmenter params (part of the segmentation cache key -> keep identical).
COMMON_OVR=(
    "segmenter.grid_side=6"
    "segmenter.score_threshold=0.5"
    "segmenter.iou_threshold=0.5"
)

# HDBSCAN (leaf selection + tuned fractions) and the propagation quality
# thresholds min_image_frequency / min_avg_sam_confidence now live in the base
# configs (unsup_hdbscan_propagation*.yaml), so they are no longer overridden here.

# Prototype propagation: tuned thresholds + SAM-weighted prototype selection
# (preserved from the best Sunnybrook config).
PROP_OVR=(
    "propagation.references_per_cluster=5"
    "propagation.quality.min_avg_labeling_confidence=0.5"
    "propagation.mask_selection.min_combined_score=0.5"
)

# Feature versions on the unsupervised axis.
FEATURE_VERSIONS=(moments)
if [[ "$WITH_EMBEDDINGS" == true ]]; then
    FEATURE_VERSIONS+=(embeddings_red embeddings_raw emb_momentos)
fi

# Few-shot reference image stems per dataset and K (verified to contain all
# annotated organs).
declare -A REFS_1 REFS_3 REFS_5
REFS_1[XRay]="100469495785351489872749036114751610212_rfyvv7"
REFS_3[XRay]="100469495785351489872749036114751610212_rfyvv7 10155709300728342918543955138521808206_f7cj92 10287653421930576798556842610982533460_vpbhw6"
REFS_5[XRay]="100469495785351489872749036114751610212_rfyvv7 10155709300728342918543955138521808206_f7cj92 10287653421930576798556842610982533460_vpbhw6 103416378058309979932405295235813040436_1g2pmw 10383960670432673238945376919735423432_hd3moq"
REFS_1[Sunnybrook]="SCD0000101_IM_0003_0059"
REFS_3[Sunnybrook]="SCD0000101_IM_0003_0059 SCD0000101_IM_0003_0079 SCD0000101_IM_0003_0099"
REFS_5[Sunnybrook]="SCD0000101_IM_0003_0059 SCD0000101_IM_0003_0079 SCD0000101_IM_0003_0099 SCD0000101_IM_0003_0119 SCD0000101_IM_0003_0139"

# ── Helpers ─────────────────────────────────────────────────────────────────
set_version_overrides() {  # populates the global VER_OVR array
    case "$1" in
        moments)
            VER_OVR=("labeler.standardize=true"
                     "extractor.features=${MOMENTS_FEATURES}"
                     "labeler.features=${MOMENTS_FEATURES}"
                     "labeler.embedding.enabled=false") ;;
        embeddings_red)
            VER_OVR=("labeler.standardize=true"
                     "labeler.features=null"
                     "labeler.embedding.enabled=true"
                     "labeler.embedding.reduction=pca"
                     "labeler.embedding.n_components=16") ;;
        embeddings_raw)
            VER_OVR=("labeler.standardize=true"
                     "labeler.features=null"
                     "labeler.embedding.enabled=true"
                     "labeler.embedding.reduction=none") ;;
        emb_momentos)
            VER_OVR=("labeler.standardize=true"
                     "extractor.features=${MOMENTS_FEATURES}"
                     "labeler.features=${MOMENTS_FEATURES}"
                     "labeler.embedding.enabled=true"
                     "labeler.embedding.reduction=pca"
                     "labeler.embedding.n_components=16") ;;
        *) echo "Unknown version: $1" >&2; exit 1 ;;
    esac
}

# run_and_eval <out_dir> <config> <ds_path> <matching> [extra main args...]
# Reads the global RUN_OVR array for the --override list.
run_and_eval() {
    local out="$1" config="$2" ds_path="$3" matching="$4"; shift 4
    local extra=("$@")
    mkdir -p "$(dirname "$out")"

    echo ""
    echo "── RUN  ${out}   ($(date '+%H:%M:%S'))"
    python -m main \
        --config "$config" \
        --dataset "${ds_path}/images" \
        --output-dir "$out" \
        --seg-cache-dir "$SEG_CACHE" \
        --compute-seg-if-missing \
        ${MAX_ARGS[@]+"${MAX_ARGS[@]}"} \
        ${extra[@]+"${extra[@]}"} \
        --override "${RUN_OVR[@]}"

    echo "── EVAL ${out}   (matching=${matching})"
    python evaluate.py \
        --gt "data/processed/${ds_path}/masks" \
        --pred "${out}/masks" \
        --output "$out" \
        --matching "$matching"
}

# Baseline (raw grid) + the two unsupervised propagation configs x versions.
run_unsup_for_dataset() {  # <ds_path> <ds_label>
    local DS="$1" LBL="$2"

    RUN_OVR=("${COMMON_OVR[@]}")
    run_and_eval "${RESULTS_ROOT}/baseline/${LBL}/unsup_baseline" \
        configs/experiments/unsup_baseline.yaml "$DS" hungarian

    for V in "${FEATURE_VERSIONS[@]}"; do
        set_version_overrides "$V"
        RUN_OVR=("${VER_OVR[@]}" "${PROP_OVR[@]}" "${COMMON_OVR[@]}")

        run_and_eval "${RESULTS_ROOT}/${V}/${LBL}/unsup_hdbscan_propagation" \
            configs/experiments/unsup_hdbscan_propagation.yaml "$DS" hungarian

        run_and_eval "${RESULTS_ROOT}/${V}/${LBL}/unsup_hdbscan_propagation_iter" \
            configs/experiments/unsup_hdbscan_propagation_iter.yaml "$DS" hungarian
    done
}

# Few-shot: independent + iterative propagation, swept over K. Once -> fs_propagation.
run_fewshot_for_dataset() {  # <ds_path> <ds_label>
    local DS="$1" LBL="$2"
    RUN_OVR=("${COMMON_OVR[@]}")

    for K in "${FS_K[@]}"; do
        case "$K" in
            1) REFS="${REFS_1[$DS]}" ;;
            3) REFS="${REFS_3[$DS]}" ;;
            5) REFS="${REFS_5[$DS]}" ;;
        esac

        run_and_eval "${RESULTS_ROOT}/fs_propagation/${LBL}/fs_indep_${K}ref" \
            configs/experiments/fs_propagation.yaml "$DS" semantic \
            --num-refs "$K" --ref-images $REFS

        run_and_eval "${RESULTS_ROOT}/fs_propagation/${LBL}/fs_iter_${K}ref" \
            configs/experiments/fs_propagation_iter.yaml "$DS" semantic \
            --num-refs "$K" --ref-images $REFS
    done
}

# Export references: build unsupervised prototypes (moments) on a dataset and
# write them as a few-shot reference tree under EXPORT_REFS_DIR, then exit
# (skips full-dataset propagation). The result is usable as the memory for a
# few_shot run on another dataset (e.g. ACDC).
run_export_refs() {  # <ds_path> <export_dir>
    local DS="$1" OUT="$2"

    set_version_overrides moments
    RUN_OVR=("${VER_OVR[@]}" "${PROP_OVR[@]}" "${COMMON_OVR[@]}")

    echo ""
    echo "── EXPORT REFS  ${DS} (moments) -> ${OUT}   ($(date '+%H:%M:%S'))"
    python -m main \
        --config configs/experiments/unsup_hdbscan_propagation.yaml \
        --dataset "${DS}/images" \
        --output-dir "${RESULTS_ROOT}/_export_refs/${DS}" \
        --seg-cache-dir "$SEG_CACHE" \
        --compute-seg-if-missing \
        ${MAX_ARGS[@]+"${MAX_ARGS[@]}"} \
        --export-references "$OUT" \
        --override "${RUN_OVR[@]}"
}

# ── Run ─────────────────────────────────────────────────────────────────────
echo "############################################################"
echo "  Experiment catalog   embeddings=${WITH_EMBEDDINGS}   max=${MAX_IMAGES:-all}"
echo "  Feature versions: ${FEATURE_VERSIONS[*]}"
echo "  Datasets:         ${DS_LABELS[*]}"
echo "  Moments set (12): ${MOMENTS_FEATURES}"
echo "  Export refs:      Sunnybrook (moments) -> ${EXPORT_REFS_DIR}"
echo "  $(date)"
echo "############################################################"

# Export Sunnybrook (moments) prototypes as a few-shot reference tree for ACDC.
run_export_refs Sunnybrook "$EXPORT_REFS_DIR"

for i in "${!DS_PATHS[@]}"; do
    run_unsup_for_dataset    "${DS_PATHS[$i]}" "${DS_LABELS[$i]}"
    # Few-shot only for datasets with in-dataset references defined.
    # ACDC has no REFS_* entries -> unsupervised only.
    if [[ -n "${REFS_1[${DS_PATHS[$i]}]:-}" ]]; then
        run_fewshot_for_dataset  "${DS_PATHS[$i]}" "${DS_LABELS[$i]}"
    else
        echo "── SKIP few-shot for ${DS_LABELS[$i]} (no references defined)"
    fi
done

echo ""
echo "############################################################"
echo "  All experiments finished at $(date)"
echo "  Results under ${RESULTS_ROOT}/"
echo "############################################################"
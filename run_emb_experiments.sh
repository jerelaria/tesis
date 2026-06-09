#!/bin/bash
# ============================================================================
# run_emb_experiments.sh
#
# Run three feature-representation versions of the unsupervised HDBSCAN
# experiment, with their evaluations:
#
#   embeddings_red  — SAM2 encoder embeddings, PCA-reduced to 16 dims, no moments.
#   embeddings_raw  — SAM2 encoder embeddings, raw 256 dims (no reduction), no moments.
#   emb_momentos    — 16 moments + PCA-reduced embeddings (hybrid).
#
# All versions share ONE segmentation cache: segmentation_cache.py always
# extracts embeddings (force_embeddings=True) and feature/embedding settings
# are NOT part of the cache key. The first run builds the cache via
# --compute-seg-if-missing; the second run reuses it.
#
# Standardization note:
#   All versions set labeler.standardize=true. Standardization is applied to the
#   final feature vector AFTER any PCA reduction (see
#   ClusteringLabeler._extract_feature_matrix). For the hybrid it is mandatory
#   (log-Hu moments vs PCA components differ by orders of magnitude under
#   HDBSCAN's Euclidean metric). For the embeddings-only versions it is kept on
#   for a clean apples-to-apples comparison; note the whitening effect is
#   stronger for embeddings_raw (256 dims, more low-variance/noise dims forced to
#   unit variance). To match the old run_final.sh behaviour, drop the
#   standardize line from the relevant *_OVR array.
#
# Output layout:  results_leaf/{version}/{dataset}_test/{exp_name}/
#   where {dataset}_test uses the _test suffix to match the results_leaf convention.
#
# Usage:
#   ./run_emb_experiments.sh                  # full dataset, both datasets
#   ./run_emb_experiments.sh --max 30         # cap at 30 images per dataset
#   ./run_emb_experiments.sh 2>&1 | tee run_emb.log
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

# ── Environment ───────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROJECT_ROOT}/venv/bin/activate"
export LD_LIBRARY_PATH="${PROJECT_ROOT}/venv/lib/python3.13/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_ROOT="results_leaf"
SEG_CACHE="results/_segmentation"
# DS_PATHS: actual data directories under data/raw/ and data/processed/
# DS_LABELS: output directory labels (with _test suffix, matching results_leaf convention)
DS_PATHS=("XRay" "Sunnybrook")
DS_LABELS=("XRay_test" "Sunnybrook_test")
CONFIG="configs/experiments/unsup_hdbscan_propagation.yaml"
EXP_NAME="unsup_hdbscan_propagation"
MATCHING="hungarian"

# 16-moment feature set (shared by the hybrid version)
ALL_MOMENTS="V,Cx,Cy,Dx,Dy,L,ecc,solidity,extent,compact,hu0,hu1,hu2,intensity_mean,intensity_std,orientation"

# Per-version overrides (feature representation).
EMBEDDINGS_RED_OVR=(
    "labeler.standardize=true"        # remove this line to match old run_final.sh
    "labeler.features=null"           # null -> embeddings-only mode
    "labeler.embedding.enabled=true"
    "labeler.embedding.reduction=pca"
    "labeler.embedding.n_components=16"
)
EMBEDDINGS_RAW_OVR=(
    "labeler.standardize=true"        # remove this line to skip whitening of raw dims
    "labeler.features=null"           # null -> embeddings-only mode
    "labeler.embedding.enabled=true"
    "labeler.embedding.reduction=none"   # none -> raw 256-dim embeddings (n_components ignored)
)
EMB_MOMENTOS_OVR=(
    "labeler.standardize=true"
    "extractor.features=${ALL_MOMENTS}"
    "labeler.features=${ALL_MOMENTS}"
    "labeler.embedding.enabled=true"
    "labeler.embedding.reduction=pca"
    "labeler.embedding.n_components=16"
)

# HDBSCAN params applied on top of every version override (hdbscan experiments).
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

run_version() {
    # run_version <version> <ds_path> <ds_label> <override...>
    #   ds_path:  actual data directory (e.g. XRay)
    #   ds_label: output label with _test suffix (e.g. XRay_test)
    local version="$1"; local ds_path="$2"; local ds_label="$3"; shift 3
    local overrides=("$@")
    local out_dir="${RESULTS_ROOT}/${version}/${ds_label}/${EXP_NAME}"
    mkdir -p "${RESULTS_ROOT}/${version}/${ds_label}"

    step "RUN  ${version} / ${ds_label} / ${EXP_NAME}"
    python -m main \
        --config        "${CONFIG}" \
        --dataset       "${ds_path}/images" \
        --output-dir    "${out_dir}" \
        --seg-cache-dir "${SEG_CACHE}" \
        --compute-seg-if-missing \
        "${MAX_ARGS[@]}" \
        --override "${overrides[@]}" "${H_OVR[@]}"

    step "EVAL ${version} / ${ds_label} / ${EXP_NAME}  (matching=${MATCHING})"
    python evaluate.py \
        --gt       "data/processed/${ds_path}/masks" \
        --pred     "${out_dir}/masks" \
        --output   "${out_dir}" \
        --matching "${MATCHING}"
}

# ── Run ───────────────────────────────────────────────────────────────────────
for i in "${!DS_PATHS[@]}"; do
    DS_PATH="${DS_PATHS[$i]}"
    DS_LABEL="${DS_LABELS[$i]}"
    run_version embeddings_red  "${DS_PATH}" "${DS_LABEL}" "${EMBEDDINGS_RED_OVR[@]}"
    run_version embeddings_raw  "${DS_PATH}" "${DS_LABEL}" "${EMBEDDINGS_RAW_OVR[@]}"
    run_version emb_momentos    "${DS_PATH}" "${DS_LABEL}" "${EMB_MOMENTOS_OVR[@]}"
done

step "ALL DONE"
echo "  Results: ${RESULTS_ROOT}/{embeddings_red,embeddings_raw,emb_momentos}/{${DS_LABELS[*]}}/${EXP_NAME}/"
#!/bin/bash
set -e

# ============================================================================
# run_versions.sh — Run all experiment versions sequentially
#
# Version naming convention:
#   v1_baseline     — Kervadec 6 features, no standardization
#   v2_standardized — Kervadec 6 features + z-score standardization
#   v3_extended     — All 16 moment features + standardization
#   v4_emb_only     — SAM2 embeddings only, no moment features (raw 256-dim)
#   v4_emb_only_red — SAM2 embeddings only, PCA-reduced to 16 dims
#   v5_ext_emb_red  — All 16 moments + PCA-reduced embeddings (hybrid)
#
# Each version applies transversal overrides to ALL experiment configs
# in run_suite.sh. The YAML configs define structural variations (algorithm,
# propagation mode, refinement), while versions define feature/normalization
# choices applied uniformly.
# ============================================================================

# Full list of 16 moment features (for override strings)
ALL_MOMENTS="V,Cx,Cy,Dx,Dy,L,ecc,solidity,extent,compact,hu0,hu1,hu2,intensity_mean,intensity_std,orientation"

VERSIONS=(
    # v1: Kervadec baseline (6 features, no standardization)
    "v1_baseline||extractor.features=V,Cx,Cy,Dx,Dy,L labeler.features=V,Cx,Cy,Dx,Dy,L"

    # v2: Same features, standardized
    "v2_standardized||labeler.standardize=true"

    # v3: All 16 moment features, standardized
    "v3_extended||labeler.standardize=true extractor.features=${ALL_MOMENTS} labeler.features=${ALL_MOMENTS}"

    # v4_emb_only: Embeddings only, no moment features, raw 256-dim
    # features=none triggers embeddings-only mode in ClusteringConfig
    "v4_emb_only||labeler.standardize=true labeler.features=none labeler.embedding.enabled=true labeler.embedding.reduction=none"

    # v4_emb_only_red: Embeddings only, PCA-reduced to 16 dims
    "v4_emb_only_red||labeler.standardize=true labeler.features=none labeler.embedding.enabled=true labeler.embedding.reduction=pca labeler.embedding.n_components=16"

    # v5: Hybrid — all 16 moments + PCA-reduced embeddings
    "v5_ext_emb_red||labeler.standardize=true extractor.features=${ALL_MOMENTS} labeler.features=${ALL_MOMENTS} labeler.embedding.enabled=true labeler.embedding.reduction=pca labeler.embedding.n_components=16"
)

# ── Parse arguments ──────────────────────────────────────────────────────────

START_FROM="${START_FROM:-}"
ONLY="${ONLY:-}"
SUITE_ARGS=("$@")

# ── Build list of versions to run ────────────────────────────────────────────

declare -a ONLY_LIST
if [ -n "$ONLY" ]; then
    IFS=',' read -ra ONLY_LIST <<< "$ONLY"
fi

should_run() {
    local name="$1"

    # ONLY filter
    if [ ${#ONLY_LIST[@]} -gt 0 ]; then
        for allowed in "${ONLY_LIST[@]}"; do
            if [ "$name" = "$allowed" ]; then
                return 0
            fi
        done
        return 1
    fi

    return 0
}

# ── Run ──────────────────────────────────────────────────────────────────────

TOTAL=${#VERSIONS[@]}
STARTED=false
RAN=0

if [ -z "$START_FROM" ]; then
    STARTED=true
fi

echo ""
echo "################################################################"
echo "  Version sweep: ${TOTAL} versions defined"
echo "  Suite args: ${SUITE_ARGS[*]:-none}"
echo "  Start from: ${START_FROM:-beginning}"
echo "  Only: ${ONLY:-all}"
echo "  $(date)"
echo "################################################################"

for i in "${!VERSIONS[@]}"; do
    IFS='|' read -r VERSION_NAME VERSION_SKIPS VERSION_OVERRIDES <<< "${VERSIONS[$i]}"

    # Handle START_FROM
    if [ "$STARTED" = false ]; then
        if [ "$VERSION_NAME" = "$START_FROM" ]; then
            STARTED=true
        else
            echo "  [SKIP] ${VERSION_NAME} (before START_FROM=${START_FROM})"
            continue
        fi
    fi

    # Handle ONLY filter
    if ! should_run "$VERSION_NAME"; then
        echo "  [SKIP] ${VERSION_NAME} (not in ONLY list)"
        continue
    fi

    echo ""
    echo "================================================================"
    echo "  Version $((i+1))/${TOTAL}: ${VERSION_NAME}"
    echo "  Overrides: ${VERSION_OVERRIDES:-none (baseline)}"
    echo "  Skips: ${VERSION_SKIPS:-none}"
    echo "  $(date)"
    echo "================================================================"

    # Build skip args from version-level skip flags
    SKIP_ARGS=()
    if [[ "$VERSION_SKIPS" == *"fs"* ]]; then
        SKIP_ARGS+=(--skip-fewshot)
    fi
    if [[ "$VERSION_SKIPS" == *"tg"* ]]; then
        SKIP_ARGS+=(--skip-textguided)
    fi
    if [[ "$VERSION_SKIPS" == *"unsup"* ]]; then
        SKIP_ARGS+=(--skip-unsup)
    fi

    # Build override args
    OVERRIDE_ARGS=()
    if [ -n "$VERSION_OVERRIDES" ]; then
        OVERRIDE_ARGS=(--override $VERSION_OVERRIDES)
    fi

    ./run_suite.sh "$VERSION_NAME" \
        "${SUITE_ARGS[@]}" \
        "${SKIP_ARGS[@]}" \
        "${OVERRIDE_ARGS[@]}"

    RAN=$((RAN + 1))
    echo ""
    echo "  Done: ${VERSION_NAME} at $(date)"
done

echo ""
echo "################################################################"
echo "  Sweep complete: ${RAN} versions ran"
echo "  $(date)"
echo "################################################################"
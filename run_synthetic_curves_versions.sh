#!/usr/bin/env bash
# run_synthetic_curves_versions.sh
# --------------------------------
# Run the full degradation-curve sweep for multiple feature-space versions.
# Each version is one row of the VERSIONS array, with the same schema as
# run_versions.sh:
#     <version_name>|<primary_cfg>|<norefine_cfg>|<override args>
#
# Empty norefine_cfg means: skip the curve D no-refine companion runs
# (used for v0_baseline where there is no Refiner to compare against).

set -euo pipefail

source /media/apoloml/DATOS_2/Tesis_Cosegmentacion/venv/bin/activate
export LD_LIBRARY_PATH=/media/apoloml/DATOS_2/Tesis_Cosegmentacion/venv/lib/python3.13/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}

ALL_MOMENTS="V,Cx,Cy,Dx,Dy,L,ecc,solidity,extent,compact,hu0,hu1,hu2,intensity_mean,intensity_std,orientation"

VERSIONS=(
    # v0: raw MedSAM2 grid only — pipeline floor. No clustering, no refiner.
    "v0_baseline|configs/experiments/unsup_baseline.yaml||"

    # v3: extended moments + standardization (current "best believed")
    "v3_extended|configs/experiments/unsup_hdbscan_refine.yaml|configs/experiments/unsup_hdbscan.yaml|--override labeler.standardize=true extractor.features=${ALL_MOMENTS} labeler.features=${ALL_MOMENTS}"

    # v4: embeddings only, raw 256-dim, no moments
    "v4_emb_only|configs/experiments/unsup_hdbscan_refine.yaml|configs/experiments/unsup_hdbscan.yaml|--override labeler.standardize=true labeler.features=none labeler.embedding.enabled=true labeler.embedding.reduction=none"    

    # v5: hybrid — extended moments + PCA-reduced embeddings
    "v5_ext_emb_red|configs/experiments/unsup_hdbscan_refine.yaml|configs/experiments/unsup_hdbscan.yaml|--override labeler.standardize=true extractor.features=${ALL_MOMENTS} labeler.features=${ALL_MOMENTS} labeler.embedding.enabled=true labeler.embedding.reduction=pca labeler.embedding.n_components=16"
)

ONLY="${ONLY:-}"      # Comma-separated version names to run; empty = all
declare -a ONLY_LIST=()
if [ -n "$ONLY" ]; then
    IFS=',' read -ra ONLY_LIST <<< "$ONLY"
fi

should_run() {
    local name="$1"
    if [ ${#ONLY_LIST[@]} -gt 0 ]; then
        for v in "${ONLY_LIST[@]}"; do
            [ "$name" = "$v" ] && return 0
        done
        return 1
    fi
    return 0
}

TOTAL=${#VERSIONS[@]}
echo "################################################################"
echo "  Curves sweep over ${TOTAL} versions"
echo "  Only filter: ${ONLY:-<all>}"
echo "  $(date)"
echo "################################################################"

MAX_IMAGES="${MAX_IMAGES:-}"  # Optional: limit images per dataset for smoke tests

for entry in "${VERSIONS[@]}"; do
    IFS='|' read -r VNAME VCFG VNOREFCFG VOVERRIDES <<< "$entry"

    if ! should_run "$VNAME"; then
        echo "  [SKIP] $VNAME"
        continue
    fi

    echo ""
    echo "################################################################"
    echo "  Running curves for $VNAME"
    echo "################################################################"

    # Build args array explicitly to avoid word-splitting issues
    CALL_ARGS=("$VNAME" "$VCFG" "${VNOREFCFG:-}")
    if [ -n "${MAX_IMAGES:-}" ]; then
        CALL_ARGS+=(--max-images "$MAX_IMAGES")
    fi
    # Split VOVERRIDES into individual tokens safely
    if [ -n "$VOVERRIDES" ]; then
        read -ra OVERRIDE_TOKENS <<< "$VOVERRIDES"
        CALL_ARGS+=("${OVERRIDE_TOKENS[@]}")
    fi

    bash run_synthetic_curves.sh "${CALL_ARGS[@]}"
done

echo ""
echo "################################################################"
echo "  Multi-version curves sweep complete at $(date)"
echo "################################################################"
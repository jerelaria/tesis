#!/usr/bin/env bash
# run_synthetic_curves.sh
# -----------------------
# Block 3: run the pipeline over the 25 degradation-curve variants.
#
# Now accepts an explicit primary config + optional overrides, so the
# same script can drive any version (v0 baseline, v1 Kervadec, v3 extended,
# v4 embeddings, v5 hybrid) by changing the call.
#
# Usage:
#     ./run_synthetic_curves.sh <version_name> <primary_cfg> <norefine_cfg> [override KEY=VAL]...
#
# Examples:
#     # v1 (current state, Kervadec 6 features, no standardize):
#     ./run_synthetic_curves.sh v1_baseline \
#         configs/experiments/unsup_hdbscan_refine.yaml \
#         configs/experiments/unsup_hdbscan.yaml
#
#     # v3 (extended features + standardize):
#     ./run_synthetic_curves.sh v3_extended \
#         configs/experiments/unsup_hdbscan_refine.yaml \
#         configs/experiments/unsup_hdbscan.yaml \
#         --override labeler.standardize=true \
#                    extractor.features=V,Cx,Cy,Dx,Dy,L,ecc,solidity,extent,compact,hu0,hu1,hu2,intensity_mean,intensity_std,orientation \
#                    labeler.features=V,Cx,Cy,Dx,Dy,L,ecc,solidity,extent,compact,hu0,hu1,hu2,intensity_mean,intensity_std,orientation

set -euo pipefail

VERSION="${1:?version name required}"
BEST_CFG="${2:?primary config path required}"
NOREFINE_CFG="${3:-}"   # Optional: pass empty string "" to skip curve D no-refine runs
shift 3

# Remaining arguments are forwarded to main.py (e.g. --override ...)
EXTRA_ARGS=("$@")

CURVE_DATASETS=(
    SyntheticV1_curveA_noise_002  SyntheticV1_curveA_noise_005
    SyntheticV1_curveA_noise_01   SyntheticV1_curveA_noise_015
    SyntheticV1_curveA_noise_02

    SyntheticV1_curveB_intra_002  SyntheticV1_curveB_intra_005
    SyntheticV1_curveB_intra_01   SyntheticV1_curveB_intra_015
    SyntheticV1_curveB_intra_02

    SyntheticV1_curveC_distract_0   SyntheticV1_curveC_distract_2
    SyntheticV1_curveC_distract_5   SyntheticV1_curveC_distract_10
    SyntheticV1_curveC_distract_20

    SyntheticV1_curveD_occlude_00  SyntheticV1_curveD_occlude_01
    SyntheticV1_curveD_occlude_02  SyntheticV1_curveD_occlude_03
    SyntheticV1_curveD_occlude_05

    SyntheticV1_curveE_missing_00  SyntheticV1_curveE_missing_01
    SyntheticV1_curveE_missing_02  SyntheticV1_curveE_missing_03
    SyntheticV1_curveE_missing_04
)

START_TIME=$(date +%s)
echo "================================================================"
echo "  Curves run for version: ${VERSION}"
echo "  Primary config:  ${BEST_CFG}"
echo "  No-refine config: ${NOREFINE_CFG:-<skipped>}"
echo "  Extra args:       ${EXTRA_ARGS[*]:-<none>}"
echo "  $(date)"
echo "================================================================"

for ds in "${CURVE_DATASETS[@]}"; do
    results_dir="results/v_synthetic_curves/${VERSION}/${ds}/hdbscan_refine"
    echo ""
    echo "  [${VERSION}] ${ds} -> ${results_dir}"

    python -m main \
        --config "$BEST_CFG" \
        --dataset "synthetic_curves/${ds}/images" \
        --output-dir "$results_dir" \
        "${EXTRA_ARGS[@]}"

    if [ -d "$results_dir/masks" ]; then
        gt_dir="data/processed/synthetic_curves/${ds}/masks"
        python evaluate.py \
            --gt "$gt_dir" \
            --pred "$results_dir/masks" \
            --output "$results_dir" \
            --matching hungarian
    fi

    # Curve D special-case: also run no-refine config for the gap plot.
    # Skipped when NOREFINE_CFG is empty (e.g. v0_baseline).
    if [[ -n "$NOREFINE_CFG" && "$ds" == SyntheticV1_curveD_* ]]; then
        norefine_dir="results/v_synthetic_curves/${VERSION}/${ds}/hdbscan_norefine"
        echo "    additional: no-refine -> $norefine_dir"
        python -m main \
            --config "$NOREFINE_CFG" \
            --dataset "synthetic_curves/${ds}/images" \
            --output-dir "$norefine_dir" \
            "${EXTRA_ARGS[@]}"

        if [ -d "$norefine_dir/masks" ]; then
            gt_dir="data/processed/synthetic_curves/${ds}/masks"
            python evaluate.py \
                --gt "$gt_dir" \
                --pred "$norefine_dir/masks" \
                --output "$norefine_dir" \
                --matching hungarian
        fi
    fi
done

ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "  [${VERSION}] Done in $((ELAPSED / 60)) min"
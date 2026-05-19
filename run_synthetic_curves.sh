#!/usr/bin/env bash
# run_synthetic_curves.sh
# -----------------------
# Block 3: run the pipeline over the 25 degradation-curve variants.
#
# Each curve sweeps one parameter holding everything else fixed.
# Curve D additionally runs HDBSCAN without refinement so the Refiner's
# isolated contribution can be quantified as the gap between both lines.
#
# Usage:
#     ./run_synthetic_curves.sh <version>

set -euo pipefail

VERSION="${1:-v_synthetic_curves}"
BEST_CFG="configs/experiments/unsup_hdbscan_refine.yaml"
NOREFINE_CFG="configs/experiments/unsup_hdbscan.yaml"

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
echo "Starting synthetic curves run at $(date)"
echo "Version: $VERSION"
echo "Total datasets: ${#CURVE_DATASETS[@]}"

for ds in "${CURVE_DATASETS[@]}"; do
    results_dir="results/${VERSION}/${ds}/hdbscan_refine"
    echo ""
    echo "=================================================================="
    echo "  ${ds} -> ${results_dir}"
    echo "=================================================================="

    python -m main \
        --config "$BEST_CFG" \
        --dataset "synthetic_curves/${ds}/images" \
        --output-dir "$results_dir" \

    if [ -d "$results_dir/masks" ]; then
        gt_dir="data/processed/synthetic_curves/${ds}/masks"
        python evaluate.py \
            --gt "$gt_dir" \
            --pred "$results_dir/masks" \
            --output "$results_dir" \
            --matching hungarian
    fi

    if [[ "$ds" == SyntheticV1_curveD_* ]]; then
        norefine_dir="results/${VERSION}/${ds}/hdbscan_norefine"
        echo ""
        echo "  additional run: HDBSCAN no-refine -> $norefine_dir"
        python -m main \
            --config "$NOREFINE_CFG" \
            --dataset "${ds}/images" \
            --output-dir "$norefine_dir" \

        if [ -d "$norefine_dir/masks" ]; then
            gt_dir="data/processed/${ds}/masks"
            python evaluate.py \
                --gt "$gt_dir" \
                --pred "$norefine_dir/masks" \
                --output "$norefine_dir" \
                --matching hungarian
        fi
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
echo ""
echo "=================================================================="
echo "  Done. Total time: ${ELAPSED_MIN} min"
echo "  Next: python data/scripts/analyze_curves.py \\"
echo "        --results-root results/${VERSION}/ \\"
echo "        --output results/${VERSION}/curves_analysis/"
echo "=================================================================="
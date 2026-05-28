#!/usr/bin/env bash
# resume_curves.sh
# ----------------
# Resume the Block 3 degradation curves after the overnight run failure.
#
# Re-runs everything that is either incomplete or potentially affected
# by the prefix bug in run_synthetic_curves.sh:
#
#   - Curve D: both hdbscan_refine and hdbscan_norefine (5 datasets x 2 configs)
#     The refine results may have been computed but with stale state,
#     so we re-run them from scratch.
#   - Curve E: hdbscan_refine only (5 datasets, never executed).
#
# Datasets live under data/raw/synthetic_curves/ and data/processed/synthetic_curves/
# (custom path used in this project — the prefix is required).
#
# Usage:
#     ./resume_curves.sh [version]
#
# Default version: v_synthetic_curves.

set -euo pipefail

VERSION="${1:-v_synthetic_curves}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="resume_curves_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

BEST_CFG="configs/experiments/unsup_hdbscan_refine.yaml"
NOREFINE_CFG="configs/experiments/unsup_hdbscan.yaml"

CURVE_D_DATASETS=(
    SyntheticV1_curveD_occlude_00
    SyntheticV1_curveD_occlude_01
    SyntheticV1_curveD_occlude_02
    SyntheticV1_curveD_occlude_03
    SyntheticV1_curveD_occlude_05
)

CURVE_E_DATASETS=(
    SyntheticV1_curveE_missing_00
    SyntheticV1_curveE_missing_01
    SyntheticV1_curveE_missing_02
    SyntheticV1_curveE_missing_03
    SyntheticV1_curveE_missing_04
)

START_TIME=$(date +%s)
echo "=================================================================="
echo "  Resuming Block 3 curves at $(date)"
echo "  Version: $VERSION"
echo "  Log: $LOG_FILE"
echo "  Datasets to run: ${#CURVE_D_DATASETS[@]} x 2 (D) + ${#CURVE_E_DATASETS[@]} (E)"
echo "=================================================================="

run_single() {
    # Helper to run one (dataset, config) combination + its evaluation
    local ds="$1"
    local config="$2"
    local exp_subdir="$3"   # e.g. "hdbscan_refine" or "hdbscan_norefine"

    local results_dir="results/${VERSION}/${ds}/${exp_subdir}"
    local dataset_arg="synthetic_curves/${ds}/images"
    local gt_dir="data/processed/synthetic_curves/${ds}/masks"

    echo ""
    echo "------------------------------------------------------------------"
    echo "  ${ds} / ${exp_subdir} -> ${results_dir}"
    echo "------------------------------------------------------------------"

    # Clean previous (possibly stale) results
    if [ -d "$results_dir" ]; then
        echo "  Removing stale results at $results_dir"
        rm -rf "$results_dir"
    fi

    python -m main \
        --config "$config" \
        --dataset "$dataset_arg" \
        --output-dir "$results_dir"

    if [ -d "$results_dir/masks" ]; then
        python evaluate.py \
            --gt "$gt_dir" \
            --pred "$results_dir/masks" \
            --output "$results_dir" \
            --matching hungarian
    else
        echo "  [WARN] no masks produced at $results_dir/masks — skipping eval"
    fi
}

# ---- Curve D: both configs (refine + no-refine) -------------------
echo ""
echo "[1/2] Curve D — running refine + no-refine for ${#CURVE_D_DATASETS[@]} datasets ..."
for ds in "${CURVE_D_DATASETS[@]}"; do
    run_single "$ds" "$BEST_CFG"     "hdbscan_refine"
    run_single "$ds" "$NOREFINE_CFG" "hdbscan_norefine"
done

# ---- Curve E: refine only -----------------------------------------
echo ""
echo "[2/2] Curve E — running refine for ${#CURVE_E_DATASETS[@]} datasets ..."
for ds in "${CURVE_E_DATASETS[@]}"; do
    run_single "$ds" "$BEST_CFG" "hdbscan_refine"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))

echo ""
echo "=================================================================="
echo "  Resume done. Total time: ${ELAPSED_MIN} min"
echo ""
echo "  Next step: produce curve plots and consolidated CSV:"
echo "    python data/scripts/analyze_curves.py \\"
echo "        --results-root results/${VERSION}/ \\"
echo "        --output results/${VERSION}/curves_analysis/"
echo "=================================================================="
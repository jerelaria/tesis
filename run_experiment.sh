#!/bin/bash
set -e

# ============================================================================
# run_all.sh — Run full experiment suite across datasets
# ============================================================================
#
# Usage:
#   ./run_all.sh <version> <dataset1> [dataset2] ... [options]
#
# Options:
#   --max-images N            Limit dataset images per experiment
#   --fs-sizes K1 K2 ...      Few-shot K values to sweep (default: 1 4 7 10)
#   --ref-images name1 ...    Priority reference names for few-shot.
#                             If fewer than K, remaining are filled
#                             alphabetically from data/few_shot/{dataset}/.
#   --override KEY=VAL ...    Config overrides forwarded to main.py
#   --skip-unsup              Skip unsupervised experiments
#   --skip-fewshot            Skip few-shot experiments
#   --skip-textguided         Skip text-guided experiments
#   --skip-eval               Skip evaluation step
#   --skip-plots              Skip plot generation
#
# Examples:
#   # Basic: sweep all configs on one dataset
#   ./run_all.sh v1 XRayNicoSent/images
#
#   # With few-shot K sweep and specific refs
#   ./run_all.sh v2 XRayNicoSent/images --fs-sizes 1 4 7 10 \
#       --ref-images ref_001 ref_002 ref_003 ref_004 ref_005 ref_006 ref_007 ref_008 ref_009 ref_010
#
#   # Quick test with 10 images
#   ./run_all.sh v_test XRayNicoSent/images --max-images 10 --fs-sizes 1 2
#
#   # Multiple datasets
#   ./run_all.sh v3 XRayNicoSent/images Sunnybrook/images
#
#   # With config overrides
#   ./run_all.sh v4 XRayNicoSent/images --override segmenter.score_threshold=0.6
#
# ============================================================================

source /media/apoloml/DATOS_2/Tesis_Cosegmentacion/venv/bin/activate
export LD_LIBRARY_PATH=/media/apoloml/DATOS_2/Tesis_Cosegmentacion/venv/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH

# ── Parse arguments ──────────────────────────────────────────────────────────

VERSION=${1:?"Usage: ./run_all.sh <version> <dataset1> [dataset2] ... [options]"}
shift 1

DATASETS=()
MAX_IMAGES=""
FS_SIZES=()
REF_IMAGES=()
OVERRIDES=()
SKIP_UNSUP=false
SKIP_FEWSHOT=false
SKIP_TEXTGUIDED=false
SKIP_EVAL=false
SKIP_PLOTS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-images)
            MAX_IMAGES="$2"
            shift 2
            ;;
        --fs-sizes)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                FS_SIZES+=("$1")
                shift
            done
            ;;
        --ref-images)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                REF_IMAGES+=("$1")
                shift
            done
            ;;
        --override)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                OVERRIDES+=("$1")
                shift
            done
            ;;
        --skip-unsup)
            SKIP_UNSUP=true
            shift
            ;;
        --skip-fewshot)
            SKIP_FEWSHOT=true
            shift
            ;;
        --skip-textguided)
            SKIP_TEXTGUIDED=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --skip-plots)
            SKIP_PLOTS=true
            shift
            ;;
        *)
            DATASETS+=("$1")
            shift
            ;;
    esac
done

# Defaults
if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "Error: at least one dataset required"
    echo "Usage: ./run_all.sh <version> <dataset1> [dataset2] ... [options]"
    exit 1
fi

if [ ${#FS_SIZES[@]} -eq 0 ]; then
    FS_SIZES=(1 4 7 10)
fi

# ── Config lists ─────────────────────────────────────────────────────────────

UNSUPERVISED_CONFIGS=(
    configs/experiments/unsup_kmeans.yaml
    configs/experiments/unsup_kmeans_refine.yaml
    configs/experiments/unsup_hdbscan.yaml
    configs/experiments/unsup_hdbscan_refine.yaml
)

FEW_SHOT_CONFIGS=(
    configs/experiments/fs_indep.yaml
    configs/experiments/fs_indep_refine.yaml
    configs/experiments/fs_iter.yaml
    configs/experiments/fs_iter_refine.yaml
)

TEXT_GUIDED_CONFIGS=(
    configs/experiments/tg.yaml
)

# ── Display configuration ───────────────────────────────────────────────────

echo ""
echo "############################################################"
echo "  Experiment Suite: ${VERSION}"
echo "  $(date)"
echo "############################################################"
echo "  Datasets:       ${DATASETS[*]}"
echo "  Max images:     ${MAX_IMAGES:-all}"
echo "  FS sizes (K):   ${FS_SIZES[*]}"
echo "  Ref images:     ${REF_IMAGES[*]:-auto-discover}"
echo "  Overrides:      ${OVERRIDES[*]:-none}"
echo "  Skip unsup:     ${SKIP_UNSUP}"
echo "  Skip fewshot:   ${SKIP_FEWSHOT}"
echo "  Skip textguided: ${SKIP_TEXTGUIDED}"
echo "############################################################"

# ── Helper functions ─────────────────────────────────────────────────────────

run_experiment() {
    local cfg="$1"
    local matching="$2"
    local dataset="$3"
    shift 3
    local extra_args=("$@")

    local dataset_short="${dataset%%/*}"
    local results_base="results/${VERSION}/${dataset_short}"
    local gt_dir="data/processed/${dataset_short}/masks"

    # Build experiment name from config + optional K suffix
    local exp_name
    exp_name=$(basename "$cfg" .yaml)

    local num_refs=""
    for i in "${!extra_args[@]}"; do
        if [[ "${extra_args[$i]}" == "--num-refs" ]]; then
            num_refs="${extra_args[$((i+1))]}"
            exp_name="${exp_name}_${num_refs}ref"
            break
        fi
    done

    local results_dir="${results_base}/${exp_name}"

    echo ""
    echo "============================================================"
    echo "  Experiment: ${exp_name}"
    echo "  Config:     ${cfg}"
    echo "  Dataset:    ${dataset}"
    echo "  Output:     ${results_dir}"
    if [ -n "$num_refs" ]; then
        echo "  Num refs:   ${num_refs}"
    fi
    echo "  $(date)"
    echo "============================================================"

    python -m main \
        --config "$cfg" \
        --dataset "$dataset" \
        --output-dir "$results_dir" \
        ${MAX_IMAGES:+--max-images "$MAX_IMAGES"} \
        ${OVERRIDES:+--override "${OVERRIDES[@]}"} \
        "${extra_args[@]}" \
        2>&1 | tee "${results_base}/${exp_name}.log"

    if [ "$SKIP_EVAL" = false ]; then
        if [ -d "${results_dir}/masks" ] && [ -d "$gt_dir" ]; then
            echo ""
            echo "  Evaluating: ${exp_name} (matching=${matching})"
            echo "  ----------------------------------------------"
            python evaluate.py \
                --gt "$gt_dir" \
                --pred "${results_dir}/masks" \
                --output "$results_dir" \
                --matching "$matching" \
                2>&1 | tee -a "${results_base}/${exp_name}.log"
        else
            echo "  [SKIP] Evaluation: masks or GT directory not found"
        fi
    fi
}

# ── Main loop ────────────────────────────────────────────────────────────────

for dataset in "${DATASETS[@]}"; do
    dataset_short="${dataset%%/*}"
    results_base="results/${VERSION}/${dataset_short}"
    mkdir -p "$results_base"
    cp -r configs/experiments/ "$results_base/configs_used/"

    echo ""
    echo "############################################################"
    echo "  Dataset: ${dataset}"
    echo "############################################################"

    # ── Unsupervised (hungarian matching, no refs) ────────────────────────
    if [ "$SKIP_UNSUP" = false ]; then
        echo ""
        echo "  ── Unsupervised experiments ──"
        for cfg in "${UNSUPERVISED_CONFIGS[@]}"; do
            run_experiment "$cfg" "hungarian" "$dataset"
        done
    fi

    # ── Few-shot (semantic matching, sweep over K) ────────────────────────
    if [ "$SKIP_FEWSHOT" = false ]; then
        echo ""
        echo "  ── Few-shot experiments (K sweep: ${FS_SIZES[*]}) ──"
        for K in "${FS_SIZES[@]}"; do
            for cfg in "${FEW_SHOT_CONFIGS[@]}"; do
                if [ ${#REF_IMAGES[@]} -gt 0 ]; then
                    run_experiment "$cfg" "semantic" "$dataset" \
                        --num-refs "$K" \
                        --ref-images "${REF_IMAGES[@]}"
                else
                    run_experiment "$cfg" "semantic" "$dataset" \
                        --num-refs "$K"
                fi
            done
        done
    fi

    # ── Text-guided (semantic matching) ───────────────────────────────────
    if [ "$SKIP_TEXTGUIDED" = false ]; then
        echo ""
        echo "  ── Text-guided experiments ──"
        for cfg in "${TEXT_GUIDED_CONFIGS[@]}"; do
            run_experiment "$cfg" "semantic" "$dataset"
        done
    fi

    # ── Plots ─────────────────────────────────────────────────────────────
    if [ "$SKIP_PLOTS" = false ]; then
        echo ""
        echo "============================================================"
        echo "  Generating plots for ${dataset_short}"
        echo "============================================================"
        python plot_results.py \
            --results_dir "$results_base" \
            --output "${results_base}/plots/"
    fi
done

echo ""
echo "############################################################"
echo "  All experiments finished at $(date)"
echo "  Results saved in: results/${VERSION}/"
echo "############################################################"
#!/bin/bash
set -e
source /media/apoloml/DATOS_2/Tesis_Cosegmentacion/venv/bin/activate
export LD_LIBRARY_PATH=/media/apoloml/DATOS_2/Tesis_Cosegmentacion/venv/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH

# Usage: ./run_experiments.sh <version> <dataset1> [dataset2] ... [--max-images N]
# Example: ./run_experiments.sh v1_baseline XRayNicoSent/images --max-images 50
# Example: ./run_experiments.sh v2_full XRayNicoSent/images Sunnybrook/images

VERSION=${1:?Usage: ./run_experiments.sh <version> <dataset1> [dataset2] ... [--max-images N]}
shift 1

# Parse datasets and optional --max-images flag
DATASETS=()
MAX_IMAGES=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-images)
            MAX_IMAGES="$2"
            shift 2
            ;;
        *)
            DATASETS+=("$1")
            shift
            ;;
    esac
done

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "Error: at least one dataset required"
    echo "Usage: ./run_experiments.sh <version> <dataset1> [dataset2] ... [--max-images N]"
    exit 1
fi

SEMANTIC_CONFIGS=(
    configs/experiments/fs_indep_1ref.yaml
    configs/experiments/fs_indep_1ref_refine.yaml
    configs/experiments/fs_indep_4ref.yaml
    configs/experiments/fs_indep_4ref_refine.yaml
    configs/experiments/fs_iter_1ref.yaml
    configs/experiments/fs_iter_4ref.yaml
    configs/experiments/fs_iter_4ref_refine.yaml
    configs/experiments/tg.yaml
)

HUNGARIAN_CONFIGS=(
    configs/experiments/unsup_kmeans.yaml
    configs/experiments/unsup_hdbscan.yaml
    configs/experiments/unsup_kmeans_refine.yaml
    configs/experiments/unsup_hdbscan_refine.yaml
)

run_experiment() {
    local cfg="$1"
    local matching="$2"
    local dataset="$3"
    local dataset_short
    dataset_short=$(echo "$dataset" | cut -d'/' -f1)
    local results_base="results/${VERSION}/${dataset_short}"
    local gt_dir="data/processed/${dataset_short}/masks"
    local name
    name=$(basename "$cfg" .yaml)
    local results_dir="${results_base}/${name}"

    echo ""
    echo "============================================================"
    echo "  Running: $cfg"
    echo "  Dataset: $dataset"
    echo "  Output:  $results_dir"
    echo "  $(date)"
    echo "============================================================"
    python -m main --config "$cfg" \
        --output-dir "$results_dir" \
        --dataset "$dataset" \
        ${MAX_IMAGES:+--max-images "$MAX_IMAGES"} \
        2>&1 | tee "${results_base}/${name}.log"

    echo ""
    echo "  Evaluating: $name (matching=$matching)"
    echo "------------------------------------------------------------"
    python evaluate.py \
        --gt "$gt_dir" \
        --pred "${results_dir}/masks" \
        --output "$results_dir" \
        --matching "$matching" 2>&1 | tee -a "${results_base}/${name}.log"
}

for dataset in "${DATASETS[@]}"; do
    dataset_short=$(echo "$dataset" | cut -d'/' -f1)
    results_base="results/${VERSION}/${dataset_short}"
    mkdir -p "$results_base"
    cp -r configs/experiments/ "$results_base/configs_used/"

    echo ""
    echo "############################################################"
    echo "  Dataset: $dataset"
    echo "############################################################"

    for cfg in "${HUNGARIAN_CONFIGS[@]}"; do
        run_experiment "$cfg" "hungarian" "$dataset"
    done

    for cfg in "${SEMANTIC_CONFIGS[@]}"; do
        run_experiment "$cfg" "semantic" "$dataset"
    done

    echo ""
    echo "============================================================"
    echo "  Generating plots for $dataset_short"
    echo "============================================================"
    python plot_results.py --results_dir "$results_base" --output "${results_base}/plots/"
done

echo ""
echo "All experiments finished at $(date)"
echo "Results saved in: results/${VERSION}/"
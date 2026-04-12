#!/bin/bash
set -e
source /media/apoloml/DATOS_2/Tesis_Cosegmentacion/venv/bin/activate
export LD_LIBRARY_PATH=/media/apoloml/DATOS_2/Tesis_Cosegmentacion/venv/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH

GT_DIR="data/processed/XRayNicoSent/masks"

# Configs where predictions have semantic organ names (few-shot / text-guided)
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

# Configs where predictions have cluster_N names (unsupervised)
HUNGARIAN_CONFIGS=(
    configs/experiments/unsup_kmeans.yaml
    configs/experiments/unsup_hdbscan.yaml
    configs/experiments/unsup_kmeans_refine.yaml
    configs/experiments/unsup_hdbscan_refine.yaml
)

run_experiment() {
    local cfg="$1"
    local matching="$2"
    local name
    name=$(basename "$cfg" .yaml)
    local results_dir="results/${name}"

    echo ""
    echo "============================================================"
    echo "  Running: $cfg"
    echo "  $(date)"
    echo "============================================================"
    python -m main --config "$cfg" 2>&1 | tee "results/${name}.log"

    echo ""
    echo "  Evaluating: $name (matching=$matching)"
    echo "------------------------------------------------------------"
    python evaluate.py \
        --gt "$GT_DIR" \
        --pred "${results_dir}/masks" \
        --output "$results_dir" \
        --matching "$matching" 2>&1 | tee -a "results/${name}.log"
}

# Run unsupervised experiments (hungarian matching)
for cfg in "${HUNGARIAN_CONFIGS[@]}"; do
    run_experiment "$cfg" "hungarian"
done

# Run few-shot and text-guided experiments (semantic matching)
for cfg in "${SEMANTIC_CONFIGS[@]}"; do
    run_experiment "$cfg" "semantic"
done

echo ""
echo "All experiments finished at $(date)"
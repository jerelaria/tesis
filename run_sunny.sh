#!/bin/bash
# run_sunny_training.sh
# ---------------------
# Sequential unsupervised training runs on the full Sunnybrook dataset
# for all four feature versions (v0, v3, v4, v5).
#
# Each version runs unsup_hdbscan_refine (except v0 which runs
# unsup_baseline only, since it has no clustering).
# Evaluation is run automatically by run_experiment.sh after each experiment.
#
# Usage:
#   tmux new -s jere_sunny_training
#   ./run_sunny_training.sh
#
# Monitor progress:
#   tail -f results/v3_extended_training_sunny/SunnybrookNicoSent/unsup_hdbscan_refine.log

set -e

cd /media/apoloml/DATOS_2/Tesis_Cosegmentacion

START_TIME=$(date +%s)
echo ""
echo "########################################################"
echo "  Sunnybrook training runs"
echo "  $(date)"
echo "########################################################"

# ── v0_baseline: grid prompting only, no clustering ──────────────────────────
echo ""
echo "========================================================"
echo "  [1/4] v0_baseline_training_sunny"
echo "========================================================"

./run_experiment.sh v0_baseline_training_sunny SunnybrookNicoSent/images \
    --skip-fewshot \
    --skip-textguided \
    --unsup-configs configs/experiments/unsup_baseline.yaml

# ── v3_extended: 16 standardized moments ─────────────────────────────────────
echo ""
echo "========================================================"
echo "  [2/4] v3_extended_training_sunny"
echo "========================================================"

./run_experiment.sh v3_extended_training_sunny SunnybrookNicoSent/images \
    --skip-fewshot \
    --skip-textguided \
    --unsup-configs configs/experiments/unsup_hdbscan_refine.yaml \
    --override \
        labeler.standardize=true \
        extractor.features=V,Cx,Cy,Dx,Dy,L,ecc,solidity,extent,compact,hu0,hu1,hu2,intensity_mean,intensity_std,orientation \
        labeler.features=V,Cx,Cy,Dx,Dy,L,ecc,solidity,extent,compact,hu0,hu1,hu2,intensity_mean,intensity_std,orientation

# ── v4_emb_only: SAM2 encoder embeddings, no PCA ─────────────────────────────
echo ""
echo "========================================================"
echo "  [3/4] v4_emb_only_training_sunny"
echo "========================================================"

./run_experiment.sh v4_emb_only_training_sunny SunnybrookNicoSent/images \
    --skip-fewshot \
    --skip-textguided \
    --unsup-configs configs/experiments/unsup_hdbscan_refine.yaml \
    --override \
        labeler.standardize=true \
        labeler.features=none \
        labeler.embedding.enabled=true \
        labeler.embedding.reduction=none

# ── v5_ext_emb_red: 16 moments + PCA-reduced embeddings (16 dims) ────────────
echo ""
echo "========================================================"
echo "  [4/4] v5_ext_emb_red_training_sunny"
echo "========================================================"

./run_experiment.sh v5_ext_emb_red_training_sunny SunnybrookNicoSent/images \
    --skip-fewshot \
    --skip-textguided \
    --unsup-configs configs/experiments/unsup_hdbscan_refine.yaml \
    --override \
        labeler.standardize=true \
        extractor.features=V,Cx,Cy,Dx,Dy,L,ecc,solidity,extent,compact,hu0,hu1,hu2,intensity_mean,intensity_std,orientation \
        labeler.features=V,Cx,Cy,Dx,Dy,L,ecc,solidity,extent,compact,hu0,hu1,hu2,intensity_mean,intensity_std,orientation \
        labeler.embedding.enabled=true \
        labeler.embedding.reduction=pca \
        labeler.embedding.n_components=16

# ── Done ──────────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "########################################################"
echo "  All Sunnybrook training runs complete."
echo "  Total time: ${ELAPSED} min"
echo "  $(date)"
echo "########################################################"
echo ""
echo "Results:"
for f in results/*_training_sunny/SunnybrookNicoSent/*/summary.json; do
    organs=$(python3 -c "import json; d=json.load(open('$f')); print(list(d['per_organ'].keys()))")
    echo "  $f"
    echo "    organs: $organs"
done
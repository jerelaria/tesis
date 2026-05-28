#!/usr/bin/env bash
# run_overnight.sh
# ----------------
# Master orchestration script for the synthetic curves block (Block 3).
#
# Sequentially:
#   1. Generates the 25 curve config YAMLs.
#   2. Generates the 25 synthetic datasets.
#   3. Runs the pipeline + evaluation on all 25 datasets.
#   4. Runs the analyze_curves.py script to produce the 5 figures + CSV.
#
# Each step is idempotent and produces stdout logs to a timestamped file
# so progress can be inspected from another terminal:
#     tail -f overnight_<timestamp>.log
#
# Usage:
#     ./run_overnight.sh [version]
#
# Default version: v_synthetic_curves.

set -euo pipefail

VERSION="${1:-v_synthetic_curves}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="overnight_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================"
echo "  Overnight run started at $(date)"
echo "  Version: $VERSION"
echo "  Log: $LOG_FILE"
echo "================================================================"

# ---- Step 1: generate 25 curve config YAMLs -----------------------
echo ""
echo "[Step 1/4] Generating curve configs ..."
python data/scripts/generate_curve_configs.py

# ---- Step 2: generate 25 datasets ---------------------------------
echo ""
echo "[Step 2/4] Generating curve datasets ..."
for cfg in configs/datasets/synthetic_curves/*.yaml; do
    name=$(basename "$cfg" .yaml)
    if [ -d "data/raw/$name/images" ] && \
       [ "$(ls data/raw/$name/images | wc -l)" -gt 0 ]; then
        echo "  [SKIP] $name already exists"
    else
        echo "  Generating $name ..."
        python data/scripts/generate_synthetic.py --config "$cfg"
    fi
done

# ---- Step 3: run pipeline across all versions × 25 datasets ----------
echo ""
echo "[Step 3/4] Running pipeline + evaluation across versions ..."
bash run_synthetic_curves_versions.sh

# ---- Step 4: produce per-version curves + cross-version comparison ---
echo ""
echo "[Step 4/4] Generating curve plots ..."
# Per-version curves (current behavior, one set of 5 figures per version)
for vdir in results/v*_*/; do
    vname=$(basename "$vdir")
    if [ -d "$vdir" ]; then
        echo "  Analyzing $vname ..."
        python data/scripts/analyze_curves.py \
            --results-root "$vdir" \
            --output "${vdir}curves_analysis/"
    fi
done

echo ""
echo "================================================================"
echo "  Overnight run finished at $(date)"
echo "  Per-version plots: results/<version>/curves_analysis/"
echo "  Versions run: see results/ subdirectories"
echo "================================================================"
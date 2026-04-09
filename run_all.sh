set -e
 
CONFIGS=(
    configs/unsup.yaml
    configs/unsup_refine.yaml
    configs/unsup_refine_kmeans.yaml
    configs/fs_indep_1ref.yaml
    configs/fs_indep_1ref_refine.yaml
    configs/fs_indep_4ref.yaml
    configs/fs_indep_4ref_refine.yaml
    configs/fs_iter_1ref.yaml
    configs/fs_iter_1ref_refine.yaml
    configs/fs_iter_4ref.yaml
    configs/fs_iter_4ref_refine.yaml
    configs/fs_iter_4ref_refine_kmeans.yaml
    configs/tg.yaml
)
 
for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Running: $cfg"
    echo "  $(date)"
    echo "============================================================"
    python -m project.main --config "$cfg" 2>&1 | tee "results/$(basename "$cfg" .yaml).log"
done
 
echo ""
echo "All experiments finished at $(date)"
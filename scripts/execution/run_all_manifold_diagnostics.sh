#!/bin/bash
# Master script to run all Manifold Law diagnostic tests

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT}/src"

echo "============================================================"
echo "Manifold Law Diagnostic Suite - Full Execution"
echo "============================================================"
echo ""

# Dataset paths
ADAMSON_ADATA="data/gears_pert_data/adamson/perturb_processed.h5ad"
ADAMSON_SPLIT="results/goal_2_baselines/splits/adamson_split_seed1.json"
K562_ADATA="data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad"
K562_SPLIT="results/goal_2_baselines/splits/replogle_k562_essential_split_seed1.json"
K562_ANNOT="data/annotations/replogle_k562_functional_classes_go.tsv"

OUTPUT_BASE="results/manifold_law_diagnostics"

# Epic 1: Curvature Sweep
echo "============================================================"
echo "EPIC 1: Curvature Sweep"
echo "============================================================"
echo ""

BASELINES=("lpm_selftrained" "lpm_randomGeneEmb")

for baseline in "${BASELINES[@]}"; do
    echo "Running ${baseline} on Adamson..."
    python -m goal_3_prediction.lsft.curvature_sweep \
        --adata_path "${ADAMSON_ADATA}" \
        --split_config "${ADAMSON_SPLIT}" \
        --dataset_name adamson \
        --baseline_type "${baseline}" \
        --output_dir "${OUTPUT_BASE}/epic1_curvature" \
        --k_list 3 5 10 20 30 50 \
        --pca_dim 10 \
        --ridge_penalty 0.1 \
        --seed 1 \
        2>&1 | tee "${OUTPUT_BASE}/epic1_curvature/log_adamson_${baseline}.log" | tail -20
    
    echo ""
done

echo "Epic 1 complete!"
echo ""

# Epic 4: Direction-Flip Probe
echo "============================================================"
echo "EPIC 4: Direction-Flip Probe"
echo "============================================================"
echo ""
echo "Epic 4 requires existing LSFT results - will run after Epic 1 completes fully"
echo ""

# Epic 5: Tangent Alignment
echo "============================================================"
echo "EPIC 5: Tangent Alignment"
echo "============================================================"
echo ""
echo "Epic 5 requires existing LSFT results - will run after Epic 1 completes fully"
echo ""

echo "============================================================"
echo "Diagnostic Suite Run Complete!"
echo "============================================================"
echo ""
echo "Results saved to: ${OUTPUT_BASE}/"
echo ""
echo "Next steps:"
echo "  1. Review Epic 1 curvature plots"
echo "  2. Run Epic 4 & 5 with LSFT results"
echo "  3. Refine Epic 2 & 3 integration"
echo ""


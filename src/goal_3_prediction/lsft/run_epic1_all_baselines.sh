#!/bin/bash
# Run Epic 1 (Curvature Sweep) on all baselines for all datasets

set -e

# Get absolute path to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT}/src"

OUTPUT_DIR="${REPO_ROOT}/results/manifold_law_diagnostics/epic1_curvature"
K_LIST="3 5 10 20 30 50"

# Dataset paths (use absolute paths)
ADAMSON_ADATA="${REPO_ROOT}/data/gears_pert_data/adamson/perturb_processed.h5ad"
ADAMSON_SPLIT="${REPO_ROOT}/results/goal_2_baselines/splits/adamson_split_seed1.json"

K562_ADATA="data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad"
K562_SPLIT="${REPO_ROOT}/results/goal_2_baselines/splits/replogle_k562_essential_split_seed1.json"

RPE1_ADATA="data/gears_pert_data/replogle_rpe1_essential/perturb_processed.h5ad"
RPE1_SPLIT="${REPO_ROOT}/results/goal_2_baselines/splits/replogle_rpe1_essential_split_seed1.json"

# Baselines to test (excluding some that may need special setup)
BASELINES=(
    "lpm_selftrained"
    "lpm_randomGeneEmb"
    "lpm_randomPertEmb"
    # "lpm_scgptGeneEmb"  # May need checkpoint
    # "lpm_scFoundationGeneEmb"  # May need checkpoint
    # "lpm_gearsPertEmb"  # May need embeddings file
    # "lpm_k562PertEmb"  # Cross-dataset
    # "lpm_rpe1PertEmb"  # Cross-dataset
)

echo "============================================================"
echo "EPIC 1: Curvature Sweep - All Baselines & Datasets"
echo "============================================================"
echo ""
echo "Baselines: ${BASELINES[@]}"
echo "K values: ${K_LIST}"
echo ""

# Function to run on one dataset
run_dataset() {
    local dataset_name=$1
    local adata_path=$2
    local split_path=$3
    
    if [ ! -f "$adata_path" ] || [ ! -f "$split_path" ]; then
        echo "⚠️  Skipping ${dataset_name}: Missing files"
        return
    fi
    
    echo "============================================================"
    echo "Dataset: ${dataset_name}"
    echo "============================================================"
    echo ""
    
    for baseline in "${BASELINES[@]}"; do
        echo "  Running: ${baseline}"
        
        python -m goal_3_prediction.lsft.curvature_sweep \
            --adata_path "${adata_path}" \
            --split_config "${split_path}" \
            --dataset_name "${dataset_name}" \
            --baseline_type "${baseline}" \
            --output_dir "${OUTPUT_DIR}" \
            --k_list ${K_LIST} \
            --pca_dim 10 \
            --ridge_penalty 0.1 \
            --seed 1 \
            2>&1 | tee "${OUTPUT_DIR}/log_${dataset_name}_${baseline}.log" | grep -E "(Running|Summary|Curvature|Saved|ERROR)" || true
        
        echo ""
    done
}

# Run on all datasets
run_dataset "adamson" "${ADAMSON_ADATA}" "${ADAMSON_SPLIT}"
run_dataset "k562" "${K562_ADATA}" "${K562_SPLIT}"
run_dataset "rpe1" "${RPE1_ADATA}" "${RPE1_SPLIT}"

echo "============================================================"
echo "Epic 1 Complete!"
echo "============================================================"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"


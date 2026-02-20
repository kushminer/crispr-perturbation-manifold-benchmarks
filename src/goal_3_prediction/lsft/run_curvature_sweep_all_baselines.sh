#!/bin/bash
# Run curvature sweep on all baselines for a dataset

set -e

DATASET_NAME=${1:-"adamson"}
ADATA_PATH=${2:-"data/gears_pert_data/adamson/perturb_processed.h5ad"}
SPLIT_CONFIG=${3:-"results/goal_2_baselines/splits/adamson_split_seed1.json"}
OUTPUT_DIR=${4:-"results/manifold_law_diagnostics/epic1_curvature"}
K_LIST=${5:-"3 5 10 20 30 50"}

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

BASELINES=(
    "lpm_selftrained"
    "lpm_randomGeneEmb"
    "lpm_scgptGeneEmb"
)

echo "Running curvature sweep for ${DATASET_NAME}"
echo "Baselines: ${BASELINES[@]}"
echo "K values: ${K_LIST}"
echo ""

for baseline in "${BASELINES[@]}"; do
    echo "=========================================="
    echo "Running: ${baseline}"
    echo "=========================================="
    
    python -m goal_3_prediction.lsft.curvature_sweep \
        --adata_path "${ADATA_PATH}" \
        --split_config "${SPLIT_CONFIG}" \
        --dataset_name "${DATASET_NAME}" \
        --baseline_type "${baseline}" \
        --output_dir "${OUTPUT_DIR}" \
        --k_list ${K_LIST} \
        --pca_dim 10 \
        --ridge_penalty 0.1 \
        --seed 1 \
        2>&1 | tee "${OUTPUT_DIR}/log_${DATASET_NAME}_${baseline}.log"
    
    echo ""
done

echo "=========================================="
echo "Curvature sweep complete for ${DATASET_NAME}!"
echo "=========================================="


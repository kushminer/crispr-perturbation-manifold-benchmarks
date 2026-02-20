#!/bin/bash
# Run remaining lpm_gearsPertEmb evaluations for all 3 datasets

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Parameters (matching main script)
TOP_PCTS="0.01 0.05 0.10"
PCA_DIM=10
RIDGE_PENALTY=0.1
SEED=1
N_BOOT=1000
N_PERM=10000
OUTPUT_BASE="results/goal_3_prediction/lsft_resampling"
BASELINE="lpm_gearsPertEmb"

# Dataset paths
ADAMSON_DATA="data/gears_pert_data/adamson/perturb_processed.h5ad"
K562_DATA="data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad"
RPE1_DATA="data/gears_pert_data/replogle_rpe1_essential/perturb_processed.h5ad"

# Split paths
ADAMSON_SPLIT="results/goal_2_baselines/splits/adamson_split_seed1.json"
K562_SPLIT="results/goal_2_baselines/splits/replogle_k562_essential_split_seed1.json"
RPE1_SPLIT="results/goal_2_baselines/splits/replogle_rpe1_essential_split_seed1.json"

echo "============================================================"
echo "Running remaining lpm_gearsPertEmb evaluations"
echo "============================================================"
echo ""

# Adamson
if [ -f "$ADAMSON_DATA" ]; then
    echo "Running Adamson..."
    PYTHONPATH=src python -m goal_3_prediction.lsft.run_lsft_with_resampling \
        --adata_path "$ADAMSON_DATA" \
        --split_config "$ADAMSON_SPLIT" \
        --dataset_name adamson \
        --baseline_type "$BASELINE" \
        --output_dir "${OUTPUT_BASE}/adamson" \
        --top_pcts ${TOP_PCTS} \
        --pca_dim ${PCA_DIM} \
        --ridge_penalty ${RIDGE_PENALTY} \
        --seed ${SEED} \
        --n_boot ${N_BOOT} \
        --n_perm ${N_PERM} 2>&1 | tee "${OUTPUT_BASE}/adamson/lsft_adamson_${BASELINE}_run.log"
    echo "✅ Adamson complete"
    echo ""
fi

# K562
if [ -f "$K562_DATA" ]; then
    echo "Running K562..."
    PYTHONPATH=src python -m goal_3_prediction.lsft.run_lsft_with_resampling \
        --adata_path "$K562_DATA" \
        --split_config "$K562_SPLIT" \
        --dataset_name k562 \
        --baseline_type "$BASELINE" \
        --output_dir "${OUTPUT_BASE}/k562" \
        --top_pcts ${TOP_PCTS} \
        --pca_dim ${PCA_DIM} \
        --ridge_penalty ${RIDGE_PENALTY} \
        --seed ${SEED} \
        --n_boot ${N_BOOT} \
        --n_perm ${N_PERM} 2>&1 | tee "${OUTPUT_BASE}/k562/lsft_k562_${BASELINE}_run.log"
    echo "✅ K562 complete"
    echo ""
fi

# RPE1
if [ -f "$RPE1_DATA" ]; then
    echo "Running RPE1..."
    PYTHONPATH=src python -m goal_3_prediction.lsft.run_lsft_with_resampling \
        --adata_path "$RPE1_DATA" \
        --split_config "$RPE1_SPLIT" \
        --dataset_name rpe1 \
        --baseline_type "$BASELINE" \
        --output_dir "${OUTPUT_BASE}/rpe1" \
        --top_pcts ${TOP_PCTS} \
        --pca_dim ${PCA_DIM} \
        --ridge_penalty ${RIDGE_PENALTY} \
        --seed ${SEED} \
        --n_boot ${N_BOOT} \
        --n_perm ${N_PERM} 2>&1 | tee "${OUTPUT_BASE}/rpe1/lsft_rpe1_${BASELINE}_run.log"
    echo "✅ RPE1 complete"
    echo ""
fi

echo "============================================================"
echo "✅ All remaining evaluations complete!"
echo "============================================================"

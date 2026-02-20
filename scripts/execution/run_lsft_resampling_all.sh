#!/bin/bash
# Run LSFT evaluation with resampling on all datasets and baselines

set -e  # Exit on error

# Dataset paths
ADAMSON_DATA_PATH="data/gears_pert_data/adamson/perturb_processed.h5ad"
K562_DATA_PATH="data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad"
RPE1_DATA_PATH="data/gears_pert_data/replogle_rpe1_essential/perturb_processed.h5ad"

# Split paths
ADAMSON_SPLIT="results/goal_2_baselines/splits/adamson_split_seed1.json"
K562_SPLIT="results/goal_2_baselines/splits/replogle_k562_essential_split_seed1.json"
RPE1_SPLIT="results/goal_2_baselines/splits/replogle_rpe1_essential_split_seed1.json"

# All baselines (excluding mean_response for LSFT as it doesn't use embeddings)
BASELINES=(
    "lpm_selftrained"
    "lpm_randomPertEmb"
    "lpm_randomGeneEmb"
    "lpm_scgptGeneEmb"
    "lpm_scFoundationGeneEmb"
    "lpm_gearsPertEmb"
    "lpm_k562PertEmb"
    "lpm_rpe1PertEmb"
)

# Parameters
TOP_PCTS="0.01 0.05 0.10"
PCA_DIM=10
RIDGE_PENALTY=0.1
SEED=1
N_BOOT=1000
N_PERM=10000

# Output directory
OUTPUT_BASE="results/goal_3_prediction/lsft_resampling"

echo "============================================================"
echo "LSFT Resampling Evaluation: All Datasets & Baselines"
echo "============================================================"
echo ""
echo "Parameters:"
echo "  - Bootstrap samples: ${N_BOOT}"
echo "  - Permutations: ${N_PERM}"
echo "  - Top percentages: ${TOP_PCTS}"
echo "  - PCA dimension: ${PCA_DIM}"
echo "  - Ridge penalty: ${RIDGE_PENALTY}"
echo "  - Seed: ${SEED}"
echo ""

# Function to run LSFT with resampling
run_lsft_resampling() {
    local dataset_name=$1
    local data_path=$2
    local split_path=$3
    local baseline=$4
    local output_dir="${OUTPUT_BASE}/${dataset_name}"
    
    echo "  Running: ${dataset_name} - ${baseline}"
    echo "    Data: ${data_path}"
    echo "    Split: ${split_path}"
    echo "    Output: ${output_dir}"
    
    PYTHONPATH=src python -m goal_3_prediction.lsft.run_lsft_with_resampling \
        --adata_path "${data_path}" \
        --split_config "${split_path}" \
        --dataset_name "${dataset_name}" \
        --baseline_type "${baseline}" \
        --output_dir "${output_dir}" \
        --top_pcts ${TOP_PCTS} \
        --pca_dim ${PCA_DIM} \
        --ridge_penalty ${RIDGE_PENALTY} \
        --seed ${SEED} \
        --n_boot ${N_BOOT} \
        --n_perm ${N_PERM} 2>&1 | tee "${output_dir}/lsft_${dataset_name}_${baseline}_run.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "    ✅ Complete"
    else
        echo "    ❌ Failed"
        return 1
    fi
    echo ""
}

# Adamson
echo "============================================================"
echo "ADAMSON Dataset"
echo "============================================================"
if [ ! -f "$ADAMSON_DATA_PATH" ]; then
    echo "  ⚠️  Adamson data not found at $ADAMSON_DATA_PATH"
    echo "  Skipping Adamson..."
else
    for baseline in "${BASELINES[@]}"; do
        run_lsft_resampling "adamson" "$ADAMSON_DATA_PATH" "$ADAMSON_SPLIT" "$baseline" || echo "    ⚠️  Failed, continuing..."
    done
fi

# Replogle K562
echo "============================================================"
echo "REPLOGLE K562 Dataset"
echo "============================================================"
if [ ! -f "$K562_DATA_PATH" ]; then
    echo "  ⚠️  K562 data not found at $K562_DATA_PATH"
    echo "  Skipping K562..."
else
    for baseline in "${BASELINES[@]}"; do
        run_lsft_resampling "k562" "$K562_DATA_PATH" "$K562_SPLIT" "$baseline" || echo "    ⚠️  Failed, continuing..."
    done
fi

# Replogle RPE1
echo "============================================================"
echo "REPLOGLE RPE1 Dataset"
echo "============================================================"
if [ ! -f "$RPE1_DATA_PATH" ]; then
    echo "  ⚠️  RPE1 data not found at $RPE1_DATA_PATH"
    echo "  Skipping RPE1..."
else
    for baseline in "${BASELINES[@]}"; do
        run_lsft_resampling "rpe1" "$RPE1_DATA_PATH" "$RPE1_SPLIT" "$baseline" || echo "    ⚠️  Failed, continuing..."
    done
fi

echo ""
echo "============================================================"
echo "✅ LSFT Resampling Evaluation Complete!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - ${OUTPUT_BASE}/adamson/"
echo "  - ${OUTPUT_BASE}/k562/"
echo "  - ${OUTPUT_BASE}/rpe1/"
echo ""
echo "Each baseline has:"
echo "  - Standardized CSV/JSONL/Parquet outputs"
echo "  - Summary JSON with bootstrap CIs"
echo "  - Baseline comparisons (if multiple baselines)"
echo "  - Hardness regressions with CIs"
echo ""


#!/bin/bash
# Run LSFT evaluation on all datasets and baselines with corrected splits

set -e  # Exit on error

# Dataset paths
ADAMSON_DATA_PATH="../paper/benchmark/data/gears_pert_data/adamson/perturb_processed.h5ad"
K562_DATA_PATH="/Users/samuelminer/Documents/classes/nih_research/data_replogle_k562_essential/perturb_processed.h5ad"
RPE1_DATA_PATH="/Users/samuelminer/Documents/classes/nih_research/data_replogle_rpe1_essential/perturb_processed.h5ad"

# Split paths (corrected splits)
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

echo "=========================================="
echo "LSFT Evaluation: All Datasets & Baselines"
echo "=========================================="
echo ""
echo "This will regenerate all LSFT results with corrected splits"
echo ""

# Function to run LSFT for a dataset and baseline
run_lsft() {
    local dataset_name=$1
    local data_path=$2
    local split_path=$3
    local baseline=$4
    local output_dir="results/goal_3_prediction/lsft/${dataset_name}"
    
    echo "  Running: ${dataset_name} - ${baseline}"
    
    PYTHONPATH=src python -m goal_3_prediction.lsft.lsft \
        --adata_path "${data_path}" \
        --split_config "${split_path}" \
        --dataset_name "${dataset_name}" \
        --output_dir "${output_dir}" \
        --baseline_type "${baseline}" \
        --top_pcts ${TOP_PCTS} \
        --pca_dim ${PCA_DIM} \
        --ridge_penalty ${RIDGE_PENALTY} \
        --seed ${SEED} 2>&1 | grep -E "(Running LSFT|Baseline:|Top percentages:|Summary|Saved results|Error|Failed)" || true
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "    ✅ Complete"
    else
        echo "    ❌ Failed"
    fi
    echo ""
}

# Adamson
echo "=========================================="
echo "ADAMSON Dataset"
echo "=========================================="
if [ ! -f "$ADAMSON_DATA_PATH" ]; then
    echo "  ⚠️  Adamson data not found at $ADAMSON_DATA_PATH"
    echo "  Skipping Adamson..."
else
    for baseline in "${BASELINES[@]}"; do
        run_lsft "adamson" "$ADAMSON_DATA_PATH" "$ADAMSON_SPLIT" "$baseline"
    done
fi

# Replogle K562
echo "=========================================="
echo "REPLOGLE K562 Dataset"
echo "=========================================="
if [ ! -f "$K562_DATA_PATH" ]; then
    echo "  ⚠️  K562 data not found at $K562_DATA_PATH"
    echo "  Skipping K562..."
else
    for baseline in "${BASELINES[@]}"; do
        run_lsft "k562" "$K562_DATA_PATH" "$K562_SPLIT" "$baseline"
    done
fi

# Replogle RPE1
echo "=========================================="
echo "REPLOGLE RPE1 Dataset"
echo "=========================================="
if [ ! -f "$RPE1_DATA_PATH" ]; then
    echo "  ⚠️  RPE1 data not found at $RPE1_DATA_PATH"
    echo "  Skipping RPE1..."
else
    for baseline in "${BASELINES[@]}"; do
        run_lsft "rpe1" "$RPE1_DATA_PATH" "$RPE1_SPLIT" "$baseline"
    done
fi

echo ""
echo "=========================================="
echo "✅ LSFT Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - results/goal_3_prediction/lsft/adamson/"
echo "  - results/goal_3_prediction/lsft/k562/"
echo "  - results/goal_3_prediction/lsft/rpe1/"
echo ""


#!/bin/bash
# Run LOGO evaluation with resampling on all datasets

set -e  # Exit on error

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "LOGO Resampling Evaluation: All Datasets"
echo "============================================================"
echo ""

# Check if PYTHONPATH is set
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH="src:$PYTHONPATH"
fi

# Parameters
PCA_DIM=10
RIDGE_PENALTY=0.1
SEED=1
N_BOOT=1000
N_PERM=10000
CLASS_NAME="Transcription"

# Output directory
OUTPUT_BASE="results/goal_3_prediction/functional_class_holdout_resampling"

echo "Parameters:"
echo "  - Bootstrap samples: ${N_BOOT}"
echo "  - Permutations: ${N_PERM}"
echo "  - Class: ${CLASS_NAME}"
echo "  - PCA dimension: ${PCA_DIM}"
echo "  - Ridge penalty: ${RIDGE_PENALTY}"
echo "  - Seed: ${SEED}"
echo ""

# Adamson
echo "============================================================"
echo "1. Running LOGO with Resampling on Adamson..."
echo "   Dataset: adamson"
echo "   Class: ${CLASS_NAME}"
echo ""

ADAMSON_DATA_PATH="data/gears_pert_data/adamson/perturb_processed.h5ad"
ADAMSON_ANNOTATION_PATH="data/annotations/adamson_functional_classes_enriched.tsv"

if [ ! -f "$ADAMSON_DATA_PATH" ]; then
    echo "   ⚠️  Adamson data not found at: $ADAMSON_DATA_PATH"
    echo "   Skipping Adamson..."
else
    if [ ! -f "$ADAMSON_ANNOTATION_PATH" ]; then
        echo "   ⚠️  Adamson annotation not found at: $ADAMSON_ANNOTATION_PATH"
        echo "   Skipping Adamson..."
    else
        PYTHONPATH=src python -m goal_3_prediction.functional_class_holdout.logo_resampling \
            --adata_path "$ADAMSON_DATA_PATH" \
            --annotation_path "$ADAMSON_ANNOTATION_PATH" \
            --dataset_name adamson \
            --output_dir "${OUTPUT_BASE}/adamson" \
            --class_name "${CLASS_NAME}" \
            --pca_dim ${PCA_DIM} \
            --ridge_penalty ${RIDGE_PENALTY} \
            --seed ${SEED} \
            --n_boot ${N_BOOT} \
            --n_perm ${N_PERM} 2>&1 | tee "${OUTPUT_BASE}/adamson/logo_adamson_${CLASS_NAME}_run.log"
        
        if [ $? -eq 0 ]; then
            echo "   ✅ Adamson LOGO with resampling complete"
        else
            echo "   ❌ Adamson LOGO failed"
        fi
    fi
fi

echo ""
echo "============================================================"
echo "2. Running LOGO with Resampling on Replogle K562..."
echo "   Dataset: replogle_k562_essential"
echo "   Class: ${CLASS_NAME}"
echo ""

K562_DATA_PATH="${K562_DATA_PATH:-data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad}"
K562_ANNOTATION_PATH="data/annotations/replogle_k562_functional_classes_go.tsv"

if [ ! -f "$K562_DATA_PATH" ]; then
    echo "   ⚠️  K562 data not found at: $K562_DATA_PATH"
    echo "   Set K562_DATA_PATH environment variable or update path in script"
    echo "   Skipping K562..."
else
    if [ ! -f "$K562_ANNOTATION_PATH" ]; then
        echo "   ⚠️  K562 annotation not found at: $K562_ANNOTATION_PATH"
        echo "   Skipping K562..."
    else
        PYTHONPATH=src python -m goal_3_prediction.functional_class_holdout.logo_resampling \
            --adata_path "$K562_DATA_PATH" \
            --annotation_path "$K562_ANNOTATION_PATH" \
            --dataset_name replogle_k562_essential \
            --output_dir "${OUTPUT_BASE}/replogle_k562" \
            --class_name "${CLASS_NAME}" \
            --pca_dim ${PCA_DIM} \
            --ridge_penalty ${RIDGE_PENALTY} \
            --seed ${SEED} \
            --n_boot ${N_BOOT} \
            --n_perm ${N_PERM} 2>&1 | tee "${OUTPUT_BASE}/replogle_k562/logo_replogle_k562_${CLASS_NAME}_run.log"
        
        if [ $? -eq 0 ]; then
            echo "   ✅ K562 LOGO with resampling complete"
        else
            echo "   ❌ K562 LOGO failed"
        fi
    fi
fi

echo ""
echo "============================================================"
echo "3. Running LOGO with Resampling on Replogle RPE1..."
echo "   Dataset: replogle_rpe1_essential"
echo "   Class: ${CLASS_NAME}"
echo ""

RPE1_DATA_PATH="${RPE1_DATA_PATH:-data/gears_pert_data/replogle_rpe1_essential/perturb_processed.h5ad}"
# Use K562 annotations for RPE1 if RPE1-specific not available
RPE1_ANNOTATION_PATH="${RPE1_ANNOTATION_PATH:-data/annotations/replogle_k562_functional_classes_go.tsv}"

if [ ! -f "$RPE1_DATA_PATH" ]; then
    echo "   ⚠️  RPE1 data not found at: $RPE1_DATA_PATH"
    echo "   Set RPE1_DATA_PATH environment variable or update path in script"
    echo "   Skipping RPE1..."
else
    if [ ! -f "$RPE1_ANNOTATION_PATH" ]; then
        echo "   ⚠️  RPE1 annotation not found at: $RPE1_ANNOTATION_PATH"
        echo "   Skipping RPE1..."
    else
        PYTHONPATH=src python -m goal_3_prediction.functional_class_holdout.logo_resampling \
            --adata_path "$RPE1_DATA_PATH" \
            --annotation_path "$RPE1_ANNOTATION_PATH" \
            --dataset_name replogle_rpe1_essential \
            --output_dir "${OUTPUT_BASE}/replogle_rpe1" \
            --class_name "${CLASS_NAME}" \
            --pca_dim ${PCA_DIM} \
            --ridge_penalty ${RIDGE_PENALTY} \
            --seed ${SEED} \
            --n_boot ${N_BOOT} \
            --n_perm ${N_PERM} 2>&1 | tee "${OUTPUT_BASE}/replogle_rpe1/logo_replogle_rpe1_${CLASS_NAME}_run.log"
        
        if [ $? -eq 0 ]; then
            echo "   ✅ RPE1 LOGO with resampling complete"
        else
            echo "   ❌ RPE1 LOGO failed"
        fi
    fi
fi

echo ""
echo "============================================================"
echo "✅ LOGO Resampling Evaluation Complete!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - ${OUTPUT_BASE}/adamson/"
if [ -f "$K562_DATA_PATH" ] && [ -f "$K562_ANNOTATION_PATH" ]; then
    echo "  - ${OUTPUT_BASE}/replogle_k562/"
fi
if [ -f "$RPE1_DATA_PATH" ] && [ -f "$RPE1_ANNOTATION_PATH" ]; then
    echo "  - ${OUTPUT_BASE}/replogle_rpe1/"
fi
echo ""
echo "Each dataset has:"
echo "  - Standardized CSV/JSONL/Parquet outputs"
echo "  - Summary JSON with bootstrap CIs"
echo "  - Baseline comparisons with p-values"
echo ""


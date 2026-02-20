#!/bin/bash
# Run LOGO evaluation on all datasets (Adamson, K562, RPE1)

set -e  # Exit on error

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "LOGO Evaluation: All Datasets"
echo "=========================================="
echo ""

# Check if PYTHONPATH is set
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH="src:$PYTHONPATH"
fi

# Adamson
echo "1. Running LOGO on Adamson..."
echo "   Dataset: adamson"
echo "   Class: Transcription (5 genes)"
echo ""

PYTHONPATH=src python -m goal_3_prediction.functional_class_holdout.logo \
    --adata_path data/gears_pert_data/adamson/perturb_processed.h5ad \
    --annotation_path data/annotations/adamson_functional_classes_enriched.tsv \
    --dataset_name adamson \
    --output_dir results/goal_3_prediction/functional_class_holdout/adamson \
    --class_name Transcription \
    --pca_dim 10 \
    --ridge_penalty 0.1 \
    --seed 1

if [ $? -eq 0 ]; then
    echo "   ✅ Adamson LOGO complete"
    
    # Compare baselines
    echo "   Running baseline comparison..."
    PYTHONPATH=src python -m goal_3_prediction.functional_class_holdout.compare_baselines \
        --results_csv results/goal_3_prediction/functional_class_holdout/adamson/logo_adamson_transcription_results.csv \
        --output_dir results/goal_3_prediction/functional_class_holdout/adamson \
        --dataset_name adamson \
        --class_name Transcription
    
    echo "   ✅ Adamson comparison complete"
else
    echo "   ❌ Adamson LOGO failed"
    exit 1
fi

echo ""
echo "2. Running LOGO on Replogle K562..."
echo "   Dataset: replogle_k562_essential"
echo "   Class: Transcription (397 genes)"
echo ""

# Check if K562 data path exists
K562_DATA_PATH="${K562_DATA_PATH:-data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad}"

if [ ! -f "$K562_DATA_PATH" ]; then
    echo "   ⚠️  K562 data not found at: $K562_DATA_PATH"
    echo "   Set K562_DATA_PATH environment variable or update path in script"
    echo "   Skipping K562..."
else
    PYTHONPATH=src python -m goal_3_prediction.functional_class_holdout.logo \
        --adata_path "$K562_DATA_PATH" \
        --annotation_path ../data/annotations/replogle_k562_functional_classes_go.tsv \
        --dataset_name replogle_k562_essential \
        --output_dir results/goal_3_prediction/functional_class_holdout/replogle_k562 \
        --class_name Transcription \
        --pca_dim 10 \
        --ridge_penalty 0.1 \
        --seed 1
    
    if [ $? -eq 0 ]; then
        echo "   ✅ K562 LOGO complete"
        
        # Compare baselines
        echo "   Running baseline comparison..."
        PYTHONPATH=src python -m goal_3_prediction.functional_class_holdout.compare_baselines \
            --results_csv results/goal_3_prediction/functional_class_holdout/replogle_k562/logo_replogle_k562_essential_transcription_results.csv \
            --output_dir results/goal_3_prediction/functional_class_holdout/replogle_k562 \
            --dataset_name replogle_k562_essential \
            --class_name Transcription
        
        echo "   ✅ K562 comparison complete"
    else
        echo "   ❌ K562 LOGO failed"
    fi
fi

echo ""
echo "3. Running LOGO on Replogle RPE1..."
echo "   Dataset: replogle_rpe1_essential"
echo "   Class: Transcription (to be determined)"
echo ""

# Check if RPE1 data path exists
RPE1_DATA_PATH="${RPE1_DATA_PATH:-data/gears_pert_data/replogle_rpe1_essential/perturb_processed.h5ad}"

if [ ! -f "$RPE1_DATA_PATH" ]; then
    echo "   ⚠️  RPE1 data not found at: $RPE1_DATA_PATH"
    echo "   Set RPE1_DATA_PATH environment variable or update path in script"
    echo "   Skipping RPE1..."
else
    # Note: RPE1 may need its own annotation file
    # For now, use K562 annotations if available
    RPE1_ANNOTATION_PATH="${RPE1_ANNOTATION_PATH:-../data/annotations/replogle_k562_functional_classes_go.tsv}"
    
    if [ ! -f "$RPE1_ANNOTATION_PATH" ]; then
        echo "   ⚠️  RPE1 annotation file not found"
        echo "   Skipping RPE1..."
    else
        PYTHONPATH=src python -m goal_3_prediction.functional_class_holdout.logo \
            --adata_path "$RPE1_DATA_PATH" \
            --annotation_path "$RPE1_ANNOTATION_PATH" \
            --dataset_name replogle_rpe1_essential \
            --output_dir results/goal_3_prediction/functional_class_holdout/replogle_rpe1 \
            --class_name Transcription \
            --pca_dim 10 \
            --ridge_penalty 0.1 \
            --seed 1
        
        if [ $? -eq 0 ]; then
            echo "   ✅ RPE1 LOGO complete"
            
            # Compare baselines
            echo "   Running baseline comparison..."
            PYTHONPATH=src python -m goal_3_prediction.functional_class_holdout.compare_baselines \
                --results_csv results/goal_3_prediction/functional_class_holdout/replogle_rpe1/logo_replogle_rpe1_essential_transcription_results.csv \
                --output_dir results/goal_3_prediction/functional_class_holdout/replogle_rpe1 \
                --dataset_name replogle_rpe1_essential \
                --class_name Transcription
            
            echo "   ✅ RPE1 comparison complete"
        else
            echo "   ❌ RPE1 LOGO failed"
        fi
    fi
fi

echo ""
echo "=========================================="
echo "✅ LOGO Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - results/goal_3_prediction/functional_class_holdout/adamson/"
if [ -f "$K562_DATA_PATH" ]; then
    echo "  - results/goal_3_prediction/functional_class_holdout/replogle_k562/"
fi
if [ -f "$RPE1_DATA_PATH" ]; then
    echo "  - results/goal_3_prediction/functional_class_holdout/replogle_rpe1/"
fi


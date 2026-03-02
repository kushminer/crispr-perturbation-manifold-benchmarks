#!/bin/bash
# Run LOGO evaluation on all datasets (Adamson, K562, RPE1)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "=========================================="
echo "LOGO Evaluation: All Datasets"
echo "=========================================="
echo ""

echo "1. Running LOGO on Adamson..."
PYTHONPATH="$PYTHONPATH" python3 -m goal_3_prediction.functional_class_holdout.logo \
    --adata_path data/gears_pert_data/adamson/perturb_processed.h5ad \
    --annotation_path data/annotations/adamson_functional_classes_enriched.tsv \
    --dataset_name adamson \
    --output_dir results/goal_3_prediction/functional_class_holdout/adamson \
    --class_name Transcription \
    --pca_dim 10 \
    --ridge_penalty 0.1 \
    --seed 1

PYTHONPATH="$PYTHONPATH" python3 -m goal_3_prediction.functional_class_holdout.compare_baselines \
    --results_csv results/goal_3_prediction/functional_class_holdout/adamson/logo_adamson_transcription_results.csv \
    --output_dir results/goal_3_prediction/functional_class_holdout/adamson \
    --dataset_name adamson \
    --class_name Transcription

echo ""
echo "2. Running LOGO on Replogle K562..."
K562_DATA_PATH="${K562_DATA_PATH:-data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad}"
K562_ANNOTATION_PATH="${K562_ANNOTATION_PATH:-data/annotations/replogle_k562_functional_classes_go.tsv}"
if [ ! -f "$K562_DATA_PATH" ]; then
    echo "  Skipping K562: missing data at $K562_DATA_PATH"
else
    PYTHONPATH="$PYTHONPATH" python3 -m goal_3_prediction.functional_class_holdout.logo \
        --adata_path "$K562_DATA_PATH" \
        --annotation_path "$K562_ANNOTATION_PATH" \
        --dataset_name replogle_k562_essential \
        --output_dir results/goal_3_prediction/functional_class_holdout/replogle_k562 \
        --class_name Transcription \
        --pca_dim 10 \
        --ridge_penalty 0.1 \
        --seed 1

    PYTHONPATH="$PYTHONPATH" python3 -m goal_3_prediction.functional_class_holdout.compare_baselines \
        --results_csv results/goal_3_prediction/functional_class_holdout/replogle_k562/logo_replogle_k562_essential_transcription_results.csv \
        --output_dir results/goal_3_prediction/functional_class_holdout/replogle_k562 \
        --dataset_name replogle_k562_essential \
        --class_name Transcription
fi

echo ""
echo "3. Running LOGO on Replogle RPE1..."
RPE1_DATA_PATH="${RPE1_DATA_PATH:-data/gears_pert_data/replogle_rpe1_essential/perturb_processed.h5ad}"
RPE1_ANNOTATION_PATH="${RPE1_ANNOTATION_PATH:-data/annotations/replogle_rpe1_functional_classes_go.tsv}"
if [ ! -f "$RPE1_DATA_PATH" ]; then
    echo "  Skipping RPE1: missing data at $RPE1_DATA_PATH"
elif [ ! -f "$RPE1_ANNOTATION_PATH" ]; then
    echo "  Skipping RPE1: missing annotations at $RPE1_ANNOTATION_PATH"
else
    PYTHONPATH="$PYTHONPATH" python3 -m goal_3_prediction.functional_class_holdout.logo \
        --adata_path "$RPE1_DATA_PATH" \
        --annotation_path "$RPE1_ANNOTATION_PATH" \
        --dataset_name replogle_rpe1_essential \
        --output_dir results/goal_3_prediction/functional_class_holdout/replogle_rpe1 \
        --class_name Transcription \
        --pca_dim 10 \
        --ridge_penalty 0.1 \
        --seed 1

    PYTHONPATH="$PYTHONPATH" python3 -m goal_3_prediction.functional_class_holdout.compare_baselines \
        --results_csv results/goal_3_prediction/functional_class_holdout/replogle_rpe1/logo_replogle_rpe1_essential_transcription_results.csv \
        --output_dir results/goal_3_prediction/functional_class_holdout/replogle_rpe1 \
        --dataset_name replogle_rpe1_essential \
        --class_name Transcription
fi

echo ""
echo "=========================================="
echo "LOGO Evaluation Complete"
echo "=========================================="

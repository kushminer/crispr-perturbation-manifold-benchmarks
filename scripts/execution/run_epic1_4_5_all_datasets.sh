#!/bin/bash
# Run Epic 1, 4, 5 on ALL baselines and datasets systematically

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT}/src"

# Dataset configuration
declare -A DATASETS
DATASETS[adamson,adata]="${REPO_ROOT}/data/gears_pert_data/adamson/perturb_processed.h5ad"
DATASETS[adamson,split]="${REPO_ROOT}/results/goal_2_baselines/splits/adamson_split_seed1.json"
DATASETS[adamson,annot]="${REPO_ROOT}/data/annotations/adamson_functional_classes_go.tsv"

DATASETS[k562,adata]="data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad"
DATASETS[k562,split]="${REPO_ROOT}/results/goal_2_baselines/splits/replogle_k562_essential_split_seed1.json"
DATASETS[k562,annot]="${REPO_ROOT}/data/annotations/replogle_k562_functional_classes_go.tsv"

DATASETS[rpe1,adata]="data/gears_pert_data/replogle_rpe1_essential/perturb_processed.h5ad"
DATASETS[rpe1,split]="${REPO_ROOT}/results/goal_2_baselines/splits/replogle_rpe1_essential_split_seed1.json"
DATASETS[rpe1,annot]="${REPO_ROOT}/data/annotations/replogle_rpe1_functional_classes_go.tsv"

# Core baselines (expand as needed)
BASELINES=(
    "lpm_selftrained"
    "lpm_randomGeneEmb"
    "lpm_randomPertEmb"
)

K_LIST="3 5 10 20 30 50"
OUTPUT_BASE="${REPO_ROOT}/results/manifold_law_diagnostics"

echo "============================================================"
echo "Running Epic 1, 4, 5 on ALL Baselines & Datasets"
echo "============================================================"
echo ""
echo "Baselines: ${BASELINES[@]}"
echo "Datasets: adamson k562 rpe1"
echo ""

# Function to check if files exist
check_dataset() {
    local dataset=$1
    local adata_path="${DATASETS[${dataset},adata]}"
    local split_path="${DATASETS[${dataset},split]}"
    
    if [ ! -f "$adata_path" ] || [ ! -f "$split_path" ]; then
        echo "⚠️  Skipping ${dataset}: Missing files"
        echo "   adata: $adata_path"
        echo "   split: $split_path"
        return 1
    fi
    return 0
}

# Epic 1: Curvature Sweep
echo "============================================================"
echo "EPIC 1: Curvature Sweep"
echo "============================================================"
echo ""

for dataset in adamson k562 rpe1; do
    if ! check_dataset "$dataset"; then
        continue
    fi
    
    adata_path="${DATASETS[${dataset},adata]}"
    split_path="${DATASETS[${dataset},split]}"
    
    echo "Dataset: ${dataset}"
    
    for baseline in "${BASELINES[@]}"; do
        output_file="${OUTPUT_BASE}/epic1_curvature/curvature_sweep_summary_${dataset}_${baseline}.csv"
        
        if [ -f "$output_file" ]; then
            echo "  ✅ ${baseline} - Already exists, skipping"
            continue
        fi
        
        echo "  Running: ${baseline}"
        
        python -m goal_3_prediction.lsft.curvature_sweep \
            --adata_path "${adata_path}" \
            --split_config "${split_path}" \
            --dataset_name "${dataset}" \
            --baseline_type "${baseline}" \
            --output_dir "${OUTPUT_BASE}/epic1_curvature" \
            --k_list ${K_LIST} \
            --pca_dim 10 \
            --ridge_penalty 0.1 \
            --seed 1 \
            2>&1 | tee "${OUTPUT_BASE}/epic1_curvature/log_${dataset}_${baseline}.log" | tail -15
        
        echo ""
    done
    echo ""
done

# Epic 4: Direction-Flip Probe
echo "============================================================"
echo "EPIC 4: Direction-Flip Probe"
echo "============================================================"
echo ""

for dataset in adamson k562 rpe1; do
    if ! check_dataset "$dataset"; then
        continue
    fi
    
    adata_path="${DATASETS[${dataset},adata]}"
    split_path="${DATASETS[${dataset},split]}"
    
    echo "Dataset: ${dataset}"
    
    for baseline in "${BASELINES[@]}"; do
        output_file="${OUTPUT_BASE}/epic4_direction_flip/direction_flip_probe_${dataset}_${baseline}.csv"
        
        if [ -f "$output_file" ]; then
            echo "  ✅ ${baseline} - Already exists, skipping"
            continue
        fi
        
        echo "  Running: ${baseline}"
        
        python src/goal_3_prediction/lsft/run_epic4_from_lsft_results.py \
            --adata_path "${adata_path}" \
            --split_config "${split_path}" \
            --dataset_name "${dataset}" \
            --baseline_type "${baseline}" \
            --output_dir "${OUTPUT_BASE}/epic4_direction_flip" \
            --top_pct 0.05 \
            --pca_dim 10 \
            --seed 1 \
            2>&1 | tee "${OUTPUT_BASE}/epic4_direction_flip/log_${dataset}_${baseline}.log" | tail -5
        
        echo ""
    done
    echo ""
done

# Epic 5: Tangent Alignment
echo "============================================================"
echo "EPIC 5: Tangent Alignment"
echo "============================================================"
echo ""

for dataset in adamson k562 rpe1; do
    if ! check_dataset "$dataset"; then
        continue
    fi
    
    adata_path="${DATASETS[${dataset},adata]}"
    split_path="${DATASETS[${dataset},split]}"
    
    echo "Dataset: ${dataset}"
    
    for baseline in "${BASELINES[@]}"; do
        output_file="${OUTPUT_BASE}/epic5_tangent_alignment/tangent_alignment_${dataset}_${baseline}.csv"
        
        if [ -f "$output_file" ]; then
            echo "  ✅ ${baseline} - Already exists, skipping"
            continue
        fi
        
        echo "  Running: ${baseline}"
        
        python src/goal_3_prediction/lsft/run_epic5_from_lsft_results.py \
            --adata_path "${adata_path}" \
            --split_config "${split_path}" \
            --dataset_name "${dataset}" \
            --baseline_type "${baseline}" \
            --output_dir "${OUTPUT_BASE}/epic5_tangent_alignment" \
            --top_pct 0.05 \
            --pca_dim 10 \
            --n_components 10 \
            --seed 1 \
            2>&1 | tee "${OUTPUT_BASE}/epic5_tangent_alignment/log_${dataset}_${baseline}.log" | tail -5
        
        echo ""
    done
    echo ""
done

echo "============================================================"
echo "✅ Epic 1, 4, 5 Complete!"
echo "============================================================"


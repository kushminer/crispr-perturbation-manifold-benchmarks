#!/bin/bash
# Run ALL 5 Epics on ALL Baselines across ALL Datasets

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT}/src"

echo "============================================================"
echo "Manifold Law Diagnostic Suite - Complete Execution"
echo "ALL Epics × ALL Baselines × ALL Datasets"
echo "============================================================"
echo ""

# Dataset configuration
ADAMSON_ADATA="${REPO_ROOT}/data/gears_pert_data/adamson/perturb_processed.h5ad"
ADAMSON_SPLIT="${REPO_ROOT}/results/goal_2_baselines/splits/adamson_split_seed1.json"
# Try GO version first, fallback to enriched
if [ -f "${REPO_ROOT}/data/annotations/adamson_functional_classes_go.tsv" ]; then
    ADAMSON_ANNOT="${REPO_ROOT}/data/annotations/adamson_functional_classes_go.tsv"
elif [ -f "${REPO_ROOT}/data/annotations/adamson_functional_classes_enriched.tsv" ]; then
    ADAMSON_ANNOT="${REPO_ROOT}/data/annotations/adamson_functional_classes_enriched.tsv"
else
    ADAMSON_ANNOT="${REPO_ROOT}/data/annotations/adamson_functional_classes.tsv"
fi

K562_ADATA="data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad"
K562_SPLIT="${REPO_ROOT}/results/goal_2_baselines/splits/replogle_k562_essential_split_seed1.json"
K562_ANNOT="${REPO_ROOT}/data/annotations/replogle_k562_functional_classes_go.tsv"

RPE1_ADATA="data/gears_pert_data/replogle_rpe1_essential/perturb_processed.h5ad"
RPE1_SPLIT="${REPO_ROOT}/results/goal_2_baselines/splits/replogle_rpe1_essential_split_seed1.json"
RPE1_ANNOT="${REPO_ROOT}/data/annotations/replogle_rpe1_functional_classes_go.tsv"

# ALL 8 Baselines + mean_response (9 total)
# Note: All baselines should work. GEARS may have fewer perturbations (only those in GO graph).
# Cross-dataset baselines (K562, RPE1) work fine - they use source dataset PCA space.
ALL_BASELINES=(
    "lpm_selftrained"
    "lpm_randomGeneEmb"
    "lpm_randomPertEmb"
    "lpm_scgptGeneEmb"
    "lpm_scFoundationGeneEmb"
    "lpm_gearsPertEmb"  # GEARS GO graph embeddings - may have fewer perturbations
    "lpm_k562PertEmb"   # Cross-dataset: uses K562 perturbation PCA space
    "lpm_rpe1PertEmb"   # Cross-dataset: uses RPE1 perturbation PCA space
    # "mean_response"    # Note: mean_response is special case - may need separate handling for some epics
)

OUTPUT_BASE="${REPO_ROOT}/results/manifold_law_diagnostics"

# Check if baseline should be skipped for this dataset
# Cross-dataset baselines only work on their specific dataset
should_skip_baseline() {
    local baseline="$1"
    local dataset="$2"
    
    if [ "$baseline" == "lpm_rpe1PertEmb" ] && [ "$dataset" != "rpe1" ]; then
        return 0  # Skip
    fi
    
    if [ "$baseline" == "lpm_k562PertEmb" ] && [ "$dataset" != "k562" ]; then
        return 0  # Skip
    fi
    
    return 1  # Don't skip
}

K_LIST="3 5 10 20 30 50"

echo "Baselines to run: ${ALL_BASELINES[@]}"
echo "Datasets: adamson k562 rpe1"
echo ""
echo "This will run:"
echo "  - Epic 1: Curvature Sweep"
echo "  - Epic 2: Mechanism Ablation"
echo "  - Epic 3: Noise Injection"
echo "  - Epic 4: Direction-Flip Probe"
echo "  - Epic 5: Tangent Alignment"
echo ""
echo "Total experiments: $((${#ALL_BASELINES[@]} * 3 * 5)) (${#ALL_BASELINES[@]} baselines × 3 datasets × 5 epics)"
echo ""

# Function to check if files exist
check_dataset() {
    local dataset=$1
    local adata_path=$2
    local split_path=$3
    
    if [ ! -f "$adata_path" ] || [ ! -f "$split_path" ]; then
        echo "⚠️  Skipping ${dataset}: Missing files"
        return 1
    fi
    return 0
}

# Function to check if result already exists
check_result_exists() {
    local output_file=$1
    if [ -f "$output_file" ]; then
        return 0  # Exists
    fi
    return 1  # Doesn't exist
}

# Epic 1: Curvature Sweep
echo "============================================================"
echo "EPIC 1: Curvature Sweep"
echo "============================================================"
echo ""

for dataset in adamson k562 rpe1; do
    if [ "$dataset" == "adamson" ]; then
        adata_path="$ADAMSON_ADATA"
        split_path="$ADAMSON_SPLIT"
    elif [ "$dataset" == "k562" ]; then
        adata_path="$K562_ADATA"
        split_path="$K562_SPLIT"
    elif [ "$dataset" == "rpe1" ]; then
        adata_path="$RPE1_ADATA"
        split_path="$RPE1_SPLIT"
    fi
    
    if ! check_dataset "$dataset" "$adata_path" "$split_path"; then
        continue
    fi
    
    echo "Dataset: ${dataset}"
    
    for baseline in "${ALL_BASELINES[@]}"; do
        # Skip invalid baseline×dataset combinations
        if should_skip_baseline "$baseline" "$dataset"; then
            echo "  ⏭️  Skipping ${baseline} on ${dataset} (cross-dataset baseline only works on specific dataset)"
            continue
        fi
        
        output_file="${OUTPUT_BASE}/epic1_curvature/curvature_sweep_summary_${dataset}_${baseline}.csv"
        
        if check_result_exists "$output_file"; then
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
            2>&1 | tee "${OUTPUT_BASE}/epic1_curvature/log_${dataset}_${baseline}.log" | tail -15 || echo "  ⚠️  Failed: ${baseline}"
        
        echo ""
    done
    echo ""
done

# Epic 2: Mechanism Ablation
echo "============================================================"
echo "EPIC 2: Mechanism Ablation"
echo "============================================================"
echo ""

for dataset in adamson k562 rpe1; do
    if [ "$dataset" == "adamson" ]; then
        adata_path="$ADAMSON_ADATA"
        split_path="$ADAMSON_SPLIT"
        annot_path="$ADAMSON_ANNOT"
    elif [ "$dataset" == "k562" ]; then
        adata_path="$K562_ADATA"
        split_path="$K562_SPLIT"
        annot_path="$K562_ANNOT"
    elif [ "$dataset" == "rpe1" ]; then
        adata_path="$RPE1_ADATA"
        split_path="$RPE1_SPLIT"
        annot_path="$RPE1_ANNOT"
    fi
    
    if ! check_dataset "$dataset" "$adata_path" "$split_path"; then
        continue
    fi
    
    if [ ! -f "$annot_path" ]; then
        echo "⚠️  Skipping ${dataset}: Missing annotation file: $annot_path"
        continue
    fi
    
    echo "Dataset: ${dataset}"
    
    for baseline in "${ALL_BASELINES[@]}"; do
        # Skip invalid baseline×dataset combinations
        if should_skip_baseline "$baseline" "$dataset"; then
            echo "  ⏭️  Skipping ${baseline} on ${dataset} (cross-dataset baseline only works on specific dataset)"
            continue
        fi
        
        output_file="${OUTPUT_BASE}/epic2_mechanism_ablation/mechanism_ablation_${dataset}_${baseline}.csv"
        
        if check_result_exists "$output_file"; then
            echo "  ✅ ${baseline} - Already exists, skipping"
            continue
        fi
        
        echo "  Running: ${baseline}"
        
        python src/goal_3_prediction/lsft/run_epic2_mechanism_ablation.py \
            --adata_path "${adata_path}" \
            --split_config "${split_path}" \
            --annotation_path "${annot_path}" \
            --dataset_name "${dataset}" \
            --baseline_type "${baseline}" \
            --output_dir "${OUTPUT_BASE}/epic2_mechanism_ablation" \
            --k_list 5 10 20 \
            --pca_dim 10 \
            --seed 1 \
            2>&1 | tee "${OUTPUT_BASE}/epic2_mechanism_ablation/log_${dataset}_${baseline}.log" | tail -5 || echo "  ⚠️  Failed: ${baseline}"
        
        echo ""
    done
    echo ""
done

# Epic 3: Noise Injection
echo "============================================================"
echo "EPIC 3: Noise Injection"
echo "============================================================"
echo ""

for dataset in adamson k562 rpe1; do
    if [ "$dataset" == "adamson" ]; then
        adata_path="$ADAMSON_ADATA"
        split_path="$ADAMSON_SPLIT"
    elif [ "$dataset" == "k562" ]; then
        adata_path="$K562_ADATA"
        split_path="$K562_SPLIT"
    elif [ "$dataset" == "rpe1" ]; then
        adata_path="$RPE1_ADATA"
        split_path="$RPE1_SPLIT"
    fi
    
    if ! check_dataset "$dataset" "$adata_path" "$split_path"; then
        continue
    fi
    
    echo "Dataset: ${dataset}"
    
    for baseline in "${ALL_BASELINES[@]}"; do
        # Skip invalid baseline×dataset combinations
        if should_skip_baseline "$baseline" "$dataset"; then
            echo "  ⏭️  Skipping ${baseline} on ${dataset} (cross-dataset baseline only works on specific dataset)"
            continue
        fi
        
        output_file="${OUTPUT_BASE}/epic3_noise_injection/noise_injection_${dataset}_${baseline}.csv"
        
        if check_result_exists "$output_file"; then
            echo "  ✅ ${baseline} - Already exists, skipping"
            continue
        fi
        
        echo "  Running: ${baseline}"
        
        python src/goal_3_prediction/lsft/run_epic3_noise_injection.py \
            --adata_path "${adata_path}" \
            --split_config "${split_path}" \
            --dataset_name "${dataset}" \
            --baseline_type "${baseline}" \
            --output_dir "${OUTPUT_BASE}/epic3_noise_injection" \
            --k_list 5 10 20 \
            --noise_levels 0.01 0.05 0.1 0.2 \
            --noise_type gaussian \
            --pca_dim 10 \
            --seed 1 \
            2>&1 | tee "${OUTPUT_BASE}/epic3_noise_injection/log_${dataset}_${baseline}.log" | tail -5 || echo "  ⚠️  Failed: ${baseline}"
        
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
    if [ "$dataset" == "adamson" ]; then
        adata_path="$ADAMSON_ADATA"
        split_path="$ADAMSON_SPLIT"
    elif [ "$dataset" == "k562" ]; then
        adata_path="$K562_ADATA"
        split_path="$K562_SPLIT"
    elif [ "$dataset" == "rpe1" ]; then
        adata_path="$RPE1_ADATA"
        split_path="$RPE1_SPLIT"
    fi
    
    if ! check_dataset "$dataset" "$adata_path" "$split_path"; then
        continue
    fi
    
    echo "Dataset: ${dataset}"
    
    for baseline in "${ALL_BASELINES[@]}"; do
        # Skip invalid baseline×dataset combinations
        if should_skip_baseline "$baseline" "$dataset"; then
            echo "  ⏭️  Skipping ${baseline} on ${dataset} (cross-dataset baseline only works on specific dataset)"
            continue
        fi
        
        output_file="${OUTPUT_BASE}/epic4_direction_flip/direction_flip_probe_${dataset}_${baseline}.csv"
        
        if check_result_exists "$output_file"; then
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
            2>&1 | tee "${OUTPUT_BASE}/epic4_direction_flip/log_${dataset}_${baseline}.log" | tail -5 || echo "  ⚠️  Failed: ${baseline}"
        
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
    if [ "$dataset" == "adamson" ]; then
        adata_path="$ADAMSON_ADATA"
        split_path="$ADAMSON_SPLIT"
    elif [ "$dataset" == "k562" ]; then
        adata_path="$K562_ADATA"
        split_path="$K562_SPLIT"
    elif [ "$dataset" == "rpe1" ]; then
        adata_path="$RPE1_ADATA"
        split_path="$RPE1_SPLIT"
    fi
    
    if ! check_dataset "$dataset" "$adata_path" "$split_path"; then
        continue
    fi
    
    echo "Dataset: ${dataset}"
    
    for baseline in "${ALL_BASELINES[@]}"; do
        # Skip invalid baseline×dataset combinations
        if should_skip_baseline "$baseline" "$dataset"; then
            echo "  ⏭️  Skipping ${baseline} on ${dataset} (cross-dataset baseline only works on specific dataset)"
            continue
        fi
        
        output_file="${OUTPUT_BASE}/epic5_tangent_alignment/tangent_alignment_${dataset}_${baseline}.csv"
        
        if check_result_exists "$output_file"; then
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
            2>&1 | tee "${OUTPUT_BASE}/epic5_tangent_alignment/log_${dataset}_${baseline}.log" | tail -5 || echo "  ⚠️  Failed: ${baseline}"
        
        echo ""
    done
    echo ""
done

echo "============================================================"
echo "✅ ALL EPICS COMPLETE!"
echo "============================================================"
echo ""
echo "Results saved to: ${OUTPUT_BASE}/"
echo ""
echo "Next: Generate summary reports and visualizations"


#!/bin/bash
# Run Epic 2 for RPE1 dataset only (all 8 baselines)

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT}/src"

RPE1_ADATA="data/gears_pert_data/replogle_rpe1_essential/perturb_processed.h5ad"
RPE1_SPLIT="${REPO_ROOT}/results/goal_2_baselines/splits/replogle_rpe1_essential_split_seed1.json"

# Use K562 annotations for RPE1 (same functional classes)
RPE1_ANNOT="${REPO_ROOT}/data/annotations/replogle_rpe1_functional_classes_go.tsv"

ALL_BASELINES=(
    "lpm_selftrained"
    "lpm_randomGeneEmb"
    "lpm_randomPertEmb"
    "lpm_scgptGeneEmb"
    "lpm_scFoundationGeneEmb"
    "lpm_gearsPertEmb"
    "lpm_k562PertEmb"
    "lpm_rpe1PertEmb"
)

OUTPUT_BASE="${REPO_ROOT}/results/manifold_law_diagnostics"

echo "============================================================"
echo "EPIC 2: Mechanism Ablation - RPE1 Only"
echo "============================================================"
echo ""

# Check if annotation file exists
if [ ! -f "$RPE1_ANNOT" ]; then
    echo "❌ ERROR: RPE1 annotation file not found: $RPE1_ANNOT"
    echo "   Please create symlink: ln -s replogle_k562_functional_classes_go.tsv data/annotations/replogle_rpe1_functional_classes_go.tsv"
    exit 1
fi

echo "Dataset: rpe1"
echo "Annotation: $RPE1_ANNOT"
echo ""

for baseline in "${ALL_BASELINES[@]}"; do
    output_file="${OUTPUT_BASE}/epic2_mechanism_ablation/mechanism_ablation_rpe1_${baseline}.csv"
    
    if [ -f "$output_file" ] && [ -s "$output_file" ]; then
        echo "  ✅ ${baseline} - Already exists, skipping"
        continue
    fi
    
    echo "  Running: ${baseline}"
    
    python src/goal_3_prediction/lsft/run_epic2_mechanism_ablation.py \
        --adata_path "${RPE1_ADATA}" \
        --split_config "${RPE1_SPLIT}" \
        --annotation_path "${RPE1_ANNOT}" \
        --dataset_name "rpe1" \
        --baseline_type "${baseline}" \
        --output_dir "${OUTPUT_BASE}/epic2_mechanism_ablation" \
        --k_list 5 10 20 \
        --pca_dim 10 \
        --seed 1 \
        2>&1 | tee "${OUTPUT_BASE}/epic2_mechanism_ablation/log_rpe1_${baseline}.log" | tail -5 || echo "  ⚠️  Failed: ${baseline}"
    
    echo ""
done

echo "============================================================"
echo "✅ Epic 2 RPE1 execution complete"
echo "============================================================"


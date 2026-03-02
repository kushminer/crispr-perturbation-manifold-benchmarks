#!/bin/bash
# Optimized runner for ALL 5 Epics on ALL Baselines across ALL Datasets.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH}"

echo "============================================================"
echo "Manifold Law Diagnostic Suite - OPTIMIZED Execution"
echo "ALL Epics x ALL Baselines x ALL Datasets"
echo "============================================================"
echo ""

echo "Running validation check..."
if ! python3 scripts/utilities/validate_lsft_logic.py > /dev/null 2>&1; then
    echo "Validation failed. Check LSFT logic before proceeding."
    exit 1
fi
echo "Validation passed - logic is intact"
echo ""

ADAMSON_ADATA="${REPO_ROOT}/data/gears_pert_data/adamson/perturb_processed.h5ad"
ADAMSON_SPLIT="${REPO_ROOT}/results/goal_2_baselines/splits/adamson_split_seed1.json"
if [ -f "${REPO_ROOT}/data/annotations/adamson_functional_classes_go.tsv" ]; then
    ADAMSON_ANNOT="${REPO_ROOT}/data/annotations/adamson_functional_classes_go.tsv"
elif [ -f "${REPO_ROOT}/data/annotations/adamson_functional_classes_enriched.tsv" ]; then
    ADAMSON_ANNOT="${REPO_ROOT}/data/annotations/adamson_functional_classes_enriched.tsv"
else
    ADAMSON_ANNOT="${REPO_ROOT}/data/annotations/adamson_functional_classes.tsv"
fi

K562_ADATA="${REPO_ROOT}/data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad"
K562_SPLIT="${REPO_ROOT}/results/goal_2_baselines/splits/replogle_k562_essential_split_seed1.json"
K562_ANNOT="${REPO_ROOT}/data/annotations/replogle_k562_functional_classes_go.tsv"

RPE1_ADATA="${REPO_ROOT}/data/gears_pert_data/replogle_rpe1_essential/perturb_processed.h5ad"
RPE1_SPLIT="${REPO_ROOT}/results/goal_2_baselines/splits/replogle_rpe1_essential_split_seed1.json"
RPE1_ANNOT="${REPO_ROOT}/data/annotations/replogle_rpe1_functional_classes_go.tsv"

ALL_BASELINES=(
    "lpm_selftrained"
    "lpm_randomGeneEmb"
    "lpm_randomPertEmb"
    "lpm_scgptGeneEmb"
    "lpm_scFoundationGeneEmb"
)

OUTPUT_BASE="${REPO_ROOT}/results/manifold_law_diagnostics"
K_LIST="3 5 10 20 30 50"

echo "Baselines: ${ALL_BASELINES[*]}"
echo "Datasets: adamson k562 rpe1"
echo "Optimizations: skip completed results, cached embeddings, validation gate"
echo ""

check_dataset() {
    local adata_path=$1
    local split_path=$2
    [ -f "$adata_path" ] && [ -f "$split_path" ]
}

check_result_exists() {
    [ -f "$1" ]
}

echo "============================================================"
echo "EPIC 1: Curvature Sweep"
echo "============================================================"
echo ""

for dataset in adamson k562 rpe1; do
    case "$dataset" in
        adamson) adata_path="$ADAMSON_ADATA"; split_path="$ADAMSON_SPLIT" ;;
        k562) adata_path="$K562_ADATA"; split_path="$K562_SPLIT" ;;
        rpe1) adata_path="$RPE1_ADATA"; split_path="$RPE1_SPLIT" ;;
    esac

    if ! check_dataset "$adata_path" "$split_path"; then
        echo "Skipping ${dataset}: missing files"
        continue
    fi

    echo "Dataset: ${dataset}"
    for baseline in "${ALL_BASELINES[@]}"; do
        output_file="${OUTPUT_BASE}/epic1_curvature/curvature_sweep_summary_${dataset}_${baseline}.csv"
        if check_result_exists "$output_file"; then
            echo "  ${baseline}: skip (already exists)"
            continue
        fi

        echo "  Running ${baseline}"
        python3 -m goal_3_prediction.lsft.curvature_sweep \
            --adata_path "${adata_path}" \
            --split_config "${split_path}" \
            --dataset_name "${dataset}" \
            --baseline_type "${baseline}" \
            --output_dir "${OUTPUT_BASE}/epic1_curvature" \
            --k_list ${K_LIST} \
            --pca_dim 10 \
            --ridge_penalty 0.1 \
            --seed 1 \
            2>&1 | tee "${OUTPUT_BASE}/epic1_curvature/log_${dataset}_${baseline}.log" | tail -10 || echo "  Failed ${baseline}"
    done
    echo ""
done

echo "============================================================"
echo "Epic 1 complete or skipped"
echo "Use scripts/execution/run_all_epics_all_baselines.sh for the full Epic 2-5 run."
echo "============================================================"

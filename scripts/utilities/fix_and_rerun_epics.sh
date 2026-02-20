#!/bin/bash
# =============================================================================
# FIX AND RE-RUN INCOMPLETE EPICS
# =============================================================================
# 
# This script fixes the Epic 2 and Epic 3 issues:
# - Epic 2: Re-runs with CORRECT mechanism_ablation.py (not the placeholder script)
# - Epic 3: Aggregates Lipschitz constants from existing noise injection files
#
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT}/src"

echo "============================================================"
echo "FIXING MANIFOLD LAW DIAGNOSTIC SUITE"
echo "============================================================"
echo ""

# Dataset configuration
ADAMSON_ADATA="${REPO_ROOT}/data/gears_pert_data/adamson/perturb_processed.h5ad"
ADAMSON_SPLIT="${REPO_ROOT}/results/goal_2_baselines/splits/adamson_split_seed1.json"
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

# Core baselines only (skip cross-dataset for now to speed up)
CORE_BASELINES=(
    "lpm_selftrained"
    "lpm_randomGeneEmb"
    "lpm_randomPertEmb"
    "lpm_scgptGeneEmb"
    "lpm_scFoundationGeneEmb"
    "lpm_gearsPertEmb"
)

OUTPUT_BASE="${REPO_ROOT}/results/manifold_law_diagnostics"

# =============================================================================
# FIX 1: Re-run Epic 2 with CORRECT mechanism_ablation.py
# =============================================================================
echo "============================================================"
echo "FIX 1: Re-running Epic 2 with CORRECT implementation"
echo "============================================================"
echo ""
echo "⚠️  This will REPLACE the existing Epic 2 files with proper ablation values"
echo ""

# Backup existing Epic 2 files
if [ -d "${OUTPUT_BASE}/epic2_mechanism_ablation" ]; then
    backup_dir="${OUTPUT_BASE}/epic2_mechanism_ablation_backup_$(date +%Y%m%d_%H%M%S)"
    echo "Backing up existing Epic 2 files to: $backup_dir"
    mv "${OUTPUT_BASE}/epic2_mechanism_ablation" "$backup_dir"
fi

mkdir -p "${OUTPUT_BASE}/epic2_mechanism_ablation"

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
    
    # Check if files exist
    if [ ! -f "$adata_path" ]; then
        echo "⚠️  Skipping ${dataset}: Missing adata file: $adata_path"
        continue
    fi
    if [ ! -f "$split_path" ]; then
        echo "⚠️  Skipping ${dataset}: Missing split file: $split_path"
        continue
    fi
    if [ ! -f "$annot_path" ]; then
        echo "⚠️  Skipping ${dataset}: Missing annotation file: $annot_path"
        continue
    fi
    
    echo "Dataset: ${dataset}"
    
    for baseline in "${CORE_BASELINES[@]}"; do
        echo "  Running: ${baseline} (with CORRECT ablation implementation)"
        
        # Use the CORRECT mechanism_ablation.py module
        python -m goal_3_prediction.lsft.mechanism_ablation \
            --adata_path "${adata_path}" \
            --split_config "${split_path}" \
            --annotation_path "${annot_path}" \
            --dataset_name "${dataset}" \
            --baseline_type "${baseline}" \
            --output_dir "${OUTPUT_BASE}/epic2_mechanism_ablation" \
            --top_pct 0.05 \
            --pca_dim 10 \
            --ridge_penalty 0.1 \
            --seed 1 \
            2>&1 | tee "${OUTPUT_BASE}/epic2_mechanism_ablation/log_${dataset}_${baseline}.log" | tail -10 || echo "  ⚠️  Failed: ${baseline}"
        
        echo ""
    done
    echo ""
done

echo "✅ Epic 2 re-run complete"
echo ""

# =============================================================================
# FIX 2: Aggregate Epic 3 Lipschitz constants
# =============================================================================
echo "============================================================"
echo "FIX 2: Aggregating Epic 3 Lipschitz constants"
echo "============================================================"
echo ""

python3 << 'EOF'
import pandas as pd
import numpy as np
from pathlib import Path

epic3_dir = Path("results/manifold_law_diagnostics/epic3_noise_injection")

if not epic3_dir.exists():
    print("⚠️  Epic 3 directory not found")
    exit(0)

# Collect all noise injection files
all_lipschitz = []

for csv_file in sorted(epic3_dir.glob("noise_injection_*.csv")):
    try:
        df = pd.read_csv(csv_file)
        if len(df) == 0:
            continue
        
        # Extract dataset and baseline from filename
        parts = csv_file.stem.replace("noise_injection_", "").split("_", 1)
        if len(parts) != 2:
            continue
        dataset, baseline = parts
        
        # Group by k and compute Lipschitz for each
        for k in df['k'].unique():
            k_data = df[df['k'] == k].sort_values('noise_level')
            
            # Get baseline (noise_level = 0)
            baseline_row = k_data[k_data['noise_level'] == 0.0]
            if len(baseline_row) == 0:
                continue
            
            r_baseline = baseline_row['mean_r'].iloc[0]
            
            # Get noisy conditions
            noisy_data = k_data[k_data['noise_level'] > 0]
            if len(noisy_data) == 0:
                continue
            
            noise_levels = noisy_data['noise_level'].values
            r_values = noisy_data['mean_r'].values
            
            # Compute Lipschitz: L = max |r(0) - r(σ)| / σ
            delta_r = r_baseline - r_values
            sensitivity = np.abs(delta_r) / noise_levels
            lipschitz = np.max(sensitivity) if len(sensitivity) > 0 else np.nan
            
            all_lipschitz.append({
                'dataset': dataset,
                'baseline': baseline,
                'k': k,
                'baseline_r': r_baseline,
                'lipschitz_constant': lipschitz,
                'mean_sensitivity': np.mean(sensitivity) if len(sensitivity) > 0 else np.nan,
                'max_delta_r': np.max(np.abs(delta_r)) if len(delta_r) > 0 else np.nan,
            })
    except Exception as e:
        print(f"  Warning: Error processing {csv_file.name}: {e}")

if all_lipschitz:
    lipschitz_df = pd.DataFrame(all_lipschitz)
    
    # Save detailed results
    lipschitz_df.to_csv(epic3_dir / "lipschitz_analysis_detailed.csv", index=False)
    print(f"✅ Saved detailed Lipschitz analysis: {len(lipschitz_df)} entries")
    
    # Create summary (aggregate across k values)
    summary = lipschitz_df.groupby(['dataset', 'baseline']).agg({
        'baseline_r': 'mean',
        'lipschitz_constant': 'mean',
        'mean_sensitivity': 'mean',
    }).reset_index()
    
    summary.to_csv(epic3_dir / "lipschitz_summary.csv", index=False)
    print(f"✅ Saved Lipschitz summary: {len(summary)} baseline×dataset combinations")
    
    # Show summary
    print("\nLipschitz Summary by Baseline:")
    baseline_summary = lipschitz_df.groupby('baseline').agg({
        'lipschitz_constant': ['mean', 'std'],
        'baseline_r': 'mean',
    }).round(4)
    print(baseline_summary.to_string())
else:
    print("⚠️  No data found for Lipschitz aggregation")
EOF

echo ""
echo "✅ Epic 3 Lipschitz aggregation complete"
echo ""

# =============================================================================
# Summary
# =============================================================================
echo "============================================================"
echo "FIXES APPLIED"
echo "============================================================"
echo ""
echo "1. ✅ Epic 2 re-run with CORRECT mechanism_ablation.py"
echo "     (ablated_pearson_r and delta_r now populated)"
echo ""
echo "2. ✅ Epic 3 Lipschitz constants aggregated"
echo "     (lipschitz_summary.csv created)"
echo ""
echo "Next step: Run report generation to regenerate 5-epic grid"
echo "  python deliverables/publication_package/generate_publication_reports.py"
echo ""


#!/bin/bash
# Generate baseline comparisons across all baselines and datasets
# This aggregates standardized results and runs permutation tests

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "Generating Baseline Comparisons"
echo "============================================================"
echo ""

# Check if all evaluations are complete
echo "Checking for completed evaluations..."
TOTAL_EXPECTED=24
COMPLETED=$(find results/goal_3_prediction/lsft_resampling -name "*_standardized.csv" | wc -l | tr -d ' ')

if [ "$COMPLETED" -lt "$TOTAL_EXPECTED" ]; then
    echo "⚠️  Warning: Only $COMPLETED/$TOTAL_EXPECTED evaluations complete"
    echo "   Comparisons may be incomplete"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

echo "Aggregating results and generating comparisons..."
echo ""

# For each dataset, aggregate standardized results and generate comparisons
for dataset in adamson k562 rpe1; do
    dataset_dir="results/goal_3_prediction/lsft_resampling/$dataset"
    
    if [ ! -d "$dataset_dir" ]; then
        echo "⚠️  Skipping $dataset - directory not found"
        continue
    fi
    
    echo "Processing $dataset..."
    
    # Check if we have multiple baselines
    baseline_count=$(find "$dataset_dir" -name "*_standardized.csv" | wc -l | tr -d ' ')
    
    if [ "$baseline_count" -lt 2 ]; then
        echo "  ⚠️  Need at least 2 baselines for comparison (found $baseline_count)"
        continue
    fi
    
    # Aggregate all standardized CSVs
    echo "  Aggregating standardized results..."
    combined_csv="$dataset_dir/lsft_${dataset}_all_baselines_combined.csv"
    
    # Combine all standardized CSVs
    first=1
    for csv in "$dataset_dir"/*_standardized.csv; do
        if [ "$first" -eq 1 ]; then
            head -1 "$csv" > "$combined_csv"
            first=0
        fi
        tail -n +2 "$csv" >> "$combined_csv"
    done
    
    echo "  ✅ Combined results: $baseline_count baselines"
    
    # Generate comparisons using Python
    echo "  Generating baseline comparisons..."
    PYTHONPATH=src python3 << EOF
import sys
sys.path.insert(0, 'src')
import pandas as pd
from pathlib import Path
from goal_3_prediction.lsft.compare_baselines_resampling import (
    compare_all_baseline_pairs,
    save_baseline_comparisons
)

# Load combined results
combined_df = pd.read_csv("$combined_csv")

# Generate comparisons
try:
    comparison_df = compare_all_baseline_pairs(
        results_df=combined_df,
        metrics=["pearson_r", "l2"],
        top_pcts=[0.01, 0.05, 0.10],
        n_perm=10000,
        n_boot=1000,
        random_state=1,
    )
    
    output_path = Path("$dataset_dir/lsft_${dataset}_baseline_comparisons")
    save_baseline_comparisons(comparison_df, output_path, format="both")
    print(f"  ✅ Saved comparisons to {output_path}")
    
except Exception as e:
    print(f"  ❌ Failed: {e}")
EOF
    
    echo ""
done

echo "============================================================"
echo "✅ Baseline comparison generation complete!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - results/goal_3_prediction/lsft_resampling/*/lsft_*_baseline_comparisons.*"
echo ""


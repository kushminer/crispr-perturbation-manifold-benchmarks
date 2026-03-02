#!/usr/bin/env python3
"""
Generate baseline comparisons from standardized LSFT results.
This script aggregates all baseline results and generates comparison tables.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
from goal_3_prediction.lsft.compare_baselines_resampling import (
    compare_all_baseline_pairs,
    save_baseline_comparisons
)

def main():
    print("Generating baseline comparisons...")
    print()
    
    # Process each dataset
    for dataset in ["adamson", "k562", "rpe1"]:
        dataset_dir = Path(f"results/goal_3_prediction/lsft_resampling/{dataset}")
        
        if not dataset_dir.exists():
            print(f"⚠️  Skipping {dataset} - directory not found")
            continue
        
        print(f"Processing {dataset}...")
        
        # Aggregate standardized CSVs
        csv_files = list(dataset_dir.glob("*_standardized.csv"))
        
        if len(csv_files) < 2:
            print(f"  ⚠️  Need at least 2 baselines (found {len(csv_files)})")
            continue
        
        # Combine CSVs
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_csv = dataset_dir / f"lsft_{dataset}_all_baselines_combined.csv"
        combined_df.to_csv(combined_csv, index=False)
        print(f"  ✅ Combined {len(csv_files)} baselines")
        
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
            
            output_path = dataset_dir / f"lsft_{dataset}_baseline_comparisons"
            save_baseline_comparisons(comparison_df, output_path, format="both")
            print(f"  ✅ Saved comparisons to {output_path}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print("✅ Baseline comparison generation complete!")

if __name__ == "__main__":
    main()


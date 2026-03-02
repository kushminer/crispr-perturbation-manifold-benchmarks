#!/usr/bin/env python3
"""
Generate all visualizations with CI overlays for LSFT resampling results.
Creates publication-ready figures with uncertainty quantification.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
from goal_3_prediction.lsft.visualize_resampling import create_all_lsft_visualizations_with_ci

def main():
    print("Generating all LSFT visualizations with CI overlays...")
    print()
    
    # Process each dataset
    for dataset in ["adamson", "k562", "rpe1"]:
        print(f"Processing {dataset}...")
        
        # Load combined results
        combined_csv = Path(f"results/goal_3_prediction/lsft_resampling/{dataset}/lsft_{dataset}_all_baselines_combined.csv")
        if not combined_csv.exists():
            print(f"  ⚠️  Combined CSV not found: {combined_csv}")
            continue
        
        results_df = pd.read_csv(combined_csv)
        print(f"  ✅ Loaded {len(results_df)} results")
        
        # Create plots directory
        plots_dir = Path(f"results/goal_3_prediction/lsft_resampling/{dataset}/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # For each baseline, create visualizations
        baselines = results_df["baseline_type"].unique()
        print(f"  Generating visualizations for {len(baselines)} baselines...")
        
        for baseline in baselines:
            baseline_results = results_df[results_df["baseline_type"] == baseline]
            
            # Find summary file
            summary_path = Path(f"results/goal_3_prediction/lsft_resampling/{dataset}/lsft_{dataset}_{baseline}_summary.json")
            if not summary_path.exists():
                continue
            
            # Find regression file
            regression_path = Path(f"results/goal_3_prediction/lsft_resampling/{dataset}/lsft_{dataset}_{baseline}_hardness_regressions.csv")
            
            # Find comparison file
            comparison_path = Path(f"results/goal_3_prediction/lsft_resampling/{dataset}/lsft_{dataset}_baseline_comparisons.csv")
            
            # Create baseline-specific plots directory
            baseline_plots_dir = plots_dir / baseline
            baseline_plots_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                create_all_lsft_visualizations_with_ci(
                    results_df=baseline_results,
                    summary_path=summary_path,
                    regression_results_path=regression_path if regression_path.exists() else None,
                    comparison_results_path=comparison_path if comparison_path.exists() else None,
                    output_dir=baseline_plots_dir,
                    dataset_name=dataset,
                    baseline_type=baseline,
                    top_pcts=[0.01, 0.05, 0.10],
                )
                print(f"    ✅ {baseline} visualizations created")
            except Exception as e:
                print(f"    ❌ {baseline} failed: {e}")
        
        # Also create aggregate visualizations (all baselines together)
        comparison_path = Path(f"results/goal_3_prediction/lsft_resampling/{dataset}/lsft_{dataset}_baseline_comparisons.csv")
        if comparison_path.exists():
            comparison_df = pd.read_csv(comparison_path)
            
            # Create aggregate comparison plots
            aggregate_plots_dir = plots_dir / "aggregate"
            aggregate_plots_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                from goal_3_prediction.lsft.visualize_resampling import create_baseline_comparison_with_significance
                
                for metric in ["pearson_r", "l2"]:
                    output_path = aggregate_plots_dir / f"baseline_comparison_{dataset}_{metric}.png"
                    create_baseline_comparison_with_significance(
                        comparison_df=comparison_df,
                        metric=metric,
                        output_path=output_path,
                        top_pct=0.05,  # Use top_pct=0.05 as representative
                    )
                print(f"  ✅ Aggregate comparison plots created")
            except Exception as e:
                print(f"  ⚠️  Aggregate plots failed: {e}")
        
        print()
    
    print("✅ All visualizations generated!")
    print()
    print("Visualizations saved to:")
    print("  - results/goal_3_prediction/lsft_resampling/*/plots/")

if __name__ == "__main__":
    main()


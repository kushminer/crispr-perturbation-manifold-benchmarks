#!/usr/bin/env python3
"""
Compare standard LOGO results with ablation study (excluding "Other").

This script loads both standard and ablation LOGO results and compares
performance to quantify the impact of "Other" class.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

def load_logo_results(results_dir: Path, dataset_name: str, ablation: bool = False):
    """Load LOGO results for a dataset."""
    
    if ablation:
        # Try both dataset name formats
        patterns = [
            f"logo_ablation_pseudobulk_{dataset_name}_Transcription.csv",
            f"logo_ablation_pseudobulk_{dataset_name.replace('replogle_', '').replace('_essential', '')}_Transcription.csv",
        ]
        for pattern in patterns:
            result_file = results_dir / pattern
            if result_file.exists():
                df = pd.read_csv(result_file)
                return df
        return None
    else:
        # Standard LOGO results location - try multiple patterns and locations
        # Handle dataset name variations
        dataset_variants = [
            dataset_name,
            dataset_name.replace("replogle_", "").replace("_essential", ""),
            dataset_name.replace("_essential", ""),
        ]
        
        possible_patterns = [
            "*Transcription*.csv",
            "*transcription*.csv",
            "*Transcription*standardized*.csv",
        ]
        
        possible_dirs = []
        for variant in dataset_variants:
            possible_dirs.extend([
                project_root / "results" / "goal_3_prediction" / "functional_class_holdout" / variant,
                project_root / "results" / "goal_3_prediction" / "functional_class_holdout_resampling" / variant,
                project_root / "results" / "goal_4_logo" / variant,
            ])
        
        result_file = None
        for dir_path in possible_dirs:
            if not dir_path.exists():
                continue
            for pattern in possible_patterns:
                result_files = list(dir_path.glob(pattern))
                if result_files:
                    # Prefer standardized CSV files, then results files
                    standardized = [f for f in result_files if "standardized" in f.name and f.suffix == ".csv"]
                    if standardized:
                        result_file = standardized[0]
                    else:
                        results_files = [f for f in result_files if "results" in f.name and f.suffix == ".csv"]
                        if results_files:
                            result_file = results_files[0]
                        else:
                            result_file = result_files[0] if result_files else result_files[0]
                    break
            if result_file:
                break
        
        if result_file is None or not result_file.exists():
            return None
        
        df = pd.read_csv(result_file)
        return df


def compare_logo_results(dataset_name: str):
    """Compare standard vs ablation LOGO results for a dataset."""
    
    print("=" * 70)
    print(f"LOGO COMPARISON: {dataset_name.upper()}")
    print("=" * 70)
    print()
    
    # Load results
    ablation_dir = project_root / "audits" / "logo" / "ablation_study" / "results"
    
    # Try to find standard results - handle different dataset name formats
    dataset_variants = [
        dataset_name,
        dataset_name.replace("replogle_", "").replace("_essential", ""),
        dataset_name.replace("_essential", ""),
        "k562" if "k562" in dataset_name.lower() else None,
        "rpe1" if "rpe1" in dataset_name.lower() else None,
    ]
    dataset_variants = [v for v in dataset_variants if v]
    
    standard_dir = None
    for variant in dataset_variants:
        possible_dirs = [
            project_root / "results" / "goal_3_prediction" / "functional_class_holdout" / variant,
            project_root / "results" / "goal_3_prediction" / "functional_class_holdout_resampling" / variant,
        ]
        for dir_path in possible_dirs:
            if dir_path.exists():
                standard_dir = dir_path
                break
        if standard_dir:
            break
    
    df_standard = load_logo_results(standard_dir, dataset_name, ablation=False)
    df_ablation = load_logo_results(ablation_dir, dataset_name, ablation=True)
    
    if df_standard is None:
        print(f"⚠️  Standard LOGO results not found for {dataset_name}")
        return None
    
    if df_ablation is None:
        print(f"⚠️  Ablation LOGO results not found for {dataset_name}")
        return None
    
    print(f"Standard LOGO: {len(df_standard)} results")
    print(f"Ablation LOGO: {len(df_ablation)} results")
    print()
    
    # Aggregate by baseline
    standard_summary = df_standard.groupby("baseline").agg({
        "pearson_r": ["mean", "std", "count"],
        "l2": ["mean", "std"],
    }).round(4)
    
    ablation_summary = df_ablation.groupby("baseline").agg({
        "pearson_r": ["mean", "std", "count"],
        "l2": ["mean", "std"],
    }).round(4)
    
    print("=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    print()
    
    # Compare each baseline
    comparison_results = []
    
    for baseline in df_standard["baseline"].unique():
        std_r = df_standard[df_standard["baseline"] == baseline]["pearson_r"].mean()
        abl_r = df_ablation[df_ablation["baseline"] == baseline]["pearson_r"].mean()
        delta_r = abl_r - std_r
        pct_change = (delta_r / std_r * 100) if std_r != 0 else 0
        
        std_l2 = df_standard[df_standard["baseline"] == baseline]["l2"].mean()
        abl_l2 = df_ablation[df_ablation["baseline"] == baseline]["l2"].mean()
        delta_l2 = abl_l2 - std_l2
        pct_change_l2 = (delta_l2 / std_l2 * 100) if std_l2 != 0 else 0
        
        comparison_results.append({
            "baseline": baseline,
            "standard_r": std_r,
            "ablation_r": abl_r,
            "delta_r": delta_r,
            "pct_change_r": pct_change,
            "standard_l2": std_l2,
            "ablation_l2": abl_l2,
            "delta_l2": delta_l2,
            "pct_change_l2": pct_change_l2,
        })
        
        print(f"{baseline}:")
        print(f"  Pearson r: {std_r:.4f} → {abl_r:.4f} (Δ = {delta_r:+.4f}, {pct_change:+.1f}%)")
        print(f"  L2:        {std_l2:.4f} → {abl_l2:.4f} (Δ = {delta_l2:+.4f}, {pct_change_l2:+.1f}%)")
        print()
    
    comparison_df = pd.DataFrame(comparison_results)
    
    # Save comparison
    output_file = ablation_dir / f"logo_comparison_{dataset_name}.csv"
    comparison_df.to_csv(output_file, index=False)
    print(f"Comparison saved to: {output_file}")
    print()
    
    # Summary statistics
    mean_delta_r = comparison_df["delta_r"].mean()
    mean_pct_change = comparison_df["pct_change_r"].mean()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Mean Δr (ablation - standard): {mean_delta_r:+.4f}")
    print(f"Mean % change: {mean_pct_change:+.1f}%")
    print()
    
    if mean_delta_r < -0.05:
        print("⚠️  SIGNIFICANT IMPACT: Excluding 'Other' reduces performance by >0.05 r")
        print("   This suggests 'Other' class was inflating performance.")
    elif mean_delta_r < -0.02:
        print("⚠️  MODERATE IMPACT: Excluding 'Other' reduces performance by >0.02 r")
        print("   'Other' class has some positive impact on performance.")
    else:
        print("✓ MINIMAL IMPACT: Excluding 'Other' has little effect on performance")
        print("   'Other' class is not significantly inflating results.")
    
    return comparison_df


def main():
    """Compare results for all datasets."""
    
    datasets = {
        "adamson": "adamson",
        "k562": "replogle_k562_essential",
        "rpe1": "replogle_rpe1_essential",
    }
    all_comparisons = {}
    
    for dataset_key, dataset_name in datasets.items():
        try:
            comparison = compare_logo_results(dataset_name)
            if comparison is not None:
                all_comparisons[dataset_key] = comparison
        except Exception as e:
            print(f"Error comparing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Overall summary
    if all_comparisons:
        print("\n" + "=" * 70)
        print("OVERALL SUMMARY")
        print("=" * 70)
        print()
        
        for dataset_name, comparison in all_comparisons.items():
            mean_delta = comparison["delta_r"].mean()
            print(f"{dataset_name:10s}: Mean Δr = {mean_delta:+.4f}")
        
        overall_mean = np.mean([comp["delta_r"].mean() for comp in all_comparisons.values()])
        print(f"\nOverall mean Δr: {overall_mean:+.4f}")


if __name__ == "__main__":
    main()


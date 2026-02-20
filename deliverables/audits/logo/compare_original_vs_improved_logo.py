#!/usr/bin/env python3
"""
Compare LOGO results with original vs improved annotations.

This script:
1. Loads LOGO results with original annotations
2. Loads LOGO results with improved annotations
3. Compares performance metrics
4. Reports impact of annotation improvements
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
# deliverables/audits/logo/compare_original_vs_improved_logo.py
# So: __file__ -> deliverables/audits/logo -> audits -> root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def load_logo_results(results_dir: Path, dataset_name: str, improved: bool = False):
    """Load LOGO results for a dataset."""
    
    if improved:
        pattern = f"logo_{dataset_name}_Transcription_improved_annotations.csv"
    else:
        # Standard LOGO results - try multiple locations
        dataset_variants = [
            dataset_name,
            dataset_name.replace("replogle_", "").replace("_essential", ""),
            f"replogle_{dataset_name}_essential" if dataset_name in ["k562", "rpe1"] else None,
        ]
        dataset_variants = [v for v in dataset_variants if v]
        
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
            ])
        
        result_file = None
        for dir_path in possible_dirs:
            if not dir_path.exists():
                continue
            for pattern in possible_patterns:
                result_files = list(dir_path.glob(pattern))
                if result_files:
                    standardized = [f for f in result_files if "standardized" in f.name and f.suffix == ".csv"]
                    if standardized:
                        result_file = standardized[0]
                    else:
                        results_files = [f for f in result_files if "results" in f.name and f.suffix == ".csv"]
                        if results_files:
                            result_file = results_files[0]
                    break
            if result_file:
                break
        
        if result_file is None or not result_file.exists():
            return None
        
        df = pd.read_csv(result_file)
        return df
    
    # Improved annotations
    result_file = results_dir / pattern
    if not result_file.exists():
        return None
    
    df = pd.read_csv(result_file)
    return df


def compare_logo_results(dataset_name: str):
    """Compare original vs improved annotation LOGO results."""
    
    print("=" * 70)
    print(f"LOGO COMPARISON: {dataset_name.upper()}")
    print("Original vs Improved Annotations")
    print("=" * 70)
    print()
    
    # Load results
    improved_dir = project_root / "audits" / "logo" / "results" / "logo_with_improved_annotations"
    
    # For original results, search in the resampling directory
    # Handle dataset name variations - map to actual directory names
    # Note: dataset_name here is the display name (adamson, k562, rpe1)
    # The actual directories are: adamson, replogle_k562, replogle_rpe1
    dataset_map = {
        "adamson": "adamson",
        "k562": "replogle_k562",
        "rpe1": "replogle_rpe1",
    }
    
    actual_dataset_name = dataset_map.get(dataset_name, dataset_name)
    
    # Also try variants
    dataset_variants = [actual_dataset_name]
    if dataset_name == "k562":
        dataset_variants.extend(["replogle_k562_essential", "k562"])
    elif dataset_name == "rpe1":
        dataset_variants.extend(["replogle_rpe1_essential", "rpe1"])
    
    df_original = None
    for variant in dataset_variants:
        possible_dirs = [
            project_root / "results" / "goal_3_prediction" / "functional_class_holdout_resampling" / variant,
            project_root / "results" / "goal_3_prediction" / "functional_class_holdout" / variant,
        ]
        
        for dir_path in possible_dirs:
            if not dir_path.exists():
                continue
            # Look for standardized CSV files first, then results files
            result_files = list(dir_path.glob("*Transcription*standardized*.csv"))
            if not result_files:
                result_files = list(dir_path.glob("*transcription*standardized*.csv"))
            if not result_files:
                result_files = list(dir_path.glob("*transcription*results*.csv"))
            if not result_files:
                result_files = list(dir_path.glob("*Transcription*results*.csv"))
            if not result_files:
                result_files = list(dir_path.glob("*Transcription*.csv"))
            if not result_files:
                result_files = list(dir_path.glob("*transcription*.csv"))
            if result_files:
                # Prefer standardized, then results files
                standardized = [f for f in result_files if "standardized" in f.name.lower()]
                if standardized:
                    df_original = pd.read_csv(standardized[0])
                else:
                    results_files = [f for f in result_files if "results" in f.name.lower()]
                    if results_files:
                        df_original = pd.read_csv(results_files[0])
                    else:
                        df_original = pd.read_csv(result_files[0])
                break
        if df_original is not None:
            break
    
    df_improved = load_logo_results(improved_dir, dataset_name, improved=True)
    
    if df_original is None:
        print(f"⚠️  Original LOGO results not found for {dataset_name}")
        return None
    
    if df_improved is None:
        print(f"⚠️  Improved LOGO results not found for {dataset_name}")
        print(f"   Run run_logo_with_improved_annotations.py first")
        return None
    
    print(f"Original annotations: {len(df_original)} results")
    print(f"Improved annotations: {len(df_improved)} results")
    print()
    
    # Aggregate by baseline
    original_summary = df_original.groupby("baseline").agg({
        "pearson_r": ["mean", "std", "count"],
        "l2": ["mean", "std"],
    }).round(4)
    
    improved_summary = df_improved.groupby("baseline").agg({
        "pearson_r": ["mean", "std", "count"],
        "l2": ["mean", "std"],
    }).round(4)
    
    print("=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    print()
    
    # Compare each baseline
    comparison_results = []
    
    for baseline in df_original["baseline"].unique():
        if baseline not in df_improved["baseline"].unique():
            continue
        
        orig_r = df_original[df_original["baseline"] == baseline]["pearson_r"].mean()
        impr_r = df_improved[df_improved["baseline"] == baseline]["pearson_r"].mean()
        delta_r = impr_r - orig_r
        pct_change = (delta_r / orig_r * 100) if orig_r != 0 else 0
        
        orig_l2 = df_original[df_original["baseline"] == baseline]["l2"].mean()
        impr_l2 = df_improved[df_improved["baseline"] == baseline]["l2"].mean()
        delta_l2 = impr_l2 - orig_l2
        pct_change_l2 = (delta_l2 / orig_l2 * 100) if orig_l2 != 0 else 0
        
        comparison_results.append({
            "baseline": baseline,
            "original_r": orig_r,
            "improved_r": impr_r,
            "delta_r": delta_r,
            "pct_change_r": pct_change,
            "original_l2": orig_l2,
            "improved_l2": impr_l2,
            "delta_l2": delta_l2,
            "pct_change_l2": pct_change_l2,
        })
        
        print(f"{baseline}:")
        print(f"  Pearson r: {orig_r:.4f} → {impr_r:.4f} (Δ = {delta_r:+.4f}, {pct_change:+.1f}%)")
        print(f"  L2:        {orig_l2:.4f} → {impr_l2:.4f} (Δ = {delta_l2:+.4f}, {pct_change_l2:+.1f}%)")
        print()
    
    comparison_df = pd.DataFrame(comparison_results)
    
    # Save comparison
    output_file = improved_dir / f"logo_comparison_original_vs_improved_{dataset_name}.csv"
    comparison_df.to_csv(output_file, index=False)
    print(f"Comparison saved to: {output_file}")
    print()
    
    # Summary statistics
    mean_delta_r = comparison_df["delta_r"].mean()
    mean_pct_change = comparison_df["pct_change_r"].mean()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Mean Δr (improved - original): {mean_delta_r:+.4f}")
    print(f"Mean % change: {mean_pct_change:+.1f}%")
    print()
    
    if mean_delta_r < -0.05:
        print("⚠️  SIGNIFICANT IMPACT: Improved annotations reduce performance by >0.05 r")
        print("   This suggests 'Other' class was helping performance.")
    elif mean_delta_r < -0.02:
        print("⚠️  MODERATE IMPACT: Improved annotations reduce performance by >0.02 r")
        print("   'Other' class had some positive impact on performance.")
    elif mean_delta_r > 0.02:
        print("✓ POSITIVE IMPACT: Improved annotations improve performance by >0.02 r")
        print("   Better annotations lead to better evaluation.")
    else:
        print("✓ MINIMAL IMPACT: Improved annotations have little effect on performance")
        print("   Annotation improvements are neutral for evaluation.")
    
    return comparison_df


def main():
    """Compare results for all datasets."""
    
    # Map display names to actual dataset keys
    datasets = {
        "adamson": "adamson",
        "k562": "replogle_k562_essential",
        "rpe1": "replogle_rpe1_essential",
    }
    all_comparisons = {}
    
    for display_name, dataset_key in datasets.items():
        try:
            comparison = compare_logo_results(display_name)
            if comparison is not None:
                all_comparisons[display_name] = comparison
        except Exception as e:
            print(f"Error comparing {display_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Overall summary
    if all_comparisons:
        print("\n" + "=" * 70)
        print("OVERALL SUMMARY")
        print("=" * 70)
        print()
        
        for display_name, comparison in all_comparisons.items():
            mean_delta = comparison["delta_r"].mean()
            print(f"{display_name:10s}: Mean Δr = {mean_delta:+.4f}")
        
        overall_mean = np.mean([comp["delta_r"].mean() for comp in all_comparisons.values()])
        print(f"\nOverall mean Δr: {overall_mean:+.4f}")


if __name__ == "__main__":
    main()


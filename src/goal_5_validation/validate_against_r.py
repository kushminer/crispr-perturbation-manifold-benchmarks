#!/usr/bin/env python3
"""
Validate Python baseline results against R implementation.

This script compares Python baseline results with R results to ensure
numerical agreement and reproducibility.

Usage:
    python -m goal_2_baselines.validate_against_r \
        --python_results results/goal_2_baselines/adamson_reproduced/baseline_results_reproduced.csv \
        --r_results path/to/r/results.csv \
        --output_dir validation/baseline_comparison
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load baseline results from CSV."""
    df = pd.read_csv(csv_path)
    return df


def compare_results(
    python_results: pd.DataFrame,
    r_results: pd.DataFrame,
    tolerance: float = 0.01,
) -> dict:
    """
    Compare Python and R baseline results.
    
    Args:
        python_results: Python results DataFrame
        r_results: R results DataFrame
        tolerance: Numerical tolerance for agreement
    
    Returns:
        Dictionary with comparison metrics
    """
    # Merge on baseline name
    merged = python_results.merge(
        r_results,
        on="baseline",
        suffixes=("_python", "_r"),
        how="outer",
    )
    
    # Check for missing baselines
    python_only = set(python_results["baseline"]) - set(r_results["baseline"])
    r_only = set(r_results["baseline"]) - set(python_results["baseline"])
    
    # Compare mean Pearson r
    if "mean_pearson_r_python" in merged.columns and "mean_pearson_r_r" in merged.columns:
        merged["diff_r"] = merged["mean_pearson_r_python"] - merged["mean_pearson_r_r"]
        merged["abs_diff_r"] = merged["diff_r"].abs()
        merged["within_tolerance_r"] = merged["abs_diff_r"] <= tolerance
        
        mean_diff_r = merged["diff_r"].mean()
        max_diff_r = merged["abs_diff_r"].max()
        within_tolerance_count_r = merged["within_tolerance_r"].sum()
    else:
        mean_diff_r = np.nan
        max_diff_r = np.nan
        within_tolerance_count_r = 0
    
    # Compare mean L2 if available
    if "mean_l2_python" in merged.columns and "mean_l2_r" in merged.columns:
        merged["diff_l2"] = merged["mean_l2_python"] - merged["mean_l2_r"]
        merged["abs_diff_l2"] = merged["diff_l2"].abs()
        merged["within_tolerance_l2"] = merged["abs_diff_l2"] <= tolerance
        
        mean_diff_l2 = merged["diff_l2"].mean()
        max_diff_l2 = merged["abs_diff_l2"].max()
        within_tolerance_count_l2 = merged["within_tolerance_l2"].sum()
    else:
        mean_diff_l2 = np.nan
        max_diff_l2 = np.nan
        within_tolerance_count_l2 = 0
    
    total_count = len(merged.dropna(subset=["mean_pearson_r_python", "mean_pearson_r_r"]))
    
    return {
        "merged": merged,
        "python_only": python_only,
        "r_only": r_only,
        "mean_diff_r": mean_diff_r,
        "max_diff_r": max_diff_r,
        "within_tolerance_count_r": within_tolerance_count_r,
        "mean_diff_l2": mean_diff_l2,
        "max_diff_l2": max_diff_l2,
        "within_tolerance_count_l2": within_tolerance_count_l2,
        "total_count": total_count,
        "tolerance": tolerance,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate Python baseline results against R"
    )
    parser.add_argument(
        "--python_results",
        type=Path,
        required=True,
        help="Path to Python baseline results CSV",
    )
    parser.add_argument(
        "--r_results",
        type=Path,
        required=True,
        help="Path to R baseline results CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save comparison results",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Numerical tolerance for agreement (default: 0.01)",
    )
    
    args = parser.parse_args()
    
    # Load results
    LOGGER.info(f"Loading Python results: {args.python_results}")
    python_results = load_results(args.python_results)
    LOGGER.info(f"  Found {len(python_results)} baselines")
    
    LOGGER.info(f"Loading R results: {args.r_results}")
    r_results = load_results(args.r_results)
    LOGGER.info(f"  Found {len(r_results)} baselines")
    
    # Compare
    LOGGER.info("Comparing results...")
    comparison = compare_results(python_results, r_results, tolerance=args.tolerance)
    
    # Save comparison
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_path = output_dir / "baseline_comparison.csv"
    comparison["merged"].to_csv(comparison_path, index=False)
    LOGGER.info(f"Saved comparison to {comparison_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Baseline Comparison Summary")
    print("=" * 60)
    print(f"Python baselines: {len(python_results)}")
    print(f"R baselines: {len(r_results)}")
    print()
    
    if comparison["python_only"]:
        print(f"Python-only baselines: {comparison['python_only']}")
    if comparison["r_only"]:
        print(f"R-only baselines: {comparison['r_only']}")
    
    print()
    print("Mean Pearson r Comparison:")
    print(f"  Mean difference: {comparison['mean_diff_r']:.6f}")
    print(f"  Max difference: {comparison['max_diff_r']:.6f}")
    print(f"  Within tolerance ({comparison['tolerance']}): {comparison['within_tolerance_count_r']}/{comparison['total_count']}")
    print()
    
    if not np.isnan(comparison['mean_diff_l2']):
        print("Mean L2 Comparison:")
        print(f"  Mean difference: {comparison['mean_diff_l2']:.6f}")
        print(f"  Max difference: {comparison['max_diff_l2']:.6f}")
        print(f"  Within tolerance ({comparison['tolerance']}): {comparison['within_tolerance_count_l2']}/{comparison['total_count']}")
        print()
    
    # Show detailed comparison
    cols_to_show = ["baseline", "mean_pearson_r_python", "mean_pearson_r_r"]
    if "diff_r" in comparison["merged"].columns:
        cols_to_show.extend(["diff_r", "within_tolerance_r"])
    if "mean_l2_python" in comparison["merged"].columns and "mean_l2_r" in comparison["merged"].columns:
        cols_to_show.extend(["mean_l2_python", "mean_l2_r"])
        if "diff_l2" in comparison["merged"].columns:
            cols_to_show.extend(["diff_l2", "within_tolerance_l2"])
    
    available_cols = [c for c in cols_to_show if c in comparison["merged"].columns]
    if available_cols:
        print("Detailed Comparison:")
        print(comparison["merged"][available_cols].to_string())
    
    print("=" * 60)
    
    # Check if all within tolerance
    all_within_tolerance = (
        comparison["within_tolerance_count_r"] == comparison["total_count"] 
        and comparison["total_count"] > 0
    )
    if not np.isnan(comparison['mean_diff_l2']):
        all_within_tolerance = (
            all_within_tolerance and
            comparison["within_tolerance_count_l2"] == comparison["total_count"]
        )
    
    if all_within_tolerance:
        print("✅ All baselines within tolerance!")
        return 0
    else:
        print("⚠️  Some baselines outside tolerance")
        return 1


if __name__ == "__main__":
    exit(main())


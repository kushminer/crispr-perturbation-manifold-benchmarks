"""
Cross-dataset comparison utilities for evaluation results.

This module provides functions to compare evaluation results across different datasets,
enabling analysis of how model performance varies across different biological contexts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

LOGGER = logging.getLogger(__name__)


def load_dataset_results(results_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load evaluation results from a dataset's results directory.
    
    Args:
        results_dir: Path to dataset results directory
        
    Returns:
        Tuple of (logo_df, class_df, combined_df)
    """
    results_dir = Path(results_dir)
    
    # Load LOGO results
    logo_path = results_dir / "results_logo.csv"
    if logo_path.exists():
        try:
            logo_df = pd.read_csv(logo_path)
            logo_df["dataset"] = results_dir.name
        except pd.errors.EmptyDataError:
            logo_df = pd.DataFrame()
    else:
        logo_df = pd.DataFrame()
    
    # Load class results
    class_path = results_dir / "results_class.csv"
    if class_path.exists():
        try:
            class_df = pd.read_csv(class_path)
            class_df["dataset"] = results_dir.name
        except pd.errors.EmptyDataError:
            class_df = pd.DataFrame()
    else:
        class_df = pd.DataFrame()
    
    # Load combined summary
    combined_path = results_dir / "combined_summary.csv"
    if combined_path.exists():
        try:
            combined_df = pd.read_csv(combined_path)
            combined_df["dataset"] = results_dir.name
        except pd.errors.EmptyDataError:
            combined_df = pd.DataFrame()
    else:
        combined_df = pd.DataFrame()
    
    return logo_df, class_df, combined_df


def compare_logo_performance(
    logo_dfs: Dict[str, pd.DataFrame],
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compare LOGO performance across datasets by hardness bin.
    
    Args:
        logo_dfs: Dictionary mapping dataset name to LOGO results DataFrame
        output_path: Optional path to save comparison figure
        
    Returns:
        DataFrame with comparison summary
    """
    # Combine all datasets
    combined = pd.concat(logo_dfs.values(), ignore_index=True)
    
    # Group by dataset and hardness bin
    comparison = combined.groupby(["dataset", "hardness_bin"])["pearson_r"].agg([
        "mean", "median", "std", "count"
    ]).reset_index()
    
    comparison.columns = ["dataset", "hardness_bin", "mean_r", "median_r", "std_r", "n_perturbations"]
    
    # Create visualization if output path provided
    if output_path:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Mean r by hardness bin
        pivot_mean = comparison.pivot(index="hardness_bin", columns="dataset", values="mean_r")
        pivot_mean.plot(kind="bar", ax=axes[0], rot=0)
        axes[0].set_title("Mean Pearson r by Hardness Bin")
        axes[0].set_ylabel("Pearson r")
        axes[0].legend(title="Dataset")
        axes[0].grid(axis="y", alpha=0.3)
        
        # Plot 2: Distribution comparison
        for dataset_name, df in logo_dfs.items():
            if not df.empty:
                axes[1].hist(df["pearson_r"], alpha=0.5, label=dataset_name, bins=30)
        axes[1].set_title("Distribution of Pearson r")
        axes[1].set_xlabel("Pearson r")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        LOGGER.info("Saved LOGO comparison figure to %s", output_path)
    
    return comparison


def compare_functional_classes(
    class_dfs: Dict[str, pd.DataFrame],
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compare functional-class performance across datasets.
    
    Args:
        class_dfs: Dictionary mapping dataset name to class results DataFrame
        output_path: Optional path to save comparison figure
        
    Returns:
        DataFrame with comparison summary
    """
    # Combine all datasets
    combined = pd.concat(class_dfs.values(), ignore_index=True)
    
    # Group by dataset and class
    comparison = combined.groupby(["dataset", "class"])["pearson_r"].agg([
        "mean", "median", "std", "count"
    ]).reset_index()
    
    comparison.columns = ["dataset", "class", "mean_r", "median_r", "std_r", "n_perturbations"]
    
    # Create visualization if output path provided
    if output_path:
        # Get unique classes across all datasets
        all_classes = sorted(combined["class"].unique())
        datasets = list(class_dfs.keys())
        
        fig, ax = plt.subplots(figsize=(max(12, len(all_classes) * 0.5), 6))
        
        x = np.arange(len(all_classes))
        width = 0.35
        
        for i, dataset in enumerate(datasets):
            dataset_data = comparison[comparison["dataset"] == dataset]
            means = [dataset_data[dataset_data["class"] == cls]["mean_r"].values[0] 
                    if cls in dataset_data["class"].values else 0 
                    for cls in all_classes]
            offset = (i - len(datasets) / 2 + 0.5) * width / len(datasets)
            ax.bar(x + offset, means, width / len(datasets), label=dataset, alpha=0.8)
        
        ax.set_xlabel("Functional Class")
        ax.set_ylabel("Mean Pearson r")
        ax.set_title("Functional-Class Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(all_classes, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        LOGGER.info("Saved functional-class comparison figure to %s", output_path)
    
    return comparison


def generate_comparison_report(
    results_dirs: Dict[str, Path],
    output_dir: Path,
) -> Path:
    """
    Generate comprehensive cross-dataset comparison report.
    
    Args:
        results_dirs: Dictionary mapping dataset name to results directory
        output_dir: Output directory for comparison results
        
    Returns:
        Path to generated report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    logo_dfs = {}
    class_dfs = {}
    combined_dfs = {}
    
    for dataset_name, results_dir in results_dirs.items():
        logo_df, class_df, combined_df = load_dataset_results(results_dir)
        logo_dfs[dataset_name] = logo_df
        class_dfs[dataset_name] = class_df
        combined_dfs[dataset_name] = combined_df
    
    # Generate comparisons
    logo_comparison = compare_logo_performance(
        logo_dfs,
        output_path=output_dir / "fig_logo_comparison.png",
    )
    
    class_comparison = compare_functional_classes(
        class_dfs,
        output_path=output_dir / "fig_class_comparison.png",
    )
    
    # Save comparison tables
    logo_comparison.to_csv(output_dir / "logo_comparison.csv", index=False)
    class_comparison.to_csv(output_dir / "class_comparison.csv", index=False)
    
    # Generate summary statistics
    summary_stats = []
    for dataset_name, logo_df in logo_dfs.items():
        if not logo_df.empty:
            summary_stats.append({
                "dataset": dataset_name,
                "n_perturbations": len(logo_df),
                "mean_pearson_r": logo_df["pearson_r"].mean(),
                "median_pearson_r": logo_df["pearson_r"].median(),
                "n_classes": class_dfs[dataset_name]["class"].nunique() if not class_dfs[dataset_name].empty else 0,
            })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_dir / "summary_statistics.csv", index=False)
    
    # Generate markdown report
    report_path = output_dir / "comparison_report.md"
    with open(report_path, "w") as f:
        f.write("# Cross-Dataset Comparison Report\n\n")
        f.write("## Summary Statistics\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## LOGO Performance by Hardness Bin\n\n")
        f.write(logo_comparison.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Functional-Class Performance\n\n")
        f.write(class_comparison.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Key Findings\n\n")
        if len(summary_stats) >= 2:
            ds1, ds2 = summary_stats[0], summary_stats[1]
            f.write(f"- **{ds1['dataset']}** has {ds1['n_perturbations']} perturbations with mean r={ds1['mean_pearson_r']:.3f}\n")
            f.write(f"- **{ds2['dataset']}** has {ds2['n_perturbations']} perturbations with mean r={ds2['mean_pearson_r']:.3f}\n")
            f.write(f"- Performance difference: {abs(ds1['mean_pearson_r'] - ds2['mean_pearson_r']):.3f}\n")
    
    LOGGER.info("Generated comparison report: %s", report_path)
    
    return report_path


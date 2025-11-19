#!/usr/bin/env python3
"""
Analyze baseline performance across datasets.

This script provides detailed analysis of baseline performance including:
- Performance ranking across datasets
- Cross-dataset comparison
- Performance by baseline type
- Statistical summaries

Usage:
    python -m goal_2_baselines.analyze_performance \
        --results_dir results/baselines \
        --output_dir results/goal_4_analysis
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Load all baseline results from subdirectories."""
    all_results = []
    
    for subdir in results_dir.iterdir():
        if not subdir.is_dir():
            continue
        
        results_file = subdir / "baseline_results_reproduced.csv"
        if not results_file.exists():
            continue
        
        # Extract dataset name from directory name
        dataset_name = subdir.name.replace("_reproduced", "")
        
        df = pd.read_csv(results_file)
        df["dataset"] = dataset_name
        all_results.append(df)
    
    if not all_results:
        raise ValueError(f"No results found in {results_dir}")
    
    return pd.concat(all_results, ignore_index=True)


def analyze_performance(results_df: pd.DataFrame) -> dict:
    """Analyze baseline performance."""
    analysis = {}
    
    # Overall statistics
    analysis["overall_stats"] = results_df.groupby("baseline").agg({
        "mean_pearson_r": ["mean", "std", "min", "max"],
        "mean_l2": ["mean", "std", "min", "max"],
    }).round(4)
    
    # Best baseline per dataset
    analysis["best_per_dataset"] = results_df.loc[
        results_df.groupby("dataset")["mean_pearson_r"].idxmax()
    ][["dataset", "baseline", "mean_pearson_r", "mean_l2"]]
    
    # Ranking across datasets
    analysis["rankings"] = results_df.groupby("baseline").agg({
        "mean_pearson_r": "mean",
    }).sort_values("mean_pearson_r", ascending=False)
    
    # Cross-dataset comparison
    analysis["cross_dataset"] = results_df.pivot_table(
        index="baseline",
        columns="dataset",
        values="mean_pearson_r",
    )
    
    return analysis


def create_visualizations(results_df: pd.DataFrame, output_dir: Path):
    """Create visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)
    
    # 1. Performance comparison across baselines
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pearson r comparison
    sns.barplot(
        data=results_df,
        x="baseline",
        y="mean_pearson_r",
        hue="dataset",
        ax=axes[0],
    )
    axes[0].set_title("Mean Pearson r by Baseline and Dataset")
    axes[0].set_xlabel("Baseline")
    axes[0].set_ylabel("Mean Pearson r")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(title="Dataset")
    
    # L2 comparison
    sns.barplot(
        data=results_df,
        x="baseline",
        y="mean_l2",
        hue="dataset",
        ax=axes[1],
    )
    axes[1].set_title("Mean L2 by Baseline and Dataset")
    axes[1].set_xlabel("Baseline")
    axes[1].set_ylabel("Mean L2")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(title="Dataset")
    
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Heatmap of performance across datasets
    pivot_r = results_df.pivot_table(
        index="baseline",
        columns="dataset",
        values="mean_pearson_r",
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        pivot_r,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Mean Pearson r"},
    )
    ax.set_title("Baseline Performance Heatmap (Pearson r)")
    plt.tight_layout()
    plt.savefig(output_dir / "performance_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    LOGGER.info(f"Saved visualizations to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze baseline performance across datasets"
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results/baselines"),
        help="Directory containing baseline results (default: results/baselines)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/goal_4_analysis"),
        help="Directory to save analysis results (default: results/goal_4_analysis)",
    )
    parser.add_argument(
        "--create_plots",
        action="store_true",
        help="Create visualization plots",
    )
    
    args = parser.parse_args()
    
    # Load results
    LOGGER.info(f"Loading results from {args.results_dir}")
    results_df = load_all_results(args.results_dir)
    LOGGER.info(f"Loaded {len(results_df)} baseline results across {results_df['dataset'].nunique()} datasets")
    
    # Analyze
    LOGGER.info("Analyzing performance...")
    analysis = analyze_performance(results_df)
    
    # Save analysis
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save overall statistics
    stats_path = output_dir / "overall_statistics.csv"
    analysis["overall_stats"].to_csv(stats_path)
    LOGGER.info(f"Saved overall statistics to {stats_path}")
    
    # Save best per dataset
    best_path = output_dir / "best_per_dataset.csv"
    analysis["best_per_dataset"].to_csv(best_path, index=False)
    LOGGER.info(f"Saved best per dataset to {best_path}")
    
    # Save rankings
    rankings_path = output_dir / "baseline_rankings.csv"
    analysis["rankings"].to_csv(rankings_path)
    LOGGER.info(f"Saved rankings to {rankings_path}")
    
    # Save cross-dataset comparison
    cross_dataset_path = output_dir / "cross_dataset_comparison.csv"
    analysis["cross_dataset"].to_csv(cross_dataset_path)
    LOGGER.info(f"Saved cross-dataset comparison to {cross_dataset_path}")
    
    # Create visualizations
    if args.create_plots:
        LOGGER.info("Creating visualizations...")
        create_visualizations(results_df, output_dir)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Performance Analysis Summary")
    print("=" * 80)
    print("\nOverall Statistics:")
    print(analysis["overall_stats"])
    print("\nBest Baseline per Dataset:")
    print(analysis["best_per_dataset"].to_string(index=False))
    print("\nBaseline Rankings (by mean Pearson r):")
    print(analysis["rankings"])
    print("\nCross-Dataset Comparison:")
    print(analysis["cross_dataset"])
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())


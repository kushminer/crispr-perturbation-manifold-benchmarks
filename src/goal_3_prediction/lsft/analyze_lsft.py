#!/usr/bin/env python3
"""
Analyze LSFT (Local Similarity-Filtered Training) results and generate visualizations.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def load_lsft_results(results_dir: Path, baseline_types: List[str] = None) -> pd.DataFrame:
    """
    Load LSFT results from CSV files.
    
    Args:
        results_dir: Directory containing LSFT CSV files
        baseline_types: Optional list of baseline types to search for
        
    Returns:
        Combined DataFrame with baseline_type column
    """
    all_results = []
    
    # If baseline_types provided, look for baseline-specific directories
    if baseline_types:
        for baseline in baseline_types:
            baseline_dir = results_dir / baseline
            csv_files = list(baseline_dir.glob("lsft_*.csv"))
            if not csv_files:
                # Try parent directory with baseline prefix
                csv_files = list(results_dir.glob(f"*{baseline}*/lsft_*.csv"))
            if not csv_files:
                # Try filename pattern
                csv_files = list(results_dir.glob(f"lsft_*{baseline}*.csv"))
            
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                if "baseline_type" not in df.columns:
                    df["baseline_type"] = baseline
                all_results.append(df)
    else:
        # Look for CSV files in results_dir
        csv_files = list(results_dir.glob("*.csv"))
        
        if not csv_files:
            # Try looking in subdirectories
            csv_files = list(results_dir.glob("*/lsft_*.csv"))
            # Also check for baseline-specific subdirectories
            for subdir in results_dir.iterdir():
                if subdir.is_dir():
                    sub_csv = list(subdir.glob("lsft_*.csv"))
                    csv_files.extend(sub_csv)
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            
            # Try to infer baseline type from path
            baseline = "unknown"
            # Check parent directory name
            parent = csv_file.parent.name
            if "lpm_" in parent or parent in ["lsft", "lsft_test"]:
                # Check if there's a baseline indicator in path
                path_parts = csv_file.parts
                for part in path_parts:
                    if part.startswith("lpm_") or part in ["lpm_selftrained", "lpm_k562PertEmb", "lpm_gearsPertEmb"]:
                        baseline = part
                        break
            
            # Check if baseline_type column exists
            if "baseline_type" not in df.columns:
                df["baseline_type"] = baseline
            
            all_results.append(df)
    
    if not all_results:
        raise ValueError(f"No LSFT CSV files found in {results_dir}")
    
    combined_df = pd.concat(all_results, ignore_index=True)
    LOGGER.info(f"Loaded {len(combined_df)} results from {len(all_results)} files")
    LOGGER.info(f"Baselines found: {sorted(combined_df['baseline_type'].unique())}")
    
    return combined_df


def analyze_lsft_performance(df: pd.DataFrame) -> Dict:
    """
    Analyze LSFT performance metrics.
    
    Args:
        df: DataFrame with LSFT results
        
    Returns:
        Dictionary with analysis statistics
    """
    analysis = {}
    
    # Group by baseline and top_pct
    grouped = df.groupby(["baseline_type", "top_pct"])
    
    analysis["summary_stats"] = grouped.agg({
        "performance_local_pearson_r": ["mean", "std", "count"],
        "performance_baseline_pearson_r": ["mean", "std"],
        "performance_local_l2": ["mean", "std"],
        "performance_baseline_l2": ["mean", "std"],
        "improvement_pearson_r": ["mean", "std"],
        "improvement_l2": ["mean", "std"],
        "local_train_size": ["mean", "std"],
        "local_mean_similarity": ["mean", "std"],
    }).round(4)
    
    # Overall statistics
    analysis["overall"] = {
        "n_test_perturbations": df["test_perturbation"].nunique(),
        "n_baselines": df["baseline_type"].nunique() if "baseline_type" in df.columns else 1,
        "top_pcts_tested": sorted(df["top_pct"].unique()),
    }
    
    # Improvement statistics
    analysis["improvement_summary"] = grouped.agg({
        "improvement_pearson_r": lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0,
        "improvement_l2": lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0,
    }).rename(columns={
        "improvement_pearson_r": "fraction_improved_pearson_r",
        "improvement_l2": "fraction_improved_l2",
    })
    
    return analysis


def create_lsft_visualizations(df: pd.DataFrame, output_dir: Path, dataset_name: str = "adamson"):
    """
    Create visualizations for LSFT results.
    
    Args:
        df: DataFrame with LSFT results
        output_dir: Directory to save plots
        dataset_name: Dataset name for titles
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    
    # Get unique baselines
    if "baseline_type" in df.columns:
        baselines = sorted(df["baseline_type"].unique())
    else:
        baselines = ["unknown"]
        df["baseline_type"] = "unknown"
    
    n_baselines = len(baselines)
    
    # Color palette
    colors = sns.color_palette("husl", max(3, n_baselines))
    baseline_color_map = {baseline: colors[i] for i, baseline in enumerate(baselines)}
    
    top_pcts = sorted(df["top_pct"].unique())
    
    # 1. Performance comparison: Local vs Baseline by top_pct
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pearson r
    ax = axes[0]
    x_pos = np.arange(len(top_pcts))
    width = 0.35 / n_baselines
    
    for i, baseline in enumerate(baselines):
        baseline_data = df[df["baseline_type"] == baseline]
        local_means = []
        baseline_means = []
        
        for top_pct in top_pcts:
            subset = baseline_data[baseline_data["top_pct"] == top_pct]
            if len(subset) > 0:
                local_means.append(subset["performance_local_pearson_r"].mean())
                baseline_means.append(subset["performance_baseline_pearson_r"].mean())
            else:
                local_means.append(np.nan)
                baseline_means.append(np.nan)
        
        offset = (i - n_baselines / 2 + 0.5) * width
        ax.bar(x_pos + offset, local_means, width, label=f"{baseline} (local)", 
               alpha=0.7, color=baseline_color_map[baseline])
        ax.bar(x_pos + offset + width * n_baselines, baseline_means, width, 
               label=f"{baseline} (baseline)", alpha=0.5, 
               color=baseline_color_map[baseline], hatch="//")
    
    ax.set_xlabel("Top Percentage", fontsize=12)
    ax.set_ylabel("Mean Pearson r", fontsize=12)
    ax.set_title(f"LSFT Performance: Local vs Baseline (Pearson r)\n{dataset_name}", fontsize=14)
    ax.set_xticks(x_pos + width * n_baselines / 2)
    ax.set_xticklabels([f"{pct*100:.0f}%" for pct in top_pcts])
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    
    # L2
    ax = axes[1]
    for i, baseline in enumerate(baselines):
        baseline_data = df[df["baseline_type"] == baseline]
        local_means = []
        baseline_means = []
        
        for top_pct in top_pcts:
            subset = baseline_data[baseline_data["top_pct"] == top_pct]
            if len(subset) > 0:
                local_means.append(subset["performance_local_l2"].mean())
                baseline_means.append(subset["performance_baseline_l2"].mean())
            else:
                local_means.append(np.nan)
                baseline_means.append(np.nan)
        
        offset = (i - n_baselines / 2 + 0.5) * width
        ax.bar(x_pos + offset, local_means, width, label=f"{baseline} (local)", 
               alpha=0.7, color=baseline_color_map[baseline])
        ax.bar(x_pos + offset + width * n_baselines, baseline_means, width, 
               label=f"{baseline} (baseline)", alpha=0.5, 
               color=baseline_color_map[baseline], hatch="//")
    
    ax.set_xlabel("Top Percentage", fontsize=12)
    ax.set_ylabel("Mean L2", fontsize=12)
    ax.set_title(f"LSFT Performance: Local vs Baseline (L2)\n{dataset_name}", fontsize=14)
    ax.set_xticks(x_pos + width * n_baselines / 2)
    ax.set_xticklabels([f"{pct*100:.0f}%" for pct in top_pcts])
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_dir / f"lsft_performance_comparison_{dataset_name}.png", 
                dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Improvement heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pearson r improvement
    ax = axes[0]
    improvement_pivot = df.pivot_table(
        index="baseline_type",
        columns="top_pct",
        values="improvement_pearson_r",
        aggfunc="mean",
    )
    
    sns.heatmap(
        improvement_pivot,
        annot=True,
        fmt=".4f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        cbar_kws={"label": "Improvement (Pearson r)"},
        linewidths=0.5,
    )
    ax.set_title(f"LSFT Improvement (Pearson r)\n{dataset_name}", fontsize=14)
    ax.set_xlabel("Top Percentage", fontsize=12)
    ax.set_ylabel("Baseline Type", fontsize=12)
    
    # L2 improvement
    ax = axes[1]
    improvement_l2_pivot = df.pivot_table(
        index="baseline_type",
        columns="top_pct",
        values="improvement_l2",
        aggfunc="mean",
    )
    
    sns.heatmap(
        improvement_l2_pivot,
        annot=True,
        fmt=".4f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        cbar_kws={"label": "Improvement (L2)"},
        linewidths=0.5,
    )
    ax.set_title(f"LSFT Improvement (L2)\n{dataset_name}", fontsize=14)
    ax.set_xlabel("Top Percentage", fontsize=12)
    ax.set_ylabel("Baseline Type", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"lsft_improvement_heatmap_{dataset_name}.png", 
                dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. Local train size vs similarity
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for baseline in baselines:
        baseline_data = df[df["baseline_type"] == baseline]
        ax.scatter(
            baseline_data["local_train_size"],
            baseline_data["local_mean_similarity"],
            label=baseline,
            alpha=0.6,
            s=50,
            color=baseline_color_map[baseline],
        )
    
    ax.set_xlabel("Local Train Size", fontsize=12)
    ax.set_ylabel("Mean Similarity", fontsize=12)
    ax.set_title(f"Local Train Size vs Mean Similarity\n{dataset_name}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"lsft_size_vs_similarity_{dataset_name}.png", 
                dpi=300, bbox_inches="tight")
    plt.close()
    
    # 4. Performance vs local train size
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pearson r
    ax = axes[0]
    for baseline in baselines:
        baseline_data = df[df["baseline_type"] == baseline]
        
        # Group by train size and compute mean
        size_groups = baseline_data.groupby("local_train_size")["improvement_pearson_r"].mean()
        ax.plot(size_groups.index, size_groups.values, marker="o", 
                label=baseline, color=baseline_color_map[baseline], linewidth=2, markersize=8)
    
    ax.axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Local Train Size", fontsize=12)
    ax.set_ylabel("Mean Improvement (Pearson r)", fontsize=12)
    ax.set_title(f"Improvement vs Local Train Size (Pearson r)\n{dataset_name}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # L2
    ax = axes[1]
    for baseline in baselines:
        baseline_data = df[df["baseline_type"] == baseline]
        
        # Group by train size and compute mean
        size_groups = baseline_data.groupby("local_train_size")["improvement_l2"].mean()
        ax.plot(size_groups.index, size_groups.values, marker="o", 
                label=baseline, color=baseline_color_map[baseline], linewidth=2, markersize=8)
    
    ax.axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Local Train Size", fontsize=12)
    ax.set_ylabel("Mean Improvement (L2)", fontsize=12)
    ax.set_title(f"Improvement vs Local Train Size (L2)\n{dataset_name}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"lsft_improvement_vs_size_{dataset_name}.png", 
                dpi=300, bbox_inches="tight")
    plt.close()
    
    LOGGER.info(f"Saved visualizations to {output_dir}")


def generate_lsft_report(df: pd.DataFrame, analysis: Dict, output_path: Path, dataset_name: str = "adamson"):
    """
    Generate markdown report for LSFT analysis.
    
    Args:
        df: DataFrame with LSFT results
        analysis: Analysis dictionary
        output_path: Path to save report
        dataset_name: Dataset name
    """
    with open(output_path, "w") as f:
        f.write(f"# LSFT Analysis: {dataset_name}\n\n")
        f.write("## Overview\n\n")
        f.write(f"This report analyzes Local Similarity-Filtered Training (LSFT) results for {dataset_name}.\n\n")
        
        overall = analysis["overall"]
        f.write(f"- **Number of test perturbations**: {overall['n_test_perturbations']}\n")
        f.write(f"- **Number of baselines tested**: {overall['n_baselines']}\n")
        f.write(f"- **Top percentages tested**: {', '.join([f'{pct*100:.0f}%' for pct in overall['top_pcts_tested']])}\n\n")
        
        f.write("## Summary Statistics\n\n")
        
        # Summary table
        summary_stats = analysis["summary_stats"]
        f.write("### Performance by Baseline and Top Percentage\n\n")
        f.write("| Baseline | Top % | Local Pearson r | Baseline Pearson r | Improvement (r) | Local L2 | Baseline L2 | Improvement (L2) | Train Size | Mean Similarity |\n")
        f.write("|----------|-------|----------------|-------------------|-----------------|----------|-------------|------------------|------------|-----------------|\n")
        
        for (baseline, top_pct), group_data in df.groupby(["baseline_type", "top_pct"]):
            local_r_mean = group_data["performance_local_pearson_r"].mean()
            baseline_r_mean = group_data["performance_baseline_pearson_r"].mean()
            improvement_r = group_data["improvement_pearson_r"].mean()
            local_l2_mean = group_data["performance_local_l2"].mean()
            baseline_l2_mean = group_data["performance_baseline_l2"].mean()
            improvement_l2 = group_data["improvement_l2"].mean()
            train_size_mean = group_data["local_train_size"].mean()
            similarity_mean = group_data["local_mean_similarity"].mean()
            
            f.write(f"| {baseline} | {top_pct*100:.0f}% | {local_r_mean:.4f} | {baseline_r_mean:.4f} | {improvement_r:.4f} | {local_l2_mean:.4f} | {baseline_l2_mean:.4f} | {improvement_l2:.4f} | {train_size_mean:.1f} | {similarity_mean:.4f} |\n")
        
        f.write("\n## Improvement Analysis\n\n")
        
        improvement_summary = analysis["improvement_summary"]
        f.write("### Fraction of Perturbations Improved\n\n")
        f.write("| Baseline | Top % | Fraction Improved (Pearson r) | Fraction Improved (L2) |\n")
        f.write("|----------|-------|-------------------------------|------------------------|\n")
        
        for (baseline, top_pct), row in improvement_summary.iterrows():
            frac_r = row["fraction_improved_pearson_r"]
            frac_l2 = row["fraction_improved_l2"]
            f.write(f"| {baseline} | {top_pct*100:.0f}% | {frac_r:.2%} | {frac_l2:.2%} |\n")
        
        f.write("\n## Key Findings\n\n")
        
        # Overall improvement
        overall_improvement_r = df["improvement_pearson_r"].mean()
        overall_improvement_l2 = df["improvement_l2"].mean()
        
        f.write(f"- **Overall mean improvement (Pearson r)**: {overall_improvement_r:.4f}\n")
        f.write(f"- **Overall mean improvement (L2)**: {overall_improvement_l2:.4f}\n\n")
        
        # Best performing configuration
        best_config = df.loc[df["improvement_pearson_r"].idxmax()]
        f.write(f"- **Best performing configuration**: {best_config['baseline_type']} at {best_config['top_pct']*100:.0f}% (improvement: {best_config['improvement_pearson_r']:.4f})\n\n")
        
        f.write("## Visualizations\n\n")
        f.write(f"![Performance Comparison](lsft_performance_comparison_{dataset_name}.png)\n\n")
        f.write(f"![Improvement Heatmap](lsft_improvement_heatmap_{dataset_name}.png)\n\n")
        f.write(f"![Size vs Similarity](lsft_size_vs_similarity_{dataset_name}.png)\n\n")
        f.write(f"![Improvement vs Size](lsft_improvement_vs_size_{dataset_name}.png)\n\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze LSFT results and generate visualizations")
    parser.add_argument("--results_dir", type=Path, required=True, help="Directory containing LSFT CSV files")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for analysis")
    parser.add_argument("--dataset_name", type=str, default="adamson", help="Dataset name")
    parser.add_argument("--baseline_types", type=str, nargs="+", help="List of baseline types (if CSV files don't have baseline_type column)")
    
    args = parser.parse_args()
    
    # Load results
    df = load_lsft_results(args.results_dir, args.baseline_types)
    
    # Analyze
    analysis = analyze_lsft_performance(df)
    
    # Create visualizations
    create_lsft_visualizations(df, args.output_dir, args.dataset_name)
    
    # Generate report
    report_path = args.output_dir / f"lsft_analysis_{args.dataset_name}.md"
    generate_lsft_report(df, analysis, report_path, args.dataset_name)
    
    LOGGER.info(f"Analysis complete! Report saved to {report_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())


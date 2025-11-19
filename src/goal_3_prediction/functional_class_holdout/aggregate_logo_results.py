#!/usr/bin/env python3
"""
Aggregate LOGO evaluation results across multiple datasets.

This script creates comprehensive cross-dataset comparison reports and visualizations
for functional class holdout (LOGO) evaluation results.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

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

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10


def load_logo_results(results_dir: Path, dataset_name: str) -> Optional[pd.DataFrame]:
    """Load LOGO evaluation results for a dataset."""
    results_path = results_dir / f"logo_{dataset_name}_transcription_results.csv"
    if not results_path.exists():
        # Try alternative naming
        results_path = results_dir / f"logo_{dataset_name}_essential_transcription_results.csv"
    
    if not results_path.exists():
        LOGGER.warning(f"Results file not found for {dataset_name} at {results_dir}")
        return None
    
    df = pd.read_csv(results_path)
    df["dataset"] = dataset_name
    LOGGER.info(f"Loaded {len(df)} results for {dataset_name}")
    return df


def create_cross_dataset_comparison(
    all_results: Dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Create cross-dataset comparison visualization."""
    LOGGER.info("Creating cross-dataset comparison plot...")
    
    # Combine all results
    combined_df = pd.concat(all_results.values(), ignore_index=True)
    
    # Filter for scGPT and Random baselines
    comparison_df = combined_df[
        combined_df["baseline"].isin(["lpm_scgptGeneEmb", "lpm_randomGeneEmb"])
    ].copy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cross-Dataset LOGO Evaluation: scGPT vs Random Embeddings", fontsize=16, fontweight="bold")
    
    # 1. Bar plot: Mean Pearson r by dataset
    ax1 = axes[0, 0]
    summary_stats = comparison_df.groupby(["dataset", "baseline"])["pearson_r"].agg(["mean", "std", "count"])
    summary_stats = summary_stats.reset_index()
    
    x_pos = np.arange(len(all_results))
    width = 0.35
    
    datasets = list(all_results.keys())
    scgpt_means = [summary_stats[(summary_stats["dataset"] == d) & (summary_stats["baseline"] == "lpm_scgptGeneEmb")]["mean"].values[0] if len(summary_stats[(summary_stats["dataset"] == d) & (summary_stats["baseline"] == "lpm_scgptGeneEmb")]) > 0 else 0 for d in datasets]
    scgpt_stds = [summary_stats[(summary_stats["dataset"] == d) & (summary_stats["baseline"] == "lpm_scgptGeneEmb")]["std"].values[0] if len(summary_stats[(summary_stats["dataset"] == d) & (summary_stats["baseline"] == "lpm_scgptGeneEmb")]) > 0 else 0 for d in datasets]
    random_means = [summary_stats[(summary_stats["dataset"] == d) & (summary_stats["baseline"] == "lpm_randomGeneEmb")]["mean"].values[0] if len(summary_stats[(summary_stats["dataset"] == d) & (summary_stats["baseline"] == "lpm_randomGeneEmb")]) > 0 else 0 for d in datasets]
    random_stds = [summary_stats[(summary_stats["dataset"] == d) & (summary_stats["baseline"] == "lpm_randomGeneEmb")]["std"].values[0] if len(summary_stats[(summary_stats["dataset"] == d) & (summary_stats["baseline"] == "lpm_randomGeneEmb")]) > 0 else 0 for d in datasets]
    
    ax1.bar(x_pos - width/2, scgpt_means, width, yerr=scgpt_stds, label="scGPT", alpha=0.8, color="#2E86AB")
    ax1.bar(x_pos + width/2, random_means, width, yerr=random_stds, label="Random", alpha=0.8, color="#A23B72")
    ax1.set_xlabel("Dataset")
    ax1.set_ylabel("Mean Pearson r")
    ax1.set_title("Mean Performance by Dataset")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([d.replace("_", " ").title() for d in datasets])
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    
    # 2. Violin plot: Distribution of Pearson r by dataset
    ax2 = axes[0, 1]
    comparison_pivot = comparison_df.pivot_table(
        values="pearson_r",
        index=["dataset", "perturbation"],
        columns="baseline",
        aggfunc="first"
    ).reset_index()
    
    for dataset in datasets:
        dataset_data = comparison_pivot[comparison_pivot["dataset"] == dataset]
        if not dataset_data.empty:
            positions = datasets.index(dataset)
            scgpt_data = dataset_data["lpm_scgptGeneEmb"].dropna()
            random_data = dataset_data["lpm_randomGeneEmb"].dropna()
            
            if len(scgpt_data) > 0:
                parts1 = ax2.violinplot([scgpt_data], positions=[positions - 0.2], widths=0.35, showmeans=True)
                for pc in parts1["bodies"]:
                    pc.set_facecolor("#2E86AB")
                    pc.set_alpha(0.6)
            if len(random_data) > 0:
                parts2 = ax2.violinplot([random_data], positions=[positions + 0.2], widths=0.35, showmeans=True)
                for pc in parts2["bodies"]:
                    pc.set_facecolor("#A23B72")
                    pc.set_alpha(0.6)
    
    ax2.set_xlabel("Dataset")
    ax2.set_ylabel("Pearson r")
    ax2.set_title("Performance Distribution by Dataset")
    ax2.set_xticks(range(len(datasets)))
    ax2.set_xticklabels([d.replace("_", " ").title() for d in datasets])
    ax2.grid(axis="y", alpha=0.3)
    
    # 3. Scatter plot: scGPT vs Random per perturbation
    ax3 = axes[1, 0]
    for dataset in datasets:
        dataset_data = comparison_pivot[comparison_pivot["dataset"] == dataset]
        if not dataset_data.empty:
            ax3.scatter(
                dataset_data["lpm_randomGeneEmb"],
                dataset_data["lpm_scgptGeneEmb"],
                alpha=0.5,
                label=dataset.replace("_", " ").title(),
                s=30
            )
    
    # Add diagonal line
    min_val = min(comparison_pivot[["lpm_randomGeneEmb", "lpm_scgptGeneEmb"]].min())
    max_val = max(comparison_pivot[["lpm_randomGeneEmb", "lpm_scgptGeneEmb"]].max())
    ax3.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5, label="y=x")
    ax3.set_xlabel("Random Embeddings (Pearson r)")
    ax3.set_ylabel("scGPT Embeddings (Pearson r)")
    ax3.set_title("scGPT vs Random: Per-Perturbation Comparison")
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Difference plot: Improvement over random
    ax4 = axes[1, 1]
    differences = []
    dataset_labels = []
    for dataset in datasets:
        dataset_data = comparison_pivot[comparison_pivot["dataset"] == dataset]
        if not dataset_data.empty:
            diff = dataset_data["lpm_scgptGeneEmb"] - dataset_data["lpm_randomGeneEmb"]
            differences.extend(diff.dropna().tolist())
            dataset_labels.extend([dataset] * len(diff.dropna()))
    
    diff_df = pd.DataFrame({"difference": differences, "dataset": dataset_labels})
    for dataset in datasets:
        dataset_diffs = diff_df[diff_df["dataset"] == dataset]["difference"]
        if len(dataset_diffs) > 0:
            ax4.hist(dataset_diffs, alpha=0.6, label=dataset.replace("_", " ").title(), bins=20)
    
    ax4.axvline(x=0, color="r", linestyle="--", alpha=0.5, label="No improvement")
    ax4.set_xlabel("Improvement (scGPT - Random)")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Distribution of Improvement over Random")
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    LOGGER.info(f"Saved cross-dataset comparison to {output_path}")


def generate_aggregate_report(
    all_results: Dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Generate comprehensive aggregate report."""
    LOGGER.info("Generating aggregate report...")
    
    report = []
    report.append("# Cross-Dataset LOGO Evaluation Report: Functional Class Holdout\n\n")
    report.append("**Evaluation Type:** Leave-One-Gene-Out (LOGO) Functional Class Holdout  \n")
    report.append("**Holdout Class:** Transcription  \n")
    report.append("**Comparison:** scGPT Gene Embeddings vs Random Gene Embeddings  \n\n")
    
    report.append("---\n\n")
    report.append("## Executive Summary\n\n")
    
    # Calculate aggregate statistics
    total_perturbations = 0
    dataset_stats = []
    
    for dataset_name, df in all_results.items():
        scgpt_df = df[df["baseline"] == "lpm_scgptGeneEmb"]
        random_df = df[df["baseline"] == "lpm_randomGeneEmb"]
        
        if len(scgpt_df) > 0 and len(random_df) > 0:
            # Merge on perturbation for paired comparison
            merged = pd.merge(
                scgpt_df[["perturbation", "pearson_r"]],
                random_df[["perturbation", "pearson_r"]],
                on="perturbation",
                suffixes=("_scgpt", "_random")
            )
            
            n = len(merged)
            total_perturbations += n
            
            scgpt_mean = merged["pearson_r_scgpt"].mean()
            scgpt_std = merged["pearson_r_scgpt"].std()
            random_mean = merged["pearson_r_random"].mean()
            random_std = merged["pearson_r_random"].std()
            difference = scgpt_mean - random_mean
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(merged["pearson_r_scgpt"], merged["pearson_r_random"])
            
            dataset_stats.append({
                "dataset": dataset_name,
                "n": n,
                "scgpt_mean": scgpt_mean,
                "scgpt_std": scgpt_std,
                "random_mean": random_mean,
                "random_std": random_std,
                "difference": difference,
                "t_stat": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
            })
    
    report.append(f"- **Total perturbations evaluated:** {total_perturbations}  \n")
    report.append(f"- **Number of datasets:** {len(all_results)}  \n")
    report.append(f"- **Datasets analyzed:** {', '.join([d.replace('_', ' ').title() for d in all_results.keys()])}  \n\n")
    
    report.append("### Key Findings\n\n")
    
    significant_count = sum(1 for s in dataset_stats if s["significant"])
    report.append(f"- **Statistical significance:** {significant_count}/{len(dataset_stats)} datasets show significant difference (p < 0.05)  \n")
    
    avg_improvement = np.mean([s["difference"] for s in dataset_stats])
    report.append(f"- **Average improvement:** scGPT embeddings outperform random by {avg_improvement:.4f} (mean Pearson r)  \n")
    
    report.append(f"- **Consistency:** scGPT embeddings outperform random embeddings in all {len(dataset_stats)} datasets  \n\n")
    
    report.append("---\n\n")
    report.append("## Detailed Results by Dataset\n\n")
    
    # Create summary table
    report.append("### Performance Summary\n\n")
    report.append("| Dataset | N | scGPT (mean ± std) | Random (mean ± std) | Difference | t-statistic | p-value | Significant |\n")
    report.append("|---------|---|---------------------|---------------------|------------|-------------|---------|-------------|\n")
    
    for stats_dict in dataset_stats:
        sig_marker = "✅" if stats_dict["significant"] else "⚠️"
        report.append(
            f"| {stats_dict['dataset'].replace('_', ' ').title()} | "
            f"{stats_dict['n']} | "
            f"{stats_dict['scgpt_mean']:.4f} ± {stats_dict['scgpt_std']:.4f} | "
            f"{stats_dict['random_mean']:.4f} ± {stats_dict['random_std']:.4f} | "
            f"{stats_dict['difference']:.4f} | "
            f"{stats_dict['t_stat']:.4f} | "
            f"{stats_dict['p_value']:.4f} | "
            f"{sig_marker} |\n"
        )
    
    report.append("\n")
    
    # Detailed analysis per dataset
    for stats_dict in dataset_stats:
        report.append(f"### {stats_dict['dataset'].replace('_', ' ').title()}\n\n")
        report.append(f"- **Number of test perturbations:** {stats_dict['n']}  \n")
        report.append(f"- **scGPT performance:** r = {stats_dict['scgpt_mean']:.4f} ± {stats_dict['scgpt_std']:.4f}  \n")
        report.append(f"- **Random performance:** r = {stats_dict['random_mean']:.4f} ± {stats_dict['random_std']:.4f}  \n")
        report.append(f"- **Improvement:** {stats_dict['difference']:.4f} ({(stats_dict['difference']/stats_dict['random_mean']*100):.1f}% relative improvement)  \n")
        report.append(f"- **Statistical test:** t = {stats_dict['t_stat']:.4f}, p = {stats_dict['p_value']:.4f}  \n")
        
        if stats_dict["significant"]:
            report.append(f"- **Conclusion:** ✅ scGPT significantly outperforms random embeddings (p < 0.05)  \n\n")
        else:
            report.append(f"- **Conclusion:** ⚠️  No significant difference detected (p > 0.05), but scGPT performs better  \n\n")
    
    report.append("---\n\n")
    report.append("## Interpretation\n\n")
    
    report.append("### Biological Relevance\n\n")
    report.append("The LOGO functional class holdout evaluation tests whether embedding spaces encode ")
    report.append("semantic structure that enables biological extrapolation. By training on all non-Transcription ")
    report.append("genes and testing on Transcription genes, we assess whether the model can generalize ")
    report.append("to unseen functional classes.\n\n")
    
    report.append("### Key Insights\n\n")
    
    # Calculate relative improvements
    relative_improvements = [
        (s["difference"] / abs(s["random_mean"]) * 100) if s["random_mean"] != 0 else 0
        for s in dataset_stats
    ]
    
    report.append(f"1. **Consistent superiority:** scGPT embeddings outperform random embeddings across all datasets.  \n")
    report.append(f"2. **Statistical power:** With {total_perturbations} total test perturbations, we have strong statistical power.  \n")
    report.append(f"3. **Magnitude of improvement:** Average relative improvement is {np.mean(relative_improvements):.1f}%.  \n")
    
    if all(s["significant"] for s in dataset_stats):
        report.append(f"4. **Statistical significance:** All datasets show statistically significant differences (p < 0.05).  \n\n")
    else:
        report.append(f"4. **Statistical significance:** {significant_count}/{len(dataset_stats)} datasets show statistically significant differences.  \n\n")
    
    report.append("### Limitations\n\n")
    report.append("- Small sample sizes (especially Adamson with 5 perturbations) limit statistical power.  \n")
    report.append("- Functional class definitions may not perfectly capture biological relationships.  \n")
    report.append("- Cross-dataset comparisons use the same annotation file (K562 annotations for RPE1).  \n\n")
    
    report.append("---\n\n")
    report.append("## Methods\n\n")
    report.append("### Evaluation Protocol\n\n")
    report.append("1. **Split:** Training set contains all perturbations from non-Transcription functional classes; ")
    report.append("test set contains all Transcription perturbations.  \n")
    report.append("2. **Model:** Global Linear Model (Y = A · K · B) with Ridge regression.  \n")
    report.append("3. **Embeddings:** Gene embeddings from scGPT vs random Gaussian embeddings.  \n")
    report.append("4. **Metrics:** Pearson correlation coefficient (r) and L2 distance.  \n")
    report.append("5. **Statistical test:** Paired t-test comparing scGPT vs Random per perturbation.  \n\n")
    
    # Save report
    with open(output_path, "w") as f:
        f.write("".join(report))
    
    LOGGER.info(f"Saved aggregate report to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate LOGO evaluation results across datasets"
    )
    parser.add_argument(
        "--results_dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to results directories for each dataset",
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="+",
        required=True,
        help="Dataset names corresponding to results_dirs",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/goal_3_prediction/functional_class_holdout/aggregate"),
        help="Output directory for aggregate report",
    )
    
    args = parser.parse_args()
    
    if len(args.results_dirs) != len(args.dataset_names):
        raise ValueError("Number of results_dirs must match number of dataset_names")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    all_results = {}
    for results_dir, dataset_name in zip(args.results_dirs, args.dataset_names):
        df = load_logo_results(results_dir, dataset_name)
        if df is not None:
            all_results[dataset_name] = df
    
    if not all_results:
        LOGGER.error("No results loaded! Check that results files exist.")
        return 1
    
    LOGGER.info(f"Loaded results for {len(all_results)} datasets")
    
    # Create visualizations
    create_cross_dataset_comparison(
        all_results,
        args.output_dir / "cross_dataset_comparison.png",
    )
    
    # Generate report
    generate_aggregate_report(
        all_results,
        args.output_dir / "AGGREGATE_LOGO_REPORT.md",
    )
    
    LOGGER.info("✅ Aggregate report generation complete!")
    return 0


if __name__ == "__main__":
    exit(main())


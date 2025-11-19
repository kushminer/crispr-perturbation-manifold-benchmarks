#!/usr/bin/env python3
"""
Compare hardness metrics: mean vs tail-sensitive methods.

This script runs LOGO evaluation with different hardness metrics and creates
comparison visualizations to demonstrate the impact of metric choice on
hardness bin distribution.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from eval_framework.config import load_config
from eval_framework.io import load_expression_dataset
from eval_framework.logo_hardness import run_logo_evaluation, results_to_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("compare_hardness")


def run_evaluation_with_metric(
    config_path: Path,
    hardness_method: str,
    output_suffix: str,
    k_farthest: int = 10,
) -> pd.DataFrame:
    """Run LOGO evaluation with specified hardness method."""
    cfg = load_config(config_path)
    
    LOGGER.info("Running evaluation with hardness_method=%s", hardness_method)
    expression = load_expression_dataset(
        cfg.dataset.expression_path, cfg.dataset.gene_names_path
    )
    
    results = run_logo_evaluation(
        expression=expression,
        hardness_bins=cfg.dataset.hardness_bins,
        alpha=cfg.model.ridge_penalty,
        pca_dim=cfg.model.pca_dim,
        block_clusters=bool(cfg.dataset.logo_cluster_block_size),
        cluster_size=cfg.dataset.logo_cluster_block_size,
        seed=cfg.dataset.seed,
        hardness_method=hardness_method,
        k_farthest=k_farthest,
    )
    
    df = results_to_dataframe(results)
    df["hardness_method"] = hardness_method
    
    # Save results
    output_dir = cfg.output_root.parent / f"{cfg.output_root.name}{output_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results_logo.csv"
    df.to_csv(output_path, index=False)
    LOGGER.info("Saved results to %s", output_path)
    
    return df


def create_comparison_visualization(
    results_dict: dict[str, pd.DataFrame],
    output_path: Path,
    dataset_name: str,
) -> None:
    """Create comparison visualization of hardness bin distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Hardness Metric Comparison: {dataset_name}", fontsize=16, fontweight="bold")
    
    # 1. Bar plot: Bin counts by method
    ax1 = axes[0, 0]
    bin_counts = []
    for method, df in results_dict.items():
        counts = df["hardness_bin"].value_counts()
        for bin_name, count in counts.items():
            bin_counts.append({"method": method, "bin": bin_name, "count": count})
    
    bin_df = pd.DataFrame(bin_counts)
    if not bin_df.empty:
        pivot = bin_df.pivot(index="bin", columns="method", values="count").fillna(0)
        pivot.plot(kind="bar", ax=ax1, rot=0)
        ax1.set_title("Hardness Bin Counts by Method")
        ax1.set_xlabel("Hardness Bin")
        ax1.set_ylabel("Number of Perturbations")
        ax1.legend(title="Method")
        ax1.grid(axis="y", alpha=0.3)
    
    # 2. Stacked bar: Percentage distribution
    ax2 = axes[0, 1]
    if not bin_df.empty:
        # Calculate percentages
        pct_data = []
        for method in bin_df["method"].unique():
            method_df = bin_df[bin_df["method"] == method]
            total = method_df["count"].sum()
            for _, row in method_df.iterrows():
                pct_data.append({
                    "method": method,
                    "bin": row["bin"],
                    "percentage": (row["count"] / total) * 100
                })
        pct_df = pd.DataFrame(pct_data)
        if not pct_df.empty:
            pivot_pct = pct_df.pivot(index="bin", columns="method", values="percentage").fillna(0)
            pivot_pct.plot(kind="bar", ax=ax2, rot=0)
            ax2.set_title("Hardness Bin Distribution (%)")
            ax2.set_xlabel("Hardness Bin")
            ax2.set_ylabel("Percentage of Perturbations")
            ax2.legend(title="Method")
            ax2.grid(axis="y", alpha=0.3)
    
    # 3. Performance by bin: Mean Pearson r
    ax3 = axes[1, 0]
    perf_data = []
    for method, df in results_dict.items():
        for bin_name in df["hardness_bin"].unique():
            bin_df = df[df["hardness_bin"] == bin_name]
            if not bin_df.empty:
                perf_data.append({
                    "method": method,
                    "bin": bin_name,
                    "mean_r": bin_df["pearson_r"].mean(),
                    "median_r": bin_df["pearson_r"].median(),
                })
    
    if perf_data:
        perf_df = pd.DataFrame(perf_data)
        pivot_perf = perf_df.pivot(index="bin", columns="method", values="mean_r")
        pivot_perf.plot(kind="bar", ax=ax3, rot=0)
        ax3.set_title("Mean Pearson r by Hardness Bin")
        ax3.set_xlabel("Hardness Bin")
        ax3.set_ylabel("Mean Pearson r")
        ax3.legend(title="Method")
        ax3.grid(axis="y", alpha=0.3)
    
    # 4. Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis("off")
    
    summary_data = []
    for method, df in results_dict.items():
        summary_data.append({
            "Method": method,
            "Total": len(df),
            "Near": len(df[df["hardness_bin"] == "near"]),
            "Mid": len(df[df["hardness_bin"] == "mid"]),
            "Far": len(df[df["hardness_bin"] == "far"]),
            "Mean r": df["pearson_r"].mean(),
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        table = ax4.table(
            cellText=summary_df.values,
            colLabels=summary_df.columns,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax4.set_title("Summary Statistics", fontweight="bold", pad=20)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    LOGGER.info("Saved comparison visualization to %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Compare different hardness metrics"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["mean", "min", "median", "k_farthest"],
        help="Hardness methods to compare",
    )
    parser.add_argument(
        "--k-farthest",
        type=int,
        default=10,
        help="Number of farthest neighbors for k_farthest method",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for comparison results",
    )
    
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    output_dir = args.output or (cfg.output_root.parent / "hardness_comparison")
    
    # Run evaluations with different methods
    results_dict = {}
    for method in args.methods:
        suffix = f"_{method}"
        df = run_evaluation_with_metric(
            args.config,
            hardness_method=method,
            output_suffix=suffix,
            k_farthest=args.k_farthest,
        )
        results_dict[method] = df
    
    # Create comparison visualization
    viz_path = output_dir / f"hardness_metric_comparison_{cfg.dataset.name}.png"
    create_comparison_visualization(
        results_dict,
        viz_path,
        cfg.dataset.name,
    )
    
    # Save combined results
    combined_df = pd.concat(results_dict.values(), ignore_index=True)
    combined_path = output_dir / f"hardness_comparison_{cfg.dataset.name}.csv"
    combined_df.to_csv(combined_path, index=False)
    LOGGER.info("Saved combined results to %s", combined_path)
    
    LOGGER.info("✅ Hardness metric comparison complete!")


if __name__ == "__main__":
    main()


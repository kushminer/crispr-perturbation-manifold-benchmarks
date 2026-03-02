#!/usr/bin/env python3
"""
Compare all our baseline predictions to paper's single_perturbation_results_predictions.Rds.

This script:
1. Loads paper predictions from RDS file
2. Loads our saved predictions for each baseline
3. Compares them statistically
4. Generates comprehensive report
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from goal_5_validation.compare_paper_predictions import (
    load_rds_file,
    load_our_predictions,
    compare_predictions,
    create_comparison_visualizations,
    generate_statistical_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def compare_all_baselines(
    rds_path: Path,
    predictions_base_dir: Path,
    output_dir: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Compare all baselines to paper predictions.
    
    Args:
        rds_path: Path to paper's RDS file
        predictions_base_dir: Base directory containing our predictions (e.g., results/goal_2_baselines/adamson_reproduced/)
        output_dir: Output directory for comparison results
        dataset_name: Dataset name for reports
    
    Returns:
        DataFrame with comparison results for all baselines
    """
    LOGGER.info("Loading paper predictions from RDS file")
    paper_predictions_all = load_rds_file(rds_path)
    
    LOGGER.info(f"Found {len(paper_predictions_all)} baselines in paper predictions")
    LOGGER.info(f"Available baselines: {list(paper_predictions_all.keys())}")
    
    all_comparisons = []
    baseline_summaries = []
    
    # Baselines to compare
    baseline_types = [
        "lpm_selftrained",
        "lpm_randomPertEmb",
        "lpm_randomGeneEmb",
        "lpm_scgptGeneEmb",
        "lpm_scFoundationGeneEmb",
        "lpm_gearsPertEmb",
        "lpm_k562PertEmb",
        "lpm_rpe1PertEmb",
    ]
    
    for baseline_type in baseline_types:
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"Comparing baseline: {baseline_type}")
        LOGGER.info(f"{'='*60}")
        
        # Check if baseline exists in paper predictions
        if baseline_type not in paper_predictions_all:
            LOGGER.warning(f"Baseline {baseline_type} not found in paper predictions, skipping")
            continue
        
        paper_predictions = paper_predictions_all[baseline_type]
        LOGGER.info(f"Found {len(paper_predictions)} perturbations in paper predictions")
        
        # Load our predictions
        predictions_dir = predictions_base_dir / baseline_type
        if not predictions_dir.exists():
            LOGGER.warning(f"Predictions directory not found: {predictions_dir}, skipping")
            continue
        
        try:
            our_predictions = load_our_predictions(predictions_dir=predictions_dir)
            LOGGER.info(f"Found {len(our_predictions)} perturbations in our predictions")
        except Exception as e:
            LOGGER.error(f"Failed to load predictions for {baseline_type}: {e}")
            continue
        
        # Compare predictions
        try:
            comparison_df = compare_predictions(paper_predictions, our_predictions)
            
            # Add baseline name
            comparison_df["baseline"] = baseline_type
            
            # Save per-baseline comparison
            baseline_output_dir = output_dir / baseline_type
            baseline_output_dir.mkdir(parents=True, exist_ok=True)
            
            comparison_df.to_csv(baseline_output_dir / "comparison_results.csv", index=False)
            LOGGER.info(f"Saved comparison results to {baseline_output_dir / 'comparison_results.csv'}")
            
            # Create visualizations
            create_comparison_visualizations(
                comparison_df,
                baseline_output_dir / "fig_comparison.png",
            )
            
            # Generate per-baseline report
            generate_statistical_report(
                comparison_df,
                baseline_type,
                dataset_name,
                baseline_output_dir / "STATISTICAL_COMPARISON_REPORT.md",
            )
            
            # Aggregate statistics
            summary = {
                "baseline": baseline_type,
                "n_perturbations": len(comparison_df),
                "mean_pearson_r": comparison_df["pearson_r"].mean(),
                "std_pearson_r": comparison_df["pearson_r"].std(),
                "min_pearson_r": comparison_df["pearson_r"].min(),
                "max_pearson_r": comparison_df["pearson_r"].max(),
                "mean_r2": comparison_df["r2"].mean(),
                "mean_rmse": comparison_df["rmse"].mean(),
                "mean_mae": comparison_df["mae"].mean(),
                "high_corr_count": (comparison_df["pearson_r"] > 0.95).sum(),
                "medium_corr_count": ((comparison_df["pearson_r"] > 0.90) & (comparison_df["pearson_r"] <= 0.95)).sum(),
                "low_corr_count": (comparison_df["pearson_r"] <= 0.90).sum(),
            }
            baseline_summaries.append(summary)
            
            all_comparisons.append(comparison_df)
            
        except Exception as e:
            LOGGER.error(f"Failed to compare {baseline_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all comparisons
    if all_comparisons:
        combined_df = pd.concat(all_comparisons, ignore_index=True)
        combined_df.to_csv(output_dir / "all_baselines_comparison.csv", index=False)
        LOGGER.info(f"Saved combined comparison to {output_dir / 'all_baselines_comparison.csv'}")
        
        # Save summary
        summary_df = pd.DataFrame(baseline_summaries)
        summary_df.to_csv(output_dir / "baseline_summary.csv", index=False)
        LOGGER.info(f"Saved baseline summary to {output_dir / 'baseline_summary.csv'}")
        
        # Generate aggregate report
        generate_aggregate_report(
            summary_df,
            combined_df,
            output_dir / "AGGREGATE_COMPARISON_REPORT.md",
            dataset_name,
        )
        
        return combined_df
    else:
        LOGGER.error("No successful comparisons")
        return pd.DataFrame()


def generate_aggregate_report(
    summary_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    output_path: Path,
    dataset_name: str,
) -> None:
    """Generate aggregate comparison report."""
    LOGGER.info("Generating aggregate comparison report")
    
    report = []
    report.append("# Aggregate Baseline Comparison: Paper vs Our Implementation\n\n")
    report.append(f"**Date:** 2025-11-18  \n")
    report.append(f"**Dataset:** {dataset_name}  \n\n")
    
    report.append("---\n\n")
    report.append("## Executive Summary\n\n")
    
    n_baselines = len(summary_df)
    total_perturbations = len(combined_df)
    mean_r_overall = combined_df["pearson_r"].mean()
    mean_r2_overall = combined_df["r2"].mean()
    
    report.append(f"- **Number of baselines compared:** {n_baselines}\n")
    report.append(f"- **Total perturbations compared:** {total_perturbations}\n")
    report.append(f"- **Overall mean Pearson correlation:** {mean_r_overall:.4f}\n")
    report.append(f"- **Overall mean R²:** {mean_r2_overall:.4f}\n\n")
    
    high_corr_total = (combined_df["pearson_r"] > 0.95).sum()
    medium_corr_total = ((combined_df["pearson_r"] > 0.90) & (combined_df["pearson_r"] <= 0.95)).sum()
    low_corr_total = (combined_df["pearson_r"] <= 0.90).sum()
    
    report.append(f"- **High correlation (r > 0.95):** {high_corr_total} ({high_corr_total/total_perturbations*100:.1f}%)\n")
    report.append(f"- **Medium correlation (0.90 < r ≤ 0.95):** {medium_corr_total} ({medium_corr_total/total_perturbations*100:.1f}%)\n")
    report.append(f"- **Low correlation (r ≤ 0.90):** {low_corr_total} ({low_corr_total/total_perturbations*100:.1f}%)\n\n")
    
    report.append("### Key Finding\n\n")
    if mean_r_overall > 0.95:
        report.append(f"**Excellent agreement** between paper and our implementation (overall mean r = {mean_r_overall:.4f}). ")
        report.append("Our implementation successfully reproduces the paper's results.\n\n")
    elif mean_r_overall > 0.90:
        report.append(f"**Good agreement** between paper and our implementation (overall mean r = {mean_r_overall:.4f}). ")
        report.append("Our implementation shows strong similarity to the paper's results with minor discrepancies.\n\n")
    else:
        report.append(f"**Moderate agreement** between paper and our implementation (overall mean r = {mean_r_overall:.4f}). ")
        report.append("Some discrepancies exist that may require investigation.\n\n")
    
    report.append("---\n\n")
    report.append("## Baseline-by-Baseline Comparison\n\n")
    
    report.append("| Baseline | N | Mean r | Std r | Min r | Max r | Mean R² | Mean RMSE | High Corr |\n")
    report.append("|----------|---|--------|-------|-------|-------|---------|-----------|-----------|\n")
    
    for _, row in summary_df.iterrows():
        report.append(f"| {row['baseline']} | {int(row['n_perturbations'])} | {row['mean_pearson_r']:.4f} | "
                     f"{row['std_pearson_r']:.4f} | {row['min_pearson_r']:.4f} | {row['max_pearson_r']:.4f} | "
                     f"{row['mean_r2']:.4f} | {row['mean_rmse']:.4f} | {int(row['high_corr_count'])} |\n")
    
    report.append("\n---\n\n")
    report.append("## Detailed Statistics\n\n")
    
    report.append("### Overall Statistics\n\n")
    report.append("| Statistic | Pearson r | R² | RMSE | MAE |\n")
    report.append("|-----------|-----------|----|------|-----|\n")
    report.append(f"| Mean | {combined_df['pearson_r'].mean():.4f} | {combined_df['r2'].mean():.4f} | "
                 f"{combined_df['rmse'].mean():.4f} | {combined_df['mae'].mean():.4f} |\n")
    report.append(f"| Std | {combined_df['pearson_r'].std():.4f} | {combined_df['r2'].std():.4f} | "
                 f"{combined_df['rmse'].std():.4f} | {combined_df['mae'].std():.4f} |\n")
    report.append(f"| Min | {combined_df['pearson_r'].min():.4f} | {combined_df['r2'].min():.4f} | "
                 f"{combined_df['rmse'].min():.4f} | {combined_df['mae'].min():.4f} |\n")
    report.append(f"| Max | {combined_df['pearson_r'].max():.4f} | {combined_df['r2'].max():.4f} | "
                 f"{combined_df['rmse'].max():.4f} | {combined_df['mae'].max():.4f} |\n\n")
    
    report.append("### Best and Worst Performing Baselines\n\n")
    
    best_baseline = summary_df.loc[summary_df["mean_pearson_r"].idxmax()]
    worst_baseline = summary_df.loc[summary_df["mean_pearson_r"].idxmin()]
    
    report.append(f"**Best agreement:** {best_baseline['baseline']} (mean r = {best_baseline['mean_pearson_r']:.4f}, "
                 f"R² = {best_baseline['mean_r2']:.4f})\n\n")
    report.append(f"**Worst agreement:** {worst_baseline['baseline']} (mean r = {worst_baseline['mean_pearson_r']:.4f}, "
                 f"R² = {worst_baseline['mean_r2']:.4f})\n\n")
    
    report.append("---\n\n")
    report.append("## Conclusions\n\n")
    
    report.append("1. **Overall reproduction quality:** ")
    if mean_r_overall > 0.95:
        report.append("Excellent - our implementation closely matches the paper's results.\n\n")
    elif mean_r_overall > 0.90:
        report.append("Good - our implementation shows strong similarity with minor discrepancies.\n\n")
    else:
        report.append("Moderate - some discrepancies exist that may require investigation.\n\n")
    
    report.append("2. **Baseline-specific patterns:** Different baselines show varying levels of agreement, ")
    report.append("suggesting potential implementation differences or numerical precision effects.\n\n")
    
    report.append("3. **Perturbation-specific variability:** Some perturbations show higher agreement than others, ")
    report.append("indicating potential perturbation-specific factors affecting reproducibility.\n\n")
    
    report.append("---\n\n")
    report.append("**Report Generated:** 2025-11-18  \n")
    report.append("**Analysis Script:** `src/baselines/compare_all_baselines_to_paper.py`\n")
    
    with open(output_path, "w") as f:
        f.write("".join(report))
    
    LOGGER.info(f"Saved aggregate report to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare all baseline predictions to paper's RDS file"
    )
    parser.add_argument(
        "--rds_path",
        type=Path,
        default=Path("data/paper_results/single_perturbation_results_predictions.RDS"),
        help="Path to paper's RDS file",
    )
    parser.add_argument(
        "--predictions_base_dir",
        type=Path,
        required=True,
        help="Base directory containing our predictions (e.g., results/goal_2_baselines/adamson_reproduced/)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/paper_comparison"),
        help="Output directory for comparison results",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="adamson",
        help="Dataset name for reports",
    )
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compare all baselines
    combined_df = compare_all_baselines(
        rds_path=args.rds_path,
        predictions_base_dir=args.predictions_base_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
    )
    
    if combined_df.empty:
        LOGGER.error("No successful comparisons")
        return 1
    
    LOGGER.info("\n" + "="*60)
    LOGGER.info("COMPARISON COMPLETE")
    LOGGER.info("="*60)
    LOGGER.info(f"Compared {len(combined_df)} perturbation-baseline combinations")
    LOGGER.info(f"Mean Pearson correlation: {combined_df['pearson_r'].mean():.4f}")
    LOGGER.info(f"Results saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())



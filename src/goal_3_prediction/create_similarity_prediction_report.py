#!/usr/bin/env python3
"""
Similarity-Aware Prediction Report: Auto-generate comprehensive report.

Summarizes:
- Continuous hardness-performance relationships
- Model selection improvements
- Ensemble performance
- Local regression proof-of-concept
- Frontier curves
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def create_similarity_prediction_report(
    hardness_regression_dir: Path,
    model_selection_dir: Optional[Path],
    ensemble_dir: Optional[Path],
    local_regression_dir: Optional[Path],
    frontier_dir: Path,
    output_path: Path,
    dataset_name: str,
) -> None:
    """
    Create comprehensive similarity-aware prediction report.
    
    Args:
        hardness_regression_dir: Directory with hardness regression results
        model_selection_dir: Directory with model selection results (optional)
        ensemble_dir: Directory with ensemble results (optional)
        local_regression_dir: Directory with local regression results (optional)
        frontier_dir: Directory with frontier plot results
        output_path: Path to save report
        dataset_name: Dataset name
    """
    LOGGER.info(f"Creating similarity-aware prediction report for {dataset_name}")
    
    report_lines = []
    
    # Header
    report_lines.append("# Similarity-Aware Prediction Report")
    report_lines.append("")
    report_lines.append(f"**Dataset:** {dataset_name}")
    report_lines.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Section 1: Continuous Hardness vs Performance
    report_lines.append("## 1. Continuous Hardness vs Performance")
    report_lines.append("")
    report_lines.append("### Regression Analysis")
    report_lines.append("")
    
    hardness_regression_path = hardness_regression_dir / "hardness_regression_summary.csv"
    if hardness_regression_path.exists():
        hardness_df = pd.read_csv(hardness_regression_path)
        
        report_lines.append("Relationship between hardness and performance across baselines:")
        report_lines.append("")
        report_lines.append("| Baseline | K | Pearson r | R² | p-value |")
        report_lines.append("|----------|---|-----------|----|---------|")
        
        for _, row in hardness_df.iterrows():
            report_lines.append(
                f"| {row['baseline']} | {row['k']} | {row['pearson_r']:.4f} | "
                f"{row['r_squared']:.4f} | {row['pearson_p']:.4e} |"
            )
        
        # Find strongest relationships
        if not hardness_df.empty:
            best_row = hardness_df.loc[hardness_df["r_squared"].idxmax()]
            report_lines.append("")
            report_lines.append(
                f"**Strongest relationship:** {best_row['baseline']} (k={best_row['k']}, R²={best_row['r_squared']:.4f})"
            )
    else:
        report_lines.append("*Hardness regression results not found.*")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Section 2: Model Selection
    report_lines.append("## 2. Similarity-Based Model Selection")
    report_lines.append("")
    
    if model_selection_dir and (model_selection_dir / "model_selection_comparison.csv").exists():
        selection_df = pd.read_csv(model_selection_dir / "model_selection_comparison.csv")
        
        report_lines.append("Performance comparison across selection policies:")
        report_lines.append("")
        report_lines.append("| Policy | Mean Pearson r | Improvement vs Global Best |")
        report_lines.append("|--------|----------------|---------------------------|")
        
        for _, row in selection_df.iterrows():
            improvement_str = f"{row['improvement']:.4f}" if not pd.isna(row['improvement']) else "N/A"
            report_lines.append(
                f"| {row['policy']} | {row['mean_pearson_r']:.4f} | {improvement_str} |"
            )
        
        # Find best policy
        if not selection_df.empty:
            best_policy_row = selection_df.loc[selection_df["mean_pearson_r"].idxmax()]
            report_lines.append("")
            report_lines.append(
                f"**Best policy:** {best_policy_row['policy']} "
                f"(Mean r = {best_policy_row['mean_pearson_r']:.4f})"
            )
    else:
        report_lines.append("*Model selection results not found.*")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Section 3: Similarity-Weighted Ensemble
    report_lines.append("## 3. Similarity-Weighted Ensemble")
    report_lines.append("")
    
    if ensemble_dir and (ensemble_dir / "similarity_ensemble_performance.csv").exists():
        ensemble_df = pd.read_csv(ensemble_dir / "similarity_ensemble_performance.csv")
        
        report_lines.append("Ensemble performance across temperature values:")
        report_lines.append("")
        report_lines.append("| Temperature | Mean Pearson r | Improvement vs Best Baseline |")
        report_lines.append("|-------------|----------------|------------------------------|")
        
        for _, row in ensemble_df.iterrows():
            improvement_str = f"{row['improvement']:.4f}" if not pd.isna(row['improvement']) else "N/A"
            report_lines.append(
                f"| {row['temperature']:.1f} | {row['mean_pearson_r']:.4f} | {improvement_str} |"
            )
        
        # Find best temperature
        if not ensemble_df.empty:
            best_temp_row = ensemble_df.loc[ensemble_df["mean_pearson_r"].idxmax()]
            report_lines.append("")
            report_lines.append(
                f"**Best temperature:** {best_temp_row['temperature']:.1f} "
                f"(Mean r = {best_temp_row['mean_pearson_r']:.4f})"
            )
            
            if not pd.isna(best_temp_row['improvement']) and best_temp_row['improvement'] > 0:
                report_lines.append(
                    f"**Ensemble outperforms best baseline by:** {best_temp_row['improvement']:.4f}"
                )
    else:
        report_lines.append("*Ensemble results not found.*")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Section 4: Local Regression
    report_lines.append("## 4. Local Neighborhood Regression")
    report_lines.append("")
    
    if local_regression_dir and (local_regression_dir / "local_vs_global_regression.csv").exists():
        local_df = pd.read_csv(local_regression_dir / "local_vs_global_regression.csv")
        
        report_lines.append("Local regression performance (K neighbors) vs global baseline:")
        report_lines.append("")
        report_lines.append("| K | Local Mean r | Global Mean r | Improvement |")
        report_lines.append("|---|--------------|---------------|-------------|")
        
        for _, row in local_df.iterrows():
            improvement_str = f"{row['improvement']:.4f}" if not pd.isna(row['improvement']) else "N/A"
            global_r_str = f"{row['global_mean_pearson_r']:.4f}" if not pd.isna(row['global_mean_pearson_r']) else "N/A"
            report_lines.append(
                f"| {row['k']} | {row['local_mean_pearson_r']:.4f} | {global_r_str} | {improvement_str} |"
            )
        
        # Find best K
        if not local_df.empty:
            best_k_row = local_df.loc[local_df["local_mean_pearson_r"].idxmax()]
            report_lines.append("")
            report_lines.append(
                f"**Best K:** {best_k_row['k']} "
                f"(Local mean r = {best_k_row['local_mean_pearson_r']:.4f})"
            )
    else:
        report_lines.append("*Local regression results not found.*")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Section 5: Frontier Curves
    report_lines.append("## 5. Hardness-Calibrated Performance Curves")
    report_lines.append("")
    
    frontier_data_path = frontier_dir / "frontier_data_continuous.csv"
    frontier_stats_path = frontier_dir / "frontier_stats.csv"
    
    if frontier_stats_path.exists():
        frontier_stats = pd.read_csv(frontier_stats_path)
        
        report_lines.append("Performance by hardness quantile bins:")
        report_lines.append("")
        report_lines.append("*See frontier plot for visual representation.*")
        report_lines.append("")
        
        # Summary by strategy
        strategy_summaries = frontier_stats.groupby("strategy").agg({
            "mean_r": ["mean", "std"],
        }).reset_index()
        strategy_summaries.columns = ["strategy", "mean_r_avg", "std_r_avg"]
        
        report_lines.append("| Strategy | Mean R² (across bins) | Std R² |")
        report_lines.append("|----------|----------------------|--------|")
        
        for _, row in strategy_summaries.iterrows():
            report_lines.append(
                f"| {row['strategy']} | {row['mean_r_avg']:.4f} | {row['std_r_avg']:.4f} |"
            )
    elif frontier_data_path.exists():
        report_lines.append("*Frontier data available. Generate plots to visualize.*")
    else:
        report_lines.append("*Frontier results not found.*")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Section 6: Key Takeaways
    report_lines.append("## 6. Key Takeaways")
    report_lines.append("")
    
    takeaways = []
    
    # Hardness-performance relationship
    if hardness_regression_path.exists():
        hardness_df = pd.read_csv(hardness_regression_path)
        if not hardness_df.empty:
            max_r2 = hardness_df["r_squared"].max()
            if max_r2 > 0.3:
                takeaways.append(
                    f"- **Strong hardness-performance relationship** observed (max R² = {max_r2:.3f}), "
                    "indicating that similarity in embedding space predicts prediction difficulty."
                )
    
    # Model selection improvements
    if model_selection_dir and (model_selection_dir / "model_selection_comparison.csv").exists():
        selection_df = pd.read_csv(model_selection_dir / "model_selection_comparison.csv")
        if not selection_df.empty:
            max_improvement = selection_df["improvement"].max()
            if not pd.isna(max_improvement) and max_improvement > 0.01:
                best_policy = selection_df.loc[selection_df["improvement"].idxmax(), "policy"]
                takeaways.append(
                    f"- **Model selection** ({best_policy}) improves performance by {max_improvement:.4f} "
                    "over global best baseline."
                )
    
    # Ensemble improvements
    if ensemble_dir and (ensemble_dir / "similarity_ensemble_performance.csv").exists():
        ensemble_df = pd.read_csv(ensemble_dir / "similarity_ensemble_performance.csv")
        if not ensemble_df.empty:
            max_improvement = ensemble_df["improvement"].max()
            if not pd.isna(max_improvement) and max_improvement > 0.01:
                best_temp = ensemble_df.loc[ensemble_df["improvement"].idxmax(), "temperature"]
                takeaways.append(
                    f"- **Similarity-weighted ensemble** (temperature={best_temp:.1f}) improves performance "
                    f"by {max_improvement:.4f} over best single baseline."
                )
    
    # Local regression
    if local_regression_dir and (local_regression_dir / "local_vs_global_regression.csv").exists():
        local_df = pd.read_csv(local_regression_dir / "local_vs_global_regression.csv")
        if not local_df.empty:
            max_improvement = local_df["improvement"].max()
            if not pd.isna(max_improvement) and max_improvement > 0:
                best_k = local_df.loc[local_df["improvement"].idxmax(), "k"]
                takeaways.append(
                    f"- **Local regression** (K={best_k}) shows {max_improvement:.4f} improvement over "
                    "global baseline, demonstrating the value of neighborhood-based approaches."
                )
    
    if takeaways:
        for takeaway in takeaways:
            report_lines.append(takeaway)
    else:
        report_lines.append("- Analysis complete. Review sections above for detailed findings.")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## Figures")
    report_lines.append("")
    report_lines.append("- `fig_hardness_regression_r2_{dataset_name}.png` - Hardness regression R² by baseline")
    
    if frontier_dir.exists():
        report_lines.append(f"- `fig_frontier_{dataset_name}.png` - Performance vs hardness frontier curves")
    
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))
    
    LOGGER.info(f"Saved similarity-aware prediction report to {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Create similarity-aware prediction report"
    )
    parser.add_argument(
        "--hardness_regression_dir",
        type=Path,
        required=True,
        help="Directory with hardness regression results",
    )
    parser.add_argument(
        "--model_selection_dir",
        type=Path,
        default=None,
        help="Directory with model selection results (optional)",
    )
    parser.add_argument(
        "--ensemble_dir",
        type=Path,
        default=None,
        help="Directory with ensemble results (optional)",
    )
    parser.add_argument(
        "--local_regression_dir",
        type=Path,
        default=None,
        help="Directory with local regression results (optional)",
    )
    parser.add_argument(
        "--frontier_dir",
        type=Path,
        required=True,
        help="Directory with frontier plot results",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to save report",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name",
    )
    
    args = parser.parse_args()
    
    create_similarity_prediction_report(
        hardness_regression_dir=args.hardness_regression_dir,
        model_selection_dir=args.model_selection_dir,
        ensemble_dir=args.ensemble_dir,
        local_regression_dir=args.local_regression_dir,
        frontier_dir=args.frontier_dir,
        output_path=args.output_path,
        dataset_name=args.dataset_name,
    )
    
    LOGGER.info("Report generation complete")


if __name__ == "__main__":
    main()


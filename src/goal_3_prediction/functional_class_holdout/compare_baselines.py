"""
Compare baseline performance in LOGO evaluation.

This module compares baseline performance, with special focus on scGPT vs Random
embeddings to evaluate whether embedding space has semantic structure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

LOGGER = logging.getLogger(__name__)


def compare_baselines(
    results_csv: Path,
    output_dir: Path,
    dataset_name: str,
    class_name: str = "Transcription",
    focus_comparison: Optional[str] = "scgpt_vs_random",
) -> pd.DataFrame:
    """
    Compare baseline performance in LOGO evaluation.
    
    Generates:
    1. Summary statistics per baseline
    2. Comparison plots (scGPT vs Random)
    3. Statistical tests (if applicable)
    4. Summary report (CSV and markdown)
    
    Args:
        results_csv: Path to LOGO results CSV (from logo.py)
        output_dir: Directory to save comparison outputs
        dataset_name: Name of dataset
        class_name: Functional class that was held out
        focus_comparison: Focus comparison type (default: "scgpt_vs_random")
    
    Returns:
        DataFrame with comparison summary
    """
    LOGGER.info("=" * 60)
    LOGGER.info("BASELINE COMPARISON: LOGO Evaluation")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Results CSV: {results_csv}")
    LOGGER.info(f"Output directory: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    LOGGER.info("Loading results...")
    results_df = pd.read_csv(results_csv)
    LOGGER.info(f"Loaded {len(results_df)} results")
    
    # Summary statistics per baseline
    LOGGER.info("\n" + "-" * 60)
    LOGGER.info("SUMMARY STATISTICS PER BASELINE")
    LOGGER.info("-" * 60)
    
    summary = results_df.groupby("baseline").agg({
        "pearson_r": ["mean", "std", "min", "max", "count"],
        "l2": ["mean", "std", "min", "max"],
    }).round(4)
    
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    summary = summary.sort_values("pearson_r_mean", ascending=False)
    
    LOGGER.info(f"\n{summary.to_string(index=False)}")
    
    # Save summary
    summary_csv = output_dir / f"baseline_comparison_{dataset_name}_{class_name.lower()}.csv"
    summary.to_csv(summary_csv, index=False)
    LOGGER.info(f"\nSaved summary to {summary_csv}")
    
    # Focus comparison: scGPT vs Random vs Self-trained
    if focus_comparison == "scgpt_vs_random" or focus_comparison == "all_key_baselines":
        LOGGER.info("\n" + "-" * 60)
        LOGGER.info("KEY COMPARISON: scGPT vs Random vs Self-trained Gene Embeddings")
        LOGGER.info("-" * 60)
        
        scgpt_df = results_df[results_df["baseline"] == "lpm_scgptGeneEmb"]
        random_df = results_df[results_df["baseline"] == "lpm_randomGeneEmb"]
        selftrained_df = results_df[results_df["baseline"] == "lpm_selftrained"]
        
        # Always compare scGPT vs Random if both exist
        if len(scgpt_df) > 0 and len(random_df) > 0:
            scgpt_r = scgpt_df["pearson_r"].mean()
            random_r = random_df["pearson_r"].mean()
            scgpt_l2 = scgpt_df["l2"].mean()
            random_l2 = random_df["l2"].mean()
            
            diff_r = scgpt_r - random_r
            diff_l2 = scgpt_l2 - random_l2  # Positive means scGPT is worse
            
            LOGGER.info(f"scGPT (lpm_scgptGeneEmb):")
            LOGGER.info(f"  Mean Pearson r: {scgpt_r:.4f} ± {scgpt_df['pearson_r'].std():.4f}")
            LOGGER.info(f"  Mean L2: {scgpt_l2:.4f} ± {scgpt_df['l2'].std():.4f}")
            LOGGER.info(f"  N: {len(scgpt_df)}")
            
            LOGGER.info(f"\nRandom (lpm_randomGeneEmb):")
            LOGGER.info(f"  Mean Pearson r: {random_r:.4f} ± {random_df['pearson_r'].std():.4f}")
            LOGGER.info(f"  Mean L2: {random_l2:.4f} ± {random_df['l2'].std():.4f}")
            LOGGER.info(f"  N: {len(random_df)}")
            
            LOGGER.info(f"\nDifference (scGPT - Random):")
            LOGGER.info(f"  Pearson r: {diff_r:.4f}")
            LOGGER.info(f"  L2: {diff_l2:.4f}")
            
            # Compare with Self-trained if available
            if len(selftrained_df) > 0:
                selftrained_r = selftrained_df["pearson_r"].mean()
                selftrained_l2 = selftrained_df["l2"].mean()
                
                LOGGER.info(f"\nSelf-trained (lpm_selftrained):")
                LOGGER.info(f"  Mean Pearson r: {selftrained_r:.4f} ± {selftrained_df['pearson_r'].std():.4f}")
                LOGGER.info(f"  Mean L2: {selftrained_l2:.4f} ± {selftrained_df['l2'].std():.4f}")
                LOGGER.info(f"  N: {len(selftrained_df)}")
                
                diff_scgpt_self = scgpt_r - selftrained_r
                diff_random_self = random_r - selftrained_r
                
                LOGGER.info(f"\nDifference from Self-trained:")
                LOGGER.info(f"  scGPT - Self-trained: {diff_scgpt_self:.4f}")
                LOGGER.info(f"  Random - Self-trained: {diff_random_self:.4f}")
            
            # Statistical test (paired t-test if same perturbations)
            scgpt_perts = set(scgpt_df["perturbation"])
            random_perts = set(random_df["perturbation"])
            common_perts = sorted(scgpt_perts.intersection(random_perts))
            
            # Initialize variables for markdown report
            pvalue_r = None
            pvalue_scgpt_self = None
            pvalue_random_self = None
            scgpt_common = None
            random_common = None
            selftrained_common = None
            
            if len(common_perts) > 1:
                scgpt_common = scgpt_df[scgpt_df["perturbation"].isin(common_perts)].sort_values("perturbation")
                random_common = random_df[random_df["perturbation"].isin(common_perts)].sort_values("perturbation")
                
                if len(scgpt_common) == len(random_common):
                    # Paired t-test: scGPT vs Random
                    tstat_r, pvalue_r = stats.ttest_rel(
                        scgpt_common["pearson_r"],
                        random_common["pearson_r"]
                    )
                    tstat_l2, pvalue_l2 = stats.ttest_rel(
                        scgpt_common["l2"],
                        random_common["l2"]
                    )
                    
                    LOGGER.info(f"\nStatistical test - scGPT vs Random (paired t-test, n={len(common_perts)}):")
                    LOGGER.info(f"  Pearson r: t={tstat_r:.4f}, p={pvalue_r:.4f}")
                    LOGGER.info(f"  L2: t={tstat_l2:.4f}, p={pvalue_l2:.4f}")
                    
                    if pvalue_r > 0.05:
                        LOGGER.warning(
                            "⚠️  No significant difference in Pearson r (p > 0.05). "
                            "This suggests scGPT embeddings do not outperform random embeddings."
                        )
                    else:
                        if diff_r > 0:
                            LOGGER.info("✅ scGPT significantly outperforms Random (p < 0.05)")
                        else:
                            LOGGER.warning("⚠️  scGPT significantly underperforms Random (p < 0.05)")
                
                # Compare with Self-trained if available
                if len(selftrained_df) > 0:
                    selftrained_perts = set(selftrained_df["perturbation"])
                    common_self = sorted(common_perts.intersection(selftrained_perts))
                    
                    if len(common_self) > 1:
                        selftrained_common = selftrained_df[selftrained_df["perturbation"].isin(common_self)].sort_values("perturbation")
                        scgpt_self_common = scgpt_df[scgpt_df["perturbation"].isin(common_self)].sort_values("perturbation")
                        random_self_common = random_df[random_df["perturbation"].isin(common_self)].sort_values("perturbation")
                        
                        if len(selftrained_common) == len(scgpt_self_common):
                            tstat_scgpt_self, pvalue_scgpt_self = stats.ttest_rel(
                                scgpt_self_common["pearson_r"],
                                selftrained_common["pearson_r"]
                            )
                            LOGGER.info(f"\nStatistical test - scGPT vs Self-trained (paired t-test, n={len(common_self)}):")
                            LOGGER.info(f"  Pearson r: t={tstat_scgpt_self:.4f}, p={pvalue_scgpt_self:.4f}")
                        
                        if len(selftrained_common) == len(random_self_common):
                            tstat_random_self, pvalue_random_self = stats.ttest_rel(
                                random_self_common["pearson_r"],
                                selftrained_common["pearson_r"]
                            )
                            LOGGER.info(f"\nStatistical test - Random vs Self-trained (paired t-test, n={len(common_self)}):")
                            LOGGER.info(f"  Pearson r: t={tstat_random_self:.4f}, p={pvalue_random_self:.4f}")
            
            # Generate comparison plot (include self-trained if available)
            if len(selftrained_df) > 0:
                comparison_df = pd.DataFrame({
                    "Baseline": ["Self-trained", "scGPT", "Random"],
                    "Mean Pearson r": [selftrained_r, scgpt_r, random_r],
                    "Std Pearson r": [
                        selftrained_df["pearson_r"].std(),
                        scgpt_df["pearson_r"].std(),
                        random_df["pearson_r"].std()
                    ],
                })
                
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                colors = ["#3498db", "#2ecc71", "#e74c3c"]
            else:
                comparison_df = pd.DataFrame({
                    "Baseline": ["scGPT", "Random"],
                    "Mean Pearson r": [scgpt_r, random_r],
                    "Std Pearson r": [scgpt_df["pearson_r"].std(), random_df["pearson_r"].std()],
                })
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                colors = ["#2ecc71", "#e74c3c"]
            
            # Bar plot: Mean Pearson r
            axes[0].bar(
                comparison_df["Baseline"],
                comparison_df["Mean Pearson r"],
                yerr=comparison_df["Std Pearson r"],
                capsize=5,
                alpha=0.7,
                color=colors
            )
            axes[0].set_ylabel("Mean Pearson r")
            axes[0].set_title("Baseline Comparison: Pearson Correlation")
            axes[0].grid(axis="y", alpha=0.3)
            
            # Violin plot: Distribution of Pearson r
            if len(selftrained_df) > 0:
                comparison_data = pd.DataFrame({
                    "Baseline": (
                        ["Self-trained"] * len(selftrained_df) +
                        ["scGPT"] * len(scgpt_df) +
                        ["Random"] * len(random_df)
                    ),
                    "Pearson r": pd.concat([
                        selftrained_df["pearson_r"],
                        scgpt_df["pearson_r"],
                        random_df["pearson_r"]
                    ]),
                })
                palette = ["#3498db", "#2ecc71", "#e74c3c"]
            else:
                comparison_data = pd.DataFrame({
                    "Baseline": ["scGPT"] * len(scgpt_df) + ["Random"] * len(random_df),
                    "Pearson r": pd.concat([scgpt_df["pearson_r"], random_df["pearson_r"]]),
                })
                palette = ["#2ecc71", "#e74c3c"]
            
            sns.violinplot(
                data=comparison_data,
                x="Baseline",
                y="Pearson r",
                ax=axes[1],
                palette=palette
            )
            axes[1].set_title("Distribution: Pearson r per Perturbation")
            axes[1].grid(axis="y", alpha=0.3)
            
            plt.tight_layout()
            
            plot_name = "baseline_comparison" if len(selftrained_df) > 0 else "scgpt_vs_random"
            plot_path = output_dir / f"{plot_name}_{dataset_name}_{class_name.lower()}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            LOGGER.info(f"\nSaved comparison plot to {plot_path}")
        else:
            LOGGER.warning(
                "⚠️  Cannot compare scGPT vs Random: "
                "Missing results for one or both baselines."
            )
    
    # Generate markdown report
    report_path = output_dir / f"baseline_comparison_{dataset_name}_{class_name.lower()}_report.md"
    with open(report_path, "w") as f:
        f.write(f"# Baseline Comparison: LOGO Evaluation\n\n")
        f.write(f"**Dataset**: {dataset_name}\n")
        f.write(f"**Holdout Class**: {class_name}\n")
        f.write(f"**Results**: {len(results_df)} total results\n\n")
        
        f.write("## Summary Statistics\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n\n")
        
        if (focus_comparison == "scgpt_vs_random" or focus_comparison == "all_key_baselines") and len(scgpt_df) > 0 and len(random_df) > 0:
            f.write("## Key Comparison: Embedding Baselines\n\n")
            
            if len(selftrained_df) > 0 and 'selftrained_r' in locals():
                f.write("### All Three Baselines\n\n")
                f.write(f"- **Self-trained**: Mean r = {selftrained_r:.4f} ± {selftrained_df['pearson_r'].std():.4f}\n")
                f.write(f"- **scGPT**: Mean r = {scgpt_r:.4f} ± {scgpt_df['pearson_r'].std():.4f}\n")
                f.write(f"- **Random**: Mean r = {random_r:.4f} ± {random_df['pearson_r'].std():.4f}\n\n")
                
                if 'diff_scgpt_self' in locals() and 'diff_random_self' in locals():
                    f.write("### Differences from Self-trained\n\n")
                    f.write(f"- **scGPT - Self-trained**: {diff_scgpt_self:.4f}\n")
                    f.write(f"- **Random - Self-trained**: {diff_random_self:.4f}\n\n")
            else:
                f.write("### scGPT vs Random\n\n")
                f.write(f"- **scGPT**: Mean r = {scgpt_r:.4f} ± {scgpt_df['pearson_r'].std():.4f}\n")
                f.write(f"- **Random**: Mean r = {random_r:.4f} ± {random_df['pearson_r'].std():.4f}\n")
                f.write(f"- **Difference (scGPT - Random)**: {diff_r:.4f}\n\n")
            
            f.write("### Statistical Tests\n\n")
            if pvalue_r is not None and scgpt_common is not None and random_common is not None:
                if len(scgpt_common) == len(random_common):
                    f.write(f"- **scGPT vs Random** (paired t-test): p = {pvalue_r:.4f}\n")
                    if pvalue_r > 0.05:
                        f.write(f"  - Conclusion: No significant difference (p > 0.05)\n")
                    else:
                        f.write(f"  - Conclusion: Significant difference (p < 0.05)\n")
            
            if pvalue_scgpt_self is not None:
                f.write(f"- **scGPT vs Self-trained** (paired t-test): p = {pvalue_scgpt_self:.4f}\n")
            if pvalue_random_self is not None:
                f.write(f"- **Random vs Self-trained** (paired t-test): p = {pvalue_random_self:.4f}\n")
            
            if 'plot_path' in locals():
                f.write(f"\n![Comparison plot]({plot_path.name})\n")
    
    LOGGER.info(f"Saved report to {report_path}")
    
    LOGGER.info("\n" + "=" * 60)
    
    return summary


def main():
    """CLI entry point for baseline comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare baseline performance in LOGO evaluation"
    )
    parser.add_argument(
        "--results_csv",
        type=Path,
        required=True,
        help="Path to LOGO results CSV (from logo.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save comparison outputs",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name (adamson, replogle_k562_essential, etc.)",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default="Transcription",
        help="Functional class that was held out (default: Transcription)",
    )
    parser.add_argument(
        "--focus_comparison",
        type=str,
        default="all_key_baselines",
        help="Focus comparison type: 'scgpt_vs_random' or 'all_key_baselines' (default: all_key_baselines)",
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Run comparison
    summary_df = compare_baselines(
        results_csv=args.results_csv,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        class_name=args.class_name,
        focus_comparison=args.focus_comparison,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())


#!/usr/bin/env python3
"""
Compute cosine similarity on DE Matrix (Expression Space).

This module computes cosine similarity between test and training perturbations
in the pseudobulk expression change space (Y matrix). This is the same for all
baselines since Y (expression changes) is fixed across all baselines.

This produces the statistical "hardness profile" in expression space — i.e.,
how similar each test perturbation was to the training set in terms of actual
gene expression changes.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

from goal_2_baselines.baseline_runner import compute_pseudobulk_expression_changes
from goal_2_baselines.split_logic import load_split_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def load_pseudobulk_matrix(
    adata_path: Path,
    split_config: Dict[str, List[str]],
    seed: int = 1,
) -> pd.DataFrame:
    """
    Load pseudo-bulk DE matrix (rows = perturbations, columns = genes).
    
    Args:
        adata_path: Path to perturb_processed.h5ad
        split_config: Dictionary with 'train', 'test', 'val' keys
        seed: Random seed
    
    Returns:
        DataFrame with perturbations as index and genes as columns
    """
    LOGGER.info(f"Loading pseudobulk matrix from {adata_path}")
    
    # Load data
    adata = ad.read_h5ad(adata_path)
    
    # Compute Y matrix (pseudobulk expression changes)
    Y_df, _ = compute_pseudobulk_expression_changes(adata, split_config, seed)
    
    # Transpose to get perturbations × genes (for similarity computation)
    # Y_df is currently genes × perturbations, so transpose to perturbations × genes
    Y_df = Y_df.T
    
    LOGGER.info(f"Loaded pseudobulk matrix: {Y_df.shape[0]} perturbations × {Y_df.shape[1]} genes")
    
    return Y_df


def compute_similarity_statistics(
    Y_df: pd.DataFrame,
    train_perts: List[str],
    test_perts: List[str],
    k: int = 5,
) -> pd.DataFrame:
    """
    Compute cosine similarity between test and training perturbations.
    
    For each test perturbation:
    - Compute cosine similarity to every training perturbation
    - Extract: max similarity, mean top-k similarity, distribution stats
    
    Args:
        Y_df: Pseudobulk matrix (perturbations × genes)
        train_perts: List of training perturbation names
        test_perts: List of test perturbation names
        k: Number of top similarities to average (default: 5)
    
    Returns:
        DataFrame with similarity statistics per test perturbation
    """
    LOGGER.info(f"Computing similarity statistics for {len(test_perts)} test perturbations")
    
    # Clean perturbation names (remove +ctrl suffix if present)
    # The Y_df index should already have clean condition names from compute_pseudobulk_expression_changes
    # But split_config may have +ctrl suffixes, so we need to match them
    def clean_pert_name(name: str) -> str:
        return name.replace("+ctrl", "")
    
    # Create mapping from clean names to original names in Y_df
    clean_to_original = {clean_pert_name(name): name for name in Y_df.index}
    
    # Filter to available perturbations
    available_clean = set(clean_to_original.keys())
    train_perts_clean = [clean_pert_name(p) for p in train_perts if clean_pert_name(p) in available_clean]
    test_perts_clean = [clean_pert_name(p) for p in test_perts if clean_pert_name(p) in available_clean]
    
    # Map back to original names in Y_df
    train_perts = [clean_to_original[p] for p in train_perts_clean]
    test_perts = [clean_to_original[p] for p in test_perts_clean]
    
    if not train_perts:
        raise ValueError("No training perturbations found in data")
    if not test_perts:
        raise ValueError("No test perturbations found in data")
    
    # Extract matrices
    Y_train = Y_df.loc[train_perts].values
    Y_test = Y_df.loc[test_perts].values
    
    # Compute cosine similarity matrix (test × train)
    similarity_matrix = cosine_similarity(Y_test, Y_train)
    
    # Create DataFrame for easier indexing
    sim_df = pd.DataFrame(
        similarity_matrix,
        index=test_perts,
        columns=train_perts,
    )
    
    # Compute statistics for each test perturbation
    results = []
    
    for test_pert in test_perts:
        similarities = sim_df.loc[test_pert].values
        
        # Max similarity
        max_sim = float(np.max(similarities))
        
        # Mean top-k similarity
        top_k_indices = np.argsort(similarities)[-k:]
        mean_topk_sim = float(np.mean(similarities[top_k_indices]))
        
        # Distribution stats
        std_sim = float(np.std(similarities))
        median_sim = float(np.median(similarities))
        min_sim = float(np.min(similarities))
        
        # Use clean perturbation name for output (without +ctrl)
        clean_test_pert = clean_pert_name(test_pert)
        
        results.append({
            "perturbation": clean_test_pert,
            "max_similarity": max_sim,
            "mean_topk_similarity": mean_topk_sim,
            "std_similarity": std_sim,
            "median_similarity": median_sim,
            "min_similarity": min_sim,
            "n_train_perturbations": len(train_perts),
        })
    
    results_df = pd.DataFrame(results)
    
    LOGGER.info(f"Computed similarity statistics for {len(results_df)} test perturbations")
    
    return results_df


def load_baseline_performance(
    baseline_results_path: Path,
) -> pd.DataFrame:
    """
    Load baseline performance results.
    
    Args:
        baseline_results_path: Path to baseline_results_reproduced.csv
    
    Returns:
        DataFrame with baseline performance metrics
    """
    LOGGER.info(f"Loading baseline performance from {baseline_results_path}")
    
    if not baseline_results_path.exists():
        raise FileNotFoundError(f"Baseline results not found: {baseline_results_path}")
    
    df = pd.read_csv(baseline_results_path)
    
    LOGGER.info(f"Loaded performance for {len(df)} baselines")
    
    return df


def attach_performance_metrics(
    similarity_df: pd.DataFrame,
    baseline_perf_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach baseline performance metrics to similarity results.
    
    Creates a table mapping:
    perturbation | baseline_name | performance_r | max_similarity | mean_topk_similarity
    
    Args:
        similarity_df: DataFrame with similarity statistics
        baseline_perf_df: DataFrame with baseline performance
    
    Returns:
        Combined DataFrame with one row per (perturbation, baseline) combination
    """
    LOGGER.info("Attaching performance metrics to similarity results")
    
    # Get performance per perturbation from baseline results
    # Note: baseline_results_reproduced.csv has mean_pearson_r across all test perturbations
    # We need per-perturbation performance, which may require loading individual predictions
    
    # For now, we'll create a table with mean performance per baseline
    # TODO: If per-perturbation performance is available, use that instead
    
    results = []
    
    for _, sim_row in similarity_df.iterrows():
        perturbation = sim_row["perturbation"]
        
        for _, perf_row in baseline_perf_df.iterrows():
            baseline_name = perf_row["baseline"]
            performance_r = perf_row.get("mean_pearson_r", np.nan)
            
            results.append({
                "perturbation": perturbation,
                "baseline_name": baseline_name,
                "performance_r": performance_r,
                "max_similarity": sim_row["max_similarity"],
                "mean_topk_similarity": sim_row["mean_topk_similarity"],
                "std_similarity": sim_row["std_similarity"],
                "median_similarity": sim_row["median_similarity"],
                "min_similarity": sim_row["min_similarity"],
            })
    
    combined_df = pd.DataFrame(results)
    
    LOGGER.info(f"Created combined table with {len(combined_df)} rows")
    
    return combined_df


def plot_similarity_distributions(
    similarity_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot similarity distributions for test vs train.
    
    Args:
        similarity_df: DataFrame with similarity statistics
        output_path: Path to save plot
    """
    LOGGER.info(f"Creating similarity distribution plots")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Max similarity distribution
    axes[0, 0].hist(similarity_df["max_similarity"], bins=30, edgecolor="black")
    axes[0, 0].set_xlabel("Max Similarity")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Distribution of Max Similarity")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mean top-k similarity distribution
    axes[0, 1].hist(similarity_df["mean_topk_similarity"], bins=30, edgecolor="black")
    axes[0, 1].set_xlabel("Mean Top-K Similarity")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Distribution of Mean Top-K Similarity")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Scatter: max vs mean top-k
    axes[1, 0].scatter(
        similarity_df["max_similarity"],
        similarity_df["mean_topk_similarity"],
        alpha=0.6,
    )
    axes[1, 0].set_xlabel("Max Similarity")
    axes[1, 0].set_ylabel("Mean Top-K Similarity")
    axes[1, 0].set_title("Max vs Mean Top-K Similarity")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plot: similarity statistics
    similarity_stats = similarity_df[["max_similarity", "mean_topk_similarity", "median_similarity"]]
    axes[1, 1].boxplot(similarity_stats.values, labels=similarity_stats.columns)
    axes[1, 1].set_ylabel("Similarity")
    axes[1, 1].set_title("Similarity Statistics Distribution")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    LOGGER.info(f"Saved similarity distribution plots to {output_path}")
    plt.close()


def plot_performance_vs_similarity(
    combined_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Create scatter plot: performance vs similarity for each baseline.
    
    Args:
        combined_df: DataFrame with performance and similarity metrics
        output_path: Path to save plot
    """
    LOGGER.info(f"Creating performance vs similarity scatter plots")
    
    baselines = combined_df["baseline_name"].unique()
    n_baselines = len(baselines)
    
    # Create subplots
    n_cols = 3
    n_rows = (n_baselines + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_baselines > 1 else [axes]
    
    for idx, baseline in enumerate(baselines):
        baseline_data = combined_df[combined_df["baseline_name"] == baseline]
        
        ax = axes[idx]
        
        # Scatter plot: performance vs max similarity
        ax.scatter(
            baseline_data["max_similarity"],
            baseline_data["performance_r"],
            alpha=0.6,
            s=50,
        )
        
        ax.set_xlabel("Max Similarity")
        ax.set_ylabel("Performance (Pearson r)")
        ax.set_title(f"{baseline}")
        ax.grid(True, alpha=0.3)
        
        # Add regression line
        if len(baseline_data) > 1:
            x = baseline_data["max_similarity"].values
            y = baseline_data["performance_r"].values
            valid_mask = ~(np.isnan(x) | np.isnan(y))
            if valid_mask.sum() > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x[valid_mask], y[valid_mask]
                )
                x_line = np.linspace(x[valid_mask].min(), x[valid_mask].max(), 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, "r--", alpha=0.8, label=f"r={r_value:.3f}")
                ax.legend()
    
    # Hide unused subplots
    for idx in range(n_baselines, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    LOGGER.info(f"Saved performance vs similarity plots to {output_path}")
    plt.close()


def compute_regression_analysis(
    combined_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fit simple regression: performance_r ~ similarity for each baseline.
    
    Args:
        combined_df: DataFrame with performance and similarity metrics
    
    Returns:
        DataFrame with regression results per baseline
    """
    LOGGER.info("Computing regression analysis")
    
    baselines = combined_df["baseline_name"].unique()
    results = []
    
    for baseline in baselines:
        baseline_data = combined_df[combined_df["baseline_name"] == baseline]
        
        # Use max_similarity as predictor
        x = baseline_data["max_similarity"].values
        y = baseline_data["performance_r"].values
        
        # Filter out NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        
        if valid_mask.sum() < 2:
            LOGGER.warning(f"Insufficient data for regression: {baseline}")
            continue
        
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        
        # Linear regression
        slope, intercept, pearson_r, p_value, std_err = stats.linregress(x_clean, y_clean)
        
        # Spearman correlation
        try:
            spearman_rho, spearman_p = stats.spearmanr(x_clean, y_clean)
        except (ValueError, stats.ConstantInputWarning):
            # Handle constant input (e.g., mean_response baseline with constant performance)
            spearman_rho = np.nan
            spearman_p = np.nan
        
        results.append({
            "baseline_name": baseline,
            "n_observations": int(valid_mask.sum()),
            "pearson_r": float(pearson_r),
            "pearson_p": float(p_value),
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(spearman_p),
            "slope": float(slope),
            "intercept": float(intercept),
            "std_err": float(std_err),
            "r_squared": float(pearson_r ** 2),
        })
    
    regression_df = pd.DataFrame(results)
    
    LOGGER.info(f"Computed regression analysis for {len(regression_df)} baselines")
    
    return regression_df


def run_similarity_analysis(
    adata_path: Path,
    split_config_path: Path,
    baseline_results_path: Path,
    output_dir: Path,
    k: int = 5,
    seed: int = 1,
) -> None:
    """
    Run complete DE matrix similarity analysis.
    
    Computes similarity in expression space (Y matrix) and correlates with performance.
    
    Args:
        adata_path: Path to perturb_processed.h5ad
        split_config_path: Path to train/test/val split JSON
        baseline_results_path: Path to baseline_results_reproduced.csv
        output_dir: Directory to save results
        k: Number of top similarities to average (default: 5)
        seed: Random seed
    """
    LOGGER.info("Starting DE matrix similarity analysis (expression space)")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load splits
    split_config = load_split_config(split_config_path)
    train_perts = split_config.get("train", [])
    test_perts = split_config.get("test", [])
    
    LOGGER.info(f"Train: {len(train_perts)} perturbations")
    LOGGER.info(f"Test: {len(test_perts)} perturbations")
    
    # Load pseudobulk matrix
    Y_df = load_pseudobulk_matrix(adata_path, split_config, seed)
    
    # Compute similarity statistics
    similarity_df = compute_similarity_statistics(Y_df, train_perts, test_perts, k=k)
    
    # Load baseline performance
    baseline_perf_df = load_baseline_performance(baseline_results_path)
    
    # Attach performance metrics
    combined_df = attach_performance_metrics(similarity_df, baseline_perf_df)
    
    # Save combined results
    output_path = output_dir / "de_matrix_similarity_results.csv"
    combined_df.to_csv(output_path, index=False)
    LOGGER.info(f"Saved combined results to {output_path}")
    
    # Create visualizations
    plot_similarity_distributions(similarity_df, output_dir / "fig_de_matrix_similarity_distributions.png")
    plot_performance_vs_similarity(combined_df, output_dir / "fig_de_matrix_performance_vs_similarity.png")
    
    # Compute regression analysis
    regression_df = compute_regression_analysis(combined_df)
    
    # Save regression results
    regression_path = output_dir / "de_matrix_regression_analysis.csv"
    regression_df.to_csv(regression_path, index=False)
    LOGGER.info(f"Saved regression analysis to {regression_path}")
    
    # Generate summary report
    generate_summary_report(
        similarity_df,
        combined_df,
        regression_df,
        output_dir / "de_matrix_similarity_report.md",
    )
    
    LOGGER.info("DE matrix similarity analysis complete")


def generate_summary_report(
    similarity_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Generate summary report with statistics.
    
    Args:
        similarity_df: DataFrame with similarity statistics
        combined_df: DataFrame with performance and similarity
        regression_df: DataFrame with regression results
        output_path: Path to save report
    """
    LOGGER.info("Generating summary report")
    
    with open(output_path, "w") as f:
        f.write("# DE Matrix Similarity Analysis (Expression Space)\n\n")
        f.write("This analysis computes cosine similarity between test and training perturbations\n")
        f.write("in the pseudobulk expression change space (Y matrix). This is the same for all\n")
        f.write("baselines since Y (expression changes) is fixed across all baselines.\n\n")
        f.write("## Summary Statistics\n\n")
        
        f.write(f"- **Number of test perturbations**: {len(similarity_df)}\n")
        f.write(f"- **Number of training perturbations**: {similarity_df['n_train_perturbations'].iloc[0]}\n")
        f.write(f"- **Number of baselines**: {len(combined_df['baseline_name'].unique())}\n\n")
        
        f.write("### Similarity Statistics\n\n")
        f.write("| Statistic | Mean | Std | Min | Max |\n")
        f.write("|-----------|-----|-----|-----|-----|\n")
        f.write(f"| Max Similarity | {similarity_df['max_similarity'].mean():.4f} | "
                f"{similarity_df['max_similarity'].std():.4f} | "
                f"{similarity_df['max_similarity'].min():.4f} | "
                f"{similarity_df['max_similarity'].max():.4f} |\n")
        f.write(f"| Mean Top-K Similarity | {similarity_df['mean_topk_similarity'].mean():.4f} | "
                f"{similarity_df['mean_topk_similarity'].std():.4f} | "
                f"{similarity_df['mean_topk_similarity'].min():.4f} | "
                f"{similarity_df['mean_topk_similarity'].max():.4f} |\n\n")
        
        f.write("## Regression Analysis\n\n")
        f.write("Performance vs Similarity (max_similarity as predictor):\n\n")
        f.write("| Baseline | N | Pearson r | p-value | Spearman ρ | p-value | R² |\n")
        f.write("|----------|---|-----------|---------|------------|---------|----|\n")
        
        for _, row in regression_df.iterrows():
            f.write(f"| {row['baseline_name']} | {row['n_observations']} | "
                    f"{row['pearson_r']:.4f} | {row['pearson_p']:.4e} | "
                    f"{row['spearman_rho']:.4f} | {row['spearman_p']:.4e} | "
                    f"{row['r_squared']:.4f} |\n")
        
        f.write("\n## Key Findings\n\n")
        
        # Identify baselines with strongest similarity dependence
        regression_df_sorted = regression_df.sort_values("pearson_r", key=abs, ascending=False)
        top_baseline = regression_df_sorted.iloc[0]
        
        f.write(f"- **Baseline with strongest similarity dependence**: {top_baseline['baseline_name']} "
                f"(Pearson r = {top_baseline['pearson_r']:.4f}, p = {top_baseline['pearson_p']:.4e})\n")
        
        # Count significant correlations
        significant = regression_df[regression_df["pearson_p"] < 0.05]
        f.write(f"- **Number of baselines with significant correlation** (p < 0.05): {len(significant)}\n")
        
        if len(significant) > 0:
            f.write("\n### Baselines with Significant Similarity Dependence\n\n")
            for _, row in significant.iterrows():
                f.write(f"- **{row['baseline_name']}**: r = {row['pearson_r']:.4f}, "
                        f"ρ = {row['spearman_rho']:.4f}, p = {row['pearson_p']:.4e}\n")
    
    LOGGER.info(f"Saved summary report to {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute cosine similarity on DE matrix (expression space)"
    )
    parser.add_argument(
        "--adata_path",
        type=Path,
        required=True,
        help="Path to perturb_processed.h5ad",
    )
    parser.add_argument(
        "--split_config",
        type=Path,
        required=True,
        help="Path to train/test/val split JSON",
    )
    parser.add_argument(
        "--baseline_results",
        type=Path,
        required=True,
        help="Path to baseline_results_reproduced.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/goal_1_similarity/de_matrix_similarity"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of top similarities to average (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: 1)",
    )
    
    args = parser.parse_args()
    
    run_similarity_analysis(
        adata_path=args.adata_path,
        split_config_path=args.split_config,
        baseline_results_path=args.baseline_results,
        output_dir=args.output_dir,
        k=args.k,
        seed=args.seed,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())


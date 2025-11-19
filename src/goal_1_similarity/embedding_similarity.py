#!/usr/bin/env python3
"""
Compute cosine similarity in baseline-specific embedding spaces.

This module computes cosine similarity between test and training perturbations
in each baseline's embedding space (B matrix). This is different for each
baseline because each baseline uses different perturbation embeddings:
- lpm_selftrained: PCA on training data
- lpm_k562PertEmb: K562 PCA embeddings
- lpm_gearsPertEmb: GEARS GO embeddings
- etc.

This produces baseline-specific "hardness profiles" — i.e., how similar each
test perturbation is to the training set as represented by each baseline's
embedding space.
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

from goal_2_baselines.baseline_runner import run_single_baseline, compute_pseudobulk_expression_changes
from goal_2_baselines.baseline_types import BaselineType, get_baseline_config
from goal_2_baselines.split_logic import load_split_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def compute_embedding_similarity_statistics(
    B_train: np.ndarray,
    B_test: np.ndarray,
    train_pert_names: List[str],
    test_pert_names: List[str],
    k: int = 5,
) -> pd.DataFrame:
    """
    Compute cosine similarity between test and training perturbations in embedding space.
    
    For each test perturbation:
    - Compute cosine similarity to every training perturbation in embedding space
    - Extract: max similarity, mean top-k similarity, distribution stats
    
    Args:
        B_train: Training perturbation embeddings (d × train_perturbations)
        B_test: Test perturbation embeddings (d × test_perturbations)
        train_pert_names: List of training perturbation names
        test_pert_names: List of test perturbation names
        k: Number of top similarities to average (default: 5)
    
    Returns:
        DataFrame with similarity statistics per test perturbation
    """
    LOGGER.info(f"Computing embedding similarity statistics for {len(test_pert_names)} test perturbations")
    
    # Transpose to get perturbations × dimensions for cosine_similarity
    # B_train is (d × train_perts), B_test is (d × test_perts)
    # cosine_similarity expects (n_samples, n_features), so we transpose
    B_train_T = B_train.T  # train_perts × d
    B_test_T = B_test.T    # test_perts × d
    
    # Compute cosine similarity matrix (test × train)
    similarity_matrix = cosine_similarity(B_test_T, B_train_T)
    
    # Create DataFrame for easier indexing
    sim_df = pd.DataFrame(
        similarity_matrix,
        index=test_pert_names,
        columns=train_pert_names,
    )
    
    # Compute statistics for each test perturbation
    results = []
    
    for test_pert in test_pert_names:
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
        
        results.append({
            "perturbation": test_pert,
            "max_similarity": max_sim,
            "mean_topk_similarity": mean_topk_sim,
            "std_similarity": std_sim,
            "median_similarity": median_sim,
            "min_similarity": min_sim,
            "n_train_perturbations": len(train_pert_names),
        })
    
    results_df = pd.DataFrame(results)
    
    LOGGER.info(f"Computed embedding similarity statistics for {len(results_df)} test perturbations")
    
    return results_df


def extract_perturbation_embeddings(
    adata_path: Path,
    split_config: Dict[str, List[str]],
    baseline_type: BaselineType,
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
    gene_name_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], Dict]:
    """
    Extract perturbation embeddings (B matrices) for a baseline.
    
    Also runs the baseline to get performance metrics for correlation analysis.
    
    Args:
        adata_path: Path to perturb_processed.h5ad
        split_config: Dictionary with 'train', 'test', 'val' keys
        baseline_type: Baseline type to analyze
        pca_dim: PCA dimension
        ridge_penalty: Ridge penalty
        seed: Random seed
        gene_name_mapping: Optional gene name mapping for embeddings
    
    Returns:
        Tuple of (B_train, B_test, train_pert_names, test_pert_names, baseline_result)
    """
    LOGGER.info(f"Extracting embeddings for baseline {baseline_type.value}")
    
    # Load data
    adata = ad.read_h5ad(adata_path)
    
    # Compute Y matrix
    Y_df, split_labels = compute_pseudobulk_expression_changes(adata, split_config, seed)
    
    # Split Y into train/test
    train_perts = split_labels.get("train", [])
    test_perts = split_labels.get("test", [])
    
    Y_train = Y_df[train_perts]
    Y_test = Y_df[test_perts] if test_perts else pd.DataFrame()
    
    if Y_test.empty:
        raise ValueError(f"No test perturbations available for {baseline_type.value}")
    
    # Get gene names
    gene_names = Y_df.index.tolist()
    
    # Get gene name mapping if available
    if gene_name_mapping is None and "gene_name" in adata.var.columns:
        gene_name_mapping = dict(zip(adata.var_names, adata.var["gene_name"]))
    
    # Get baseline config
    config = get_baseline_config(
        baseline_type,
        pca_dim=pca_dim,
        ridge_penalty=ridge_penalty,
        seed=seed,
    )
    
    # Extract B matrices by calling construct_pert_embeddings directly
    from goal_2_baselines.baseline_runner import construct_pert_embeddings
    
    Y_train_np = Y_train.values
    Y_test_np = Y_test.values
    train_pert_names = Y_train.columns.tolist()
    test_pert_names = Y_test.columns.tolist()
    
    # Get perturbation embedding args
    pert_embedding_args = config.pert_embedding_args.copy() if config.pert_embedding_args else {}
    if config.pert_embedding_source in ["k562_pca", "rpe1_pca"]:
        pert_embedding_args["target_gene_names"] = gene_names
    
    B_train, _, _, B_test, _ = construct_pert_embeddings(
        source=config.pert_embedding_source,
        train_data=Y_train_np,
        pert_names=train_pert_names,
        pca_dim=pca_dim,
        seed=seed,
        embedding_args=pert_embedding_args,
        test_data=Y_test_np,
        test_pert_names=test_pert_names,
    )
    
    if B_test is None:
        raise ValueError(f"Baseline {baseline_type.value} did not produce test embeddings")
    
    # Also run baseline to get performance metrics
    LOGGER.info(f"Running baseline {baseline_type.value} to get performance metrics")
    result = run_single_baseline(
        Y_train=Y_train,
        Y_test=Y_test,
        config=config,
        gene_names=gene_names,
        gene_name_mapping=gene_name_mapping,
        adata_path=adata_path,
        split_config=split_config,
    )
    
    return B_train, B_test, train_pert_names, test_pert_names, result


def attach_performance_metrics(
    similarity_df: pd.DataFrame,
    baseline_result: Dict,
    baseline_name: str,
) -> pd.DataFrame:
    """
    Attach baseline performance metrics to similarity results.
    
    Args:
        similarity_df: DataFrame with similarity statistics
        baseline_result: Dictionary with baseline results (contains metrics)
        baseline_name: Baseline name
    
    Returns:
        Combined DataFrame with performance and similarity
    """
    LOGGER.info(f"Attaching performance metrics for {baseline_name}")
    
    # Extract per-perturbation performance from baseline_result
    metrics = baseline_result.get("metrics", {})
    
    results = []
    
    for _, sim_row in similarity_df.iterrows():
        perturbation = sim_row["perturbation"]
        
        # Get performance for this perturbation
        if perturbation in metrics:
            pert_metrics = metrics[perturbation]
            performance_r = pert_metrics.get("pearson_r", np.nan)
        else:
            # Try with +ctrl suffix
            pert_with_ctrl = f"{perturbation}+ctrl"
            if pert_with_ctrl in metrics:
                pert_metrics = metrics[pert_with_ctrl]
                performance_r = pert_metrics.get("pearson_r", np.nan)
            else:
                performance_r = np.nan
        
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


def plot_embedding_similarity_distributions(
    similarity_df: pd.DataFrame,
    baseline_name: str,
    output_path: Path,
) -> None:
    """
    Plot similarity distributions for test vs train in embedding space.
    
    Args:
        similarity_df: DataFrame with similarity statistics
        baseline_name: Baseline name for title
        output_path: Path to save plot
    """
    LOGGER.info(f"Creating embedding similarity distribution plots for {baseline_name}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Max similarity distribution
    axes[0, 0].hist(similarity_df["max_similarity"], bins=30, edgecolor="black")
    axes[0, 0].set_xlabel("Max Similarity (Embedding Space)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title(f"Max Similarity Distribution\n{baseline_name}")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mean top-k similarity distribution
    axes[0, 1].hist(similarity_df["mean_topk_similarity"], bins=30, edgecolor="black")
    axes[0, 1].set_xlabel("Mean Top-K Similarity (Embedding Space)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title(f"Mean Top-K Similarity Distribution\n{baseline_name}")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Scatter: max vs mean top-k
    axes[1, 0].scatter(
        similarity_df["max_similarity"],
        similarity_df["mean_topk_similarity"],
        alpha=0.6,
    )
    axes[1, 0].set_xlabel("Max Similarity")
    axes[1, 0].set_ylabel("Mean Top-K Similarity")
    axes[1, 0].set_title(f"Max vs Mean Top-K Similarity\n{baseline_name}")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plot: similarity statistics
    similarity_stats = similarity_df[["max_similarity", "mean_topk_similarity", "median_similarity"]]
    bp = axes[1, 1].boxplot(similarity_stats.values, labels=similarity_stats.columns)
    axes[1, 1].set_xticklabels(similarity_stats.columns)
    axes[1, 1].set_ylabel("Similarity (Embedding Space)")
    axes[1, 1].set_title(f"Similarity Statistics Distribution\n{baseline_name}")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    LOGGER.info(f"Saved embedding similarity distribution plots to {output_path}")
    plt.close()


def plot_embedding_performance_vs_similarity(
    combined_df: pd.DataFrame,
    baseline_name: str,
    output_path: Path,
) -> None:
    """
    Create scatter plot: performance vs similarity in embedding space.
    
    Args:
        combined_df: DataFrame with performance and similarity metrics
        baseline_name: Baseline name for title
        output_path: Path to save plot
    """
    LOGGER.info(f"Creating embedding performance vs similarity plot for {baseline_name}")
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Scatter plot: performance vs max similarity
    ax.scatter(
        combined_df["max_similarity"],
        combined_df["performance_r"],
        alpha=0.6,
        s=100,
    )
    
    ax.set_xlabel("Max Similarity (Embedding Space)")
    ax.set_ylabel("Performance (Pearson r)")
    ax.set_title(f"Performance vs Embedding Similarity\n{baseline_name}")
    ax.grid(True, alpha=0.3)
    
    # Add regression line
    if len(combined_df) > 1:
        x = combined_df["max_similarity"].values
        y = combined_df["performance_r"].values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        if valid_mask.sum() > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x[valid_mask], y[valid_mask]
            )
            x_line = np.linspace(x[valid_mask].min(), x[valid_mask].max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, "r--", alpha=0.8, label=f"r={r_value:.3f}, p={p_value:.3e}")
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    LOGGER.info(f"Saved embedding performance vs similarity plot to {output_path}")
    plt.close()


def compute_embedding_regression_analysis(
    combined_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fit simple regression: performance_r ~ similarity for embedding space.
    
    Args:
        combined_df: DataFrame with performance and similarity metrics
    
    Returns:
        DataFrame with regression results
    """
    LOGGER.info("Computing embedding regression analysis")
    
    # Use max_similarity as predictor
    x = combined_df["max_similarity"].values
    y = combined_df["performance_r"].values
    
    # Filter out NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    
    if valid_mask.sum() < 2:
        LOGGER.warning("Insufficient data for regression")
        return pd.DataFrame()
    
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    
    # Linear regression
    slope, intercept, pearson_r, p_value, std_err = stats.linregress(x_clean, y_clean)
    
    # Spearman correlation
    try:
        spearman_rho, spearman_p = stats.spearmanr(x_clean, y_clean)
    except (ValueError, stats.ConstantInputWarning):
        spearman_rho = np.nan
        spearman_p = np.nan
    
    results = {
        "baseline_name": combined_df["baseline_name"].iloc[0],
        "n_observations": int(valid_mask.sum()),
        "pearson_r": float(pearson_r),
        "pearson_p": float(p_value),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "slope": float(slope),
        "intercept": float(intercept),
        "std_err": float(std_err),
        "r_squared": float(pearson_r ** 2),
    }
    
    regression_df = pd.DataFrame([results])
    
    LOGGER.info(f"Computed embedding regression analysis")
    
    return regression_df


def run_embedding_similarity_analysis(
    adata_path: Path,
    split_config_path: Path,
    baseline_types: List[BaselineType],
    output_dir: Path,
    k: int = 5,
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
) -> None:
    """
    Run complete embedding similarity analysis for multiple baselines.
    
    Args:
        adata_path: Path to perturb_processed.h5ad
        split_config_path: Path to train/test/val split JSON
        baseline_types: List of baseline types to analyze
        output_dir: Directory to save results
        k: Number of top similarities to average (default: 5)
        pca_dim: PCA dimension
        ridge_penalty: Ridge penalty
        seed: Random seed
    """
    LOGGER.info("Starting embedding similarity analysis")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load splits
    split_config = load_split_config(split_config_path)
    
    # Load data for gene name mapping
    adata = ad.read_h5ad(adata_path)
    gene_name_mapping = None
    if "gene_name" in adata.var.columns:
        gene_name_mapping = dict(zip(adata.var_names, adata.var["gene_name"]))
    
    all_combined_results = []
    all_regression_results = []
    
    for baseline_type in baseline_types:
        try:
            LOGGER.info(f"Processing baseline: {baseline_type.value}")
            
            # Extract embeddings and run baseline for performance metrics
            B_train, B_test, train_pert_names, test_pert_names, baseline_result = (
                extract_perturbation_embeddings(
                    adata_path=adata_path,
                    split_config=split_config,
                    baseline_type=baseline_type,
                    pca_dim=pca_dim,
                    ridge_penalty=ridge_penalty,
                    seed=seed,
                    gene_name_mapping=gene_name_mapping,
                )
            )
            
            # Compute similarity statistics in embedding space
            similarity_df = compute_embedding_similarity_statistics(
                B_train=B_train,
                B_test=B_test,
                train_pert_names=train_pert_names,
                test_pert_names=test_pert_names,
                k=k,
            )
            
            # Attach performance metrics
            combined_df = attach_performance_metrics(
                similarity_df=similarity_df,
                baseline_result=baseline_result,
                baseline_name=baseline_type.value,
            )
            
            all_combined_results.append(combined_df)
            
            # Create visualizations
            baseline_output_dir = output_dir / baseline_type.value
            baseline_output_dir.mkdir(parents=True, exist_ok=True)
            
            plot_embedding_similarity_distributions(
                similarity_df,
                baseline_type.value,
                baseline_output_dir / "fig_embedding_similarity_distributions.png",
            )
            
            plot_embedding_performance_vs_similarity(
                combined_df,
                baseline_type.value,
                baseline_output_dir / "fig_embedding_performance_vs_similarity.png",
            )
            
            # Compute regression analysis
            regression_df = compute_embedding_regression_analysis(combined_df)
            if not regression_df.empty:
                all_regression_results.append(regression_df)
            
            # Save per-baseline results
            combined_df.to_csv(
                baseline_output_dir / "embedding_similarity_results.csv",
                index=False,
            )
            regression_df.to_csv(
                baseline_output_dir / "embedding_regression_analysis.csv",
                index=False,
            )
            
            LOGGER.info(f"Completed analysis for {baseline_type.value}")
            
        except Exception as e:
            LOGGER.error(f"Failed to process {baseline_type.value}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all results
    if all_combined_results:
        combined_all = pd.concat(all_combined_results, ignore_index=True)
        combined_all.to_csv(output_dir / "embedding_similarity_all_baselines.csv", index=False)
        LOGGER.info(f"Saved combined results to {output_dir / 'embedding_similarity_all_baselines.csv'}")
    
    if all_regression_results:
        regression_all = pd.concat(all_regression_results, ignore_index=True)
        regression_all.to_csv(output_dir / "embedding_regression_analysis_all_baselines.csv", index=False)
        LOGGER.info(f"Saved combined regression analysis to {output_dir / 'embedding_regression_analysis_all_baselines.csv'}")
    
    # Generate summary report
    if all_combined_results and all_regression_results:
        generate_embedding_summary_report(
            all_combined_results,
            all_regression_results,
            output_dir / "embedding_similarity_report.md",
        )
    
    LOGGER.info("Embedding similarity analysis complete")


def generate_embedding_summary_report(
    all_combined_results: List[pd.DataFrame],
    all_regression_results: List[pd.DataFrame],
    output_path: Path,
) -> None:
    """
    Generate summary report for embedding similarity analysis.
    
    Args:
        all_combined_results: List of DataFrames with combined results per baseline
        all_regression_results: List of DataFrames with regression results per baseline
        output_path: Path to save report
    """
    LOGGER.info("Generating embedding similarity summary report")
    
    combined_all = pd.concat(all_combined_results, ignore_index=True)
    regression_all = pd.concat(all_regression_results, ignore_index=True)
    
    with open(output_path, "w") as f:
        f.write("# Embedding Similarity Analysis (Baseline-Specific Embedding Spaces)\n\n")
        f.write("This analysis computes cosine similarity between test and training perturbations\n")
        f.write("in each baseline's embedding space (B matrix). Each baseline uses different\n")
        f.write("perturbation embeddings, so similarity is baseline-specific.\n\n")
        
        f.write("## Summary Statistics\n\n")
        f.write(f"- **Number of baselines analyzed**: {len(all_combined_results)}\n")
        f.write(f"- **Total test perturbations**: {combined_all['perturbation'].nunique()}\n\n")
        
        f.write("### Similarity Statistics by Baseline\n\n")
        for baseline_name in combined_all["baseline_name"].unique():
            baseline_data = combined_all[combined_all["baseline_name"] == baseline_name]
            f.write(f"#### {baseline_name}\n\n")
            f.write("| Statistic | Mean | Std | Min | Max |\n")
            f.write("|-----------|-----|-----|-----|-----|\n")
            f.write(f"| Max Similarity | {baseline_data['max_similarity'].mean():.4f} | "
                    f"{baseline_data['max_similarity'].std():.4f} | "
                    f"{baseline_data['max_similarity'].min():.4f} | "
                    f"{baseline_data['max_similarity'].max():.4f} |\n")
            f.write(f"| Mean Top-K Similarity | {baseline_data['mean_topk_similarity'].mean():.4f} | "
                    f"{baseline_data['mean_topk_similarity'].std():.4f} | "
                    f"{baseline_data['mean_topk_similarity'].min():.4f} | "
                    f"{baseline_data['mean_topk_similarity'].max():.4f} |\n\n")
        
        f.write("## Regression Analysis\n\n")
        f.write("Performance vs Embedding Similarity (max_similarity as predictor):\n\n")
        f.write("| Baseline | N | Pearson r | p-value | Spearman ρ | p-value | R² |\n")
        f.write("|----------|---|-----------|---------|------------|---------|----|\n")
        
        for _, row in regression_all.iterrows():
            f.write(f"| {row['baseline_name']} | {row['n_observations']} | "
                    f"{row['pearson_r']:.4f} | {row['pearson_p']:.4e} | "
                    f"{row['spearman_rho']:.4f} | {row['spearman_p']:.4e} | "
                    f"{row['r_squared']:.4f} |\n")
        
        f.write("\n## Key Findings\n\n")
        
        # Identify baselines with strongest similarity dependence
        regression_sorted = regression_all.sort_values("pearson_r", key=abs, ascending=False)
        if len(regression_sorted) > 0:
            top_baseline = regression_sorted.iloc[0]
            f.write(f"- **Baseline with strongest similarity dependence**: {top_baseline['baseline_name']} "
                    f"(Pearson r = {top_baseline['pearson_r']:.4f}, p = {top_baseline['pearson_p']:.4e})\n")
        
        # Count significant correlations
        significant = regression_all[regression_all["pearson_p"] < 0.05]
        f.write(f"- **Number of baselines with significant correlation** (p < 0.05): {len(significant)}\n")
        
        if len(significant) > 0:
            f.write("\n### Baselines with Significant Embedding Similarity Dependence\n\n")
            for _, row in significant.iterrows():
                f.write(f"- **{row['baseline_name']}**: r = {row['pearson_r']:.4f}, "
                        f"ρ = {row['spearman_rho']:.4f}, p = {row['pearson_p']:.4e}\n")
    
    LOGGER.info(f"Saved embedding similarity summary report to {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute cosine similarity in baseline-specific embedding spaces"
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
        "--baselines",
        nargs="+",
        choices=[bt.value for bt in BaselineType],
        default=[bt.value for bt in BaselineType if bt != BaselineType.MEAN_RESPONSE],
        help="Baselines to analyze (default: all except mean_response)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/goal_1_similarity/embedding_similarity"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of top similarities to average (default: 5)",
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=10,
        help="PCA dimension (default: 10)",
    )
    parser.add_argument(
        "--ridge_penalty",
        type=float,
        default=0.1,
        help="Ridge penalty (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: 1)",
    )
    
    args = parser.parse_args()
    
    # Convert baseline names to BaselineType enums
    baseline_types = [BaselineType(b) for b in args.baselines]
    
    run_embedding_similarity_analysis(
        adata_path=args.adata_path,
        split_config_path=args.split_config,
        baseline_types=baseline_types,
        output_dir=args.output_dir,
        k=args.k,
        pca_dim=args.pca_dim,
        ridge_penalty=args.ridge_penalty,
        seed=args.seed,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())


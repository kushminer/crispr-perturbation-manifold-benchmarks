#!/usr/bin/env python3
"""
Similarity-Based Evaluation: Continuous Hardness vs Performance Regression.

This module computes hardness metrics (using the hardness API) and fits regression
models to quantify how well hardness predicts performance across datasets and baselines.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from goal_2_baselines.baseline_runner import compute_pseudobulk_expression_changes
from goal_2_baselines.baseline_types import BaselineType, get_baseline_config
from goal_2_baselines.split_logic import load_split_config
from goal_1_similarity.embedding_similarity import extract_perturbation_embeddings
from goal_1_similarity.hardness_api import compute_multiple_targets_similarity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def compute_hardness_regression(
    adata_path: Path,
    split_config_path: Path,
    baseline_types: List[BaselineType],
    dataset_name: str,
    output_dir: Path,
    k_values: List[int] = [1, 5, 10, 20],
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
) -> pd.DataFrame:
    """
    Compute hardness vs performance regression for each baseline and k value.
    
    Args:
        adata_path: Path to perturb_processed.h5ad
        split_config_path: Path to train/test/val split JSON
        baseline_types: List of baseline types to analyze
        dataset_name: Name of dataset (e.g., "adamson", "replogle_k562_essential")
        output_dir: Directory to save results
        k_values: List of k values for computing mean_topk similarity
        pca_dim: PCA dimension
        ridge_penalty: Ridge penalty
        seed: Random seed
        
    Returns:
        DataFrame with regression metrics per baseline and k
    """
    LOGGER.info(f"Computing hardness regression for dataset: {dataset_name}")
    
    # Load splits
    split_config = load_split_config(split_config_path)
    
    # Load data for gene name mapping
    adata = ad.read_h5ad(adata_path)
    gene_name_mapping = None
    if "gene_name" in adata.var.columns:
        gene_name_mapping = dict(zip(adata.var_names, adata.var["gene_name"]))
    
    all_results = []
    
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
            
            # Get performance metrics per perturbation
            performance_dict = baseline_result.get("metrics", {})
            if not performance_dict:
                LOGGER.warning(f"No performance metrics for {baseline_type.value}")
                continue
            
            # Compute hardness metrics for all k values
            # B_train is (d, n_train), B_test is (d, n_test)
            # We need to transpose to (n_train, d) and (n_test, d) for the API
            B_train_T = B_train.T  # (n_train, d)
            B_test_T = B_test.T    # (n_test, d)
            
            hardness_results = compute_multiple_targets_similarity(
                target_embeddings=B_test_T,
                train_embeddings=B_train_T,
                target_names=test_pert_names,
                k_values=k_values,
            )
            
            # Combine hardness and performance
            combined_data = []
            for pert_name in test_pert_names:
                # Clean perturbation name (remove +ctrl if present)
                clean_pert_name = pert_name.replace("+ctrl", "")
                
                # Get performance metrics
                if pert_name in performance_dict:
                    perf_metrics = performance_dict[pert_name]
                    performance_r = perf_metrics.get("pearson_r", np.nan)
                else:
                    performance_r = np.nan
                
                # Get hardness metrics
                if clean_pert_name in hardness_results:
                    hardness_data = hardness_results[clean_pert_name]
                    hardness_max = hardness_data["hardness_max"]
                    
                    # Compute regression for each k
                    for k in k_values:
                        hardness_k = hardness_data["hardness_k"].get(k, np.nan)
                        mean_topk = hardness_data["mean_topk"].get(k, np.nan)
                        
                        combined_data.append({
                            "perturbation": clean_pert_name,
                            "baseline": baseline_type.value,
                            "k": k,
                            "performance_r": performance_r,
                            "hardness_k": hardness_k,
                            "hardness_max": hardness_max,
                            "mean_topk": mean_topk,
                            "max_sim": hardness_data["max_sim"],
                        })
            
            # Fit regression: performance_r ~ hardness_k for each k
            combined_df = pd.DataFrame(combined_data)
            
            for k in k_values:
                subset = combined_df[combined_df["k"] == k].dropna(subset=["hardness_k", "performance_r"])
                
                if len(subset) < 2:
                    LOGGER.warning(f"Insufficient data for regression: baseline={baseline_type.value}, k={k}")
                    continue
                
                x = subset["hardness_k"].values
                y = subset["performance_r"].values
                
                # Linear regression
                slope, intercept, pearson_r, p_value, std_err = stats.linregress(x, y)
                
                # Spearman correlation
                try:
                    spearman_rho, spearman_p = stats.spearmanr(x, y)
                except (ValueError, stats.ConstantInputWarning):
                    spearman_rho = np.nan
                    spearman_p = np.nan
                
                # R²
                r_squared = pearson_r ** 2
                
                all_results.append({
                    "dataset": dataset_name,
                    "baseline": baseline_type.value,
                    "k": k,
                    "n_perturbations": len(subset),
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(p_value),
                    "spearman_rho": float(spearman_rho),
                    "spearman_p": float(spearman_p),
                    "r_squared": float(r_squared),
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "std_err": float(std_err),
                })
            
            LOGGER.info(f"Completed analysis for {baseline_type.value}")
            
        except Exception as e:
            LOGGER.error(f"Failed to process {baseline_type.value}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    results_df = pd.DataFrame(all_results)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "hardness_regression_summary.csv"
    results_df.to_csv(results_path, index=False)
    LOGGER.info(f"Saved hardness regression summary to {results_path}")
    
    return results_df


def plot_hardness_vs_performance(
    results_df: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
    k_values: List[int] = [1, 5, 10, 20],
) -> None:
    """
    Create scatter plots: hardness vs performance for each k value.
    
    Args:
        results_df: DataFrame with hardness and performance data
        dataset_name: Name of dataset
        output_dir: Directory to save plots
        k_values: List of k values to plot
    """
    LOGGER.info(f"Creating hardness vs performance plots for {dataset_name}")
    
    # Load combined data (need to reconstruct from results)
    # For now, we'll create plots from regression summary
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots for each k
    n_k = len(k_values)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        
        # Filter to this k
        k_data = results_df[results_df["k"] == k]
        
        if len(k_data) == 0:
            continue
        
        # Plot R² vs baseline
        baselines = k_data["baseline"].unique()
        r2_values = [k_data[k_data["baseline"] == bl]["r_squared"].iloc[0] for bl in baselines]
        
        ax.bar(range(len(baselines)), r2_values, alpha=0.7)
        ax.set_xticks(range(len(baselines)))
        ax.set_xticklabels(baselines, rotation=45, ha="right")
        ax.set_ylabel("R²")
        ax.set_title(f"Hardness vs Performance R² (k={k})")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, max(r2_values) * 1.1 if max(r2_values) > 0 else 1.0])
    
    plt.tight_layout()
    plt.savefig(output_dir / f"fig_hardness_regression_r2_{dataset_name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    LOGGER.info(f"Saved hardness vs performance plots to {output_dir}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute continuous hardness vs performance regression"
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
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name (e.g., 'adamson', 'replogle_k562_essential')",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save results",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=[bt.value for bt in BaselineType],
        default=[
            BaselineType.SELFTRAINED.value,
            BaselineType.K562_PERT_EMB.value,
            BaselineType.GEARS_PERT_EMB.value,
            BaselineType.RPE1_PERT_EMB.value,
        ],
        help="Baselines to analyze",
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="K values for computing mean_topk similarity",
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=10,
        help="PCA dimension",
    )
    parser.add_argument(
        "--ridge_penalty",
        type=float,
        default=0.1,
        help="Ridge penalty",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    # Convert baseline names to BaselineType enums
    baseline_types = [BaselineType(bt) for bt in args.baselines]
    
    # Compute hardness regression
    results_df = compute_hardness_regression(
        adata_path=args.adata_path,
        split_config_path=args.split_config,
        baseline_types=baseline_types,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        k_values=args.k_values,
        pca_dim=args.pca_dim,
        ridge_penalty=args.ridge_penalty,
        seed=args.seed,
    )
    
    # Create plots
    if not results_df.empty:
        plot_hardness_vs_performance(
            results_df,
            args.dataset_name,
            args.output_dir,
            args.k_values,
        )
    
    LOGGER.info("Hardness regression analysis complete")


if __name__ == "__main__":
    main()


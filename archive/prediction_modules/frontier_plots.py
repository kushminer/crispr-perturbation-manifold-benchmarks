#!/usr/bin/env python3
"""
Hardness-Calibrated Performance Curves: Frontier plots showing performance vs hardness.

Generates plots showing R² vs hardness for different strategies:
- Original baselines
- Similarity-based model selection
- Similarity-weighted ensemble
- Local regression
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from baselines.baseline_runner import compute_pseudobulk_expression_changes
from baselines.baseline_types import BaselineType
from baselines.split_logic import load_split_config
from similarity.embedding_similarity import extract_perturbation_embeddings
from similarity.hardness_api import compute_multiple_targets_similarity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def compute_frontier_data(
    adata_path: Path,
    split_config_path: Path,
    baseline_types: List[BaselineType],
    predictions_base_dir: Path,
    model_selection_results_dir: Optional[Path],
    ensemble_results_dir: Optional[Path],
    local_regression_results_dir: Optional[Path],
    dataset_name: str,
    output_dir: Path,
    k: int = 10,
    n_bins: int = 10,
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
) -> pd.DataFrame:
    """
    Compute per-target hardness and performance for all strategies.
    
    Args:
        adata_path: Path to perturb_processed.h5ad
        split_config_path: Path to train/test/val split JSON
        baseline_types: List of baseline types
        predictions_base_dir: Base directory containing baseline predictions
        model_selection_results_dir: Directory with model selection results
        ensemble_results_dir: Directory with ensemble results
        local_regression_results_dir: Directory with local regression results
        dataset_name: Dataset name
        output_dir: Directory to save results
        k: K value for hardness computation (using mean_topk)
        n_bins: Number of bins for quantile binning (only for plotting)
        pca_dim: PCA dimension
        ridge_penalty: Ridge penalty
        seed: Random seed
        
    Returns:
        DataFrame with per-target hardness and performance for each strategy
    """
    LOGGER.info(f"Computing frontier data for dataset: {dataset_name}")
    
    # Load data
    adata = ad.read_h5ad(adata_path)
    split_config = load_split_config(split_config_path)
    
    # Compute Y matrix
    Y_df, split_labels = compute_pseudobulk_expression_changes(adata, split_config, seed)
    test_perts = split_labels.get("test", [])
    Y_test = Y_df[test_perts]
    
    # Get gene name mapping
    gene_name_mapping = None
    if "gene_name" in adata.var.columns:
        gene_name_mapping = dict(zip(adata.var_names, adata.var["gene_name"]))
    
    # Compute hardness for all baselines
    baseline_hardness = {}
    
    for baseline_type in baseline_types:
        baseline_name = baseline_type.value
        try:
            B_train, B_test, train_pert_names, test_pert_names, _ = (
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
            
            B_train_T = B_train.T
            B_test_T = B_test.T
            
            hardness_results = compute_multiple_targets_similarity(
                target_embeddings=B_test_T,
                train_embeddings=B_train_T,
                target_names=[p.replace("+ctrl", "") for p in test_pert_names],
                k_values=[k],
            )
            
            baseline_hardness[baseline_name] = {
                pert: metrics.get("hardness_k", {}).get(k, np.nan)
                for pert, metrics in hardness_results.items()
            }
            
        except Exception as e:
            LOGGER.warning(f"Failed to compute hardness for {baseline_name}: {e}")
            continue
    
    # Get baseline performance per target
    baseline_performance = {}
    test_pert_map = {pert.replace("+ctrl", ""): pert for pert in test_perts}
    
    for baseline_type in baseline_types:
        baseline_name = baseline_type.value
        baseline_dir = predictions_base_dir / baseline_name
        
        predictions_path = baseline_dir / "predictions.json"
        if not predictions_path.exists():
            continue
        
        with open(predictions_path, "r") as f:
            predictions_dict = json.load(f)
        
        gene_names_path = baseline_dir / "gene_names.json"
        with open(gene_names_path, "r") as f:
            pred_gene_names = json.load(f)
        
        common_genes = [g for g in Y_test.index if g in pred_gene_names]
        baseline_performance[baseline_name] = {}
        
        from eval_framework.metrics import compute_metrics
        
        for clean_pert_name, pred_values in predictions_dict.items():
            if clean_pert_name not in test_pert_map:
                continue
            
            pert_name = test_pert_map[clean_pert_name]
            y_true = Y_test.loc[common_genes, pert_name].values
            
            pred_array = np.array(pred_values)
            if len(pred_array) != len(pred_gene_names):
                continue
            
            pred_idx_map = {g: i for i, g in enumerate(pred_gene_names)}
            common_genes_aligned = [g for g in common_genes if g in pred_idx_map]
            pred_aligned = np.array([pred_array[pred_idx_map[g]] for g in common_genes_aligned])
            
            # Ensure y_true matches
            y_true_aligned = Y_test.loc[common_genes_aligned, pert_name].values
            
            metrics = compute_metrics(y_true_aligned, pred_aligned)
            baseline_performance[baseline_name][clean_pert_name] = metrics["pearson_r"]
    
    # Collect frontier data
    frontier_data = []
    
    # Baseline strategies
    for baseline_name in baseline_types:
        baseline_name = baseline_name.value
        if baseline_name not in baseline_hardness or baseline_name not in baseline_performance:
            continue
        
        for clean_pert_name in test_pert_map.keys():
            if clean_pert_name in baseline_hardness[baseline_name] and clean_pert_name in baseline_performance[baseline_name]:
                frontier_data.append({
                    "perturbation": clean_pert_name,
                    "strategy": f"baseline_{baseline_name}",
                    "hardness": baseline_hardness[baseline_name][clean_pert_name],
                    "performance_r": baseline_performance[baseline_name][clean_pert_name],
                })
    
    # Model selection strategies (if available)
    if model_selection_results_dir and model_selection_results_dir.exists():
        # Would need to load per-target selections
        # For now, skip - can be added later
        pass
    
    # Ensemble strategies (if available)
    if ensemble_results_dir and ensemble_results_dir.exists():
        # Would need per-target ensemble performance
        # For now, skip - can be added later
        pass
    
    frontier_df = pd.DataFrame(frontier_data)
    
    # Save continuous data
    output_dir.mkdir(parents=True, exist_ok=True)
    frontier_df.to_csv(output_dir / "frontier_data_continuous.csv", index=False)
    
    # Compute binned data for plotting
    if not frontier_df.empty and "hardness" in frontier_df.columns:
        frontier_df_binned = frontier_df.copy()
        frontier_df_binned["hardness_bin"] = pd.qcut(
            frontier_df_binned["hardness"],
            q=n_bins,
            duplicates="drop",
        )
        
        # Compute mean per bin per strategy
        binned_summary = frontier_df_binned.groupby(["strategy", "hardness_bin"]).agg({
            "performance_r": ["mean", "std", "count"],
        }).reset_index()
        binned_summary.columns = ["strategy", "hardness_bin", "mean_r", "std_r", "n"]
        
        binned_summary["hardness_bin_center"] = binned_summary["hardness_bin"].apply(
            lambda x: x.mid if hasattr(x, "mid") else np.nan
        )
        
        binned_summary.to_csv(output_dir / "frontier_stats.csv", index=False)
    
    return frontier_df


def plot_frontier_curves(
    frontier_df: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    n_bins: int = 10,
) -> None:
    """
    Generate frontier plots: performance vs hardness for each strategy.
    
    Args:
        frontier_df: DataFrame with frontier data
        output_dir: Directory to save plots
        dataset_name: Dataset name
        n_bins: Number of quantile bins for binning
    """
    LOGGER.info(f"Generating frontier plots for {dataset_name}")
    
    if frontier_df.empty:
        LOGGER.warning("No frontier data available")
        return
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    
    # Bin data for plotting
    frontier_df_clean = frontier_df.dropna(subset=["hardness", "performance_r"])
    
    if frontier_df_clean.empty:
        LOGGER.warning("No valid frontier data after filtering")
        return
    
    # Create bins
    frontier_df_clean["hardness_bin"] = pd.qcut(
        frontier_df_clean["hardness"],
        q=n_bins,
        duplicates="drop",
    )
    
    # Compute mean per bin per strategy
    binned = frontier_df_clean.groupby(["strategy", "hardness_bin"]).agg({
        "performance_r": ["mean", "std"],
    }).reset_index()
    binned.columns = ["strategy", "hardness_bin", "mean_r", "std_r"]
    
    binned["hardness_bin_center"] = binned["hardness_bin"].apply(
        lambda x: x.mid if hasattr(x, "mid") else np.nan
    )
    
    # Separate baselines from other strategies
    baseline_strategies = [s for s in binned["strategy"].unique() if s.startswith("baseline_")]
    other_strategies = [s for s in binned["strategy"].unique() if not s.startswith("baseline_")]
    
    # Plot 1: All strategies
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot baselines
    for strategy in baseline_strategies:
        strategy_data = binned[binned["strategy"] == strategy].sort_values("hardness_bin_center")
        baseline_name = strategy.replace("baseline_", "")
        ax.plot(
            strategy_data["hardness_bin_center"],
            strategy_data["mean_r"],
            marker="o",
            label=baseline_name,
            alpha=0.7,
            linewidth=2,
        )
    
    # Plot other strategies
    for strategy in other_strategies:
        strategy_data = binned[binned["strategy"] == strategy].sort_values("hardness_bin_center")
        ax.plot(
            strategy_data["hardness_bin_center"],
            strategy_data["mean_r"],
            marker="s",
            label=strategy,
            alpha=0.7,
            linewidth=2,
            linestyle="--",
        )
    
    ax.set_xlabel("Hardness (Quantile Bins)", fontsize=12)
    ax.set_ylabel("Mean Performance (Pearson r)", fontsize=12)
    ax.set_title(f"Performance vs Hardness Frontier\n{dataset_name}", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"fig_frontier_{dataset_name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    LOGGER.info(f"Saved frontier plot to {output_dir / f'fig_frontier_{dataset_name}.png'}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate hardness-calibrated performance curves (frontier plots)"
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
        "--predictions_base_dir",
        type=Path,
        required=True,
        help="Base directory containing baseline predictions",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name",
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
        help="Baselines to include",
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
        "--k",
        type=int,
        default=10,
        help="K value for hardness computation",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=10,
        help="Number of quantile bins for plotting",
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
    
    # Compute frontier data
    frontier_df = compute_frontier_data(
        adata_path=args.adata_path,
        split_config_path=args.split_config,
        baseline_types=baseline_types,
        predictions_base_dir=args.predictions_base_dir,
        model_selection_results_dir=args.model_selection_dir,
        ensemble_results_dir=args.ensemble_dir,
        local_regression_results_dir=args.local_regression_dir,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        k=args.k,
        n_bins=args.n_bins,
        pca_dim=args.pca_dim,
        ridge_penalty=args.ridge_penalty,
        seed=args.seed,
    )
    
    # Generate plots
    if not frontier_df.empty:
        plot_frontier_curves(
            frontier_df,
            args.output_dir,
            args.dataset_name,
            args.n_bins,
        )
    
    LOGGER.info("Frontier plot generation complete")


if __name__ == "__main__":
    main()


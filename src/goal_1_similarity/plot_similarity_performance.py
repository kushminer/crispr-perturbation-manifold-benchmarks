#!/usr/bin/env python3
"""
Create scatterplots of embedding similarity vs performance metrics.

Plots:
1. mean_topk_similarity x performance_r
2. max_similarity x performance_r
3. mean_topk_similarity x L2
4. max_similarity x L2
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from goal_2_baselines.baseline_runner import compute_pseudobulk_expression_changes
from goal_2_baselines.split_logic import load_split_config
from shared.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def compute_per_perturbation_l2(
    adata_path: Path,
    split_config_path: Path,
    predictions_base_dir: Path,
    baseline_names: list[str],
    seed: int = 1,
) -> pd.DataFrame:
    """Compute L2 per perturbation for each baseline from saved predictions."""
    LOGGER.info(f"Computing per-perturbation L2 for {len(baseline_names)} baselines")
    
    # Load data to get true values
    adata = ad.read_h5ad(adata_path)
    split_config = load_split_config(split_config_path)
    
    # Compute Y matrix (pseudobulk expression changes)
    Y_df, split_labels = compute_pseudobulk_expression_changes(adata, split_config, seed)
    
    # Get test perturbations
    test_perts = split_labels.get("test", [])
    Y_test = Y_df[test_perts]
    
    # Load gene names mapping (predictions use cleaned names without +ctrl)
    test_pert_map = {pert.replace("+ctrl", ""): pert for pert in test_perts}
    
    results = []
    
    for baseline_name in baseline_names:
        baseline_dir = predictions_base_dir / baseline_name
        
        # Load predictions
        predictions_path = baseline_dir / "predictions.json"
        if not predictions_path.exists():
            LOGGER.warning(f"Predictions not found for {baseline_name}, skipping")
            continue
        
        with open(predictions_path, "r") as f:
            predictions_dict = json.load(f)
        
        # Load gene names
        gene_names_path = baseline_dir / "gene_names.json"
        with open(gene_names_path, "r") as f:
            pred_gene_names = json.load(f)
        
        # Align genes between predictions and true values
        common_genes = [g for g in Y_test.index if g in pred_gene_names]
        if len(common_genes) == 0:
            LOGGER.warning(f"No common genes for {baseline_name}, skipping")
            continue
        
        # Compute L2 for each test perturbation
        for clean_pert_name, pred_values in predictions_dict.items():
            if clean_pert_name not in test_pert_map:
                continue
            
            pert_name = test_pert_map[clean_pert_name]
            
            # Get true values
            y_true = Y_test.loc[common_genes, pert_name].values
            
            # Get predictions (aligned to common genes)
            pred_array = np.array(pred_values)
            if len(pred_array) != len(pred_gene_names):
                LOGGER.warning(f"Length mismatch for {baseline_name}/{clean_pert_name}")
                continue
            
            # Align predictions to common genes
            pred_idx_map = {g: i for i, g in enumerate(pred_gene_names)}
            pred_aligned = np.array([pred_array[pred_idx_map[g]] for g in common_genes])
            
            # Compute L2
            metrics = compute_metrics(y_true, pred_aligned)
            
            results.append({
                "perturbation": clean_pert_name,
                "baseline_name": baseline_name,
                "l2": metrics["l2"],
                "performance_r": metrics["pearson_r"],
            })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Plot embedding similarity vs performance scatterplots"
    )
    parser.add_argument(
        "--similarity_csv",
        type=Path,
        required=True,
        help="Path to embedding_similarity_all_baselines.csv",
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
        help="Path to split config JSON",
    )
    parser.add_argument(
        "--predictions_base_dir",
        type=Path,
        required=True,
        help="Base directory containing baseline prediction subdirectories",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save plots",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    # Load similarity data
    LOGGER.info(f"Loading similarity data from {args.similarity_csv}")
    similarity_df = pd.read_csv(args.similarity_csv)
    
    # Get unique baseline names
    baseline_names = similarity_df["baseline_name"].unique().tolist()
    LOGGER.info(f"Found {len(baseline_names)} baselines: {baseline_names}")
    
    # Compute per-perturbation L2
    l2_df = compute_per_perturbation_l2(
        adata_path=args.adata_path,
        split_config_path=args.split_config,
        predictions_base_dir=args.predictions_base_dir,
        baseline_names=baseline_names,
        seed=args.seed,
    )
    
    # Merge similarity and L2 data
    # Note: similarity_df has performance_r, we'll use that (or recompute from L2_df)
    # Merge on perturbation and baseline_name
    merged_df = similarity_df.merge(
        l2_df[["perturbation", "baseline_name", "l2"]],
        on=["perturbation", "baseline_name"],
        how="left",
    )
    
    # Use performance_r from similarity_df (or from l2_df if missing in similarity)
    if "performance_r" not in merged_df.columns:
        merged_df = merged_df.merge(
            l2_df[["perturbation", "baseline_name", "performance_r"]],
            on=["perturbation", "baseline_name"],
            how="left",
        )
    
    LOGGER.info(f"Merged data: {len(merged_df)} rows")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    
    # Color palette for baselines
    n_baselines = len(baseline_names)
    colors = sns.color_palette("husl", n_baselines)
    baseline_color_map = {name: colors[i] for i, name in enumerate(sorted(baseline_names))}
    
    # Plot 1: mean_topk_similarity x performance_r
    fig, ax = plt.subplots(figsize=(10, 7))
    for baseline in baseline_names:
        subset = merged_df[merged_df["baseline_name"] == baseline]
        ax.scatter(
            subset["mean_topk_similarity"],
            subset["performance_r"],
            label=baseline,
            color=baseline_color_map[baseline],
            alpha=0.7,
            s=80,
        )
    
    # Add correlation line
    valid = merged_df.dropna(subset=["mean_topk_similarity", "performance_r"])
    if len(valid) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid["mean_topk_similarity"], valid["performance_r"]
        )
        x_line = np.linspace(valid["mean_topk_similarity"].min(), valid["mean_topk_similarity"].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, "k--", alpha=0.5, linewidth=2, label=f"r={r_value:.3f}, p={p_value:.3e}")
    
    ax.set_xlabel("Mean Top-K Similarity", fontsize=12)
    ax.set_ylabel("Performance (Pearson r)", fontsize=12)
    ax.set_title("Mean Top-K Similarity vs Performance", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output_dir / "fig_mean_topk_similarity_vs_performance_r.png")
    plt.close()
    LOGGER.info("Saved: fig_mean_topk_similarity_vs_performance_r.png")
    
    # Plot 2: max_similarity x performance_r
    fig, ax = plt.subplots(figsize=(10, 7))
    for baseline in baseline_names:
        subset = merged_df[merged_df["baseline_name"] == baseline]
        ax.scatter(
            subset["max_similarity"],
            subset["performance_r"],
            label=baseline,
            color=baseline_color_map[baseline],
            alpha=0.7,
            s=80,
        )
    
    # Add correlation line
    valid = merged_df.dropna(subset=["max_similarity", "performance_r"])
    if len(valid) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid["max_similarity"], valid["performance_r"]
        )
        x_line = np.linspace(valid["max_similarity"].min(), valid["max_similarity"].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, "k--", alpha=0.5, linewidth=2, label=f"r={r_value:.3f}, p={p_value:.3e}")
    
    ax.set_xlabel("Max Similarity", fontsize=12)
    ax.set_ylabel("Performance (Pearson r)", fontsize=12)
    ax.set_title("Max Similarity vs Performance", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output_dir / "fig_max_similarity_vs_performance_r.png")
    plt.close()
    LOGGER.info("Saved: fig_max_similarity_vs_performance_r.png")
    
    # Plot 3: mean_topk_similarity x L2
    fig, ax = plt.subplots(figsize=(10, 7))
    for baseline in baseline_names:
        subset = merged_df[merged_df["baseline_name"] == baseline].dropna(subset=["mean_topk_similarity", "l2"])
        if len(subset) > 0:
            ax.scatter(
                subset["mean_topk_similarity"],
                subset["l2"],
                label=baseline,
                color=baseline_color_map[baseline],
                alpha=0.7,
                s=80,
            )
    
    # Add correlation line
    valid = merged_df.dropna(subset=["mean_topk_similarity", "l2"])
    if len(valid) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid["mean_topk_similarity"], valid["l2"]
        )
        x_line = np.linspace(valid["mean_topk_similarity"].min(), valid["mean_topk_similarity"].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, "k--", alpha=0.5, linewidth=2, label=f"r={r_value:.3f}, p={p_value:.3e}")
    
    ax.set_xlabel("Mean Top-K Similarity", fontsize=12)
    ax.set_ylabel("L2 Distance", fontsize=12)
    ax.set_title("Mean Top-K Similarity vs L2 Distance", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output_dir / "fig_mean_topk_similarity_vs_l2.png")
    plt.close()
    LOGGER.info("Saved: fig_mean_topk_similarity_vs_l2.png")
    
    # Plot 4: max_similarity x L2
    fig, ax = plt.subplots(figsize=(10, 7))
    for baseline in baseline_names:
        subset = merged_df[merged_df["baseline_name"] == baseline].dropna(subset=["max_similarity", "l2"])
        if len(subset) > 0:
            ax.scatter(
                subset["max_similarity"],
                subset["l2"],
                label=baseline,
                color=baseline_color_map[baseline],
                alpha=0.7,
                s=80,
            )
    
    # Add correlation line
    valid = merged_df.dropna(subset=["max_similarity", "l2"])
    if len(valid) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid["max_similarity"], valid["l2"]
        )
        x_line = np.linspace(valid["max_similarity"].min(), valid["max_similarity"].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, "k--", alpha=0.5, linewidth=2, label=f"r={r_value:.3f}, p={p_value:.3e}")
    
    ax.set_xlabel("Max Similarity", fontsize=12)
    ax.set_ylabel("L2 Distance", fontsize=12)
    ax.set_title("Max Similarity vs L2 Distance", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output_dir / "fig_max_similarity_vs_l2.png")
    plt.close()
    LOGGER.info("Saved: fig_max_similarity_vs_l2.png")
    
    LOGGER.info("All plots saved to %s", args.output_dir)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Validation script to ensure different baselines produce different embeddings.

This script runs two baselines (SELFTRAINED and GEARS_PERT_EMB) and verifies
that they produce different embeddings and predictions.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from goal_2_baselines.baseline_runner_single_cell import run_single_baseline_single_cell
from goal_2_baselines.baseline_types import BaselineConfig, BaselineType, get_baseline_config
from goal_2_baselines.single_cell_loader import compute_single_cell_expression_changes
from goal_2_baselines.split_logic import load_split_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def validate_baseline_differences(
    adata_path: Path,
    split_config_path: Path,
    baseline_a: BaselineType,
    baseline_b: BaselineType,
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
    n_cells_per_pert: int = 50,
) -> bool:
    """
    Validate that two baselines produce different embeddings and predictions.
    
    Returns:
        True if baselines are different, False if they're identical
    """
    import anndata as ad
    
    LOGGER.info(f"Validating {baseline_a.value} vs {baseline_b.value}")
    
    # Load data
    adata = ad.read_h5ad(adata_path)
    split_config = load_split_config(split_config_path)
    
    # Compute single-cell Y matrix
    Y_df, split_labels, cell_to_pert = compute_single_cell_expression_changes(
        adata, split_config, n_cells_per_pert=n_cells_per_pert, seed=seed
    )
    
    train_cells = split_labels.get("train", [])
    test_cells = split_labels.get("test", [])
    
    Y_train = Y_df[train_cells] if train_cells else None
    Y_test = Y_df[test_cells] if test_cells else None
    
    if Y_train is None or Y_test is None:
        raise ValueError("Need both train and test cells for validation")
    
    cell_to_pert_train = {c: cell_to_pert[c] for c in train_cells if c in cell_to_pert}
    cell_to_pert_test = {c: cell_to_pert[c] for c in test_cells if c in cell_to_pert}
    
    gene_names = Y_df.index.tolist()
    gene_name_mapping = None
    if "gene_name" in adata.var.columns:
        gene_name_mapping = dict(zip(adata.var_names, adata.var["gene_name"]))
    
    # Run baseline A
    config_a = get_baseline_config(baseline_a, pca_dim=pca_dim, ridge_penalty=ridge_penalty, seed=seed)
    result_a = run_single_baseline_single_cell(
        Y_train=Y_train,
        Y_test=Y_test,
        config=config_a,
        gene_names=gene_names,
        cell_to_pert_train=cell_to_pert_train,
        cell_to_pert_test=cell_to_pert_test,
        gene_name_mapping=gene_name_mapping,
    )
    
    # Run baseline B
    config_b = get_baseline_config(baseline_b, pca_dim=pca_dim, ridge_penalty=ridge_penalty, seed=seed)
    result_b = run_single_baseline_single_cell(
        Y_train=Y_train,
        Y_test=Y_test,
        config=config_b,
        gene_names=gene_names,
        cell_to_pert_train=cell_to_pert_train,
        cell_to_pert_test=cell_to_pert_test,
        gene_name_mapping=gene_name_mapping,
    )
    
    # Compare embeddings
    B_train_a = result_a["B_train"]
    B_train_b = result_b["B_train"]
    
    if B_train_a.shape != B_train_b.shape:
        LOGGER.info(f"✓ Embedding shapes differ: {B_train_a.shape} vs {B_train_b.shape}")
        return True
    
    max_diff = np.abs(B_train_a - B_train_b).max()
    mean_diff = np.abs(B_train_a - B_train_b).mean()
    
    LOGGER.info(f"Embedding comparison:")
    LOGGER.info(f"  Max difference: {max_diff:.10f}")
    LOGGER.info(f"  Mean difference: {mean_diff:.10f}")
    
    if max_diff < 1e-6:
        LOGGER.error(f"✗ CRITICAL: Embeddings are IDENTICAL!")
        return False
    elif max_diff < 1e-3:
        LOGGER.warning(f"⚠ Embeddings are VERY SIMILAR (max_diff={max_diff:.10f})")
        return False
    else:
        LOGGER.info(f"✓ Embeddings are sufficiently different")
    
    # Compare predictions if available
    if "predictions" in result_a and "predictions" in result_b:
        pred_a = result_a["predictions"]
        pred_b = result_b["predictions"]
        
        if pred_a.shape == pred_b.shape and pred_a.size > 0:
            pred_max_diff = np.abs(pred_a - pred_b).max()
            pred_mean_diff = np.abs(pred_a - pred_b).mean()
            
            LOGGER.info(f"Prediction comparison:")
            LOGGER.info(f"  Max difference: {pred_max_diff:.10f}")
            LOGGER.info(f"  Mean difference: {pred_mean_diff:.10f}")
            
            if pred_max_diff < 1e-6:
                LOGGER.error(f"✗ CRITICAL: Predictions are IDENTICAL!")
                return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate single-cell baseline differences")
    parser.add_argument("--adata_path", type=Path, required=True, help="Path to adata file")
    parser.add_argument("--split_config_path", type=Path, required=True, help="Path to split config JSON")
    parser.add_argument("--baseline_a", type=str, default="lpm_selftrained", help="First baseline to compare")
    parser.add_argument("--baseline_b", type=str, default="lpm_gearsPertEmb", help="Second baseline to compare")
    parser.add_argument("--pca_dim", type=int, default=10, help="PCA dimension")
    parser.add_argument("--ridge_penalty", type=float, default=0.1, help="Ridge penalty")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--n_cells_per_pert", type=int, default=50, help="Number of cells per perturbation")
    
    args = parser.parse_args()
    
    baseline_a = BaselineType(args.baseline_a)
    baseline_b = BaselineType(args.baseline_b)
    
    try:
        are_different = validate_baseline_differences(
            adata_path=args.adata_path,
            split_config_path=args.split_config_path,
            baseline_a=baseline_a,
            baseline_b=baseline_b,
            pca_dim=args.pca_dim,
            ridge_penalty=args.ridge_penalty,
            seed=args.seed,
            n_cells_per_pert=args.n_cells_per_pert,
        )
        
        if are_different:
            LOGGER.info("✓ Validation PASSED: Baselines produce different embeddings")
            return 0
        else:
            LOGGER.error("✗ Validation FAILED: Baselines produce identical or very similar embeddings")
            return 1
    except Exception as e:
        LOGGER.error(f"Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())


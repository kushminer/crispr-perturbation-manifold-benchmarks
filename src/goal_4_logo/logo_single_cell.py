#!/usr/bin/env python3
"""
LOGO Single-Cell: Leave-One-GO-Out evaluation for single-cell data.

Hold out cells belonging to perturbations of a specific functional class
(e.g., Transcription) and train on cells from all other classes.

This tests biological extrapolation at the single-cell level:
Can the model predict responses for a novel functional class?
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from shared.io import load_annotations
from shared.linear_model import solve_y_axb
from shared.metrics import compute_metrics
from goal_2_baselines.baseline_runner import construct_gene_embeddings, construct_pert_embeddings
from goal_2_baselines.baseline_types import BaselineConfig, BaselineType, get_baseline_config
from goal_2_baselines.single_cell_loader import aggregate_cell_metrics_by_perturbation, compute_cell_level_pseudobulk

LOGGER = logging.getLogger(__name__)


def compute_single_cell_expression_changes_logo(
    adata: ad.AnnData,
    holdout_targets: List[str],
    n_cells_per_pert: int = 50,
    seed: int = 1,
    min_cells_required: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, str]]:
    """
    Compute single-cell expression changes with LOGO split.
    
    Args:
        adata: AnnData object
        holdout_targets: List of perturbation names to hold out (test set)
        n_cells_per_pert: Number of cells to sample per perturbation
        seed: Random seed
        min_cells_required: Minimum cells required
        
    Returns:
        Tuple of:
        - Y matrix as DataFrame (genes × cells)
        - split_labels: Dict with 'train' and 'test' cell IDs
        - cell_to_pert: Dict mapping cell IDs to perturbation names
    """
    LOGGER.info(f"Computing single-cell expression changes (LOGO split)")
    LOGGER.info(f"Holdout perturbations: {len(holdout_targets)}")
    
    np.random.seed(seed)
    
    # Clean condition names
    adata.obs["clean_condition"] = (
        adata.obs["condition"].astype(str).str.replace(r"\+ctrl", "", regex=True)
    )
    
    # Compute baseline (mean expression in control)
    ctrl_mask = (adata.obs["condition"] == "ctrl").values
    if ctrl_mask.sum() == 0:
        raise ValueError("No control condition found in data")
    
    ctrl_expr = adata.X[ctrl_mask]
    if hasattr(ctrl_expr, "toarray"):
        ctrl_expr = ctrl_expr.toarray()
    baseline = np.asarray(ctrl_expr.mean(axis=0)).ravel()
    
    LOGGER.info(f"Control cells: {ctrl_mask.sum()}")
    
    # Get unique perturbation conditions (excluding ctrl)
    unique_conditions = [c for c in adata.obs["clean_condition"].unique() if c != "ctrl"]
    
    # Sample cells for each perturbation
    cell_data = []
    cell_ids = []
    cell_to_pert = {}
    
    train_cells = []
    test_cells = []
    
    for cond in unique_conditions:
        cond_mask = (adata.obs["clean_condition"] == cond).values
        n_cells_available = cond_mask.sum()
        
        if n_cells_available < min_cells_required:
            continue
        
        # Get indices of cells for this condition
        cond_indices = np.where(cond_mask)[0]
        
        # Sample cells
        n_to_sample = min(n_cells_per_pert, n_cells_available)
        sampled_indices = np.random.choice(cond_indices, n_to_sample, replace=False)
        
        # Get expression data
        cond_expr = adata.X[sampled_indices]
        if hasattr(cond_expr, "toarray"):
            cond_expr = cond_expr.toarray()
        
        # Compute change from baseline for each cell
        cell_changes = np.asarray(cond_expr) - baseline
        
        # Determine if this perturbation is in holdout
        is_test = cond in holdout_targets
        
        for i, idx in enumerate(sampled_indices):
            cell_id = f"{cond}_{i}"
            cell_data.append(cell_changes[i])
            cell_ids.append(cell_id)
            cell_to_pert[cell_id] = cond
            
            if is_test:
                test_cells.append(cell_id)
            else:
                train_cells.append(cell_id)
    
    # Create Y matrix (genes × cells)
    Y = np.vstack(cell_data).T
    
    # Create DataFrame
    gene_names = adata.var_names.tolist()
    Y_df = pd.DataFrame(Y, index=gene_names, columns=cell_ids)
    
    LOGGER.info(f"Y matrix shape: {Y_df.shape} (genes × cells)")
    LOGGER.info(f"Train cells: {len(train_cells)}")
    LOGGER.info(f"Test cells (holdout): {len(test_cells)}")
    
    split_labels = {
        "train": train_cells,
        "test": test_cells,
    }
    
    return Y_df, split_labels, cell_to_pert


def run_logo_single_cell(
    adata_path: Path,
    annotation_path: Path,
    dataset_name: str,
    output_dir: Path,
    class_name: str = "Transcription",
    baseline_types: Optional[List[BaselineType]] = None,
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
    n_cells_per_pert: int = 50,
) -> pd.DataFrame:
    """
    Run LOGO evaluation on single-cell data.
    
    Args:
        adata_path: Path to perturb_processed.h5ad
        annotation_path: Path to functional class annotations TSV
        dataset_name: Dataset name
        output_dir: Output directory
        class_name: Functional class to hold out as test set
        baseline_types: List of baseline types (None = core baselines)
        pca_dim: PCA dimension
        ridge_penalty: Ridge penalty
        seed: Random seed
        n_cells_per_pert: Number of cells per perturbation
        
    Returns:
        DataFrame with results
    """
    LOGGER.info("=" * 60)
    LOGGER.info("LOGO SINGLE-CELL EVALUATION")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Dataset: {dataset_name}")
    LOGGER.info(f"Holdout class: {class_name}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    LOGGER.info(f"Loading annotations from {annotation_path}")
    annotations = load_annotations(annotation_path)
    annotations["target"] = annotations["target"].astype(str)
    
    # Identify holdout class perturbations
    holdout_targets = annotations.loc[
        annotations["class"] == class_name, "target"
    ].unique().tolist()
    LOGGER.info(f"Found {len(holdout_targets)} perturbations in class '{class_name}'")
    
    if len(holdout_targets) == 0:
        raise ValueError(f"No perturbations found for class '{class_name}'")
    
    # Load data
    LOGGER.info(f"Loading data from {adata_path}")
    adata = ad.read_h5ad(adata_path)
    
    # Compute single-cell Y matrix with LOGO split
    Y_df, split_labels, cell_to_pert = compute_single_cell_expression_changes_logo(
        adata, holdout_targets, n_cells_per_pert=n_cells_per_pert, seed=seed
    )
    
    train_cells = split_labels["train"]
    test_cells = split_labels["test"]
    
    if len(train_cells) < 10:
        raise ValueError(f"Insufficient training cells: {len(train_cells)}")
    if len(test_cells) == 0:
        raise ValueError(f"No test cells for holdout class '{class_name}'")
    
    Y_train = Y_df[train_cells]
    Y_test = Y_df[test_cells]
    
    Y_train_np = Y_train.values
    Y_test_np = Y_test.values
    
    gene_names = Y_df.index.tolist()
    
    # Create cell-to-pert mappings
    cell_to_pert_train = {c: cell_to_pert[c] for c in train_cells}
    cell_to_pert_test = {c: cell_to_pert[c] for c in test_cells}
    
    # Default baselines
    if baseline_types is None:
        baseline_types = [
            BaselineType.SELFTRAINED,
            BaselineType.RANDOM_GENE_EMB,
            BaselineType.RANDOM_PERT_EMB,
        ]
    
    # Get gene name mapping
    gene_name_mapping = None
    if "gene_name" in adata.var.columns:
        gene_name_mapping = dict(zip(adata.var_names, adata.var["gene_name"]))
    
    all_results = []
    
    for baseline_type in baseline_types:
        LOGGER.info(f"\nRunning LOGO single-cell for {baseline_type.value}")
        
        try:
            config = get_baseline_config(
                baseline_type, pca_dim=pca_dim, ridge_penalty=ridge_penalty, seed=seed
            )
            
            # Construct gene embeddings (A matrix)
            embedding_args = config.gene_embedding_args.copy() if config.gene_embedding_args else {}
            if gene_name_mapping and config.gene_embedding_source in ["scgpt", "scfoundation"]:
                embedding_args["gene_name_mapping"] = gene_name_mapping
            
            A, _ = construct_gene_embeddings(
                source=config.gene_embedding_source,
                train_data=Y_train_np,
                gene_names=gene_names,
                pca_dim=pca_dim,
                seed=seed,
                embedding_args=embedding_args,
            )
            
            # Construct cell embeddings - FIXED: use proper embedding source
            if config.pert_embedding_source == "training_data":
                # Use cell-level PCA (self-trained baseline)
                LOGGER.debug(f"Using cell-level PCA for {baseline_type.value}")
                pca = PCA(n_components=pca_dim, random_state=seed)
                B_train = pca.fit_transform(Y_train_np.T).T
                B_test = pca.transform(Y_test_np.T).T
            else:
                # Use perturbation-level embeddings mapped to cells
                LOGGER.debug(f"Using perturbation embeddings ({config.pert_embedding_source}) for {baseline_type.value}")
                
                # Compute pseudobulk for embedding construction
                Y_train_df = pd.DataFrame(Y_train_np, index=gene_names, columns=train_cells)
                Y_test_df = pd.DataFrame(Y_test_np, index=gene_names, columns=test_cells)
                
                Y_train_pseudobulk = compute_cell_level_pseudobulk(Y_train_df, cell_to_pert_train)
                Y_test_pseudobulk = compute_cell_level_pseudobulk(Y_test_df, cell_to_pert_test)
                
                train_pert_names = Y_train_pseudobulk.columns.tolist()
                test_pert_names = Y_test_pseudobulk.columns.tolist()
                
                # Construct perturbation embeddings
                pert_embedding_args = config.pert_embedding_args.copy() if config.pert_embedding_args else {}
                if config.pert_embedding_source in ["k562_pca", "rpe1_pca"]:
                    pert_embedding_args["target_gene_names"] = gene_names
                
                B_train_pert, _, _, B_test_pert, _ = construct_pert_embeddings(
                    source=config.pert_embedding_source,
                    train_data=Y_train_pseudobulk.values,
                    pert_names=train_pert_names,
                    pca_dim=pca_dim,
                    seed=seed,
                    embedding_args=pert_embedding_args,
                    test_data=Y_test_pseudobulk.values,
                    test_pert_names=test_pert_names,
                )
                
                # Map perturbation embeddings to cells
                B_train = np.zeros((pca_dim, len(train_cells)))
                for i, cell_id in enumerate(train_cells):
                    pert = cell_to_pert_train[cell_id]
                    if pert in train_pert_names:
                        pert_idx = train_pert_names.index(pert)
                        B_train[:, i] = B_train_pert[:, pert_idx]
                
                B_test = np.zeros((pca_dim, len(test_cells)))
                for i, cell_id in enumerate(test_cells):
                    pert = cell_to_pert_test[cell_id]
                    if pert in test_pert_names and B_test_pert is not None:
                        pert_idx = test_pert_names.index(pert)
                        B_test[:, i] = B_test_pert[:, pert_idx]
                    elif pert in train_pert_names:
                        pert_idx = train_pert_names.index(pert)
                        B_test[:, i] = B_train_pert[:, pert_idx]
            
            # Train model on non-holdout cells
            center = Y_train_np.mean(axis=1, keepdims=True)
            Y_centered = Y_train_np - center
            
            solution = solve_y_axb(
                Y=Y_centered,
                A=A,
                B=B_train,
                ridge_penalty=ridge_penalty,
            )
            K = solution["K"]
            
            # Predict test cells
            Y_pred = A @ K @ B_test + center
            
            # Compute cell-level metrics
            cell_metrics = {}
            for i, cell_id in enumerate(test_cells):
                y_true = Y_test_np[:, i]
                y_pred = Y_pred[:, i]
                cell_metrics[cell_id] = compute_metrics(y_true, y_pred)
            
            # Aggregate to perturbation level
            pert_metrics = aggregate_cell_metrics_by_perturbation(cell_metrics, cell_to_pert_test)
            
            # Compute summary
            cell_mean_r = np.mean([m["pearson_r"] for m in cell_metrics.values()])
            cell_mean_l2 = np.mean([m["l2"] for m in cell_metrics.values()])
            pert_mean_r = np.mean([m["pearson_r"] for m in pert_metrics.values()])
            pert_mean_l2 = np.mean([m["l2"] for m in pert_metrics.values()])
            
            # Store results
            for cell_id, metrics in cell_metrics.items():
                all_results.append({
                    "cell_id": cell_id,
                    "perturbation": cell_to_pert_test[cell_id],
                    "baseline_type": baseline_type.value,
                    "holdout_class": class_name,
                    "pearson_r": metrics["pearson_r"],
                    "l2": metrics["l2"],
                })
            
            LOGGER.info(f"  {baseline_type.value}: cell_r={cell_mean_r:.3f}, pert_r={pert_mean_r:.3f}")
            
        except Exception as e:
            LOGGER.error(f"Failed {baseline_type.value}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_df.to_csv(
        output_dir / f"logo_single_cell_{dataset_name}_{class_name}.csv",
        index=False
    )
    
    # Compute and save summary
    summary = []
    for baseline_type in baseline_types:
        baseline_results = results_df[results_df["baseline_type"] == baseline_type.value]
        if len(baseline_results) == 0:
            continue
        
        pert_summary = baseline_results.groupby("perturbation").agg({
            "pearson_r": "mean",
            "l2": "mean",
        }).reset_index()
        
        summary.append({
            "baseline_type": baseline_type.value,
            "holdout_class": class_name,
            "n_test_cells": len(baseline_results),
            "n_test_perts": len(pert_summary),
            "cell_mean_pearson_r": baseline_results["pearson_r"].mean(),
            "cell_mean_l2": baseline_results["l2"].mean(),
            "pert_mean_pearson_r": pert_summary["pearson_r"].mean(),
            "pert_mean_l2": pert_summary["l2"].mean(),
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(
        output_dir / f"logo_single_cell_summary_{dataset_name}_{class_name}.csv",
        index=False
    )
    
    LOGGER.info(f"\nLOGO Single-Cell Summary ({dataset_name}, holdout={class_name}):")
    LOGGER.info(f"\n{summary_df.to_string()}")
    
    return results_df


def run_logo_single_cell_all_baselines(
    adata_path: Path,
    annotation_path: Path,
    output_dir: Path,
    dataset_name: str,
    class_name: str = "Transcription",
    baseline_types: Optional[List[BaselineType]] = None,
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
    n_cells_per_pert: int = 50,
) -> pd.DataFrame:
    """
    Run LOGO single-cell for all baselines.
    
    This is a convenience wrapper for run_logo_single_cell.
    """
    return run_logo_single_cell(
        adata_path=adata_path,
        annotation_path=annotation_path,
        dataset_name=dataset_name,
        output_dir=output_dir,
        class_name=class_name,
        baseline_types=baseline_types,
        pca_dim=pca_dim,
        ridge_penalty=ridge_penalty,
        seed=seed,
        n_cells_per_pert=n_cells_per_pert,
    )


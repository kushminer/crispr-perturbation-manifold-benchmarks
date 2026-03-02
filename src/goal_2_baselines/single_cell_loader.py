#!/usr/bin/env python3
"""
Single-Cell Data Loader

Replaces pseudobulk aggregation with cell-level sampling for single-cell analysis.
Instead of averaging cells per perturbation, we sample N cells per perturbation
and return cell-level expression changes.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def compute_single_cell_expression_changes(
    adata: ad.AnnData,
    split_config: Dict[str, List[str]],
    n_cells_per_pert: int = 50,
    seed: int = 1,
    min_cells_required: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, str]]:
    """
    Compute single-cell expression changes (Y matrix at cell level).
    
    Instead of pseudobulking (averaging), we:
    1. Sample N cells per perturbation
    2. Return cell-level expression changes (vs control mean)
    3. Track cell-to-perturbation mapping for aggregation
    
    Y_{i,c} = expression of gene i in cell c - mean(expression of gene i in ctrl)
    
    Args:
        adata: AnnData object with expression data
        split_config: Dictionary with 'train', 'test', 'val' keys mapping to condition lists
        n_cells_per_pert: Number of cells to sample per perturbation
        seed: Random seed for reproducibility
        min_cells_required: Minimum cells required for a perturbation (skip if fewer)
    
    Returns:
        Tuple of:
        - Y matrix as DataFrame (genes × cells)
        - split_labels: Dict mapping split names to cell IDs
        - cell_to_pert: Dict mapping cell IDs to perturbation names
    """
    LOGGER.info(f"Computing single-cell expression changes (n_cells_per_pert={n_cells_per_pert})")
    
    np.random.seed(seed)
    
    # Filter to valid conditions
    all_conditions = []
    for conditions in split_config.values():
        all_conditions.extend(conditions)
    all_conditions = list(set(all_conditions))
    
    adata = adata[adata.obs["condition"].isin(all_conditions)].copy()
    
    # Clean condition names (remove +ctrl suffix)
    adata.obs["clean_condition"] = (
        adata.obs["condition"].astype(str).str.replace(r"\+ctrl", "", regex=True)
    )
    
    # Compute baseline (mean expression in control)
    ctrl_mask = (adata.obs["condition"] == "ctrl").values  # Convert to numpy array
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
    
    skipped_perts = []
    
    for cond in unique_conditions:
        cond_mask = (adata.obs["clean_condition"] == cond).values  # Convert to numpy array
        n_cells_available = cond_mask.sum()
        
        if n_cells_available < min_cells_required:
            skipped_perts.append((cond, n_cells_available))
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
        
        for i, idx in enumerate(sampled_indices):
            cell_id = f"{cond}_{i}"
            cell_data.append(cell_changes[i])
            cell_ids.append(cell_id)
            cell_to_pert[cell_id] = cond
    
    if skipped_perts:
        LOGGER.warning(f"Skipped {len(skipped_perts)} perturbations with <{min_cells_required} cells:")
        for cond, n in skipped_perts[:5]:
            LOGGER.warning(f"  - {cond}: {n} cells")
        if len(skipped_perts) > 5:
            LOGGER.warning(f"  ... and {len(skipped_perts) - 5} more")
    
    # Create Y matrix (genes × cells)
    Y = np.vstack(cell_data).T
    
    # Create DataFrame
    gene_names = adata.var_names.tolist()
    Y_df = pd.DataFrame(Y, index=gene_names, columns=cell_ids)
    
    LOGGER.info(f"Y matrix shape: {Y_df.shape} (genes × cells)")
    LOGGER.info(f"Total perturbations included: {len(set(cell_to_pert.values()))}")
    
    # Map cells to splits based on their perturbation
    clean_split_config = {}
    for split_name, conditions in split_config.items():
        clean_split_config[split_name] = [
            cond.replace("+ctrl", "") if "+ctrl" in cond else cond
            for cond in conditions
        ]
    
    split_labels = {}
    for split_name, clean_conditions in clean_split_config.items():
        split_labels[split_name] = [
            cell_id for cell_id in cell_ids 
            if cell_to_pert.get(cell_id) in clean_conditions
        ]
    
    LOGGER.info(f"Train cells: {len(split_labels.get('train', []))}")
    LOGGER.info(f"Test cells: {len(split_labels.get('test', []))}")
    LOGGER.info(f"Val cells: {len(split_labels.get('val', []))}")
    
    return Y_df, split_labels, cell_to_pert


def aggregate_cell_metrics_by_perturbation(
    cell_metrics: Dict[str, Dict],
    cell_to_pert: Dict[str, str],
) -> Dict[str, Dict]:
    """
    Aggregate cell-level metrics to perturbation level.
    
    This enables fair comparison with pseudobulk results.
    
    Args:
        cell_metrics: Dict mapping cell IDs to metric dicts
        cell_to_pert: Dict mapping cell IDs to perturbation names
        
    Returns:
        Dict mapping perturbation names to aggregated metric dicts
    """
    # Group cells by perturbation
    pert_cells = {}
    for cell_id, pert in cell_to_pert.items():
        if pert not in pert_cells:
            pert_cells[pert] = []
        if cell_id in cell_metrics:
            pert_cells[pert].append(cell_metrics[cell_id])
    
    # Aggregate metrics
    pert_metrics = {}
    for pert, cell_metric_list in pert_cells.items():
        if not cell_metric_list:
            continue
        
        # Get all metric names
        metric_names = cell_metric_list[0].keys()
        
        # Compute mean for each metric
        pert_metrics[pert] = {}
        for metric_name in metric_names:
            values = [m[metric_name] for m in cell_metric_list if metric_name in m]
            if values:
                pert_metrics[pert][metric_name] = np.mean(values)
                pert_metrics[pert][f"{metric_name}_std"] = np.std(values)
                pert_metrics[pert][f"{metric_name}_n"] = len(values)
    
    return pert_metrics


def get_perturbation_cell_matrix(
    Y_df: pd.DataFrame,
    cell_to_pert: Dict[str, str],
    pert_name: str,
) -> pd.DataFrame:
    """
    Extract the expression matrix for all cells from a specific perturbation.
    
    Args:
        Y_df: Full Y matrix (genes × cells)
        cell_to_pert: Mapping from cell IDs to perturbations
        pert_name: Perturbation to extract
        
    Returns:
        DataFrame (genes × cells_from_pert)
    """
    pert_cells = [cell_id for cell_id, pert in cell_to_pert.items() if pert == pert_name]
    return Y_df[pert_cells]


def compute_cell_level_pseudobulk(
    Y_df: pd.DataFrame,
    cell_to_pert: Dict[str, str],
) -> pd.DataFrame:
    """
    Compute pseudobulk from single-cell data for comparison.
    
    This creates the same format as the original pseudobulk pipeline,
    enabling direct comparison of results.
    
    Args:
        Y_df: Single-cell Y matrix (genes × cells)
        cell_to_pert: Mapping from cell IDs to perturbations
        
    Returns:
        Pseudobulk DataFrame (genes × perturbations)
    """
    # Group cells by perturbation
    pert_to_cells = {}
    for cell_id, pert in cell_to_pert.items():
        if pert not in pert_to_cells:
            pert_to_cells[pert] = []
        pert_to_cells[pert].append(cell_id)
    
    # Compute mean expression per perturbation
    pseudobulk_data = {}
    for pert, cell_ids in pert_to_cells.items():
        pseudobulk_data[pert] = Y_df[cell_ids].mean(axis=1)
    
    return pd.DataFrame(pseudobulk_data)


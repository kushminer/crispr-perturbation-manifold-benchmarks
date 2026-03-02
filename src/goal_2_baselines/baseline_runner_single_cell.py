#!/usr/bin/env python3
"""
Single-Cell Baseline Runner

Adapts the baseline runner for single-cell data analysis.
Instead of training on pseudobulked data, we train on sampled single cells
and evaluate predictions at the cell level.

Key differences from pseudobulk baseline runner:
1. Y matrix is genes × cells (not genes × perturbations)
2. Perturbation embeddings are derived from cell-level data
3. Metrics are computed per-cell and aggregated per-perturbation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from shared.linear_model import solve_y_axb
from shared.metrics import compute_metrics

from .baseline_types import BaselineConfig, BaselineType, get_baseline_config
from .split_logic import load_split_config
from .single_cell_loader import (
    compute_single_cell_expression_changes,
    aggregate_cell_metrics_by_perturbation,
    compute_cell_level_pseudobulk,
)
from .baseline_runner import construct_gene_embeddings, construct_pert_embeddings

LOGGER = logging.getLogger(__name__)
def _log_matrix_summary(name: str, matrix: Optional[np.ndarray], labels: Optional[List[str]] = None) -> None:
    """Log basic statistics for an embedding matrix to help with audits."""
    if matrix is None or matrix.size == 0:
        LOGGER.info(f"{name}: empty")
        return
    
    LOGGER.info(
        "%s: shape=%s, mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        name,
        matrix.shape,
        float(matrix.mean()),
        float(matrix.std()),
        float(matrix.min()),
        float(matrix.max()),
    )
    if labels:
        sample = labels[:5]
        LOGGER.info("%s labels (first 5): %s%s", name, sample, "..." if len(labels) > 5 else "")


def _expand_per_pert_embeddings_to_cells(
    B_perts: np.ndarray,
    pert_labels: List[str],
    cell_ids: List[str],
    cell_to_pert: Dict[str, str],
    space_name: str,
) -> np.ndarray:
    """Map perturbation-level embeddings to cell-level embeddings by lookup."""
    if B_perts.size == 0:
        return np.zeros((0, len(cell_ids)))
    
    label_to_idx = {p: i for i, p in enumerate(pert_labels)}
    B_cells = np.zeros((B_perts.shape[0], len(cell_ids)))
    missing_perts = set()
    
    for j, cell_id in enumerate(cell_ids):
        pert = cell_to_pert.get(cell_id)
        if pert is None:
            missing_perts.add("undefined")
            continue
        idx = label_to_idx.get(pert)
        if idx is None:
            missing_perts.add(pert)
            continue
        B_cells[:, j] = B_perts[:, idx]
    
    if missing_perts:
        LOGGER.warning(
            "%s: %d cells mapped to perturbations without embeddings (examples: %s)",
            space_name,
            sum(cell_to_pert.get(cid) in missing_perts for cid in cell_ids),
            list(missing_perts)[:5],
        )
    return B_cells



def construct_cell_embeddings(
    Y_cells: np.ndarray,
    cell_ids: List[str],
    cell_to_pert: Dict[str, str],
    pca_dim: int,
    seed: int,
    method: str = "cell_pca",
) -> Tuple[np.ndarray, PCA]:
    """
    Construct embeddings for cells (analogous to perturbation embeddings).
    
    Args:
        Y_cells: Cell expression matrix (genes × cells)
        cell_ids: List of cell IDs
        cell_to_pert: Mapping from cell IDs to perturbations
        pca_dim: PCA dimension
        seed: Random seed
        method: Embedding method ("cell_pca" or "pert_pca")
        
    Returns:
        Tuple of (B matrix of cell embeddings, fitted PCA object)
    """
    np.random.seed(seed)
    
    if method == "cell_pca":
        # PCA directly on cells
        pca = PCA(n_components=pca_dim, random_state=seed)
        B = pca.fit_transform(Y_cells.T).T  # (pca_dim, n_cells)
        return B, pca
    
    elif method == "pert_pca":
        # First average cells to perturbation level, then PCA, then map back
        # Group cells by perturbation
        pert_to_cells = {}
        for cell_id in cell_ids:
            pert = cell_to_pert[cell_id]
            if pert not in pert_to_cells:
                pert_to_cells[pert] = []
            pert_to_cells[pert].append(cell_ids.index(cell_id))
        
        # Compute perturbation-level means
        perts = list(pert_to_cells.keys())
        Y_pert = np.zeros((Y_cells.shape[0], len(perts)))
        for i, pert in enumerate(perts):
            cell_indices = pert_to_cells[pert]
            Y_pert[:, i] = Y_cells[:, cell_indices].mean(axis=1)
        
        # PCA on perturbations
        pca = PCA(n_components=pca_dim, random_state=seed)
        B_pert = pca.fit_transform(Y_pert.T).T  # (pca_dim, n_perts)
        
        # Map perturbation embeddings back to cells
        pert_to_idx = {p: i for i, p in enumerate(perts)}
        B = np.zeros((pca_dim, len(cell_ids)))
        for j, cell_id in enumerate(cell_ids):
            pert = cell_to_pert[cell_id]
            B[:, j] = B_pert[:, pert_to_idx[pert]]
        
        return B, pca
    
    else:
        raise ValueError(f"Unknown embedding method: {method}")


def run_single_baseline_single_cell(
    Y_train: pd.DataFrame,
    Y_test: pd.DataFrame,
    config: BaselineConfig,
    gene_names: List[str],
    cell_to_pert_train: Dict[str, str],
    cell_to_pert_test: Dict[str, str],
    gene_name_mapping: Optional[Dict[str, str]] = None,
    cell_embedding_method: str = "cell_pca",
) -> Dict:
    """
    Run a single baseline model on single-cell data.
    
    Args:
        Y_train: Training Y matrix (genes × train_cells)
        Y_test: Test Y matrix (genes × test_cells)
        config: Baseline configuration
        gene_names: List of gene names
        cell_to_pert_train: Mapping from train cell IDs to perturbations
        cell_to_pert_test: Mapping from test cell IDs to perturbations
        gene_name_mapping: Optional mapping from Ensembl IDs to gene symbols
        cell_embedding_method: Method for cell embeddings ("cell_pca" or "pert_pca")
    
    Returns:
        Dictionary with predictions and metrics
    """
    LOGGER.info(f"Running single-cell baseline: {config.baseline_type.value}")
    LOGGER.info(f"Cell embedding method: {cell_embedding_method}")
    
    # Convert to numpy
    Y_train_np = Y_train.values
    Y_test_np = Y_test.values
    
    train_cell_ids = Y_train.columns.tolist()
    test_cell_ids = Y_test.columns.tolist()
    
    LOGGER.info(f"Train cells: {len(train_cell_ids)}, Test cells: {len(test_cell_ids)}")
    
    # Construct A (gene embeddings) - same as pseudobulk
    embedding_args_with_mapping = config.gene_embedding_args.copy() if config.gene_embedding_args else {}
    if gene_name_mapping and config.gene_embedding_source in ["scgpt", "scfoundation"]:
        embedding_args_with_mapping["gene_name_mapping"] = gene_name_mapping
    
    A, gene_labels = construct_gene_embeddings(
        source=config.gene_embedding_source,
        train_data=Y_train_np,
        gene_names=gene_names,
        pca_dim=config.pca_dim,
        seed=config.seed,
        embedding_args=embedding_args_with_mapping,
    )
    _log_matrix_summary(f"A ({config.baseline_type.value})", A, gene_labels)
    
    # Construct B (cell embeddings) for training cells
    LOGGER.info(f"[{config.baseline_type.value}] Constructing cell embeddings using pert_embedding_source='{config.pert_embedding_source}'")
    
    if config.pert_embedding_source == "training_data":
        LOGGER.info(f"[{config.baseline_type.value}] Using training_data path: constructing cell embeddings via PCA")
        B_train, pca = construct_cell_embeddings(
            Y_cells=Y_train_np,
            cell_ids=train_cell_ids,
            cell_to_pert=cell_to_pert_train,
            pca_dim=config.pca_dim,
            seed=config.seed,
            method=cell_embedding_method,
        )
        B_test = None
        if not Y_test.empty:
            if pca is not None:
                B_test = pca.transform(Y_test_np.T).T
            else:
                np.random.seed(config.seed + 1)
                B_test = np.random.randn(config.pca_dim, len(test_cell_ids))
    elif config.pert_embedding_source == "random":
        LOGGER.info(f"[{config.baseline_type.value}] Using random path: generating random cell embeddings")
        # Random embeddings for cells
        np.random.seed(config.seed)
        B_train = np.random.randn(config.pca_dim, len(train_cell_ids))
        pca = None
        B_test = None
        if not Y_test.empty:
            np.random.seed(config.seed + 1)
            B_test = np.random.randn(config.pca_dim, len(test_cell_ids))
    else:
        # Use perturbation-level embeddings (e.g., GEARS, cross-dataset PCA)
        LOGGER.info(f"[{config.baseline_type.value}] Using perturbation-level embedding path: source='{config.pert_embedding_source}'")
        LOGGER.info(f"[{config.baseline_type.value}] Computing pseudobulk from single-cell data...")
        train_pseudobulk = compute_cell_level_pseudobulk(Y_train, cell_to_pert_train)
        test_pseudobulk = compute_cell_level_pseudobulk(Y_test, cell_to_pert_test) if not Y_test.empty else None
        LOGGER.info(f"[{config.baseline_type.value}] Train pseudobulk: {train_pseudobulk.shape} (genes × perts)")
        if test_pseudobulk is not None:
            LOGGER.info(f"[{config.baseline_type.value}] Test pseudobulk: {test_pseudobulk.shape} (genes × perts)")
        
        pert_embedding_args = config.pert_embedding_args.copy() if config.pert_embedding_args else {}
        # Provide target gene names only for cross-dataset loaders
        if config.pert_embedding_source in {"k562_pca", "rpe1_pca"}:
            pert_embedding_args.setdefault("target_gene_names", gene_labels)
        
        # Log embedding args for debugging
        if config.pert_embedding_source == "gears":
            LOGGER.info(f"[{config.baseline_type.value}] GEARS embedding args: {pert_embedding_args}")
            if "source_csv" in pert_embedding_args:
                # Resolve path the same way construct_pert_embeddings does
                source_csv_path = Path(pert_embedding_args["source_csv"])
                if not source_csv_path.is_absolute():
                    # Resolve relative to repository root (same as baseline_runner.py)
                    repo_root = Path(__file__).parent.parent.parent
                    source_csv_path = repo_root / source_csv_path
                source_csv_path = source_csv_path.resolve()
                
                if source_csv_path.exists():
                    LOGGER.info(f"[{config.baseline_type.value}] GEARS CSV file exists: {source_csv_path}")
                else:
                    LOGGER.error(f"[{config.baseline_type.value}] GEARS CSV file NOT FOUND: {source_csv_path}")
                    raise FileNotFoundError(f"GEARS CSV file not found: {source_csv_path}")
        
        LOGGER.info(f"[{config.baseline_type.value}] Calling construct_pert_embeddings with source='{config.pert_embedding_source}'")
        try:
            B_pert_train, train_pert_labels, _, B_pert_test, test_pert_labels = construct_pert_embeddings(
                source=config.pert_embedding_source,
                train_data=train_pseudobulk.values,
                pert_names=train_pseudobulk.columns.tolist(),
                pca_dim=config.pca_dim,
                seed=config.seed,
                embedding_args=pert_embedding_args,
                test_data=test_pseudobulk.values if test_pseudobulk is not None else None,
                test_pert_names=test_pseudobulk.columns.tolist() if test_pseudobulk is not None else None,
            )
            LOGGER.info(f"[{config.baseline_type.value}] construct_pert_embeddings returned:")
            LOGGER.info(f"  B_pert_train shape: {B_pert_train.shape if B_pert_train is not None else None}")
            LOGGER.info(f"  train_pert_labels: {len(train_pert_labels) if train_pert_labels else 0} perturbations")
            LOGGER.info(f"  B_pert_test shape: {B_pert_test.shape if B_pert_test is not None else None}")
        except Exception as e:
            LOGGER.error(f"[{config.baseline_type.value}] FAILED to construct perturbation embeddings: {e}")
            LOGGER.error(f"[{config.baseline_type.value}] This is a critical error - embeddings must be constructed successfully")
            raise
        
        LOGGER.info(f"[{config.baseline_type.value}] Expanding perturbation embeddings to cell level...")
        B_train = _expand_per_pert_embeddings_to_cells(
            B_pert_train,
            train_pert_labels,
            train_cell_ids,
            cell_to_pert_train,
            f"{config.baseline_type.value}-train",
        )
        train_pert_data = (B_pert_train, train_pert_labels)
        test_pert_data = (B_pert_test, test_pert_labels)
        B_test = None
        if not Y_test.empty:
            if B_pert_test is None:
                LOGGER.warning(f"[{config.baseline_type.value}] B_pert_test is None, falling back to training embeddings for test cells")
                # Fall back to training perturbation embeddings if overlap exists
                B_test = _expand_per_pert_embeddings_to_cells(
                    train_pert_data[0],
                    train_pert_data[1],
                    test_cell_ids,
                    cell_to_pert_test,
                    f"{config.baseline_type.value}-test-shared",
                )
            else:
                B_test = _expand_per_pert_embeddings_to_cells(
                    test_pert_data[0],
                    (test_pert_data[1] or []),
                    test_cell_ids,
                    cell_to_pert_test,
                    f"{config.baseline_type.value}-test",
                )
        pca = None
    
    _log_matrix_summary(f"B_train ({config.baseline_type.value})", B_train, train_cell_ids)
    
    # VALIDATION: Ensure embeddings are not identical to training_data embeddings
    # This catches bugs where GEARS or other baselines silently fall back to training_data
    # Only validate for baselines that should use different embeddings (not training_data or random)
    if config.pert_embedding_source not in {"training_data", "random"}:
        # Compute what training_data embeddings would produce for comparison
        LOGGER.info(f"[{config.baseline_type.value}] Validating embeddings are different from training_data...")
        try:
            B_train_training_data, _ = construct_cell_embeddings(
                Y_cells=Y_train_np,
                cell_ids=train_cell_ids,
                cell_to_pert=cell_to_pert_train,
                pca_dim=config.pca_dim,
                seed=config.seed,
                method=cell_embedding_method,
            )
            
            # Check if embeddings are identical (within floating point precision)
            if B_train.shape == B_train_training_data.shape:
                max_diff = np.abs(B_train - B_train_training_data).max()
                mean_diff = np.abs(B_train - B_train_training_data).mean()
                
                LOGGER.info(f"[{config.baseline_type.value}] Comparison with training_data embeddings:")
                LOGGER.info(f"  Max difference: {max_diff:.10f}")
                LOGGER.info(f"  Mean difference: {mean_diff:.10f}")
                
                if max_diff < 1e-6:
                    error_msg = (
                        f"CRITICAL ERROR: {config.baseline_type.value} embeddings are IDENTICAL to training_data embeddings!\n"
                        f"This indicates a bug where {config.pert_embedding_source} embeddings are not being used.\n"
                        f"Max difference: {max_diff:.10e}, Mean difference: {mean_diff:.10e}\n"
                        f"B_train stats: mean={B_train.mean():.6f}, std={B_train.std():.6f}\n"
                        f"B_train_training_data stats: mean={B_train_training_data.mean():.6f}, std={B_train_training_data.std():.6f}"
                    )
                    LOGGER.error(error_msg)
                    raise ValueError(error_msg)
                elif max_diff < 1e-3:
                    warning_msg = (
                        f"WARNING: {config.baseline_type.value} embeddings are VERY SIMILAR to training_data embeddings.\n"
                        f"Max difference: {max_diff:.10e}, Mean difference: {mean_diff:.10e}\n"
                        f"This may indicate a problem with {config.pert_embedding_source} embedding loading."
                    )
                    LOGGER.warning(warning_msg)
                else:
                    LOGGER.info(f"[{config.baseline_type.value}] ✓ Embeddings are sufficiently different from training_data (max_diff={max_diff:.6f})")
            else:
                LOGGER.info(f"[{config.baseline_type.value}] Embedding shapes differ from training_data (expected): {B_train.shape} vs {B_train_training_data.shape}")
        except Exception as e:
            LOGGER.warning(f"[{config.baseline_type.value}] Could not validate against training_data embeddings: {e}")
    
    # Solve for K using training data
    center = Y_train_np.mean(axis=1, keepdims=True)
    Y_centered = Y_train_np - center
    
    solution = solve_y_axb(
        Y=Y_centered,
        A=A,
        B=B_train,
        ridge_penalty=config.ridge_penalty,
    )
    K = solution["K"]
    
    # Construct B for test cells
    if Y_test.empty:
        Y_pred_test = np.array([]).reshape(Y_train_np.shape[0], 0)
        cell_metrics = {}
        pert_metrics = {}
    else:
        if B_test is None and pca is not None:
            B_test = pca.transform(Y_test_np.T).T
        elif B_test is None:
            LOGGER.warning(
                "No embedding projection available for test cells in %s; using zeros",
                config.baseline_type.value,
            )
            B_test = np.zeros((B_train.shape[0], len(test_cell_ids)))
        
        _log_matrix_summary(f"B_test ({config.baseline_type.value})", B_test, test_cell_ids)
        
        # Make predictions: Y_pred = A @ K @ B_test + center
        Y_pred_test = A @ K @ B_test + center
        
        # Compute cell-level metrics
        cell_metrics = {}
        for i, cell_id in enumerate(test_cell_ids):
            y_true = Y_test_np[:, i]
            y_pred = Y_pred_test[:, i]
            cell_metrics[cell_id] = compute_metrics(y_true, y_pred)
        
        # Aggregate to perturbation level
        pert_metrics = aggregate_cell_metrics_by_perturbation(
            cell_metrics, cell_to_pert_test
        )
    
    return {
        "baseline_type": config.baseline_type.value,
        "predictions": Y_pred_test,
        "cell_metrics": cell_metrics,
        "pert_metrics": pert_metrics,  # Aggregated for comparison with pseudobulk
        "K": K,
        "A": A,
        "B_train": B_train,
        "n_train_cells": len(train_cell_ids),
        "n_test_cells": len(test_cell_ids),
    }


def run_all_baselines_single_cell(
    adata_path: Path,
    split_config_path: Path,
    output_dir: Path,
    baseline_types: Optional[List[BaselineType]] = None,
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
    n_cells_per_pert: int = 50,
    cell_embedding_method: str = "cell_pca",
) -> pd.DataFrame:
    """
    Run all baseline models on single-cell data and save results.
    
    Args:
        adata_path: Path to perturb_processed.h5ad
        split_config_path: Path to train/test/val split JSON
        output_dir: Directory to save results
        baseline_types: List of baseline types to run (None = core baselines)
        pca_dim: PCA dimension
        ridge_penalty: Ridge penalty
        seed: Random seed
        n_cells_per_pert: Number of cells to sample per perturbation
        cell_embedding_method: Method for cell embeddings
    
    Returns:
        DataFrame with results summary
    """
    LOGGER.info("Starting single-cell baseline evaluation")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    LOGGER.info(f"Loading data from {adata_path}")
    adata = ad.read_h5ad(adata_path)
    
    # Load splits
    LOGGER.info(f"Loading splits from {split_config_path}")
    split_config = load_split_config(split_config_path)
    
    # Compute single-cell Y matrix
    LOGGER.info(f"Computing single-cell expression changes (n_cells_per_pert={n_cells_per_pert})")
    Y_df, split_labels, cell_to_pert = compute_single_cell_expression_changes(
        adata, split_config, n_cells_per_pert=n_cells_per_pert, seed=seed
    )
    
    # Split Y into train/test/val
    train_cells = split_labels.get("train", [])
    test_cells = split_labels.get("test", [])
    
    Y_train = Y_df[train_cells] if train_cells else pd.DataFrame()
    Y_test = Y_df[test_cells] if test_cells else pd.DataFrame()
    
    # Create cell-to-pert mappings for train and test
    cell_to_pert_train = {c: cell_to_pert[c] for c in train_cells if c in cell_to_pert}
    cell_to_pert_test = {c: cell_to_pert[c] for c in test_cells if c in cell_to_pert}
    
    LOGGER.info(f"Train: {len(train_cells)} cells, Test: {len(test_cells)} cells")
    
    # Default to core baselines for single-cell
    if baseline_types is None:
        baseline_types = [
            BaselineType.SELFTRAINED,
            BaselineType.RANDOM_GENE_EMB,
            BaselineType.RANDOM_PERT_EMB,
        ]
    
    gene_names = Y_df.index.tolist()
    
    # Get gene name mapping if available
    gene_name_mapping = None
    if "gene_name" in adata.var.columns:
        gene_name_mapping = dict(zip(adata.var_names, adata.var["gene_name"]))
    
    results = []
    
    for baseline_type in baseline_types:
        try:
            config = get_baseline_config(
                baseline_type,
                pca_dim=pca_dim,
                ridge_penalty=ridge_penalty,
                seed=seed,
            )
            
            result = run_single_baseline_single_cell(
                Y_train=Y_train,
                Y_test=Y_test,
                config=config,
                gene_names=gene_names,
                cell_to_pert_train=cell_to_pert_train,
                cell_to_pert_test=cell_to_pert_test,
                gene_name_mapping=gene_name_mapping,
                cell_embedding_method=cell_embedding_method,
            )
            
            # Compute summary metrics from perturbation-level aggregation
            pert_metrics = result["pert_metrics"]
            if pert_metrics:
                mean_r = np.mean([m["pearson_r"] for m in pert_metrics.values()])
                mean_l2 = np.mean([m["l2"] for m in pert_metrics.values()])
                
                # Also get cell-level means
                cell_metrics = result["cell_metrics"]
                cell_mean_r = np.mean([m["pearson_r"] for m in cell_metrics.values()])
                cell_mean_l2 = np.mean([m["l2"] for m in cell_metrics.values()])
            else:
                mean_r = np.nan
                mean_l2 = np.nan
                cell_mean_r = np.nan
                cell_mean_l2 = np.nan
            
            results.append({
                "baseline": baseline_type.value,
                "pert_mean_pearson_r": mean_r,
                "pert_mean_l2": mean_l2,
                "cell_mean_pearson_r": cell_mean_r,
                "cell_mean_l2": cell_mean_l2,
                "n_test_cells": result["n_test_cells"],
                "n_test_perturbations": len(pert_metrics),
            })
            
            # Save detailed results
            baseline_output_dir = output_dir / baseline_type.value
            baseline_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save cell-level metrics
            cell_metrics_df = pd.DataFrame.from_dict(result["cell_metrics"], orient="index")
            cell_metrics_df["perturbation"] = [cell_to_pert_test.get(c, "unknown") for c in cell_metrics_df.index]
            cell_metrics_df.to_csv(baseline_output_dir / "cell_metrics.csv")
            
            # Save perturbation-level metrics
            pert_metrics_df = pd.DataFrame.from_dict(result["pert_metrics"], orient="index")
            pert_metrics_df.to_csv(baseline_output_dir / "pert_metrics.csv")
            
            LOGGER.info(f"  {baseline_type.value}: cell_r={cell_mean_r:.3f}, pert_r={mean_r:.3f}")
            
        except Exception as e:
            LOGGER.error(f"Failed to run baseline {baseline_type.value}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "baseline": baseline_type.value,
                "pert_mean_pearson_r": np.nan,
                "pert_mean_l2": np.nan,
                "cell_mean_pearson_r": np.nan,
                "cell_mean_l2": np.nan,
                "n_test_cells": 0,
                "n_test_perturbations": 0,
                "error": str(e),
            })
    
    # Save summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "single_cell_baseline_summary.csv", index=False)
    
    LOGGER.info(f"\nResults saved to {output_dir}")
    LOGGER.info(f"Summary:\n{results_df.to_string()}")
    
    return results_df

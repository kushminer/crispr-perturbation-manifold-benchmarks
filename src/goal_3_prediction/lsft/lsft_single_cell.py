#!/usr/bin/env python3
"""
LSFT Single-Cell: Local Similarity-Filtered Training for single-cell data.

For each test cell c:
1. Compute similarity between c and all training cells (in cell embedding space)
2. Filter training cells to top K% most similar
3. Retrain the LPM model using only filtered training cells
4. Evaluate on test cell c
5. Compare to baseline performance (trained on all cells)

Key differences from perturbation-level LSFT:
- Similarity computed between cells (not perturbations)
- Filter training cells (not perturbations)
- May select cells from different perturbations
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from goal_2_baselines.baseline_runner import construct_gene_embeddings, construct_pert_embeddings
from goal_2_baselines.baseline_types import BaselineConfig, BaselineType, get_baseline_config
from goal_2_baselines.split_logic import load_split_config
from goal_2_baselines.single_cell_loader import (
    compute_single_cell_expression_changes,
    aggregate_cell_metrics_by_perturbation,
    compute_cell_level_pseudobulk,
)
from shared.linear_model import solve_y_axb
from shared.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def compute_cell_similarities(
    B_test: np.ndarray,
    B_train: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between all test and training cells.
    
    Args:
        B_test: Test cell embeddings (pca_dim, n_test_cells)
        B_train: Training cell embeddings (pca_dim, n_train_cells)
    
    Returns:
        Similarity matrix (n_test_cells, n_train_cells)
    """
    # Transpose to (n_samples, n_features) for cosine_similarity
    B_test_T = B_test.T  # (n_test_cells, pca_dim)
    B_train_T = B_train.T  # (n_train_cells, pca_dim)
    
    # Compute all similarities at once
    similarity_matrix = cosine_similarity(B_test_T, B_train_T)
    
    return similarity_matrix


def filter_training_cells(
    similarities: np.ndarray,
    train_cell_ids: List[str],
    top_pct: float,
) -> Tuple[List[str], List[int], np.ndarray]:
    """
    Filter training cells to top K% most similar.
    
    Args:
        similarities: Similarities to all training cells (n_train_cells,)
        train_cell_ids: List of training cell IDs
        top_pct: Top percentage to keep (e.g., 0.05 for 5%)
    
    Returns:
        Tuple of (filtered cell IDs, indices, selected similarities)
    """
    n_select = max(1, int(np.ceil(len(train_cell_ids) * top_pct)))
    
    # Get top-K indices
    top_k_indices = np.argsort(similarities)[-n_select:]
    
    # Sort by similarity (descending)
    top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
    
    filtered_ids = [train_cell_ids[i] for i in top_k_indices]
    selected_similarities = similarities[top_k_indices]
    
    return filtered_ids, list(top_k_indices), selected_similarities


def evaluate_lsft_single_cell(
    adata_path: Path,
    split_config_path: Path,
    baseline_type: BaselineType,
    dataset_name: str,
    output_dir: Path,
    top_pcts: List[float] = [0.01, 0.05, 0.10],
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
    n_cells_per_pert: int = 50,
) -> pd.DataFrame:
    """
    Evaluate LSFT on single-cell data.
    
    For each test cell and each top_pct:
    - Filter training cells to top_pct most similar
    - Retrain LPM model on filtered cells
    - Evaluate on test cell
    - Compare to baseline (trained on all cells)
    
    Args:
        adata_path: Path to adata file
        split_config_path: Path to split config JSON
        baseline_type: Baseline type
        dataset_name: Dataset name
        output_dir: Output directory
        top_pcts: List of top percentages to try
        pca_dim: PCA dimension
        ridge_penalty: Ridge penalty
        seed: Random seed
        n_cells_per_pert: Number of cells to sample per perturbation
    
    Returns:
        DataFrame with results for each test cell and top_pct
    """
    LOGGER.info(f"Running single-cell LSFT evaluation for {dataset_name}")
    LOGGER.info(f"Baseline: {baseline_type.value}")
    LOGGER.info(f"Top percentages: {top_pcts}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    adata = ad.read_h5ad(adata_path)
    split_config = load_split_config(split_config_path)
    
    # Compute single-cell Y matrix
    Y_df, split_labels, cell_to_pert = compute_single_cell_expression_changes(
        adata, split_config, n_cells_per_pert=n_cells_per_pert, seed=seed
    )
    
    # Get train/test cells
    train_cell_ids = split_labels.get("train", [])
    test_cell_ids = split_labels.get("test", [])
    
    if not train_cell_ids or not test_cell_ids:
        raise ValueError(f"No train/test splits. Train: {len(train_cell_ids)}, Test: {len(test_cell_ids)}")
    
    Y_train = Y_df[train_cell_ids]
    Y_test = Y_df[test_cell_ids]
    
    Y_train_np = Y_train.values
    Y_test_np = Y_test.values
    
    gene_names = Y_df.index.tolist()
    
    LOGGER.info(f"Train: {len(train_cell_ids)} cells, Test: {len(test_cell_ids)} cells")
    
    # Get baseline config
    config = get_baseline_config(baseline_type, pca_dim=pca_dim, ridge_penalty=ridge_penalty, seed=seed)
    
    # Construct gene embeddings (A matrix)
    gene_name_mapping = None
    if "gene_name" in adata.var.columns:
        gene_name_mapping = dict(zip(adata.var_names, adata.var["gene_name"]))
    
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
    
    # Construct cell embeddings for similarity computation
    LOGGER.info("Computing cell embeddings...")
    
    if config.pert_embedding_source == "training_data":
        # Use cell-level PCA (original behavior)
        LOGGER.info("Using cell-level PCA for embeddings")
        pca = PCA(n_components=pca_dim, random_state=seed)
        B_train = pca.fit_transform(Y_train_np.T).T  # (pca_dim, n_train)
        B_test = pca.transform(Y_test_np.T).T  # (pca_dim, n_test)
    else:
        # Use perturbation-level embeddings mapped to cells
        LOGGER.info(f"Using perturbation embeddings from {config.pert_embedding_source}")
        
        # Filter cell_to_pert for train and test cells separately
        cell_to_pert_train = {c: cell_to_pert[c] for c in train_cell_ids if c in cell_to_pert}
        cell_to_pert_test = {c: cell_to_pert[c] for c in test_cell_ids if c in cell_to_pert}
        
        # 1. Compute pseudobulk training data for embedding construction
        Y_train_pseudobulk = compute_cell_level_pseudobulk(Y_train, cell_to_pert_train)
        train_pert_names = Y_train_pseudobulk.columns.tolist()
        Y_train_pseudobulk_np = Y_train_pseudobulk.values
        
        # 2. Compute pseudobulk test data (if available)
        Y_test_pseudobulk = compute_cell_level_pseudobulk(Y_test, cell_to_pert_test)
        test_pert_names = Y_test_pseudobulk.columns.tolist()
        Y_test_pseudobulk_np = Y_test_pseudobulk.values
        
        # 3. Construct perturbation embeddings
        pert_embedding_args = config.pert_embedding_args.copy() if config.pert_embedding_args else {}
        if config.pert_embedding_source in ["k562_pca", "rpe1_pca"]:
            pert_embedding_args["target_gene_names"] = gene_names

        B_train_pert, _, _, B_test_pert, _ = construct_pert_embeddings(
            source=config.pert_embedding_source,
            train_data=Y_train_pseudobulk_np,
            pert_names=train_pert_names,
            pca_dim=pca_dim,
            seed=seed,
            embedding_args=pert_embedding_args,
            test_data=Y_test_pseudobulk_np,
            test_pert_names=test_pert_names,
        )
        
        # 4. Map perturbation embeddings to cells
        # B_train_pert is (d, n_perts), we need (d, n_cells)
        B_train = np.zeros((pca_dim, len(train_cell_ids)))
        for i, cell_id in enumerate(train_cell_ids):
            pert = cell_to_pert[cell_id]
            if pert in train_pert_names:
                pert_idx = train_pert_names.index(pert)
                B_train[:, i] = B_train_pert[:, pert_idx]
            else:
                LOGGER.warning(f"Training cell {cell_id} has unknown perturbation {pert}")
                
        B_test = np.zeros((pca_dim, len(test_cell_ids)))
        for i, cell_id in enumerate(test_cell_ids):
            pert = cell_to_pert[cell_id]
            # Check if in test perts (it should be)
            if pert in test_pert_names and B_test_pert is not None:
                pert_idx = test_pert_names.index(pert)
                B_test[:, i] = B_test_pert[:, pert_idx]
            # Fallback: check if in train perts (e.g. if test cell is from a training perturbation)
            elif pert in train_pert_names:
                pert_idx = train_pert_names.index(pert)
                B_test[:, i] = B_train_pert[:, pert_idx]
            else:
                LOGGER.warning(f"Test cell {cell_id} has unknown perturbation {pert}")
    
    # Compute baseline model (trained on all cells)
    LOGGER.info("Computing baseline predictions (all training cells)...")
    center_baseline = Y_train_np.mean(axis=1, keepdims=True)
    Y_centered_baseline = Y_train_np - center_baseline
    
    solution_baseline = solve_y_axb(
        Y=Y_centered_baseline,
        A=A,
        B=B_train,
        ridge_penalty=ridge_penalty,
    )
    K_baseline = solution_baseline["K"]
    Y_pred_baseline = A @ K_baseline @ B_test + center_baseline
    
    # Compute all similarities at once
    LOGGER.info("Computing cell similarities...")
    all_similarities = compute_cell_similarities(B_test, B_train)
    
    # Create mapping from cell ID to index
    train_cell_to_idx = {c: i for i, c in enumerate(train_cell_ids)}
    
    # Collect results
    results = []
    
    # For each test cell
    n_test = len(test_cell_ids)
    log_interval = max(1, n_test // 10)
    
    for test_idx, test_cell_id in enumerate(test_cell_ids):
        if test_idx % log_interval == 0:
            LOGGER.info(f"  Processing test cell {test_idx+1}/{n_test}")
        
        test_pert = cell_to_pert.get(test_cell_id, "unknown")
        y_true = Y_test_np[:, test_idx]
        
        # Baseline prediction
        y_pred_baseline = Y_pred_baseline[:, test_idx]
        baseline_metrics = compute_metrics(y_true, y_pred_baseline)
        
        # Get similarities for this test cell
        similarities = all_similarities[test_idx]
        
        # For each top_pct
        for top_pct in top_pcts:
            # Filter training cells
            filtered_ids, filtered_indices, selected_sims = filter_training_cells(
                similarities, train_cell_ids, top_pct
            )
            
            # Get filtered training data
            Y_train_filtered = Y_train_np[:, filtered_indices]
            B_train_filtered = B_train[:, filtered_indices]
            
            # Retrain model on filtered cells
            # IMPORTANT: Use the GLOBAL center (center_baseline) for fair comparison
            # Both baseline and LSFT should use the same centering reference
            Y_centered = Y_train_filtered - center_baseline
            
            solution = solve_y_axb(
                Y=Y_centered,
                A=A,
                B=B_train_filtered,
                ridge_penalty=ridge_penalty,
            )
            K = solution["K"]
            
            # Predict test cell (use global center for consistency with baseline)
            test_cell_embedding = B_test[:, test_idx:test_idx+1]
            y_pred = (A @ K @ test_cell_embedding + center_baseline).flatten()
            
            # Compute metrics
            lsft_metrics = compute_metrics(y_true, y_pred)
            
            results.append({
                "cell_id": test_cell_id,
                "perturbation": test_pert,
                "top_pct": top_pct,
                "n_neighbors": len(filtered_indices),
                "mean_similarity": float(selected_sims.mean()),
                "baseline_pearson_r": baseline_metrics["pearson_r"],
                "baseline_l2": baseline_metrics["l2"],
                "lsft_pearson_r": lsft_metrics["pearson_r"],
                "lsft_l2": lsft_metrics["l2"],
                "delta_r": lsft_metrics["pearson_r"] - baseline_metrics["pearson_r"],
                "delta_l2": lsft_metrics["l2"] - baseline_metrics["l2"],
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save cell-level results
    results_df.to_csv(output_dir / f"lsft_single_cell_{dataset_name}_{baseline_type.value}.csv", index=False)
    
    # Compute summary statistics
    summary = []
    for top_pct in top_pcts:
        pct_results = results_df[results_df["top_pct"] == top_pct]
        
        # Aggregate by perturbation
        pert_summary = pct_results.groupby("perturbation").agg({
            "baseline_pearson_r": "mean",
            "lsft_pearson_r": "mean",
            "delta_r": "mean",
            "baseline_l2": "mean",
            "lsft_l2": "mean",
            "mean_similarity": "mean",
        }).reset_index()
        
        summary.append({
            "baseline_type": baseline_type.value,
            "top_pct": top_pct,
            "n_test_cells": len(pct_results),
            "n_test_perts": len(pct_results["perturbation"].unique()),
            "cell_mean_baseline_r": pct_results["baseline_pearson_r"].mean(),
            "cell_mean_lsft_r": pct_results["lsft_pearson_r"].mean(),
            "cell_mean_delta_r": pct_results["delta_r"].mean(),
            "pert_mean_baseline_r": pert_summary["baseline_pearson_r"].mean(),
            "pert_mean_lsft_r": pert_summary["lsft_pearson_r"].mean(),
            "pert_mean_delta_r": pert_summary["delta_r"].mean(),
            "mean_similarity": pct_results["mean_similarity"].mean(),
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / f"lsft_single_cell_summary_{dataset_name}_{baseline_type.value}.csv", index=False)
    
    LOGGER.info(f"\nSingle-cell LSFT Summary ({dataset_name}, {baseline_type.value}):")
    LOGGER.info(f"\n{summary_df.to_string()}")
    
    return results_df


def run_lsft_single_cell_all_baselines(
    adata_path: Path,
    split_config_path: Path,
    output_dir: Path,
    dataset_name: str,
    baseline_types: Optional[List[BaselineType]] = None,
    top_pcts: List[float] = [0.01, 0.05, 0.10],
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
    n_cells_per_pert: int = 50,
) -> pd.DataFrame:
    """
    Run single-cell LSFT for all baseline types.
    
    Args:
        adata_path: Path to adata file
        split_config_path: Path to split config JSON
        output_dir: Output directory
        dataset_name: Dataset name
        baseline_types: List of baseline types (None = core baselines)
        top_pcts: List of top percentages
        pca_dim: PCA dimension
        ridge_penalty: Ridge penalty
        seed: Random seed
        n_cells_per_pert: Number of cells per perturbation
    
    Returns:
        DataFrame with all results
    """
    if baseline_types is None:
        baseline_types = [
            BaselineType.SELFTRAINED,
            BaselineType.RANDOM_GENE_EMB,
            BaselineType.RANDOM_PERT_EMB,
        ]
    
    all_results = []
    
    for baseline_type in baseline_types:
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"Running LSFT single-cell for {baseline_type.value}")
        LOGGER.info(f"{'='*60}")
        
        try:
            results = evaluate_lsft_single_cell(
                adata_path=adata_path,
                split_config_path=split_config_path,
                baseline_type=baseline_type,
                dataset_name=dataset_name,
                output_dir=output_dir,
                top_pcts=top_pcts,
                pca_dim=pca_dim,
                ridge_penalty=ridge_penalty,
                seed=seed,
                n_cells_per_pert=n_cells_per_pert,
            )
            results["baseline_type"] = baseline_type.value
            all_results.append(results)
        except Exception as e:
            LOGGER.error(f"Failed {baseline_type.value}: {e}")
            import traceback
            traceback.print_exc()
    
    if all_results:
        all_results_df = pd.concat(all_results, ignore_index=True)
        all_results_df.to_csv(output_dir / f"lsft_single_cell_all_{dataset_name}.csv", index=False)
        return all_results_df
    else:
        return pd.DataFrame()


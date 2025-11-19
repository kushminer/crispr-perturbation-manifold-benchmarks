#!/usr/bin/env python3
"""
LSFT: Local Similarity-Filtered Training for perturbation prediction.

For each test perturbation p:
1. Compute similarity between p and all training perturbations (in B embedding space)
2. Filter training perturbations to top K% most similar (1%, 5%, 10%)
3. Retrain the full LPM model (Y = A·K·B) using only filtered training perturbations
4. Evaluate on test perturbation p
5. Compare to baseline performance (trained on all perturbations)

Key points:
- Similarity computed between perturbations (not genes) using B matrix
- Filter training perturbations (not genes)
- Retrain full LPM model per test perturbation
- Same architecture and hyperparameters as baseline
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from goal_2_baselines.baseline_runner import (
    compute_pseudobulk_expression_changes,
    construct_gene_embeddings,
    construct_pert_embeddings,
)
from goal_2_baselines.baseline_types import BaselineConfig, BaselineType, get_baseline_config
from goal_2_baselines.split_logic import load_split_config
from shared.linear_model import solve_y_axb
from shared.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def compute_perturbation_similarities(
    test_pert_embedding: np.ndarray,
    train_pert_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between test perturbation and all training perturbations.
    
    Args:
        test_pert_embedding: Test perturbation embedding (d,) or (1, d)
        train_pert_embeddings: Training perturbation embeddings (d, n_train_perts) or (n_train_perts, d)
    
    Returns:
        Similarities array (n_train_perts,)
    """
    # Ensure test_pert_embedding is 2D (1, d)
    if test_pert_embedding.ndim == 1:
        test_pert_embedding = test_pert_embedding.reshape(1, -1)
    
    # Handle train_pert_embeddings shape
    # B_train is typically (d, n_train_perts), but cosine_similarity expects (n_samples, n_features)
    if train_pert_embeddings.ndim == 1:
        train_pert_embeddings = train_pert_embeddings.reshape(1, -1)
    
    # Check if we need to transpose
    if train_pert_embeddings.shape[0] < train_pert_embeddings.shape[1]:
        # Likely (d, n_train_perts) format, transpose to (n_train_perts, d)
        train_pert_embeddings = train_pert_embeddings.T
    
    # Compute cosine similarity
    similarities = cosine_similarity(test_pert_embedding, train_pert_embeddings).flatten()
    
    return similarities


def compute_all_perturbation_similarities(
    B_test: np.ndarray,
    B_train: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between all test and training perturbations at once (OPTIMIZED).
    
    Args:
        B_test: Test perturbation embeddings (d, n_test_perts)
        B_train: Training perturbation embeddings (d, n_train_perts)
    
    Returns:
        Similarity matrix (n_test_perts, n_train_perts)
    """
    # Transpose to (n_samples, n_features) for cosine_similarity
    B_test_T = B_test.T  # (n_test_perts, d)
    B_train_T = B_train.T  # (n_train_perts, d)
    
    # Compute all similarities at once
    similarity_matrix = cosine_similarity(B_test_T, B_train_T)  # (n_test_perts, n_train_perts)
    
    return similarity_matrix


def filter_training_perturbations(
    similarities: np.ndarray,
    train_pert_names: List[str],
    top_pct: float = 0.05,
) -> Tuple[List[str], np.ndarray]:
    """
    Filter training perturbations to top K% most similar.
    
    Args:
        similarities: Similarities to all training perturbations (n_train_perts,)
        train_pert_names: List of training perturbation names
        top_pct: Top percentage to keep (e.g., 0.01 for 1%, 0.05 for 5%, 0.10 for 10%)
    
    Returns:
        Tuple of (filtered perturbation names, selected similarities)
    """
    n_select = max(1, int(np.ceil(len(train_pert_names) * top_pct)))
    
    # Get top-K indices
    top_k_indices = np.argsort(similarities)[-n_select:]
    
    # Sort by similarity (descending)
    top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
    
    filtered_names = [train_pert_names[i] for i in top_k_indices]
    selected_similarities = similarities[top_k_indices]
    
    return filtered_names, selected_similarities


def retrain_lpm_on_filtered_data(
    Y_train_filtered: pd.DataFrame,
    config: BaselineConfig,
    gene_names: List[str],
    pca_dim: int,
    ridge_penalty: float,
    seed: int,
    gene_name_mapping: Optional[Dict[str, str]] = None,
    test_pert_data: Optional[np.ndarray] = None,
    # OPTIMIZATION: Cached components to avoid recomputation
    A_baseline_cached: Optional[np.ndarray] = None,
    B_train_baseline: Optional[np.ndarray] = None,
    train_pert_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Retrain LPM model on filtered training data (OPTIMIZED).
    
    Optimizations:
    - Reuse cached gene embeddings (A) for pretrained embeddings (scGPT, scFoundation)
    - For cross-dataset B embeddings: subset existing B_train_baseline instead of recomputing
    - Only recompute what's necessary
    
    Args:
        Y_train_filtered: Filtered training Y matrix (genes × filtered_train_perturbations)
        config: Baseline configuration
        gene_names: List of gene names
        pca_dim: PCA dimension
        ridge_penalty: Ridge penalty
        seed: Random seed
        gene_name_mapping: Optional gene name mapping
        test_pert_data: Optional test perturbation data (genes × 1) for transforming in local space
        A_baseline_cached: Cached gene embeddings from baseline (for pretrained embeddings)
        B_train_baseline: Baseline B_train matrix (for cross-dataset embeddings)
        train_pert_indices: Indices of filtered perturbations in original train set
    
    Returns:
        Tuple of (A, K, B_train, center, B_test)
        A: Gene embeddings (genes × d_g)
        K: Interaction matrix (d_g × d_p)
        B_train: Perturbation embeddings (d_p × filtered_train_perts)
        center: Center vector (genes,)
        B_test: Test perturbation embedding in local space (d_p × 1) or None
    """
    Y_train_filtered_np = Y_train_filtered.values  # (genes, filtered_train_perts)
    
    # Adjust PCA dimension based on filtered training size
    n_filtered_perts = Y_train_filtered_np.shape[1]
    n_genes = Y_train_filtered_np.shape[0]
    
    effective_gene_pca_dim = min(pca_dim, n_genes, n_filtered_perts)
    effective_pert_pca_dim = min(pca_dim, n_filtered_perts, n_genes)
    
    # OPTIMIZATION 1: Reuse cached gene embeddings for pretrained embeddings
    if config.gene_embedding_source in ["scgpt", "scfoundation"] and A_baseline_cached is not None:
        # Pretrained embeddings don't change with filtered data - reuse!
        A = A_baseline_cached
    else:
        # Need to recompute (training_data, random, or no cache available)
        gene_embedding_args = config.gene_embedding_args.copy() if config.gene_embedding_args else {}
        if gene_name_mapping and config.gene_embedding_source in ["scgpt", "scfoundation"]:
            gene_embedding_args["gene_name_mapping"] = gene_name_mapping
        
        A, gene_names_A = construct_gene_embeddings(
            source=config.gene_embedding_source,
            train_data=Y_train_filtered_np,
            gene_names=gene_names,
            pca_dim=effective_gene_pca_dim,
            seed=seed,
            embedding_args=gene_embedding_args,
        )
        
        # Ensure A aligns with gene_names
        if gene_names_A != gene_names:
            gene_idx_map = {g: i for i, g in enumerate(gene_names_A)}
            A_aligned = np.zeros((len(gene_names), A.shape[1]))
            for i, g in enumerate(gene_names):
                if g in gene_idx_map:
                    A_aligned[i, :] = A[gene_idx_map[g], :]
            A = A_aligned
    
    # OPTIMIZATION 2: For cross-dataset embeddings, subset existing B_train_baseline
    if config.pert_embedding_source in ["k562_pca", "rpe1_pca", "gears"] and B_train_baseline is not None and train_pert_indices is not None:
        # Cross-dataset embeddings are fixed - just subset!
        B_train = B_train_baseline[:, train_pert_indices]  # (d, filtered_train_perts)
        # For test perturbation, use baseline embedding (cross-dataset space is fixed)
        B_test = None
    else:
        # Need to recompute (training_data, random, or no cache available)
        pert_embedding_args = config.pert_embedding_args.copy() if config.pert_embedding_args else {}
        
        if config.pert_embedding_source in ["k562_pca", "rpe1_pca", "file"]:
            pert_embedding_args["target_gene_names"] = gene_names
        
        B_train, pert_labels_train, pert_pca, B_test, _ = construct_pert_embeddings(
            source=config.pert_embedding_source,
            train_data=Y_train_filtered_np,
            pert_names=Y_train_filtered.columns.tolist(),
            pca_dim=effective_pert_pca_dim,
            seed=seed,
            embedding_args=pert_embedding_args,
            test_data=test_pert_data,
            test_pert_names=["test"] if test_pert_data is not None else None,
        )
    
    # Center Y
    center = Y_train_filtered_np.mean(axis=1, keepdims=True)  # (genes, 1)
    Y_centered = Y_train_filtered_np - center
    
    # Solve for K: Y = A @ K @ B_train
    solution = solve_y_axb(
        Y=Y_centered,
        A=A,
        B=B_train,
        ridge_penalty=ridge_penalty,
    )
    K = solution["K"]
    
    return A, K, B_train, center[:, 0], B_test  # Flatten center


def evaluate_lsft(
    adata_path: Path,
    split_config_path: Path,
    baseline_type: BaselineType,
    dataset_name: str,
    output_dir: Path,
    top_pcts: List[float] = [0.01, 0.05, 0.10],
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
) -> pd.DataFrame:
    """
    Evaluate LSFT: Local Similarity-Filtered Training.
    
    For each test perturbation and each top_pct:
    - Filter training perturbations to top_pct most similar
    - Retrain LPM model on filtered data
    - Evaluate on test perturbation
    - Compare to baseline (trained on all data)
    
    Args:
        adata_path: Path to adata file
        split_config_path: Path to split config JSON
        baseline_type: Baseline type (determines embedding space)
        dataset_name: Dataset name
        output_dir: Output directory
        top_pcts: List of top percentages to try (e.g., [0.01, 0.05, 0.10])
        pca_dim: PCA dimension
        ridge_penalty: Ridge penalty
        seed: Random seed
    
    Returns:
        DataFrame with results for each test perturbation and top_pct
    """
    LOGGER.info(f"Running LSFT evaluation for {dataset_name}")
    LOGGER.info(f"Baseline: {baseline_type.value}")
    LOGGER.info(f"Top percentages: {top_pcts}")
    
    # Load data
    adata = ad.read_h5ad(adata_path)
    split_config = load_split_config(split_config_path)
    
    # Compute Y matrix
    Y_df, split_labels = compute_pseudobulk_expression_changes(adata, split_config, seed=seed)
    
    # Get splits
    train_pert_names = split_labels.get("train", [])
    test_pert_names = split_labels.get("test", [])
    
    if not train_pert_names or not test_pert_names:
        raise ValueError(f"No train/test splits found. Train: {len(train_pert_names)}, Test: {len(test_pert_names)}")
    
    # Get Y_train and Y_test
    Y_train = Y_df[train_pert_names]  # (genes, train_perts)
    Y_test = Y_df[test_pert_names]    # (genes, test_perts)
    
    gene_names = Y_df.index.tolist()
    LOGGER.info(f"Loaded {len(gene_names)} genes, {len(train_pert_names)} train perts, {len(test_pert_names)} test perts")
    
    # Get gene name mapping if available (for embeddings that use gene symbols)
    gene_name_mapping = None
    if "gene_name" in adata.var.columns:
        gene_name_mapping = dict(zip(adata.var_names, adata.var["gene_name"]))
        LOGGER.info(f"Found gene_name column: {len(gene_name_mapping)} mappings")
    
    # Get baseline config
    config = get_baseline_config(baseline_type, pca_dim=pca_dim, ridge_penalty=ridge_penalty, seed=seed)
    
    # Construct baseline model (for comparison and to get B matrices)
    LOGGER.info("Constructing baseline model for comparison...")
    Y_train_np = Y_train.values
    
    gene_embedding_args = config.gene_embedding_args.copy() if config.gene_embedding_args else {}
    if gene_name_mapping and config.gene_embedding_source in ["scgpt", "scfoundation"]:
        gene_embedding_args["gene_name_mapping"] = gene_name_mapping
    
    A_baseline, gene_names_A = construct_gene_embeddings(
        source=config.gene_embedding_source,
        train_data=Y_train_np,
        gene_names=gene_names,
        pca_dim=pca_dim,
        seed=seed,
        embedding_args=gene_embedding_args,
    )
    
    if gene_names_A != gene_names:
        gene_idx_map = {g: i for i, g in enumerate(gene_names_A)}
        A_baseline_aligned = np.zeros((len(gene_names), A_baseline.shape[1]))
        for i, g in enumerate(gene_names):
            if g in gene_idx_map:
                A_baseline_aligned[i, :] = A_baseline[gene_idx_map[g], :]
        A_baseline = A_baseline_aligned
    
    pert_embedding_args = config.pert_embedding_args.copy() if config.pert_embedding_args else {}
    
    # For cross-dataset PCA embeddings, need to provide target_gene_names
    if config.pert_embedding_source in ["k562_pca", "rpe1_pca", "file"]:
        pert_embedding_args["target_gene_names"] = gene_names
    
    B_train_baseline, _, _, B_test_baseline, _ = construct_pert_embeddings(
        source=config.pert_embedding_source,
        train_data=Y_train_np,
        pert_names=train_pert_names,
        pca_dim=pca_dim,
        seed=seed,
        embedding_args=pert_embedding_args,
        test_data=Y_test.values if not Y_test.empty else None,
        test_pert_names=test_pert_names if not Y_test.empty else None,
    )
    
    # Solve baseline model
    center_baseline = Y_train_np.mean(axis=1, keepdims=True)
    Y_centered_baseline = Y_train_np - center_baseline
    
    solution_baseline = solve_y_axb(
        Y=Y_centered_baseline,
        A=A_baseline,
        B=B_train_baseline,
        ridge_penalty=ridge_penalty,
    )
    K_baseline = solution_baseline["K"]
    
    # Get baseline predictions for comparison
    Y_pred_baseline = A_baseline @ K_baseline @ B_test_baseline + center_baseline
    
    # OPTIMIZATION 3: Pre-compute all similarities at once (much faster!)
    LOGGER.info("Pre-computing all perturbation similarities...")
    all_similarities = compute_all_perturbation_similarities(B_test_baseline, B_train_baseline)
    # all_similarities is (n_test_perts, n_train_perts)
    
    # Create mapping from train perturbation names to indices (for subsetting B_train_baseline)
    train_pert_name_to_idx = {name: i for i, name in enumerate(train_pert_names)}
    
    # For each test perturbation
    results = []
    
    for test_idx, test_pert_name in enumerate(test_pert_names):
        if test_idx % 10 == 0:
            LOGGER.info(f"Processing test perturbation {test_idx}/{len(test_pert_names)}: {test_pert_name}")
        
        # Get test perturbation embedding
        test_pert_embedding = B_test_baseline[:, test_idx]  # (d,)
        
        # Get similarities for this test perturbation (pre-computed)
        similarities = all_similarities[test_idx, :]  # (n_train_perts,)
        
        # Get baseline performance for this perturbation
        y_true = Y_test.loc[:, test_pert_name].values
        y_pred_baseline = Y_pred_baseline[:, test_idx]
        baseline_metrics = compute_metrics(y_true, y_pred_baseline)
        
        # For each top_pct
        for top_pct in top_pcts:
            # Filter training perturbations
            filtered_train_names, selected_similarities = filter_training_perturbations(
                similarities, train_pert_names, top_pct=top_pct
            )
            
            if len(filtered_train_names) == 0:
                LOGGER.warning(f"No training perturbations selected for {test_pert_name} at {top_pct*100}%")
                continue
            
            # Get indices of filtered perturbations in original train set (for subsetting B_train_baseline)
            train_pert_indices = [train_pert_name_to_idx[name] for name in filtered_train_names]
            
            # Retrain LPM on filtered data
            try:
                Y_train_filtered = Y_train[filtered_train_names]
                
                # Retrain model on filtered training data
                # Get test perturbation expression data for transformation in local space
                # This is needed for self-trained embeddings where PCA is retrained
                test_pert_data = Y_test.loc[:, test_pert_name].values.reshape(-1, 1)  # (genes, 1)
                
                A_local, K_local, B_train_local, center_local, B_test_local = retrain_lpm_on_filtered_data(
                    Y_train_filtered=Y_train_filtered,
                    config=config,
                    gene_names=gene_names,
                    pca_dim=pca_dim,
                    ridge_penalty=ridge_penalty,
                    seed=seed,
                    gene_name_mapping=gene_name_mapping,
                    test_pert_data=test_pert_data,  # Transform test pert in local space
                    # OPTIMIZATION: Pass cached components
                    A_baseline_cached=A_baseline if config.gene_embedding_source in ["scgpt", "scfoundation"] else None,
                    B_train_baseline=B_train_baseline if config.pert_embedding_source in ["k562_pca", "rpe1_pca", "gears"] else None,
                    train_pert_indices=train_pert_indices if config.pert_embedding_source in ["k562_pca", "rpe1_pca", "gears"] else None,
                )
                
                # For prediction, use local test embedding if available (for self-trained embeddings)
                # Otherwise use baseline embedding (for cross-dataset embeddings like GEARS, K562)
                if B_test_local is not None:
                    # Local embedding was computed (self-trained case)
                    test_pert_embedding_local = B_test_local[:, 0]  # (d_local,)
                    
                    # Check dimension compatibility
                    if K_local.shape[1] != len(test_pert_embedding_local):
                        LOGGER.warning(
                            f"Dimension mismatch for {test_pert_name} at {top_pct*100}%: "
                            f"K_local shape {K_local.shape}, B_test_local shape {B_test_local.shape}"
                        )
                        # Skip this case
                        continue
                else:
                    # For cross-dataset embeddings, use baseline embedding
                    # But check if dimensions match
                    if K_local.shape[1] != test_pert_embedding.shape[0]:
                        LOGGER.warning(
                            f"Dimension mismatch for {test_pert_name} at {top_pct*100}%: "
                            f"K_local shape {K_local.shape}, test_pert_embedding shape {test_pert_embedding.shape}"
                        )
                        # Skip this case
                        continue
                    test_pert_embedding_local = test_pert_embedding
                
                # Predict using local model
                y_pred_local = A_local @ K_local @ test_pert_embedding_local.reshape(-1, 1) + center_local.reshape(-1, 1)
                y_pred_local = y_pred_local.flatten()
                
                # Compute metrics
                local_metrics = compute_metrics(y_true, y_pred_local)
                
                # Store results
                result_row = {
                    "test_perturbation": test_pert_name,
                    "top_pct": top_pct,
                    "local_train_size": len(filtered_train_names),
                    "local_mean_similarity": float(np.mean(selected_similarities)),
                    "local_max_similarity": float(np.max(selected_similarities)),
                    "local_min_similarity": float(np.min(selected_similarities)),
                    "performance_local_pearson_r": local_metrics["pearson_r"],
                    "performance_local_l2": local_metrics["l2"],
                    "performance_baseline_pearson_r": baseline_metrics["pearson_r"],
                    "performance_baseline_l2": baseline_metrics["l2"],
                    "improvement_pearson_r": local_metrics["pearson_r"] - baseline_metrics["pearson_r"],
                    "improvement_l2": baseline_metrics["l2"] - local_metrics["l2"],  # Positive = improvement
                }
                results.append(result_row)
                
            except Exception as e:
                LOGGER.warning(f"Error processing {test_pert_name} at {top_pct*100}%: {e}")
                # Store error row
                result_row = {
                    "test_perturbation": test_pert_name,
                    "top_pct": top_pct,
                    "local_train_size": len(filtered_train_names),
                    "local_mean_similarity": float(np.mean(selected_similarities)),
                    "local_max_similarity": float(np.max(selected_similarities)),
                    "local_min_similarity": float(np.min(selected_similarities)),
                    "performance_local_pearson_r": np.nan,
                    "performance_local_l2": np.nan,
                    "performance_baseline_pearson_r": baseline_metrics["pearson_r"],
                    "performance_baseline_l2": baseline_metrics["l2"],
                    "improvement_pearson_r": np.nan,
                    "improvement_l2": np.nan,
                }
                results.append(result_row)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Add baseline_type column
    results_df["baseline_type"] = baseline_type.value
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"lsft_{dataset_name}_{baseline_type.value}.csv"
    results_df.to_csv(output_path, index=False)
    LOGGER.info(f"Saved results to {output_path}")
    
    # Print summary
    LOGGER.info(f"\nSummary for {dataset_name}:")
    for top_pct in top_pcts:
        subset = results_df[results_df["top_pct"] == top_pct]
        if len(subset) > 0:
            LOGGER.info(f"\nTop {top_pct*100}%:")
            LOGGER.info(f"  Mean local train size: {subset['local_train_size'].mean():.1f}")
            LOGGER.info(f"  Mean local similarity: {subset['local_mean_similarity'].mean():.4f}")
            LOGGER.info(f"  Mean baseline Pearson r: {subset['performance_baseline_pearson_r'].mean():.4f}")
            LOGGER.info(f"  Mean local Pearson r: {subset['performance_local_pearson_r'].mean():.4f}")
            LOGGER.info(f"  Mean improvement Pearson r: {subset['improvement_pearson_r'].mean():.4f}")
            LOGGER.info(f"  Mean improvement L2: {subset['improvement_l2'].mean():.4f}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="LSFT: Local Similarity-Filtered Training for perturbation prediction"
    )
    parser.add_argument("--adata_path", type=Path, required=True, help="Path to adata file")
    parser.add_argument("--split_config", type=Path, required=True, help="Path to split config JSON")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--baseline_type", type=str, required=True, help="Baseline type")
    parser.add_argument(
        "--top_pcts",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.10],
        help="Top percentages to use (e.g., 0.01 0.05 0.10 for 1%%, 5%%, 10%%)",
    )
    parser.add_argument("--pca_dim", type=int, default=10, help="PCA dimension")
    parser.add_argument("--ridge_penalty", type=float, default=0.1, help="Ridge penalty")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    
    args = parser.parse_args()
    
    baseline_type = BaselineType(args.baseline_type)
    
    results_df = evaluate_lsft(
        adata_path=args.adata_path,
        split_config_path=args.split_config,
        baseline_type=baseline_type,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        top_pcts=args.top_pcts,
        pca_dim=args.pca_dim,
        ridge_penalty=args.ridge_penalty,
        seed=args.seed,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())

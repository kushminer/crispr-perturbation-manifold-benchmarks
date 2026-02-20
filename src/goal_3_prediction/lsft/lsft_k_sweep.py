"""
LSFT with k-list support for curvature sweep analysis.

This module extends LSFT to support exact k counts (neighborhood sizes)
instead of just percentages, enabling curvature sweep analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd

from goal_2_baselines.baseline_runner import (
    compute_pseudobulk_expression_changes,
    construct_gene_embeddings,
    construct_pert_embeddings,
)
from goal_2_baselines.baseline_types import BaselineConfig, BaselineType, get_baseline_config
from goal_2_baselines.split_logic import load_split_config
from shared.linear_model import solve_y_axb
from shared.metrics import compute_metrics

from goal_3_prediction.lsft.lsft import (
    compute_all_perturbation_similarities,
    retrain_lpm_on_filtered_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def filter_training_perturbations_by_k(
    similarities: np.ndarray,
    train_pert_names: List[str],
    k: int,
    functional_annotations: Optional[Dict[str, str]] = None,
    test_pert_name: Optional[str] = None,
    remove_same_class: bool = False,
) -> Tuple[List[str], np.ndarray]:
    """
    Filter training perturbations to top k most similar (exact count).
    Optionally remove neighbors with same functional class (for mechanism ablation).
    
    Args:
        similarities: Similarities to all training perturbations (n_train_perts,)
        train_pert_names: List of training perturbation names
        k: Exact number of neighbors to keep
        functional_annotations: Optional dict mapping perturbation names to functional classes
        test_pert_name: Optional test perturbation name (needed for same-class removal)
        remove_same_class: If True, remove neighbors with same functional class as test
    
    Returns:
        Tuple of (filtered perturbation names, selected similarities)
    """
    n_select = min(k, len(train_pert_names))
    
    # Get top-K indices
    top_k_indices = np.argsort(similarities)[-n_select:]
    
    # Sort by similarity (descending)
    top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
    
    # Apply functional class filtering if requested
    if remove_same_class and functional_annotations and test_pert_name:
        test_class = functional_annotations.get(test_pert_name, None)
        if test_class is not None:
            filtered_indices = []
            for idx in top_k_indices:
                train_pert = train_pert_names[idx]
                train_class = functional_annotations.get(train_pert, None)
                if train_class != test_class:
                    filtered_indices.append(idx)
            
            if len(filtered_indices) > 0:
                top_k_indices = np.array(filtered_indices)
            else:
                LOGGER.warning(f"All neighbors removed for {test_pert_name}, keeping original top-K")
    
    filtered_names = [train_pert_names[i] for i in top_k_indices]
    selected_similarities = similarities[top_k_indices]
    
    return filtered_names, selected_similarities


def evaluate_lsft_with_k_list(
    adata_path: Path,
    split_config_path: Path,
    baseline_type: BaselineType,
    dataset_name: str,
    output_dir: Path,
    k_list: List[int] = [3, 5, 10, 20, 50, 100],
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
    functional_annotations: Optional[Dict[str, str]] = None,
    remove_same_class: bool = False,
    # EPIC 3: Noise injection parameters
    noise_level: float = 0.0,
    noise_type: str = "none",
    noise_target: str = "embedding",
) -> pd.DataFrame:
    """
    Evaluate LSFT with exact k values (for curvature sweep).
    
    This is similar to evaluate_lsft but uses exact k counts instead of percentages.
    
    Args:
        adata_path: Path to adata file
        split_config_path: Path to split config JSON
        baseline_type: Baseline type (determines embedding space)
        dataset_name: Dataset name
        output_dir: Output directory
        k_list: List of exact k values (neighborhood sizes) to try
        pca_dim: PCA dimension
        ridge_penalty: Ridge penalty
        seed: Random seed
    
    Returns:
        DataFrame with results for each test perturbation and k value
    """
    LOGGER.info(f"Running LSFT evaluation with k-list for {dataset_name}")
    LOGGER.info(f"Baseline: {baseline_type.value}")
    LOGGER.info(f"K values: {k_list}")
    
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
    
    # Get gene name mapping if available
    gene_name_mapping = None
    if "gene_name" in adata.var.columns:
        gene_name_mapping = dict(zip(adata.var_names, adata.var["gene_name"]))
    
    # Get baseline config
    config = get_baseline_config(baseline_type, pca_dim=pca_dim, ridge_penalty=ridge_penalty, seed=seed)
    
    # Construct baseline model
    LOGGER.info("Constructing baseline model...")
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
    
    if config.pert_embedding_source in ["k562_pca", "rpe1_pca", "file"]:
        pert_embedding_args["target_gene_names"] = gene_names
    
    B_train_baseline, _, _, B_test_baseline, _ = construct_pert_embeddings(
        source=config.pert_embedding_source,
        train_data=Y_train_np,
        pert_names=train_pert_names,
        pca_dim=pca_dim,
        seed=seed,
        embedding_args=pert_embedding_args,
        test_data=Y_test.values,
        test_pert_names=test_pert_names,
    )
    
    # Solve baseline model
    center_baseline = Y_train_np.mean(axis=1, keepdims=True)
    Y_train_centered = Y_train_np - center_baseline
    
    solution_baseline = solve_y_axb(Y_train_centered, A_baseline, B_train_baseline, ridge_penalty=ridge_penalty)
    K_baseline = solution_baseline["K"]
    
    # Diagnostic: Check if embeddings loaded correctly
    if B_test_baseline is None:
        LOGGER.error(
            f"⚠️  CRITICAL: B_test_baseline is None for baseline {baseline_type.value} "
            f"on dataset {dataset_name}. Test embeddings failed to load."
        )
        # Return empty DataFrame - can't proceed without test embeddings
        results_df = pd.DataFrame()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"lsft_k_sweep_{dataset_name}_{baseline_type.value}.csv"
        results_df.to_csv(output_path, index=False)
        return results_df
    
    if B_test_baseline.shape[1] != len(test_pert_names):
        LOGGER.warning(
            f"⚠️  Test embedding shape mismatch: B_test_baseline has {B_test_baseline.shape[1]} columns, "
            f"but there are {len(test_pert_names)} test perturbations. Some test perts may have no embeddings."
        )
        # Count how many have non-zero embeddings
        non_zero_perts = np.sum(np.any(B_test_baseline != 0, axis=0))
        LOGGER.info(f"Test perturbations with non-zero embeddings: {non_zero_perts}/{len(test_pert_names)}")
    
    # Baseline predictions
    Y_pred_baseline = A_baseline @ K_baseline @ B_test_baseline + center_baseline
    
    # Pre-compute all similarities (optimization)
    LOGGER.info("Computing perturbation similarities...")
    all_similarities = compute_all_perturbation_similarities(B_test_baseline, B_train_baseline)
    
    # Create mapping for train perturbations
    train_pert_name_to_idx = {name: i for i, name in enumerate(train_pert_names)}
    
    # For each test perturbation and k value
    results = []
    
    for test_idx, test_pert_name in enumerate(test_pert_names):
        if test_idx % 10 == 0:
            LOGGER.info(f"Processing test perturbation {test_idx}/{len(test_pert_names)}: {test_pert_name}")
        
        # Get test perturbation embedding
        test_pert_embedding = B_test_baseline[:, test_idx]  # (d,)
        
        # Diagnostic: Check if test perturbation has valid embedding
        if np.all(test_pert_embedding == 0):
            LOGGER.warning(
                f"⚠️  Test perturbation {test_pert_name} has all-zero embedding. "
                f"This suggests it's not in the embedding vocabulary (e.g., not in GEARS GO graph)."
            )
            # Skip this perturbation - can't compute meaningful similarity or predictions
            continue
        
        # Get similarities for this test perturbation
        similarities = all_similarities[test_idx, :]  # (n_train_perts,)
        
        # Get baseline performance for this perturbation
        y_true = Y_test.loc[:, test_pert_name].values
        y_pred_baseline = Y_pred_baseline[:, test_idx]
        baseline_metrics = compute_metrics(y_true, y_pred_baseline)
        
        # For each k value
        for k in k_list:
            # Filter training perturbations by exact k
            filtered_train_names, selected_similarities = filter_training_perturbations_by_k(
                similarities, train_pert_names, k=k,
                functional_annotations=functional_annotations,
                test_pert_name=test_pert_name,
                remove_same_class=remove_same_class,
            )
            
            if len(filtered_train_names) == 0:
                LOGGER.warning(f"No training perturbations selected for {test_pert_name} at k={k}")
                continue
            
            # Get indices of filtered perturbations
            train_pert_indices = [train_pert_name_to_idx[name] for name in filtered_train_names]
            
            # Retrain LPM on filtered data
            try:
                Y_train_filtered = Y_train[filtered_train_names]
                
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
                    # OPTIMIZATION: Pass cached components (same as evaluate_lsft)
                    A_baseline_cached=A_baseline if config.gene_embedding_source in ["scgpt", "scfoundation"] else None,
                    B_train_baseline=B_train_baseline if config.pert_embedding_source in ["k562_pca", "rpe1_pca", "gears"] else None,
                    train_pert_indices=train_pert_indices if config.pert_embedding_source in ["k562_pca", "rpe1_pca", "gears"] else None,
                    # EPIC 3: Noise injection
                    noise_level=noise_level,
                    noise_type=noise_type,
                    noise_target=noise_target,
                    # CENTERING FIX: Use global center for fair comparison
                    center_global=center_baseline[:, 0],
                )
                
                # For prediction, use local test embedding if available (for self-trained embeddings)
                # Otherwise use baseline embedding (for cross-dataset embeddings like GEARS, K562)
                if B_test_local is not None:
                    # Local embedding was computed (self-trained case)
                    test_pert_embedding_local = B_test_local[:, 0]  # (d_local,)
                    
                    # Check dimension compatibility
                    if K_local.shape[1] != len(test_pert_embedding_local):
                        LOGGER.warning(
                            f"Dimension mismatch for {test_pert_name} at k={k}: "
                            f"K_local shape {K_local.shape}, B_test_local shape {B_test_local.shape}"
                        )
                        continue
                else:
                    # For cross-dataset embeddings (GEARS, K562, RPE1), use baseline embedding
                    test_pert_embedding_local = test_pert_embedding  # Use baseline embedding
                    
                    # Check dimension compatibility
                    if K_local.shape[1] != test_pert_embedding_local.shape[0]:
                        LOGGER.warning(
                            f"Dimension mismatch for {test_pert_name} at k={k}: "
                            f"K_local shape {K_local.shape}, test_pert_embedding shape {test_pert_embedding_local.shape}"
                        )
                        continue
                
                # Predict using appropriate embedding
                y_pred_local = A_local @ K_local @ test_pert_embedding_local.reshape(-1, 1) + center_local.reshape(-1, 1)
                y_pred_local = y_pred_local.flatten()
                
                # Compute metrics
                local_metrics = compute_metrics(y_true, y_pred_local)
                
                # Store results
                results.append({
                    "dataset": dataset_name,
                    "baseline_type": baseline_type.value,
                    "test_perturbation": test_pert_name,
                    "k": k,
                    "local_train_size": len(filtered_train_names),
                    "local_mean_similarity": float(np.mean(selected_similarities)),
                    "local_max_similarity": float(np.max(selected_similarities)),
                    "local_min_similarity": float(np.min(selected_similarities)),
                    "performance_baseline_pearson_r": baseline_metrics["pearson_r"],
                    "performance_baseline_l2": baseline_metrics["l2"],
                    "performance_local_pearson_r": local_metrics["pearson_r"],
                    "performance_local_l2": local_metrics["l2"],
                    "improvement_pearson_r": local_metrics["pearson_r"] - baseline_metrics["pearson_r"],
                    "improvement_l2": baseline_metrics["l2"] - local_metrics["l2"],  # L2 improvement is reduction
                    "pearson_r": local_metrics["pearson_r"],  # For compatibility
                    "l2": local_metrics["l2"],  # For compatibility
                })
                
            except Exception as e:
                import traceback
                error_msg = f"Failed to retrain for {test_pert_name} at k={k}: {e}\n"
                error_msg += f"Traceback: {traceback.format_exc()}"
                LOGGER.error(error_msg)
                # Log detailed context
                LOGGER.error(
                    f"Context: baseline={baseline_type.value}, "
                    f"filtered_train_size={len(filtered_train_names)}, "
                    f"test_pert={test_pert_name}"
                )
                continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"lsft_k_sweep_{dataset_name}_{baseline_type.value}.csv"
    results_df.to_csv(output_path, index=False)
    LOGGER.info(f"Saved results to {output_path}")
    
    # Print summary
    LOGGER.info(f"\nSummary for {dataset_name}:")
    
    if len(results_df) == 0:
        LOGGER.warning("No results generated - all experiments may have failed")
        return results_df
    
    if "k" not in results_df.columns:
        LOGGER.error(f"Results DataFrame missing 'k' column. Columns: {results_df.columns.tolist()}")
        return results_df
    
    for k in k_list:
        subset = results_df[results_df["k"] == k]
        if len(subset) > 0:
            LOGGER.info(f"\nK={k}:")
            LOGGER.info(f"  Mean local train size: {subset['local_train_size'].mean():.1f}")
            LOGGER.info(f"  Mean local similarity: {subset['local_mean_similarity'].mean():.4f}")
            LOGGER.info(f"  Mean baseline Pearson r: {subset['performance_baseline_pearson_r'].mean():.4f}")
            LOGGER.info(f"  Mean local Pearson r: {subset['performance_local_pearson_r'].mean():.4f}")
            LOGGER.info(f"  Mean improvement Pearson r: {subset['improvement_pearson_r'].mean():.4f}")
    
    return results_df


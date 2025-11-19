#!/usr/bin/env python3
"""
Local Embedding Filtering: Neighborhood-restricted training for local manifold regression.

For each target, find top-K most similar training perturbations in embedding space
and train a local model only on those neighbors.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from baselines.baseline_runner import compute_pseudobulk_expression_changes
from baselines.baseline_types import BaselineType
from baselines.split_logic import load_split_config
from similarity.embedding_similarity import extract_perturbation_embeddings
from similarity.hardness_api import compute_target_similarity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def fit_local_model(
    target_embedding: np.ndarray,
    train_embeddings: np.ndarray,
    train_responses: np.ndarray,
    k: int = 25,
    ridge_penalty: float = 0.1,
) -> Tuple[Ridge, List[int]]:
    """
    Fit a local ridge regression model using top-K most similar training perturbations.
    
    Args:
        target_embedding: Target perturbation embedding (d,)
        train_embeddings: Training perturbation embeddings (n_train, d)
        train_responses: Training responses (n_train, n_genes)
        k: Number of neighbors to use
        ridge_penalty: Ridge penalty
        
    Returns:
        Tuple of (fitted Ridge model, list of selected neighbor indices)
    """
    # Compute similarities to find top-K neighbors
    from sklearn.metrics.pairwise import cosine_similarity
    
    target_emb_2d = target_embedding.reshape(1, -1) if target_embedding.ndim == 1 else target_embedding
    similarities = cosine_similarity(target_emb_2d, train_embeddings).flatten()
    
    # Select top-K neighbors
    top_k_indices = np.argsort(similarities)[-k:]
    
    # Get local training data
    X_local = train_embeddings[top_k_indices, :]  # (k, d)
    y_local = train_responses[top_k_indices, :]  # (k, n_genes)
    
    # Fit ridge regression
    model = Ridge(alpha=ridge_penalty)
    model.fit(X_local, y_local)
    
    return model, top_k_indices.tolist()


def fit_weighted_local_model(
    target_embedding: np.ndarray,
    train_embeddings: np.ndarray,
    train_responses: np.ndarray,
    temperature: float = 1.0,
    ridge_penalty: float = 0.1,
    min_weight: float = 0.0,
) -> Tuple[Ridge, np.ndarray]:
    """
    Fit a weighted local ridge regression model using all training perturbations
    weighted by similarity.
    
    Uses softmax over similarity / temperature to weight training samples:
    w_i = softmax(sim_i / temperature)
    
    Args:
        target_embedding: Target perturbation embedding (d,)
        train_embeddings: Training perturbation embeddings (n_train, d)
        train_responses: Training responses (n_train, n_genes)
        temperature: Temperature parameter (low = focus on most similar, high = uniform)
        ridge_penalty: Ridge penalty
        min_weight: Minimum weight threshold (samples below this are excluded)
        
    Returns:
        Tuple of (fitted Ridge model, sample weights array)
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Compute similarities to all training perturbations
    target_emb_2d = target_embedding.reshape(1, -1) if target_embedding.ndim == 1 else target_embedding
    similarities = cosine_similarity(target_emb_2d, train_embeddings).flatten()
    
    # Apply temperature and softmax to get weights
    scaled_similarities = similarities / temperature
    
    # Softmax for numerical stability
    exp_sims = np.exp(scaled_similarities - np.max(scaled_similarities))
    weights = exp_sims / np.sum(exp_sims)
    
    # Filter by minimum weight if specified
    if min_weight > 0:
        mask = weights >= min_weight
        weights = weights[mask]
        train_embeddings = train_embeddings[mask, :]
        train_responses = train_responses[mask, :]
        # Renormalize weights
        weights = weights / np.sum(weights)
    
    # Fit weighted ridge regression
    # Ridge doesn't support sample_weight directly for multi-output, so we'll use
    # a workaround: multiply inputs by sqrt of weights
    weights_sqrt = np.sqrt(weights)
    X_weighted = train_embeddings * weights_sqrt[:, np.newaxis]
    y_weighted = train_responses * weights_sqrt[:, np.newaxis]
    
    model = Ridge(alpha=ridge_penalty)
    model.fit(X_weighted, y_weighted)
    
    return model, weights


def evaluate_local_regression(
    adata_path: Path,
    split_config_path: Path,
    baseline_type: BaselineType,
    dataset_name: str,
    output_dir: Path,
    k_values: List[int] = [10, 25, 50],
    use_weighted: bool = False,
    temperatures: Optional[List[float]] = None,
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
) -> pd.DataFrame:
    """
    Evaluate local regression performance vs global baseline.
    
    Args:
        adata_path: Path to perturb_processed.h5ad
        split_config_path: Path to train/test/val split JSON
        baseline_type: Baseline type to use for embeddings
        dataset_name: Dataset name
        output_dir: Directory to save results
        k_values: List of K values (neighbors) to try
        pca_dim: PCA dimension
        ridge_penalty: Ridge penalty
        seed: Random seed
        
    Returns:
        DataFrame with local vs global comparison
    """
    LOGGER.info(f"Evaluating local regression for dataset: {dataset_name}, baseline: {baseline_type.value}")
    
    import anndata as ad
    
    # Load data
    adata = ad.read_h5ad(adata_path)
    split_config = load_split_config(split_config_path)
    
    # Compute Y matrix
    Y_df, split_labels = compute_pseudobulk_expression_changes(adata, split_config, seed)
    train_perts = split_labels.get("train", [])
    test_perts = split_labels.get("test", [])
    
    Y_train = Y_df[train_perts]
    Y_test = Y_df[test_perts]
    
    # Get gene name mapping
    gene_name_mapping = None
    if "gene_name" in adata.var.columns:
        gene_name_mapping = dict(zip(adata.var_names, adata.var["gene_name"]))
    
    # Extract embeddings
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
    
    # B_train is (d, n_train), B_test is (d, n_test)
    # Convert to (n_train, d) and (n_test, d) for local regression
    B_train_T = B_train.T  # (n_train, d)
    B_test_T = B_test.T    # (n_test, d)
    
    # Get global baseline predictions for comparison
    global_metrics = baseline_result.get("metrics", {})
    
    # Get training responses (Y_train aligned to train perturbations)
    # Y_train is (n_genes, n_train_perts)
    Y_train_np = Y_train.values.T  # (n_train_perts, n_genes)
    
    # Need to align train_pert_names with Y_train columns
    train_pert_map = {p.replace("+ctrl", ""): p for p in train_perts}
    train_indices_aligned = []
    for pert_name in train_pert_names:
        clean_name = pert_name.replace("+ctrl", "")
        if clean_name in train_pert_map:
            orig_name = train_pert_map[clean_name]
            if orig_name in Y_train.columns:
                train_indices_aligned.append(list(Y_train.columns).index(orig_name))
        else:
            train_indices_aligned.append(None)
    
    # Filter to valid alignments
    valid_train_indices = [i for i in train_indices_aligned if i is not None]
    if len(valid_train_indices) != len(B_train_T):
        LOGGER.warning(f"Alignment mismatch: {len(valid_train_indices)} train indices vs {len(B_train_T)} embeddings")
        # Use first n_train indices that align
        Y_train_aligned = Y_train_np[valid_train_indices[:len(B_train_T)], :]
    else:
        Y_train_aligned = Y_train_np[valid_train_indices, :]
    
    # Evaluate local regression
    results = []
    
    from eval_framework.metrics import compute_metrics
    
    if use_weighted:
        # Weighted approach: use all training data with similarity-based weights
        if temperatures is None:
            temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        for temperature in temperatures:
            LOGGER.info(f"Evaluating weighted local regression with temperature={temperature}")
            
            local_metrics = []
            
            for i, test_pert_name in enumerate(test_pert_names):
                clean_test_name = test_pert_name.replace("+ctrl", "")
                
                # Get true values
                if test_pert_name not in Y_test.columns:
                    continue
                
                y_true = Y_test.loc[:, test_pert_name].values
                
                # Get target embedding
                target_embedding = B_test_T[i, :]  # (d,)
                
                # Fit weighted local model
                try:
                    local_model, weights = fit_weighted_local_model(
                        target_embedding=target_embedding,
                        train_embeddings=B_train_T,
                        train_responses=Y_train_aligned,
                        temperature=temperature,
                        ridge_penalty=ridge_penalty,
                    )
                    
                    # Predict
                    target_emb_2d = target_embedding.reshape(1, -1)
                    y_pred_local = local_model.predict(target_emb_2d).flatten()
                    
                    # Compute metrics
                    metrics_local = compute_metrics(y_true, y_pred_local)
                    local_metrics.append({
                        "perturbation": clean_test_name,
                        "temperature": temperature,
                        **metrics_local,
                    })
                    
                except Exception as e:
                    LOGGER.warning(f"Failed to fit weighted local model for {clean_test_name}: {e}")
                    continue
            
            if local_metrics:
                local_df = pd.DataFrame(local_metrics)
                
                # Compare to global baseline
                global_r_values = []
                for clean_test_name in local_df["perturbation"]:
                    # Find matching test_pert_name
                    for test_pert in test_pert_names:
                        if test_pert.replace("+ctrl", "") == clean_test_name:
                            if test_pert in global_metrics:
                                global_r_values.append(global_metrics[test_pert]["pearson_r"])
                            break
                
                mean_local_r = local_df["pearson_r"].mean()
                mean_local_l2 = local_df["l2"].mean()
                mean_global_r = np.mean(global_r_values) if global_r_values else np.nan
                
                improvement = mean_local_r - mean_global_r if not np.isnan(mean_global_r) else np.nan
                
                results.append({
                    "method": "weighted",
                    "temperature": temperature,
                    "n_perturbations": len(local_metrics),
                    "local_mean_pearson_r": mean_local_r,
                    "local_mean_l2": mean_local_l2,
                    "global_mean_pearson_r": mean_global_r,
                    "improvement": improvement,
                })
    else:
        # Hard cutoff approach: use top-K neighbors
        for k in k_values:
            LOGGER.info(f"Evaluating local regression with k={k}")
            
            local_metrics = []
            
            for i, test_pert_name in enumerate(test_pert_names):
                clean_test_name = test_pert_name.replace("+ctrl", "")
                
                # Get true values
                if test_pert_name not in Y_test.columns:
                    continue
                
                y_true = Y_test.loc[:, test_pert_name].values
                
                # Get target embedding
                target_embedding = B_test_T[i, :]  # (d,)
                
                # Fit local model
                try:
                    local_model, neighbor_indices = fit_local_model(
                        target_embedding=target_embedding,
                        train_embeddings=B_train_T,
                        train_responses=Y_train_aligned,
                        k=k,
                        ridge_penalty=ridge_penalty,
                    )
                    
                    # Predict
                    target_emb_2d = target_embedding.reshape(1, -1)
                    y_pred_local = local_model.predict(target_emb_2d).flatten()
                    
                    # Compute metrics
                    metrics_local = compute_metrics(y_true, y_pred_local)
                    local_metrics.append({
                        "perturbation": clean_test_name,
                        "k": k,
                        **metrics_local,
                    })
                    
                except Exception as e:
                    LOGGER.warning(f"Failed to fit local model for {clean_test_name}: {e}")
                    continue
            
            if local_metrics:
                local_df = pd.DataFrame(local_metrics)
                
                # Compare to global baseline
                global_r_values = []
                for clean_test_name in local_df["perturbation"]:
                    # Find matching test_pert_name
                    for test_pert in test_pert_names:
                        if test_pert.replace("+ctrl", "") == clean_test_name:
                            if test_pert in global_metrics:
                                global_r_values.append(global_metrics[test_pert]["pearson_r"])
                            break
                
                mean_local_r = local_df["pearson_r"].mean()
                mean_local_l2 = local_df["l2"].mean()
                mean_global_r = np.mean(global_r_values) if global_r_values else np.nan
                
                improvement = mean_local_r - mean_global_r if not np.isnan(mean_global_r) else np.nan
                
                results.append({
                    "method": "k_neighbors",
                    "k": k,
                    "n_perturbations": len(local_metrics),
                    "local_mean_pearson_r": mean_local_r,
                    "local_mean_l2": mean_local_l2,
                    "global_mean_pearson_r": mean_global_r,
                    "improvement": improvement,
                })
    
    results_df = pd.DataFrame(results)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    if use_weighted:
        results_path = output_dir / "weighted_local_vs_global_regression.csv"
    else:
        results_path = output_dir / "local_vs_global_regression.csv"
    results_df.to_csv(results_path, index=False)
    LOGGER.info(f"Saved local regression comparison to {results_path}")
    
    return results_df


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate local neighborhood regression"
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
        "--baseline_type",
        type=str,
        choices=[bt.value for bt in BaselineType],
        default=BaselineType.SELFTRAINED.value,
        help="Baseline type to use for embeddings",
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
        "--k_values",
        type=int,
        nargs="+",
        default=[10, 25, 50],
        help="K values (neighbors) to try (for hard cutoff method)",
    )
    parser.add_argument(
        "--use_weighted",
        action="store_true",
        help="Use weighted regression instead of hard cutoff",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=None,
        help="Temperature values to try (for weighted method, default: [0.1, 0.5, 1.0, 2.0, 5.0])",
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
    
    baseline_type = BaselineType(args.baseline_type)
    
    # Evaluate local regression
    results_df = evaluate_local_regression(
        adata_path=args.adata_path,
        split_config_path=args.split_config,
        baseline_type=baseline_type,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        k_values=args.k_values,
        use_weighted=args.use_weighted,
        temperatures=args.temperatures,
        pca_dim=args.pca_dim,
        ridge_penalty=args.ridge_penalty,
        seed=args.seed,
    )
    
    LOGGER.info("Local regression evaluation complete")


if __name__ == "__main__":
    main()


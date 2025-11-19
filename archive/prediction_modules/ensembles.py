#!/usr/bin/env python3
"""
Similarity-Weighted Ensemble: Combine baseline predictions using similarity-weighted averaging.

Each baseline's contribution is weighted by its similarity to the target in its embedding space.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

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


def similarity_weighted_ensemble(
    target_id: str,
    baseline_predictions: Dict[str, np.ndarray],
    baseline_similarities: Dict[str, float],
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Compute similarity-weighted ensemble prediction.
    
    Uses softmax over similarity / temperature to weight baselines:
    w_b = softmax(sim_b / temperature)
    
    Args:
        target_id: Target perturbation ID
        baseline_predictions: Dict mapping baseline_name -> prediction array
        baseline_similarities: Dict mapping baseline_name -> similarity score for this target
        temperature: Temperature parameter (low = pick best, high = average)
        
    Returns:
        Ensemble prediction array
    """
    if not baseline_predictions:
        raise ValueError("baseline_predictions cannot be empty")
    
    if not baseline_similarities:
        # Fall back to simple average if no similarities
        predictions = list(baseline_predictions.values())
        if len(set(p.shape for p in predictions)) != 1:
            raise ValueError("All predictions must have the same shape")
        return np.mean(predictions, axis=0)
    
    # Filter to baselines with both predictions and similarities
    available_baselines = [
        bl for bl in baseline_predictions.keys()
        if bl in baseline_similarities and baseline_similarities[bl] is not None
    ]
    
    if not available_baselines:
        # Fall back to simple average
        predictions = list(baseline_predictions.values())
        if len(set(p.shape for p in predictions)) != 1:
            raise ValueError("All predictions must have the same shape")
        return np.mean(predictions, axis=0)
    
    # Extract similarities and convert to weights using softmax
    similarities = np.array([baseline_similarities[bl] for bl in available_baselines])
    
    # Apply temperature
    scaled_similarities = similarities / temperature
    
    # Softmax
    exp_sims = np.exp(scaled_similarities - np.max(scaled_similarities))  # Numerical stability
    weights = exp_sims / np.sum(exp_sims)
    
    # Weighted average
    predictions = [baseline_predictions[bl] for bl in available_baselines]
    
    # Ensure all predictions have same shape
    if len(set(p.shape for p in predictions)) != 1:
        raise ValueError("All predictions must have the same shape")
    
    ensemble_pred = np.average(predictions, axis=0, weights=weights)
    
    return ensemble_pred


def evaluate_similarity_ensemble(
    adata_path: Path,
    split_config_path: Path,
    baseline_types: List[BaselineType],
    predictions_base_dir: Path,
    dataset_name: str,
    output_dir: Path,
    temperatures: List[float] = [0.5, 1.0, 2.0],
    k: int = 5,
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
) -> pd.DataFrame:
    """
    Evaluate similarity-weighted ensemble performance.
    
    Args:
        adata_path: Path to perturb_processed.h5ad
        split_config_path: Path to train/test/val split JSON
        baseline_types: List of baseline types to ensemble
        predictions_base_dir: Base directory containing baseline predictions
        dataset_name: Dataset name
        output_dir: Directory to save results
        temperatures: List of temperature values to try
        k: K value for hardness computation
        pca_dim: PCA dimension
        ridge_penalty: Ridge penalty
        seed: Random seed
        
    Returns:
        DataFrame with ensemble performance results
    """
    LOGGER.info(f"Evaluating similarity-weighted ensemble for dataset: {dataset_name}")
    
    import json
    import anndata as ad
    
    # Load data
    adata = ad.read_h5ad(adata_path)
    split_config = load_split_config(split_config_path)
    
    # Compute Y matrix for true values
    Y_df, split_labels = compute_pseudobulk_expression_changes(adata, split_config, seed)
    test_perts = split_labels.get("test", [])
    Y_test = Y_df[test_perts]
    
    # Get gene name mapping
    gene_name_mapping = None
    if "gene_name" in adata.var.columns:
        gene_name_mapping = dict(zip(adata.var_names, adata.var["gene_name"]))
    
    # Load predictions for all baselines
    baseline_predictions_all = {}
    test_pert_map = {pert.replace("+ctrl", ""): pert for pert in test_perts}
    
    for baseline_type in baseline_types:
        baseline_name = baseline_type.value
        baseline_dir = predictions_base_dir / baseline_name
        
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
        
        # Convert to numpy arrays aligned to common genes
        common_genes = [g for g in Y_test.index if g in pred_gene_names]
        baseline_predictions_all[baseline_name] = {}
        
        for clean_pert_name, pred_values in predictions_dict.items():
            if clean_pert_name not in test_pert_map:
                continue
            
            pred_array = np.array(pred_values)
            if len(pred_array) != len(pred_gene_names):
                LOGGER.warning(f"Length mismatch for {baseline_name}/{clean_pert_name}: pred={len(pred_array)}, genes={len(pred_gene_names)}")
                continue
            
            pred_idx_map = {g: i for i, g in enumerate(pred_gene_names)}
            common_genes_aligned = [g for g in common_genes if g in pred_idx_map]
            pred_aligned = np.array([pred_array[pred_idx_map[g]] for g in common_genes_aligned])
            baseline_predictions_all[baseline_name][clean_pert_name] = pred_aligned
    
    # Compute similarity metrics for all baselines
    baseline_similarities_all = {}
    
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
            
            # Extract max similarities for ensemble weighting
            baseline_similarities_all[baseline_name] = {
                pert: metrics.get("max_sim", None)
                for pert, metrics in hardness_results.items()
            }
            
        except Exception as e:
            LOGGER.warning(f"Failed to compute similarities for {baseline_name}: {e}")
            continue
    
    # Evaluate ensemble for each temperature
    results = []
    
    # Load baseline performance for comparison
    baseline_results_path = predictions_base_dir / "baseline_results_reproduced.csv"
    if baseline_results_path.exists():
        baseline_results_df = pd.read_csv(baseline_results_path)
        baseline_performance = dict(
            zip(baseline_results_df["baseline"], baseline_results_df["mean_pearson_r"])
        )
        best_baseline_r = max(baseline_performance.values())
        best_baseline_name = max(baseline_performance.items(), key=lambda x: x[1])[0]
    else:
        baseline_performance = {}
        best_baseline_r = np.nan
        best_baseline_name = None
    
    from eval_framework.metrics import compute_metrics
    
    for temperature in temperatures:
        LOGGER.info(f"Evaluating ensemble with temperature={temperature}")
        
        ensemble_metrics = []
        
        for clean_pert_name in test_pert_map.keys():
            pert_name = test_pert_map[clean_pert_name]
            y_true = Y_test.loc[:, pert_name].values
            
            # Get predictions for this target
            target_predictions = {
                bl: preds[clean_pert_name]
                for bl, preds in baseline_predictions_all.items()
                if clean_pert_name in preds
            }
            
            if not target_predictions:
                continue
            
            # Get similarities for this target
            target_similarities = {
                bl: sims.get(clean_pert_name, None)
                for bl, sims in baseline_similarities_all.items()
            }
            
            try:
                # Compute ensemble prediction
                ensemble_pred = similarity_weighted_ensemble(
                    target_id=clean_pert_name,
                    baseline_predictions=target_predictions,
                    baseline_similarities=target_similarities,
                    temperature=temperature,
                )
                
                # Compute metrics
                metrics = compute_metrics(y_true, ensemble_pred)
                ensemble_metrics.append(metrics)
                
            except Exception as e:
                LOGGER.warning(f"Failed to compute ensemble for {clean_pert_name}: {e}")
                continue
        
        if ensemble_metrics:
            mean_r = np.mean([m["pearson_r"] for m in ensemble_metrics])
            mean_l2 = np.mean([m["l2"] for m in ensemble_metrics])
            
            improvement = mean_r - best_baseline_r if not np.isnan(best_baseline_r) else np.nan
            
            results.append({
                "temperature": temperature,
                "n_perturbations": len(ensemble_metrics),
                "mean_pearson_r": mean_r,
                "mean_l2": mean_l2,
                "best_baseline": best_baseline_name,
                "best_baseline_r": best_baseline_r,
                "improvement": improvement,
            })
    
    results_df = pd.DataFrame(results)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "similarity_ensemble_performance.csv"
    results_df.to_csv(results_path, index=False)
    LOGGER.info(f"Saved ensemble performance to {results_path}")
    
    return results_df


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate similarity-weighted ensemble"
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
        help="Baselines to ensemble",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
        help="Temperature values to try",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="K value for hardness computation",
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
    
    # Evaluate ensemble
    results_df = evaluate_similarity_ensemble(
        adata_path=args.adata_path,
        split_config_path=args.split_config,
        baseline_types=baseline_types,
        predictions_base_dir=args.predictions_base_dir,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        temperatures=args.temperatures,
        k=args.k,
        pca_dim=args.pca_dim,
        ridge_penalty=args.ridge_penalty,
        seed=args.seed,
    )
    
    LOGGER.info("Similarity-weighted ensemble evaluation complete")


if __name__ == "__main__":
    main()


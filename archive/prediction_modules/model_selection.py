#!/usr/bin/env python3
"""
Similarity-Based Model Selection: Choose best baseline per target based on hardness/similarity.

Implements multiple selection policies:
1. Global best: Always choose baseline with best global R²
2. Hardness-based: Choose based on hardness threshold
3. Similarity-max: Choose baseline with highest similarity to target
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


def select_baseline_global_best(
    baseline_performance: Dict[str, float],
) -> str:
    """
    Select baseline with best global mean performance.
    
    Args:
        baseline_performance: Dict mapping baseline_name -> mean_pearson_r
        
    Returns:
        Name of best baseline
    """
    if not baseline_performance:
        raise ValueError("baseline_performance cannot be empty")
    
    best_baseline = max(baseline_performance.items(), key=lambda x: x[1])[0]
    return best_baseline


def select_baseline_hardness_threshold(
    target_id: str,
    hardness_metrics: Dict[str, Dict[str, float]],
    baseline_performance: Dict[str, float],
    hardness_threshold: float = 0.5,
    k: int = 5,
    easy_baseline: Optional[str] = None,
    hard_baseline: Optional[str] = None,
) -> str:
    """
    Select baseline based on hardness threshold.
    
    If hardness_k <= threshold → use easy_baseline (or global best if not specified)
    If hardness_k > threshold → use hard_baseline (or second best if not specified)
    
    Args:
        target_id: Target perturbation ID
        hardness_metrics: Dict mapping baseline_name -> {hardness_k: {...}, ...}
        baseline_performance: Dict mapping baseline_name -> mean_pearson_r
        hardness_threshold: Threshold for hardness_k
        k: K value to use for hardness_k
        easy_baseline: Baseline to use for easy targets (default: global best)
        hard_baseline: Baseline to use for hard targets (default: second best or specified)
        
    Returns:
        Selected baseline name
    """
    if not hardness_metrics:
        # Fall back to global best if no hardness metrics
        return select_baseline_global_best(baseline_performance)
    
    # Default: use first baseline's hardness if target-specific not available
    # Try to get hardness for target from first available baseline
    target_hardness = None
    for baseline_name, metrics in hardness_metrics.items():
        if target_id in metrics:
            hardness_data = metrics[target_id]
            target_hardness = hardness_data.get("hardness_k", {}).get(k, None)
            if target_hardness is not None:
                break
    
    if target_hardness is None:
        # Fall back to global best
        return select_baseline_global_best(baseline_performance)
    
    # Determine easy/hard baselines
    if easy_baseline is None:
        easy_baseline = select_baseline_global_best(baseline_performance)
    
    if hard_baseline is None:
        # Use second best
        sorted_baselines = sorted(
            baseline_performance.items(), key=lambda x: x[1], reverse=True
        )
        if len(sorted_baselines) > 1:
            hard_baseline = sorted_baselines[1][0]
        else:
            hard_baseline = sorted_baselines[0][0]
    
    # Select based on hardness
    if target_hardness <= hardness_threshold:
        return easy_baseline
    else:
        return hard_baseline


def select_baseline_similarity_max(
    target_id: str,
    baseline_similarities: Dict[str, float],
    baseline_performance: Optional[Dict[str, float]] = None,
) -> str:
    """
    Select baseline with highest similarity to target in its embedding space.
    
    Args:
        target_id: Target perturbation ID
        baseline_similarities: Dict mapping baseline_name -> similarity score for this target
        baseline_performance: Optional dict for tie-breaking
        
    Returns:
        Baseline name with highest similarity
    """
    if not baseline_similarities:
        if baseline_performance:
            return select_baseline_global_best(baseline_performance)
        raise ValueError("baseline_similarities cannot be empty")
    
    # Filter to available baselines for this target
    target_sims = {bl: sim for bl, sim in baseline_similarities.items() if sim is not None}
    
    if not target_sims:
        if baseline_performance:
            return select_baseline_global_best(baseline_performance)
        raise ValueError(f"No similarity metrics available for target {target_id}")
    
    # Select baseline with max similarity
    # Tie-break by performance if provided
    max_sim = max(target_sims.values())
    candidates = [bl for bl, sim in target_sims.items() if sim == max_sim]
    
    if len(candidates) == 1:
        return candidates[0]
    
    # Tie-break by performance
    if baseline_performance:
        best_candidate = max(candidates, key=lambda bl: baseline_performance.get(bl, 0.0))
        return best_candidate
    
    # If no performance data, just return first
    return candidates[0]


def select_baseline(
    target_id: str,
    dataset: str,
    hardness_metrics: Dict[str, Dict[str, Dict]],
    baseline_performance: Dict[str, float],
    policy: str = "similarity_max",
    **kwargs,
) -> str:
    """
    Select baseline for a target perturbation using specified policy.
    
    Args:
        target_id: Target perturbation ID
        dataset: Dataset name
        hardness_metrics: Dict mapping baseline_name -> {target_id: {hardness_k: {...}, max_sim: ...}}
        baseline_performance: Dict mapping baseline_name -> mean_pearson_r
        policy: Selection policy ("global_best", "hardness_threshold", "similarity_max")
        **kwargs: Additional arguments for specific policies
        
    Returns:
        Selected baseline name
    """
    if policy == "global_best":
        return select_baseline_global_best(baseline_performance)
    
    elif policy == "hardness_threshold":
        return select_baseline_hardness_threshold(
            target_id=target_id,
            hardness_metrics=hardness_metrics,
            baseline_performance=baseline_performance,
            **kwargs,
        )
    
    elif policy == "similarity_max":
        # Extract similarities for this target
        baseline_similarities = {}
        for baseline_name, target_metrics in hardness_metrics.items():
            if target_id in target_metrics:
                target_data = target_metrics[target_id]
                baseline_similarities[baseline_name] = target_data.get("max_sim", None)
        
        return select_baseline_similarity_max(
            target_id=target_id,
            baseline_similarities=baseline_similarities,
            baseline_performance=baseline_performance,
        )
    
    else:
        raise ValueError(f"Unknown policy: {policy}")


def evaluate_model_selection(
    adata_path: Path,
    split_config_path: Path,
    baseline_types: List[BaselineType],
    predictions_base_dir: Path,
    dataset_name: str,
    output_dir: Path,
    policies: List[str] = ["global_best", "similarity_max", "hardness_threshold"],
    k: int = 5,
    hardness_threshold: float = 0.5,
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
) -> pd.DataFrame:
    """
    Evaluate model selection policies and compare performance.
    
    Args:
        adata_path: Path to perturb_processed.h5ad
        split_config_path: Path to train/test/val split JSON
        baseline_types: List of baseline types to consider
        predictions_base_dir: Base directory containing baseline predictions
        dataset_name: Dataset name
        output_dir: Directory to save results
        policies: List of selection policies to evaluate
        k: K value for hardness computation
        hardness_threshold: Threshold for hardness-based selection
        pca_dim: PCA dimension
        ridge_penalty: Ridge penalty
        seed: Random seed
        
    Returns:
        DataFrame with selection results and performance comparison
    """
    LOGGER.info(f"Evaluating model selection for dataset: {dataset_name}")
    
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
    
    # Load baseline performance (global means)
    baseline_results_path = predictions_base_dir / "baseline_results_reproduced.csv"
    if baseline_results_path.exists():
        baseline_results_df = pd.read_csv(baseline_results_path)
        baseline_performance = dict(
            zip(baseline_results_df["baseline"], baseline_results_df["mean_pearson_r"])
        )
    else:
        LOGGER.warning("Baseline results not found, computing from predictions")
        baseline_performance = {}
    
    # Load predictions and metrics for all baselines
    baseline_predictions = {}
    baseline_metrics = {}
    test_pert_map = {pert.replace("+ctrl", ""): pert for pert in test_perts}
    
    for baseline_type in baseline_types:
        baseline_name = baseline_type.value
        baseline_dir = predictions_base_dir / baseline_name
        
        # Load predictions
        predictions_path = baseline_dir / "predictions.json"
        if not predictions_path.exists():
            LOGGER.warning(f"Predictions not found for {baseline_name}, skipping")
            continue
        
        with open(predictions_path, "r") as f:
            predictions_dict = json.load(f)
        baseline_predictions[baseline_name] = predictions_dict
        
        # Load or compute per-perturbation metrics
        baseline_metrics[baseline_name] = {}
        
        # Try to get metrics from baseline run if available
        # For now, we'll compute them from predictions vs true values
        from eval_framework.metrics import compute_metrics
        
        gene_names_path = baseline_dir / "gene_names.json"
        with open(gene_names_path, "r") as f:
            pred_gene_names = json.load(f)
        
        common_genes = [g for g in Y_test.index if g in pred_gene_names]
        
        for clean_pert_name, pred_values in predictions_dict.items():
            if clean_pert_name not in test_pert_map:
                continue
            
            pert_name = test_pert_map[clean_pert_name]
            y_true = Y_test.loc[common_genes, pert_name].values
            
            pred_array = np.array(pred_values)
            if len(pred_array) != len(pred_gene_names):
                LOGGER.warning(f"Length mismatch for {baseline_name}/{clean_pert_name}: pred={len(pred_array)}, genes={len(pred_gene_names)}")
                continue
            
            pred_idx_map = {g: i for i, g in enumerate(pred_gene_names)}
            pred_aligned = np.array([pred_array[pred_idx_map[g]] for g in common_genes if g in pred_idx_map])
            
            # Ensure y_true matches pred_aligned length
            y_true_aligned = y_true[:len(pred_aligned)] if len(y_true) > len(pred_aligned) else y_true
            if len(y_true_aligned) != len(pred_aligned):
                # Re-compute common genes more carefully
                common_genes_aligned = [g for g in common_genes if g in pred_idx_map]
                y_true_aligned = Y_test.loc[common_genes_aligned, pert_name].values
                pred_aligned = np.array([pred_array[pred_idx_map[g]] for g in common_genes_aligned])
            
            metrics = compute_metrics(y_true_aligned, pred_aligned)
            baseline_metrics[baseline_name][clean_pert_name] = metrics
    
    # Compute hardness metrics for all baselines
    hardness_metrics = {}
    
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
            
            hardness_metrics[baseline_name] = hardness_results
            
        except Exception as e:
            LOGGER.warning(f"Failed to compute hardness for {baseline_name}: {e}")
            continue
    
    # Evaluate each policy
    results = []
    
    for policy in policies:
        LOGGER.info(f"Evaluating policy: {policy}")
        
        selections = {}
        selected_performance = {}
        
        for clean_pert_name in test_pert_map.keys():
            try:
                selected_baseline = select_baseline(
                    target_id=clean_pert_name,
                    dataset=dataset_name,
                    hardness_metrics=hardness_metrics,
                    baseline_performance=baseline_performance,
                    policy=policy,
                    hardness_threshold=hardness_threshold,
                    k=k,
                )
                
                selections[clean_pert_name] = selected_baseline
                
                # Get performance of selected baseline for this target
                if selected_baseline in baseline_metrics:
                    if clean_pert_name in baseline_metrics[selected_baseline]:
                        selected_performance[clean_pert_name] = baseline_metrics[
                            selected_baseline
                        ][clean_pert_name]["pearson_r"]
                
            except Exception as e:
                LOGGER.warning(f"Failed to select baseline for {clean_pert_name}: {e}")
                continue
        
        # Compute aggregate performance
        if selected_performance:
            mean_r = np.mean(list(selected_performance.values()))
            mean_l2 = np.mean([
                baseline_metrics[selections[pert]][pert]["l2"]
                for pert in selections.keys()
                if pert in baseline_metrics.get(selections[pert], {})
            ]) if selections else np.nan
            
            # Compare to global best
            if baseline_performance:
                global_best_r = max(baseline_performance.values())
                improvement = mean_r - global_best_r
            else:
                global_best_r = np.nan
                improvement = np.nan
            
            results.append({
                "policy": policy,
                "n_perturbations": len(selected_performance),
                "mean_pearson_r": mean_r,
                "mean_l2": mean_l2,
                "global_best_r": global_best_r,
                "improvement": improvement,
            })
    
    results_df = pd.DataFrame(results)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "model_selection_comparison.csv"
    results_df.to_csv(results_path, index=False)
    LOGGER.info(f"Saved model selection comparison to {results_path}")
    
    return results_df


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate similarity-based model selection policies"
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
        help="Baselines to consider",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["global_best", "hardness_threshold", "similarity_max"],
        default=["global_best", "similarity_max", "hardness_threshold"],
        help="Selection policies to evaluate",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="K value for hardness computation",
    )
    parser.add_argument(
        "--hardness_threshold",
        type=float,
        default=0.5,
        help="Hardness threshold for hardness-based selection",
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
    
    # Evaluate model selection
    results_df = evaluate_model_selection(
        adata_path=args.adata_path,
        split_config_path=args.split_config,
        baseline_types=baseline_types,
        predictions_base_dir=args.predictions_base_dir,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        policies=args.policies,
        k=args.k,
        hardness_threshold=args.hardness_threshold,
        pca_dim=args.pca_dim,
        ridge_penalty=args.ridge_penalty,
        seed=args.seed,
    )
    
    LOGGER.info("Model selection evaluation complete")


if __name__ == "__main__":
    main()


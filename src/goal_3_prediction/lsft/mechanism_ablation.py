#!/usr/bin/env python3
"""
Mechanism-Mismatch Ablation: Functional-Neighborhood Alignment Analysis

This module implements Epic 2 of the Manifold Law Diagnostic Suite.

Goal: Remove neighbors sharing the same functional class as the test KO and measure
drop in accuracy. Quantifies biological-alignment smoothness (large drop ⇒ biology-aligned smoothness).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from goal_2_baselines.baseline_types import BaselineType
from goal_3_prediction.lsft.lsft import evaluate_lsft
from goal_3_prediction.lsft.lsft_k_sweep import evaluate_lsft_with_k_list

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def load_functional_annotations(annotation_path: Path) -> Dict[str, str]:
    """
    Load functional class annotations.
    
    Args:
        annotation_path: Path to TSV file with columns: target, class
        
    Returns:
        Dictionary mapping perturbation name to functional class
    """
    df = pd.read_csv(annotation_path, sep="\t")
    
    # Handle different possible column names
    target_col = None
    class_col = None
    
    for col in df.columns:
        if col.lower() in ["target", "perturbation", "pert", "gene"]:
            target_col = col
        if col.lower() in ["class", "functional_class", "category"]:
            class_col = col
    
    if target_col is None or class_col is None:
        raise ValueError(f"Could not find target and class columns in {annotation_path}. Columns: {df.columns.tolist()}")
    
    annotations = dict(zip(df[target_col], df[class_col]))
    return annotations


def filter_neighbors_by_functional_class(
    similarities: np.ndarray,
    train_pert_names: List[str],
    test_pert_name: str,
    functional_annotations: Dict[str, str],
    top_pct: float = 0.05,
    remove_same_class: bool = True,
) -> Tuple[List[str], np.ndarray]:
    """
    Filter training perturbations, optionally removing same-class neighbors.
    
    Args:
        similarities: Similarities to all training perturbations
        train_pert_names: List of training perturbation names
        test_pert_name: Test perturbation name
        functional_annotations: Dictionary mapping perturbation to functional class
        top_pct: Top percentage to keep initially
        remove_same_class: If True, remove neighbors with same functional class as test
        
    Returns:
        Tuple of (filtered perturbation names, selected similarities)
    """
    # First, get top-K% most similar
    n_select = max(1, int(np.ceil(len(train_pert_names) * top_pct)))
    top_k_indices = np.argsort(similarities)[-n_select:]
    top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
    
    # Get test perturbation's functional class
    test_class = functional_annotations.get(test_pert_name, None)
    
    if remove_same_class and test_class is not None:
        # Filter out neighbors with same class
        filtered_indices = []
        for idx in top_k_indices:
            train_pert = train_pert_names[idx]
            train_class = functional_annotations.get(train_pert, None)
            if train_class != test_class:
                filtered_indices.append(idx)
        
        if len(filtered_indices) == 0:
            # If all removed, keep original (fallback)
            LOGGER.warning(f"All neighbors removed for {test_pert_name}, keeping original top-K")
            filtered_indices = top_k_indices.tolist()
    else:
        filtered_indices = top_k_indices.tolist()
    
    filtered_names = [train_pert_names[i] for i in filtered_indices]
    selected_similarities = similarities[filtered_indices]
    
    return filtered_names, selected_similarities


def run_mechanism_ablation(
    adata_path: Path,
    split_config_path: Path,
    annotation_path: Path,
    baseline_type: BaselineType,
    dataset_name: str,
    output_dir: Path,
    top_pct: float = 0.05,
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
) -> pd.DataFrame:
    """
    Run mechanism-mismatch ablation: compare original vs same-class-removed LSFT.
    
    Args:
        adata_path: Path to adata file
        split_config_path: Path to split config JSON
        annotation_path: Path to functional class annotations TSV
        baseline_type: Baseline type
        dataset_name: Dataset name
        output_dir: Output directory
        top_pct: Top percentage for initial neighbor selection
        pca_dim: PCA dimension
        ridge_penalty: Ridge penalty
        seed: Random seed
        
    Returns:
        DataFrame with ablation results
    """
    LOGGER.info(f"Running mechanism-mismatch ablation for {dataset_name}")
    LOGGER.info(f"Baseline: {baseline_type.value}")
    
    # Load functional annotations
    functional_annotations = load_functional_annotations(annotation_path)
    LOGGER.info(f"Loaded {len(functional_annotations)} functional annotations")
    
    # Convert top_pct to approximate k for k-sweep
    # We'll use a fixed k that corresponds roughly to top_pct
    # Load data to get train size
    import anndata as ad
    from goal_2_baselines.split_logic import load_split_config
    from goal_2_baselines.baseline_runner import compute_pseudobulk_expression_changes
    
    adata = ad.read_h5ad(adata_path)
    split_config = load_split_config(split_config_path)
    Y_df, split_labels = compute_pseudobulk_expression_changes(adata, split_config, seed=seed)
    train_pert_names = split_labels.get("train", [])
    n_train = len(train_pert_names)
    k_approx = max(3, int(np.ceil(n_train * top_pct)))  # Approximate k for top_pct
    
    # Run original LSFT (control) with k-sweep
    LOGGER.info("Running original LSFT (control)...")
    from goal_3_prediction.lsft.lsft_k_sweep import evaluate_lsft_with_k_list
    
    results_original = evaluate_lsft_with_k_list(
        adata_path=adata_path,
        split_config_path=split_config_path,
        baseline_type=baseline_type,
        dataset_name=dataset_name,
        output_dir=output_dir / "original",
        k_list=[k_approx],
        pca_dim=pca_dim,
        ridge_penalty=ridge_penalty,
        seed=seed,
        functional_annotations=functional_annotations,
        remove_same_class=False,
    )
    
    # Run ablated LSFT (same-class removed)
    LOGGER.info("Running ablated LSFT (same-class neighbors removed)...")
    results_ablated = evaluate_lsft_with_k_list(
        adata_path=adata_path,
        split_config_path=split_config_path,
        baseline_type=baseline_type,
        dataset_name=dataset_name,
        output_dir=output_dir / "ablated",
        k_list=[k_approx],
        pca_dim=pca_dim,
        ridge_penalty=ridge_penalty,
        seed=seed,
        functional_annotations=functional_annotations,
        remove_same_class=True,
    )
    
    # Merge results and compute deltas
    results_original = results_original.rename(columns={
        "performance_local_pearson_r": "original_pearson_r",
        "performance_local_l2": "original_l2",
    })
    results_ablated = results_ablated.rename(columns={
        "performance_local_pearson_r": "ablated_pearson_r",
        "performance_local_l2": "ablated_l2",
    })
    
    merged = results_original.merge(
        results_ablated[["test_perturbation", "ablated_pearson_r", "ablated_l2"]],
        on="test_perturbation",
        how="inner"
    )
    
    # Add functional class and compute deltas
    ablation_results = []
    for _, row in merged.iterrows():
        test_pert = row["test_perturbation"]
        test_class = functional_annotations.get(test_pert, "Unknown")
        
        ablation_results.append({
            "dataset": dataset_name,
            "baseline_type": baseline_type.value,
            "test_perturbation": test_pert,
            "functional_class": test_class,
            "k": k_approx,
            "original_pearson_r": row["original_pearson_r"],
            "original_l2": row["original_l2"],
            "ablated_pearson_r": row["ablated_pearson_r"],
            "ablated_l2": row["ablated_l2"],
            "delta_r": row["original_pearson_r"] - row["ablated_pearson_r"],
            "delta_l2": row["ablated_l2"] - row["original_l2"],  # L2 increases are worse
        })
    
    results_df = pd.DataFrame(ablation_results)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"mechanism_ablation_{dataset_name}_{baseline_type.value}.csv"
    results_df.to_csv(output_path, index=False)
    LOGGER.info(f"Saved results to {output_path}")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mechanism-Mismatch Ablation Analysis")
    parser.add_argument("--adata_path", type=Path, required=True)
    parser.add_argument("--split_config", type=Path, required=True)
    parser.add_argument("--annotation_path", type=Path, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--baseline_type", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--top_pct", type=float, default=0.05)
    parser.add_argument("--pca_dim", type=int, default=10)
    parser.add_argument("--ridge_penalty", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    
    args = parser.parse_args()
    
    baseline_type = BaselineType(args.baseline_type)
    
    results_df = run_mechanism_ablation(
        adata_path=args.adata_path,
        split_config_path=args.split_config,
        annotation_path=args.annotation_path,
        baseline_type=baseline_type,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        top_pct=args.top_pct,
        pca_dim=args.pca_dim,
        ridge_penalty=args.ridge_penalty,
        seed=args.seed,
    )
    
    print(f"\nMechanism Ablation Results:")
    print(results_df.groupby("functional_class")["original_pearson_r"].mean())


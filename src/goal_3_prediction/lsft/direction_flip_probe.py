#!/usr/bin/env python3
"""
Direction-Flip Probe: High-Cosine but Opposite Responses

This module implements Epic 4 of the Manifold Law Diagnostic Suite.

Goal: Identify cases where cosine similarity is high but target responses
are anticorrelated. Quantifies where LSFT could break (adversarial neighborhoods).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def find_adversarial_neighbors(
    similarities: np.ndarray,
    train_pert_names: List[str],
    Y_train: pd.DataFrame,
    test_pert_name: str,
    y_test: np.ndarray,
    top_pct: float = 0.05,
    conflict_threshold: float = -0.2,
) -> List[Tuple[str, float, float]]:
    """
    Find adversarial neighbor pairs: high cosine similarity but anticorrelated targets.
    
    Args:
        similarities: Cosine similarities to training perturbations
        train_pert_names: List of training perturbation names
        Y_train: Training expression matrix (genes × perturbations)
        test_pert_name: Test perturbation name
        y_test: Test perturbation expression vector
        top_pct: Top percentage to check
        conflict_threshold: Threshold for anticorrelation (r < threshold)
        
    Returns:
        List of (perturbation_name, cosine_similarity, target_correlation) tuples
    """
    # Get top-K% neighbors
    n_select = max(1, int(np.ceil(len(train_pert_names) * top_pct)))
    top_k_indices = np.argsort(similarities)[-n_select:]
    
    adversarial = []
    
    for idx in top_k_indices:
        train_pert = train_pert_names[idx]
        cosine_sim = similarities[idx]
        
        # Get training perturbation expression vector
        if train_pert in Y_train.columns:
            y_train_vec = Y_train[train_pert].values
            
            # Compute correlation between test and train targets
            target_corr, _ = pearsonr(y_test, y_train_vec)
            
            # Check if adversarial (high cosine but anticorrelated targets)
            if cosine_sim > 0.5 and target_corr < conflict_threshold:
                adversarial.append((train_pert, float(cosine_sim), float(target_corr)))
    
    return adversarial


def run_direction_flip_probe(
    similarities_matrix: np.ndarray,
    train_pert_names: List[str],
    test_pert_names: List[str],
    Y_train: pd.DataFrame,
    Y_test: pd.DataFrame,
    output_dir: Path,
    top_pct: float = 0.05,
    conflict_threshold: float = -0.2,
) -> pd.DataFrame:
    """
    Run direction-flip probe analysis.
    
    Args:
        similarities_matrix: Similarity matrix (n_test × n_train)
        train_pert_names: List of training perturbation names
        test_pert_names: List of test perturbation names
        Y_train: Training expression matrix
        Y_test: Test expression matrix
        output_dir: Output directory
        top_pct: Top percentage to check
        conflict_threshold: Threshold for anticorrelation
        
    Returns:
        DataFrame with conflict results
    """
    LOGGER.info("Running direction-flip probe...")
    
    results = []
    
    for test_idx, test_pert_name in enumerate(test_pert_names):
        similarities = similarities_matrix[test_idx, :]
        y_test = Y_test[test_pert_name].values
        
        # Find adversarial neighbors
        adversarial = find_adversarial_neighbors(
            similarities=similarities,
            train_pert_names=train_pert_names,
            Y_train=Y_train,
            test_pert_name=test_pert_name,
            y_test=y_test,
            top_pct=top_pct,
            conflict_threshold=conflict_threshold,
        )
        
        results.append({
            "test_perturbation": test_pert_name,
            "n_adversarial": len(adversarial),
            "adversarial_rate": len(adversarial) / max(1, int(np.ceil(len(train_pert_names) * top_pct))),
            "adversarial_pairs": adversarial,
        })
    
    results_df = pd.DataFrame(results)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "direction_flip_probe_results.csv"
    results_df.to_csv(output_path, index=False)
    LOGGER.info(f"Saved results to {output_path}")
    
    return results_df


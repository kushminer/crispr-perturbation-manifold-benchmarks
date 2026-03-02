#!/usr/bin/env python3
"""
Tangent Space Alignment: Local Procrustes/CCA Alignment

This module implements Epic 5 of the Manifold Law Diagnostic Suite.

Goal: Measure manifold alignment between train and test tangent spaces.
Tests whether LSFT works because train/test live in aligned subspaces.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def compute_tangent_space_alignment(
    train_data: np.ndarray,
    test_data: np.ndarray,
    n_components: int = 10,
) -> Dict[str, float]:
    """
    Compute alignment metrics between train and test tangent spaces.
    
    Args:
        train_data: Training data (n_samples_train × n_features)
        test_data: Test data (n_samples_test × n_features)
        n_components: Number of PCA components for tangent space
        
    Returns:
        Dictionary with alignment metrics:
        - procrustes_distance
        - cca_correlation
        - principal_angles (mean)
    """
    # Compute PCA for both spaces
    pca_train = PCA(n_components=n_components)
    pca_test = PCA(n_components=n_components)
    
    train_pcs = pca_train.fit_transform(train_data)
    test_pcs = pca_test.fit_transform(test_data)
    
    # Normalize
    train_pcs = train_pcs / (np.linalg.norm(train_pcs, axis=0, keepdims=True) + 1e-10)
    test_pcs = test_pcs / (np.linalg.norm(test_pcs, axis=0, keepdims=True) + 1e-10)
    
    # Procrustes distance
    # Find optimal rotation matrix
    R, _ = orthogonal_procrustes(train_pcs.T, test_pcs.T)
    aligned_test = test_pcs @ R.T
    procrustes_distance = np.linalg.norm(train_pcs.T - aligned_test.T, 'fro')
    
    # CCA correlation
    # CCA finds linear combinations with maximum correlation
    cca = CCA(n_components=min(n_components, min(train_pcs.shape[0], test_pcs.shape[0])))
    X_c, Y_c = cca.fit_transform(train_pcs, test_pcs)
    cca_correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(X_c.shape[1])]
    mean_cca_correlation = np.mean(cca_correlations)
    
    # Principal angles (subspace angles)
    # Compute angles between corresponding principal components
    angles = []
    for i in range(min(train_pcs.shape[1], test_pcs.shape[1])):
        cos_angle = np.abs(np.dot(train_pcs[:, i], test_pcs[:, i]))
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        angles.append(angle)
    mean_principal_angle = np.mean(angles) if angles else np.nan
    
    return {
        "procrustes_distance": float(procrustes_distance),
        "mean_cca_correlation": float(mean_cca_correlation),
        "mean_principal_angle": float(mean_principal_angle),
    }


def compute_local_tangent_alignment(
    Y_train: pd.DataFrame,
    Y_test: pd.DataFrame,
    similarities: np.ndarray,
    train_pert_names: List[str],
    test_pert_name: str,
    top_pct: float = 0.05,
    n_components: int = 10,
) -> Dict[str, float]:
    """
    Compute tangent space alignment for a local neighborhood.
    
    Args:
        Y_train: Training expression matrix (genes × perturbations)
        Y_test: Test expression matrix (genes × perturbations)
        similarities: Similarities to training perturbations
        train_pert_names: List of training perturbation names
        test_pert_name: Test perturbation name
        top_pct: Top percentage for local neighborhood
        n_components: Number of PCA components
        
    Returns:
        Dictionary with alignment metrics
    """
    # Get top-K% neighbors
    n_select = max(1, int(np.ceil(len(train_pert_names) * top_pct)))
    top_k_indices = np.argsort(similarities)[-n_select:]
    
    # Get local training data
    local_train_names = [train_pert_names[i] for i in top_k_indices]
    local_train_data = Y_train[local_train_names].values.T  # (n_neighbors × genes)
    
    # Get test data
    test_data = Y_test[test_pert_name].values.reshape(1, -1)  # (1 × genes)
    
    # For test, use the single vector repeated or extend neighborhood
    # We'll compare local train PCs to test vector's projection
    # For simplicity, compute alignment on local train space vs test point
    if local_train_data.shape[0] < n_components:
        n_components = min(n_components, local_train_data.shape[0] - 1)
    
    if n_components < 2:
        return {
            "procrustes_distance": np.nan,
            "mean_cca_correlation": np.nan,
            "mean_principal_angle": np.nan,
        }
    
    # Compute PCA on local training data
    pca_local = PCA(n_components=n_components)
    local_pcs = pca_local.fit_transform(local_train_data)  # (n_neighbors × n_components)
    
    # Project test point into local PCA space
    test_projected = pca_local.transform(test_data)  # (1 × n_components)
    
    # Compute alignment metrics (simplified for single test point)
    # Use projection distance as proxy
    if local_pcs.shape[0] > 1:
        mean_local_pc = np.mean(local_pcs, axis=0)
        alignment_distance = np.linalg.norm(test_projected - mean_local_pc)
        
        # Correlation between test projection and mean local projection
        if len(mean_local_pc) > 1:
            corr = np.corrcoef(test_projected.flatten(), mean_local_pc)[0, 1]
        else:
            corr = np.nan
    else:
        alignment_distance = np.nan
        corr = np.nan
    
    return {
        "procrustes_distance": float(alignment_distance) if not np.isnan(alignment_distance) else np.nan,
        "mean_cca_correlation": float(corr) if not np.isnan(corr) else np.nan,
        "mean_principal_angle": np.nan,  # Requires paired subspaces
        "tangent_alignment_score": float(corr) if not np.isnan(corr) else np.nan,
    }


def run_tangent_alignment_analysis(
    Y_train: pd.DataFrame,
    Y_test: pd.DataFrame,
    similarities_matrix: np.ndarray,
    train_pert_names: List[str],
    test_pert_names: List[str],
    output_dir: Path,
    top_pct: float = 0.05,
    n_components: int = 10,
) -> pd.DataFrame:
    """
    Run tangent space alignment analysis for all test perturbations.
    
    Args:
        Y_train: Training expression matrix
        Y_test: Test expression matrix
        similarities_matrix: Similarity matrix (n_test × n_train)
        train_pert_names: List of training perturbation names
        test_pert_names: List of test perturbation names
        output_dir: Output directory
        top_pct: Top percentage for local neighborhood
        n_components: Number of PCA components
        
    Returns:
        DataFrame with alignment scores
    """
    LOGGER.info("Running tangent space alignment analysis...")
    
    results = []
    
    for test_idx, test_pert_name in enumerate(test_pert_names):
        similarities = similarities_matrix[test_idx, :]
        
        alignment = compute_local_tangent_alignment(
            Y_train=Y_train,
            Y_test=Y_test,
            similarities=similarities,
            train_pert_names=train_pert_names,
            test_pert_name=test_pert_name,
            top_pct=top_pct,
            n_components=n_components,
        )
        
        results.append({
            "test_perturbation": test_pert_name,
            **alignment,
        })
    
    results_df = pd.DataFrame(results)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tangent_alignment_results.csv"
    results_df.to_csv(output_path, index=False)
    LOGGER.info(f"Saved results to {output_path}")
    
    return results_df


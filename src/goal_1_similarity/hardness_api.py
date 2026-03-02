#!/usr/bin/env python3
"""
Hardness API: Compute target-specific similarity and hardness metrics.

This module provides a clean interface to compute similarity and hardness
metrics for any target perturbation in a given embedding space (baseline-specific B matrices).

Hardness is defined as:
- hardness_k = 1 - mean_topk[k]
- hardness_max = 1 - max_sim

Where similarity is computed using cosine similarity in the embedding space.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_target_similarity(
    target_embedding: np.ndarray,
    train_embeddings: np.ndarray,
    k_values: List[int] = [1, 5, 10, 20],
) -> Dict:
    """
    Compute similarity and hardness metrics for a target perturbation.
    
    Args:
        target_embedding: Target perturbation embedding vector (d,) or (1, d)
        train_embeddings: Training perturbation embeddings (d, n_train) or (n_train, d)
        k_values: List of k values for computing mean_topk similarity
        
    Returns:
        Dictionary with keys:
            - "max_sim": float, maximum similarity to any training perturbation
            - "mean_topk": dict[int, float], mean top-k similarity for each k in k_values
            - "all_sims": np.ndarray, all similarities to training perturbations (shape: (n_train,))
            - "hardness_max": float, 1 - max_sim
            - "hardness_k": dict[int, float], hardness_k = 1 - mean_topk[k] for each k
    """
    # Ensure target_embedding is 2D (1, d)
    if target_embedding.ndim == 1:
        target_embedding = target_embedding.reshape(1, -1)
    target_d = target_embedding.shape[1]  # Feature dimension
    
    # Ensure train_embeddings is 2D (n_train, d)
    if train_embeddings.ndim == 1:
        train_embeddings = train_embeddings.reshape(1, -1)
    
    # Check if we need to transpose: cosine_similarity expects (n_samples, n_features)
    # Check feature dimension matching first to determine correct orientation
    if train_embeddings.ndim == 2:
        # If feature dimensions already match, keep as is
        if train_embeddings.shape[1] == target_d:
            # Already in correct format (n_train, d)
            pass
        elif train_embeddings.shape[0] == target_d:
            # Need to transpose: (d, n_train) -> (n_train, d)
            train_embeddings = train_embeddings.T
        elif (train_embeddings.shape[0] < train_embeddings.shape[1] and 
              train_embeddings.shape[0] <= 100):
            # Heuristic: small first dimension suggests (d, n_train) format
            # But only transpose if second dimension matches target_d
            if train_embeddings.shape[1] == target_d:
                train_embeddings = train_embeddings.T
    
    # Final check: ensure feature dimensions match
    if train_embeddings.shape[1] != target_d:
        raise ValueError(
            f"Feature dimension mismatch: target has {target_d} features, "
            f"but train_embeddings has {train_embeddings.shape[1]} features. "
            f"train_embeddings shape: {train_embeddings.shape}, target shape: {target_embedding.shape}"
        )
    
    # Compute cosine similarity between target and all training perturbations
    # cosine_similarity expects (n_samples, n_features)
    similarities = cosine_similarity(target_embedding, train_embeddings)
    similarities = similarities.flatten()  # Shape: (n_train,)
    
    # Max similarity
    max_sim = float(np.max(similarities))
    
    # Mean top-k similarity for each k
    mean_topk = {}
    sorted_sims = np.sort(similarities)
    n_train = len(similarities)
    
    for k in k_values:
        k_actual = min(k, n_train)  # Don't exceed available training perturbations
        if k_actual > 0:
            top_k_sims = sorted_sims[-k_actual:]
            mean_topk[k] = float(np.mean(top_k_sims))
        else:
            mean_topk[k] = 0.0
    
    # Compute hardness metrics
    hardness_max = 1.0 - max_sim
    hardness_k = {k: 1.0 - sim for k, sim in mean_topk.items()}
    
    return {
        "max_sim": max_sim,
        "mean_topk": mean_topk,
        "all_sims": similarities,
        "hardness_max": hardness_max,
        "hardness_k": hardness_k,
    }


def compute_multiple_targets_similarity(
    target_embeddings: np.ndarray,
    train_embeddings: np.ndarray,
    target_names: List[str],
    k_values: List[int] = [1, 5, 10, 20],
) -> Dict[str, Dict]:
    """
    Compute similarity and hardness metrics for multiple target perturbations.
    
    Args:
        target_embeddings: Target perturbation embeddings (d, n_targets) or (n_targets, d)
        train_embeddings: Training perturbation embeddings (d, n_train) or (n_train, d)
        target_names: List of target perturbation names (length n_targets)
        k_values: List of k values for computing mean_topk similarity
        
    Returns:
        Dictionary mapping target_name -> similarity/hardness dict (same format as compute_target_similarity)
    """
    # Ensure target_embeddings is 2D (n_targets, d)
    if target_embeddings.ndim == 1:
        target_embeddings = target_embeddings.reshape(1, -1)
    
    # Ensure train_embeddings is 2D (n_train, d)
    if train_embeddings.ndim == 1:
        train_embeddings = train_embeddings.reshape(1, -1)
    
    # Determine correct orientation by checking feature dimensions match
    # If dimensions don't match, try transposing
    if train_embeddings.ndim == 2 and target_embeddings.ndim == 2:
        # Get feature dimension from train (should be second dimension in correct format)
        # Check both possible orientations
        if train_embeddings.shape[1] == target_embeddings.shape[1]:
            # Both already in (n_samples, d) format - keep as is
            pass
        elif train_embeddings.shape[0] == target_embeddings.shape[1]:
            # Train needs transpose: (d, n_train) -> (n_train, d)
            train_embeddings = train_embeddings.T
        elif train_embeddings.shape[1] == target_embeddings.shape[0]:
            # Targets need transpose: (d, n_targets) -> (n_targets, d)
            target_embeddings = target_embeddings.T
        elif (train_embeddings.shape[0] < train_embeddings.shape[1] and 
              train_embeddings.shape[0] <= 100 and
              train_embeddings.shape[1] == target_embeddings.shape[1]):
            # Heuristic: transpose train if it helps match features
            train_embeddings = train_embeddings.T
        elif (target_embeddings.shape[0] < target_embeddings.shape[1] and 
              target_embeddings.shape[0] <= 100 and
              target_embeddings.shape[1] == train_embeddings.shape[1]):
            # Heuristic: transpose targets if it helps match features
            target_embeddings = target_embeddings.T
    
    # Final check: ensure feature dimensions match
    if train_embeddings.shape[1] != target_embeddings.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: target has {target_embeddings.shape[1]} features, "
            f"but train_embeddings has {train_embeddings.shape[1]} features. "
            f"target shape: {target_embeddings.shape}, train shape: {train_embeddings.shape}"
        )
    
    if len(target_names) != target_embeddings.shape[0]:
        raise ValueError(
            f"Number of target names ({len(target_names)}) must match "
            f"number of target embeddings ({target_embeddings.shape[0]})"
        )
    
    # Compute all similarities at once (n_targets × n_train)
    all_similarities = cosine_similarity(target_embeddings, train_embeddings)
    
    results = {}
    n_train = train_embeddings.shape[0]
    
    for i, target_name in enumerate(target_names):
        similarities = all_similarities[i, :]
        
        # Max similarity
        max_sim = float(np.max(similarities))
        
        # Mean top-k similarity for each k
        mean_topk = {}
        sorted_sims = np.sort(similarities)
        
        for k in k_values:
            k_actual = min(k, n_train)
            if k_actual > 0:
                top_k_sims = sorted_sims[-k_actual:]
                mean_topk[k] = float(np.mean(top_k_sims))
            else:
                mean_topk[k] = 0.0
        
        # Compute hardness metrics
        hardness_max = 1.0 - max_sim
        hardness_k = {k: 1.0 - sim for k, sim in mean_topk.items()}
        
        results[target_name] = {
            "max_sim": max_sim,
            "mean_topk": mean_topk,
            "all_sims": similarities,
            "hardness_max": hardness_max,
            "hardness_k": hardness_k,
        }
    
    return results


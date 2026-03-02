#!/usr/bin/env python3
"""
Validate cross-dataset embedding parity (K562 and RPE1).

This script compares precomputed embeddings with original R/Python script outputs
to ensure numerical parity, similar to the embedding parity validation.

Usage:
    python -m goal_2_baselines.validate_cross_dataset_embeddings \
        --dataset replogle_k562_essential \
        --new_embedding results/replogle_k562_pert_emb_pca10_seed1.tsv \
        --legacy_embedding validation/legacy_runs/replogle_k562_pert_emb_pca10_seed1.tsv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def load_embedding_tsv(tsv_path: Path) -> pd.DataFrame:
    """Load embedding TSV file (rows = dimensions, columns = perturbations)."""
    df = pd.read_csv(tsv_path, sep="\t", index_col=0)
    return df


def compare_embeddings(
    new_emb: pd.DataFrame,
    legacy_emb: pd.DataFrame,
    tolerance: float = 1e-6,
) -> dict:
    """
    Compare two embedding matrices and compute parity metrics.
    
    Args:
        new_emb: New embedding (dims × perts)
        legacy_emb: Legacy embedding (dims × perts)
        tolerance: Numerical tolerance for exact matching
    
    Returns:
        Dictionary with comparison metrics
    """
    # Align by perturbation names (columns)
    common_perts = set(new_emb.columns) & set(legacy_emb.columns)
    if len(common_perts) == 0:
        raise ValueError("No common perturbations found between embeddings")
    
    new_aligned = new_emb[sorted(common_perts)]
    legacy_aligned = legacy_emb[sorted(common_perts)]
    
    # Align by dimensions (rows) if possible
    if set(new_aligned.index) != set(legacy_aligned.index):
        LOGGER.warning("Dimension names don't match, using positional alignment")
        # Use positional alignment
        min_dims = min(len(new_aligned.index), len(legacy_aligned.index))
        new_aligned = new_aligned.iloc[:min_dims]
        legacy_aligned = legacy_aligned.iloc[:min_dims]
    else:
        new_aligned = new_aligned.loc[sorted(new_aligned.index)]
        legacy_aligned = legacy_aligned.loc[sorted(legacy_aligned.index)]
    
    # Convert to numpy
    new_arr = new_aligned.values
    legacy_arr = legacy_aligned.values
    
    # Compute metrics per perturbation (column-wise)
    cosine_sims = []
    pearson_rs = []
    
    for i in range(new_arr.shape[1]):
        new_vec = new_arr[:, i]
        legacy_vec = legacy_arr[:, i]
        
        # Cosine similarity
        if np.linalg.norm(new_vec) > 0 and np.linalg.norm(legacy_vec) > 0:
            cos_sim = 1 - cosine(new_vec, legacy_vec)
            cosine_sims.append(cos_sim)
        
        # Pearson correlation
        if np.std(new_vec) > 0 and np.std(legacy_vec) > 0:
            r, _ = pearsonr(new_vec, legacy_vec)
            pearson_rs.append(r)
    
    # Compute aggregate metrics
    mean_cosine = np.mean(cosine_sims) if cosine_sims else np.nan
    min_cosine = np.min(cosine_sims) if cosine_sims else np.nan
    mean_pearson = np.mean(pearson_rs) if pearson_rs else np.nan
    
    # Check for sign flip (common in PCA)
    if mean_pearson < 0:
        LOGGER.info("Negative correlation detected, checking for sign flip...")
        # Try flipping sign
        new_arr_flipped = -new_arr
        pearson_rs_flipped = []
        for i in range(new_arr.shape[1]):
            new_vec = new_arr_flipped[:, i]
            legacy_vec = legacy_arr[:, i]
            if np.std(new_vec) > 0 and np.std(legacy_vec) > 0:
                r, _ = pearsonr(new_vec, legacy_vec)
                pearson_rs_flipped.append(r)
        mean_pearson_flipped = np.mean(pearson_rs_flipped) if pearson_rs_flipped else np.nan
        if mean_pearson_flipped > mean_pearson:
            LOGGER.info("Sign flip improves correlation, using flipped version")
            mean_pearson = mean_pearson_flipped
            new_arr = new_arr_flipped
    
    # Absolute difference
    abs_diff = np.abs(new_arr - legacy_arr)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    # Check if within tolerance
    within_tolerance = max_abs_diff < tolerance
    
    return {
        "mean_cosine": mean_cosine,
        "min_cosine": min_cosine,
        "mean_pearson": mean_pearson,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "within_tolerance": within_tolerance,
        "n_perturbations": len(common_perts),
        "n_dimensions": new_arr.shape[0],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate cross-dataset embedding parity"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (for logging)",
    )
    parser.add_argument(
        "--new_embedding",
        type=Path,
        required=True,
        help="Path to new precomputed embedding TSV",
    )
    parser.add_argument(
        "--legacy_embedding",
        type=Path,
        required=True,
        help="Path to legacy embedding TSV (from original script)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Numerical tolerance for comparison (default: 1e-6)",
    )
    
    args = parser.parse_args()
    
    # Load embeddings
    LOGGER.info(f"Loading new embedding: {args.new_embedding}")
    new_emb = load_embedding_tsv(args.new_embedding)
    LOGGER.info(f"  Shape: {new_emb.shape} (dims × perts)")
    
    LOGGER.info(f"Loading legacy embedding: {args.legacy_embedding}")
    legacy_emb = load_embedding_tsv(args.legacy_embedding)
    LOGGER.info(f"  Shape: {legacy_emb.shape} (dims × perts)")
    
    # Compare
    LOGGER.info("Comparing embeddings...")
    metrics = compare_embeddings(new_emb, legacy_emb, tolerance=args.tolerance)
    
    # Print results
    print("\n" + "=" * 60)
    print("Cross-Dataset Embedding Parity Results")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"New embedding: {args.new_embedding}")
    print(f"Legacy embedding: {args.legacy_embedding}")
    print()
    print("Metrics:")
    print(f"  Mean cosine similarity: {metrics['mean_cosine']:.8f}")
    print(f"  Min cosine similarity:  {metrics['min_cosine']:.8f}")
    print(f"  Mean Pearson r:        {metrics['mean_pearson']:.8f}")
    print(f"  Max absolute diff:     {metrics['max_abs_diff']:.2e}")
    print(f"  Mean absolute diff:    {metrics['mean_abs_diff']:.2e}")
    print()
    print(f"Perturbations compared: {metrics['n_perturbations']}")
    print(f"Dimensions: {metrics['n_dimensions']}")
    print()
    
    # Check thresholds
    thresholds_met = {
        "mean_cosine >= 0.99": metrics["mean_cosine"] >= 0.99,
        "min_cosine >= 0.95": metrics["min_cosine"] >= 0.95,
        "mean_pearson >= 0.99": abs(metrics["mean_pearson"]) >= 0.99,
        "within_tolerance": metrics["within_tolerance"],
    }
    
    print("Thresholds:")
    all_passed = True
    for threshold, passed in thresholds_met.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {threshold}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✓ All parity checks passed!")
        return 0
    else:
        print("✗ Some parity checks failed")
        return 1


if __name__ == "__main__":
    exit(main())


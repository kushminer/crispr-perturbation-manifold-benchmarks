#!/usr/bin/env python3
"""
Precompute cross-dataset perturbation embeddings (K562 and RPE1).

This script generates PCA perturbation embeddings for K562 and RPE1 datasets,
which are then used in cross-dataset baseline models (lpm_k562PertEmb, lpm_rpe1PertEmb).

Usage:
    python -m goal_2_baselines.precompute_cross_dataset_embeddings \
        --dataset replogle_k562_essential \
        --adata_path data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad \
        --output_path results/replogle_k562_pert_emb_pca10_seed1.tsv \
        --pca_dim 10 \
        --seed 1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from embeddings.registry import load

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def precompute_pca_perturbation_embedding(
    adata_path: Path,
    output_path: Path,
    pca_dim: int = 10,
    seed: int = 1,
) -> None:
    """
    Precompute PCA perturbation embeddings for a dataset.
    
    This replicates the behavior of extract_pert_embedding_pca.R/Python
    for cross-dataset baseline use.
    
    Args:
        adata_path: Path to perturb_processed.h5ad
        output_path: Path to save TSV file (rows = dimensions, columns = perturbations)
        pca_dim: PCA dimension
        seed: Random seed
    """
    LOGGER.info(f"Precomputing PCA perturbation embeddings")
    LOGGER.info(f"  Input: {adata_path}")
    LOGGER.info(f"  Output: {output_path}")
    LOGGER.info(f"  PCA dim: {pca_dim}, seed: {seed}")
    
    # Load embedding using validated loader
    result = load(
        "pca_perturbation",
        adata_path=str(adata_path),
        n_components=pca_dim,
        seed=seed,
    )
    
    # Result format: values is (dims × perturbations), item_labels are perturbation names
    # We want: rows = dimensions, columns = perturbations
    # Format matches extract_pert_embedding_pca.py output
    emb_df = pd.DataFrame(
        result.values,
        index=result.dim_labels,
        columns=result.item_labels,
    )
    
    # Save as TSV (matching original format: no index column name, but index saved)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    emb_df.to_csv(output_path, sep="\t", index=True)
    
    LOGGER.info(f"Saved embedding matrix: {emb_df.shape} (dims × perts)")
    LOGGER.info(f"  Dimensions: {len(result.dim_labels)}")
    LOGGER.info(f"  Perturbations: {len(result.item_labels)}")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute cross-dataset perturbation embeddings"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["replogle_k562_essential", "replogle_rpe1", "replogle_rpe1_essential"],
        help="Dataset name",
    )
    parser.add_argument(
        "--adata_path",
        type=Path,
        required=True,
        help="Path to perturb_processed.h5ad",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to save TSV file",
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=10,
        help="PCA dimension (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: 1)",
    )
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not args.adata_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.adata_path}")
    
    # Precompute embedding
    precompute_pca_perturbation_embedding(
        adata_path=args.adata_path,
        output_path=args.output_path,
        pca_dim=args.pca_dim,
        seed=args.seed,
    )
    
    LOGGER.info("Precomputation complete!")
    return 0


if __name__ == "__main__":
    exit(main())


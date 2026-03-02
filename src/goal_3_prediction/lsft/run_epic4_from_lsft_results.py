#!/usr/bin/env python3
"""
Run Epic 4 (Direction-Flip Probe) using existing LSFT results and data.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import anndata as ad
from scipy.stats import pearsonr

# Add src to path
base_dir = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(base_dir / "src"))

from goal_2_baselines.split_logic import load_split_config
from goal_2_baselines.baseline_runner import compute_pseudobulk_expression_changes
from goal_2_baselines.baseline_types import BaselineType, get_baseline_config
from goal_2_baselines.baseline_runner import construct_gene_embeddings, construct_pert_embeddings
from goal_3_prediction.lsft.lsft import compute_all_perturbation_similarities
from goal_3_prediction.lsft.direction_flip_probe import run_direction_flip_probe

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Direction-Flip Probe (Epic 4)")
    parser.add_argument("--adata_path", type=Path, required=True)
    parser.add_argument("--split_config", type=Path, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--baseline_type", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--top_pct", type=float, default=0.05)
    parser.add_argument("--pca_dim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    
    args = parser.parse_args()
    
    # Load data
    adata = ad.read_h5ad(args.adata_path)
    split_config = load_split_config(args.split_config)
    
    # Compute Y matrix
    Y_df, split_labels = compute_pseudobulk_expression_changes(adata, split_config, seed=args.seed)
    
    train_pert_names = split_labels.get("train", [])
    test_pert_names = split_labels.get("test", [])
    
    Y_train = Y_df[train_pert_names]
    Y_test = Y_df[test_pert_names]
    
    # Get baseline config
    baseline_type = BaselineType(args.baseline_type)
    config = get_baseline_config(baseline_type, pca_dim=args.pca_dim, ridge_penalty=0.1, seed=args.seed)
    
    # Construct embeddings to get similarity matrix
    gene_names = Y_df.index.tolist()
    gene_name_mapping = None
    if "gene_name" in adata.var.columns:
        gene_name_mapping = dict(zip(adata.var_names, adata.var["gene_name"]))
    
    gene_embedding_args = config.gene_embedding_args.copy() if config.gene_embedding_args else {}
    if gene_name_mapping and config.gene_embedding_source in ["scgpt", "scfoundation"]:
        gene_embedding_args["gene_name_mapping"] = gene_name_mapping
    
    pert_embedding_args = config.pert_embedding_args.copy() if config.pert_embedding_args else {}
    if config.pert_embedding_source in ["k562_pca", "rpe1_pca", "file"]:
        pert_embedding_args["target_gene_names"] = gene_names
    
    Y_train_np = Y_train.values
    
    _, _ = construct_gene_embeddings(
        source=config.gene_embedding_source,
        train_data=Y_train_np,
        gene_names=gene_names,
        pca_dim=args.pca_dim,
        seed=args.seed,
        embedding_args=gene_embedding_args,
    )
    
    B_train_baseline, _, _, B_test_baseline, _ = construct_pert_embeddings(
        source=config.pert_embedding_source,
        train_data=Y_train_np,
        pert_names=train_pert_names,
        pca_dim=args.pca_dim,
        seed=args.seed,
        embedding_args=pert_embedding_args,
        test_data=Y_test.values,
        test_pert_names=test_pert_names,
    )
    
    # Compute similarity matrix
    similarities_matrix = compute_all_perturbation_similarities(B_test_baseline, B_train_baseline)
    
    # Run direction-flip probe
    results_df = run_direction_flip_probe(
        similarities_matrix=similarities_matrix,
        train_pert_names=train_pert_names,
        test_pert_names=test_pert_names,
        Y_train=Y_train,
        Y_test=Y_test,
        output_dir=args.output_dir,
        top_pct=args.top_pct,
        conflict_threshold=-0.2,
    )
    
    # Save with dataset and baseline info
    results_df['dataset'] = args.dataset_name
    results_df['baseline'] = args.baseline_type
    
    output_path = args.output_dir / f"direction_flip_probe_{args.dataset_name}_{args.baseline_type}.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"\nDirection-Flip Probe Results:")
    print(f"Mean adversarial rate: {results_df['adversarial_rate'].mean():.3f}")
    print(f"Total adversarial pairs: {results_df['n_adversarial'].sum()}")
    print(f"Results saved to: {output_path}")


#!/usr/bin/env python3
"""
Run Epic 2 (Mechanism-Mismatch Ablation) using LSFT k-sweep with functional class filtering.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import anndata as ad

# Add src to path
base_dir = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(base_dir / "src"))

from goal_2_baselines.split_logic import load_split_config
from goal_2_baselines.baseline_runner import compute_pseudobulk_expression_changes
from goal_2_baselines.baseline_types import BaselineType
from goal_3_prediction.lsft.mechanism_ablation import load_functional_annotations
from goal_3_prediction.lsft.lsft_k_sweep import evaluate_lsft_with_k_list

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Mechanism-Mismatch Ablation (Epic 2)")
    parser.add_argument("--adata_path", type=Path, required=True)
    parser.add_argument("--split_config", type=Path, required=True)
    parser.add_argument("--annotation_path", type=Path, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--baseline_type", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--k_list", type=int, nargs="+", default=[3, 5, 10, 20])
    parser.add_argument("--pca_dim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    
    args = parser.parse_args()
    
    # Load functional annotations
    functional_annotations = load_functional_annotations(args.annotation_path)
    
    # Load data to get perturbation names
    adata = ad.read_h5ad(args.adata_path)
    split_config = load_split_config(args.split_config)
    
    # Compute Y matrix to get perturbation names
    Y_df, split_labels = compute_pseudobulk_expression_changes(adata, split_config, seed=args.seed)
    test_pert_names = split_labels.get("test", [])
    
    # Run original LSFT (control)
    print("Running original LSFT (control)...")
    results_original = evaluate_lsft_with_k_list(
        adata_path=args.adata_path,
        split_config_path=args.split_config,
        baseline_type=BaselineType(args.baseline_type),
        dataset_name=args.dataset_name,
        output_dir=args.output_dir / "original",
        k_list=args.k_list,
        pca_dim=args.pca_dim,
        ridge_penalty=0.1,
        seed=args.seed,
    )
    
    # Run ablated LSFT (same-class removed)
    # For now, we'll compute by modifying the filter function
    # This requires extending evaluate_lsft_with_k_list to accept functional annotations
    print("\nRunning ablated LSFT (same-class neighbors removed)...")
    print("Note: Full ablation requires extending lsft_k_sweep.py")
    
    # Create ablation results by comparing functional classes
    ablation_results = []
    
    for _, row in results_original.iterrows():
        test_pert = row["test_perturbation"]
        test_class = functional_annotations.get(test_pert, "Unknown")
        
        ablation_results.append({
            "dataset": args.dataset_name,
            "baseline_type": args.baseline_type,
            "test_perturbation": test_pert,
            "functional_class": test_class,
            "k": row["k"],
            "original_pearson_r": row["performance_local_pearson_r"],
            "original_l2": row["performance_local_l2"],
            # Placeholder - full implementation needs extended lsft_k_sweep
            "ablated_pearson_r": np.nan,
            "ablated_l2": np.nan,
            "delta_r": np.nan,
            "delta_l2": np.nan,
        })
    
    results_df = pd.DataFrame(ablation_results)
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"mechanism_ablation_{args.dataset_name}_{args.baseline_type}.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"\nSaved results to {output_path}")
    print(f"Total perturbations analyzed: {len(results_df)}")
    print(f"Classes represented: {results_df['functional_class'].nunique()}")


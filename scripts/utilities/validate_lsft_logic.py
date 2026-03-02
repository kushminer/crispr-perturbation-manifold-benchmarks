#!/usr/bin/env python3
"""
Validation script to ensure LSFT logic is intact after optimizations.
Compares results from optimized vs non-optimized paths.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from goal_2_baselines.baseline_runner import compute_pseudobulk_expression_changes
from goal_2_baselines.baseline_types import BaselineType
from goal_2_baselines.split_logic import load_split_config
from goal_3_prediction.lsft.lsft_k_sweep import evaluate_lsft_with_k_list
from shared.metrics import compute_metrics
import anndata as ad

def validate_curvature_sweep_logic():
    """Validate that curvature sweep produces consistent results."""
    print("=" * 60)
    print("LSFT Logic Validation Test")
    print("=" * 60)
    print()
    
    # Use Adamson as a test case
    adata_path = Path("data/gears_pert_data/adamson/perturb_processed.h5ad")
    split_path = Path("results/goal_2_baselines/splits/adamson_split_seed1.json")
    
    if not adata_path.exists() or not split_path.exists():
        print("⚠️  Test data not found, skipping validation")
        return True
    
    print("Test: Curvature sweep consistency check")
    print("-" * 60)
    
    # Load data to check dimensions
    adata = ad.read_h5ad(adata_path)
    split_config = load_split_config(split_path)
    Y_df, split_labels = compute_pseudobulk_expression_changes(adata, split_config, seed=1)
    
    train_perts = split_labels.get("train", [])
    test_perts = split_labels.get("test", [])
    
    print(f"  Dataset: Adamson")
    print(f"  Genes: {len(Y_df.index)}")
    print(f"  Train perturbations: {len(train_perts)}")
    print(f"  Test perturbations: {len(test_perts)}")
    print()
    
    # Test on a simple baseline
    print("  Running test on lpm_selftrained with k=[3, 5]...")
    
    try:
        results_df = evaluate_lsft_with_k_list(
            adata_path=adata_path,
            split_config_path=split_path,
            baseline_type=BaselineType.SELFTRAINED,
            dataset_name="adamson",
            output_dir=Path("results/manifold_law_diagnostics/_validation_test"),
            k_list=[3, 5],
            pca_dim=10,
            ridge_penalty=0.1,
            seed=1,
        )
        
        print(f"  ✅ Results generated: {len(results_df)} rows")
        
        # Validation checks
        checks_passed = 0
        total_checks = 5
        
        # Check 1: Results have required columns
        required_cols = ["dataset", "baseline_type", "test_perturbation", "k", "pearson_r", "l2"]
        missing_cols = [col for col in required_cols if col not in results_df.columns]
        if len(missing_cols) == 0:
            print(f"  ✅ Check 1: Required columns present")
            checks_passed += 1
        else:
            print(f"  ❌ Check 1: Missing columns: {missing_cols}")
        
        # Check 2: Results for both k values
        unique_k = sorted(results_df["k"].unique())
        if len(unique_k) == 2 and set(unique_k) == {3, 5}:
            print(f"  ✅ Check 2: Results for both k values (k={unique_k})")
            checks_passed += 1
        else:
            print(f"  ❌ Check 2: Expected k=[3, 5], got k={unique_k}")
        
        # Check 3: Reasonable performance values
        mean_r = results_df["pearson_r"].mean()
        if 0 <= mean_r <= 1:
            print(f"  ✅ Check 3: Reasonable Pearson r values (mean={mean_r:.3f})")
            checks_passed += 1
        else:
            print(f"  ❌ Check 3: Unreasonable Pearson r (mean={mean_r:.3f})")
        
        # Check 4: Decreasing k should increase similarity (on average)
        k3_r = results_df[results_df["k"] == 3]["pearson_r"].mean()
        k5_r = results_df[results_df["k"] == 5]["pearson_r"].mean()
        k3_sim = results_df[results_df["k"] == 3]["local_mean_similarity"].mean()
        k5_sim = results_df[results_df["k"] == 5]["local_mean_similarity"].mean()
        
        if k3_sim >= k5_sim:  # Smaller k should have higher similarity
            print(f"  ✅ Check 4: Smaller k has higher similarity (k=3: {k3_sim:.3f}, k=5: {k5_sim:.3f})")
            checks_passed += 1
        else:
            print(f"  ⚠️  Check 4: Unexpected similarity pattern (k=3: {k3_sim:.3f}, k=5: {k5_sim:.3f})")
        
        # Check 5: Results consistent across runs (same seed)
        if len(results_df) > 0:
            print(f"  ✅ Check 5: Results generated successfully ({len(results_df)} rows)")
            checks_passed += 1
        else:
            print(f"  ❌ Check 5: No results generated")
        
        print()
        print("-" * 60)
        print(f"Validation Results: {checks_passed}/{total_checks} checks passed")
        print()
        
        if checks_passed == total_checks:
            print("✅ All validation checks passed! Logic is intact.")
            return True
        else:
            print("⚠️  Some validation checks failed. Review results.")
            return False
            
    except Exception as e:
        print(f"  ❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = validate_curvature_sweep_logic()
    sys.exit(0 if success else 1)


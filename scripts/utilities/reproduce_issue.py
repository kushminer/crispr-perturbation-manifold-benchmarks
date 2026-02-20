
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import logging

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

from goal_3_prediction.lsft.lsft_single_cell import evaluate_lsft_single_cell
from goal_2_baselines.baseline_types import BaselineType

# Setup logging
logging.basicConfig(level=logging.INFO)

def create_synthetic_data(output_dir):
    # Create synthetic AnnData
    n_obs = 100
    n_vars = 50
    X = np.random.randn(n_obs, n_vars)
    obs = pd.DataFrame({
        "condition": ["ctrl"] * 20 + ["pert1"] * 40 + ["pert2"] * 40,
        "clean_condition": ["ctrl"] * 20 + ["pert1"] * 40 + ["pert2"] * 40
    })
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    
    adata_path = output_dir / "synthetic.h5ad"
    adata.write(adata_path)
    
    # Create split config
    split_config = {
        "train": ["pert1", "ctrl"],
        "test": ["pert2"],
        "val": []
    }
    import json
    split_path = output_dir / "split.json"
    with open(split_path, "w") as f:
        json.dump(split_config, f)
        
    return adata_path, split_path

def main():
    output_dir = Path("deliverables/reproduce_output")
    output_dir.mkdir(exist_ok=True)
    
    adata_path, split_path = create_synthetic_data(output_dir)
    
    print("Running SELFTRAINED...")
    results_self = evaluate_lsft_single_cell(
        adata_path=adata_path,
        split_config_path=split_path,
        baseline_type=BaselineType.SELFTRAINED,
        dataset_name="synthetic",
        output_dir=output_dir,
        top_pcts=[0.1],
        pca_dim=5,
        n_cells_per_pert=10
    )
    
    print("Running RANDOM_PERT_EMB...")
    # This should now produce DIFFERENT results from SELFTRAINED
    results_random = evaluate_lsft_single_cell(
        adata_path=adata_path,
        split_config_path=split_path,
        baseline_type=BaselineType.RANDOM_PERT_EMB,
        dataset_name="synthetic",
        output_dir=output_dir,
        top_pcts=[0.1],
        pca_dim=5,
        n_cells_per_pert=10
    )
    
    # Compare results
    print("\nComparison:")
    print(f"SELFTRAINED mean pearson: {results_self['lsft_pearson_r'].mean()}")
    print(f"RANDOM mean pearson: {results_random['lsft_pearson_r'].mean()}")
    
    if not np.allclose(results_self['lsft_pearson_r'], results_random['lsft_pearson_r']):
        print("\nSUCCESS: Results are different.")
    else:
        print("\nFAILURE: Results are still identical.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run only GEARS baseline for single-cell analysis across all datasets.
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from goal_2_baselines.baseline_runner_single_cell import run_all_baselines_single_cell
from goal_2_baselines.baseline_types import BaselineType
from goal_2_baselines.split_logic import load_split_config
import anndata as ad
import json

# Configuration
datasets = {
    "adamson": {
        "adata_path": "data/gears_pert_data/adamson/perturb_processed.h5ad",
    },
    "k562": {
        "adata_path": "data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad",
    },
    "rpe1": {
        "adata_path": "data/gears_pert_data/replogle_rpe1_essential/perturb_processed.h5ad",
    },
}

n_cells_per_pert = 50
pca_dim = 10
ridge_penalty = 0.1
seed = 1

results_dir = Path("results/single_cell_analysis")
results_dir.mkdir(parents=True, exist_ok=True)

# Run only GEARS
baselines = [BaselineType.GEARS_PERT_EMB]

for dataset_name, config in datasets.items():
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"Dataset: {dataset_name}")
    LOGGER.info(f"{'='*60}")
    
    adata_path = Path(config["adata_path"])
    
    if not adata_path.exists():
        LOGGER.warning(f"Data file not found: {adata_path}")
        continue
    
    # Create or load split config
    split_config_path = results_dir / f"{dataset_name}_split_config.json"
    
    if not split_config_path.exists():
        LOGGER.info("Creating train/test split...")
        adata = ad.read_h5ad(adata_path)
        
        # Get all perturbation conditions (excluding ctrl)
        conditions = [c for c in adata.obs["condition"].unique() if c != "ctrl"]
        
        # Simple random 80/20 split
        import numpy as np
        np.random.seed(seed)
        np.random.shuffle(conditions)
        
        n_train = int(0.8 * len(conditions))
        train_conditions = conditions[:n_train]
        test_conditions = conditions[n_train:]
        
        # Include ctrl in all splits
        split_config = {
            "train": list(train_conditions) + ["ctrl"],
            "test": list(test_conditions),
        }
        
        with open(split_config_path, "w") as f:
            json.dump(split_config, f, indent=2)
        
        LOGGER.info(f"Created split: {len(train_conditions)} train, {len(test_conditions)} test")
    else:
        LOGGER.info(f"Using existing split config: {split_config_path}")
    
    # Run GEARS baseline
    output_dir = results_dir / dataset_name
    
    try:
        results_df = run_all_baselines_single_cell(
            adata_path=adata_path,
            split_config_path=split_config_path,
            output_dir=output_dir,
            baseline_types=baselines,
            pca_dim=pca_dim,
            ridge_penalty=ridge_penalty,
            seed=seed,
            n_cells_per_pert=n_cells_per_pert,
            cell_embedding_method="cell_pca",
        )
        
        LOGGER.info(f"\n{dataset_name} GEARS Results:")
        LOGGER.info(f"\n{results_df.to_string()}")
        
    except Exception as e:
        LOGGER.error(f"Failed to run {dataset_name}: {e}")
        import traceback
        traceback.print_exc()

LOGGER.info("\n" + "="*60)
LOGGER.info("GEARS SINGLE-CELL BASELINE EVALUATION COMPLETE")
LOGGER.info("="*60)


#!/usr/bin/env python3
"""
Comprehensive single-cell analysis update script.
Runs all baselines, LSFT, and LOGO, then generates updated reports.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from goal_2_baselines.baseline_runner_single_cell import run_all_baselines_single_cell
from goal_2_baselines.baseline_types import BaselineType
from goal_3_prediction.lsft.lsft_single_cell import run_lsft_single_cell_all_baselines
from goal_4_logo.logo_single_cell import run_logo_single_cell
import anndata as ad
import json

# Configuration
datasets = {
    "adamson": {
        "adata_path": "data/gears_pert_data/adamson/perturb_processed.h5ad",
        "annotation_path": "data/annotations/adamson_functional_classes_enriched.tsv",
    },
    "k562": {
        "adata_path": "data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad",
    },
    "rpe1": {
        "adata_path": "data/gears_pert_data/replogle_rpe1_essential/perturb_processed.h5ad",
    },
}

# All baselines to run
all_baselines = [
    BaselineType.SELFTRAINED,
    BaselineType.RANDOM_GENE_EMB,
    BaselineType.RANDOM_PERT_EMB,
    BaselineType.SCGPT_GENE_EMB,
    BaselineType.SCFOUNDATION_GENE_EMB,
    BaselineType.GEARS_PERT_EMB,
]

# Parameters
n_cells_per_pert = 50
pca_dim = 10
ridge_penalty = 0.1
seed = 1

results_dir = Path("results/single_cell_analysis")
results_dir.mkdir(parents=True, exist_ok=True)

# Step 1: Run all baselines
LOGGER.info("="*70)
LOGGER.info("STEP 1: Running all baselines")
LOGGER.info("="*70)

for dataset_name, config in datasets.items():
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"Dataset: {dataset_name}")
    LOGGER.info(f"{'='*60}")
    
    adata_path = Path(config["adata_path"])
    
    if not adata_path.exists():
        LOGGER.warning(f"Data file not found: {adata_path}")
        continue
    
    # Use existing split config
    split_config_path = results_dir / f"{dataset_name}_split_config.json"
    
    if not split_config_path.exists():
        LOGGER.warning(f"Split config not found: {split_config_path}")
        continue
    
    output_dir = results_dir / dataset_name
    
    try:
        LOGGER.info(f"Running all baselines for {dataset_name}...")
        results_df = run_all_baselines_single_cell(
            adata_path=adata_path,
            split_config_path=split_config_path,
            output_dir=output_dir,
            baseline_types=all_baselines,
            pca_dim=pca_dim,
            ridge_penalty=ridge_penalty,
            seed=seed,
            n_cells_per_pert=n_cells_per_pert,
            cell_embedding_method="cell_pca",
        )
        
        LOGGER.info(f"\n{dataset_name} Baseline Results:")
        LOGGER.info(f"\n{results_df.to_string()}")
        
    except Exception as e:
        LOGGER.error(f"Failed to run baselines for {dataset_name}: {e}")
        import traceback
        traceback.print_exc()

# Step 2: Run LSFT for all baselines
LOGGER.info("\n" + "="*70)
LOGGER.info("STEP 2: Running LSFT for all baselines")
LOGGER.info("="*70)

for dataset_name, config in datasets.items():
    if dataset_name not in ["adamson", "k562"]:  # Skip RPE1 for LSFT if needed
        continue
        
    adata_path = Path(config["adata_path"])
    split_config_path = results_dir / f"{dataset_name}_split_config.json"
    
    if not adata_path.exists() or not split_config_path.exists():
        continue
    
    output_dir = results_dir / dataset_name / "lsft"
    
    try:
        LOGGER.info(f"\nRunning LSFT for {dataset_name}...")
        lsft_results = run_lsft_single_cell_all_baselines(
            adata_path=adata_path,
            split_config_path=split_config_path,
            output_dir=output_dir,
            dataset_name=dataset_name,
            baseline_types=all_baselines,
            top_pcts=[0.05, 0.10],
            pca_dim=pca_dim,
            ridge_penalty=ridge_penalty,
            seed=seed,
            n_cells_per_pert=n_cells_per_pert,
        )
        
        if not lsft_results.empty:
            LOGGER.info(f"\n{dataset_name} LSFT Results:")
            LOGGER.info(f"\n{lsft_results.head(20).to_string()}")
        
    except Exception as e:
        LOGGER.error(f"Failed to run LSFT for {dataset_name}: {e}")
        import traceback
        traceback.print_exc()

# Step 3: Run LOGO for datasets with annotations
LOGGER.info("\n" + "="*70)
LOGGER.info("STEP 3: Running LOGO for all baselines")
LOGGER.info("="*70)

for dataset_name, config in datasets.items():
    if "annotation_path" not in config:
        LOGGER.info(f"Skipping LOGO for {dataset_name} (no annotation file)")
        continue
    
    annotation_path = Path(config["annotation_path"])
    if not annotation_path.exists():
        LOGGER.warning(f"Annotation file not found: {annotation_path}")
        continue
    
    adata_path = Path(config["adata_path"])
    output_dir = results_dir / dataset_name / "logo"
    
    try:
        LOGGER.info(f"\nRunning LOGO for {dataset_name}...")
        logo_results = run_logo_single_cell(
            adata_path=adata_path,
            annotation_path=annotation_path,
            dataset_name=dataset_name,
            output_dir=output_dir,
            class_name="Transcription",
            baseline_types=all_baselines,
            pca_dim=pca_dim,
            ridge_penalty=ridge_penalty,
            seed=seed,
            n_cells_per_pert=n_cells_per_pert,
        )
        
        if logo_results is not None and not logo_results.empty:
            LOGGER.info(f"\n{dataset_name} LOGO Results:")
            LOGGER.info(f"\n{logo_results.head(20).to_string()}")
        
    except Exception as e:
        LOGGER.error(f"Failed to run LOGO for {dataset_name}: {e}")
        import traceback
        traceback.print_exc()

LOGGER.info("\n" + "="*70)
LOGGER.info("SINGLE-CELL ANALYSIS UPDATE COMPLETE")
LOGGER.info("="*70)
LOGGER.info(f"\nResults saved to: {results_dir}")
LOGGER.info("\nNext: Run analysis aggregation and report generation")


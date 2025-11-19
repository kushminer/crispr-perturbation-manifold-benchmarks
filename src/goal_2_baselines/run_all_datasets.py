#!/usr/bin/env python3
"""
Run all baselines on multiple datasets.

This script runs the baseline reproduction pipeline on multiple datasets
(Adamson, Replogle K562, Replogle RPE1) and generates comprehensive results.

Usage:
    python -m goal_2_baselines.run_all_datasets \
        --datasets adamson replogle_k562_essential \
        --output_base results/baselines \
        --pca_dim 10 \
        --ridge_penalty 0.1 \
        --seed 1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from goal_2_baselines.baseline_runner import run_all_baselines
from goal_2_baselines.split_logic import prepare_perturbation_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


# Dataset paths mapping
DATASET_PATHS = {
    "adamson": "../paper/benchmark/data/gears_pert_data/adamson/perturb_processed.h5ad",
    "replogle_k562_essential": "/Users/samuelminer/Documents/classes/nih_research/data_replogle_k562_essential/perturb_processed.h5ad",
    "replogle_rpe1_essential": "/Users/samuelminer/Documents/classes/nih_research/data_replogle_rpe1_essential/perturb_processed.h5ad",
}


def main():
    parser = argparse.ArgumentParser(
        description="Run all baselines on multiple datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["adamson"],
        choices=list(DATASET_PATHS.keys()),
        help="Datasets to run (default: adamson)",
    )
    parser.add_argument(
        "--output_base",
        type=Path,
        default=Path("results/baselines"),
        help="Base directory for results (default: results/baselines)",
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=10,
        help="PCA dimension (default: 10)",
    )
    parser.add_argument(
        "--ridge_penalty",
        type=float,
        default=0.1,
        help="Ridge penalty (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: 1)",
    )
    
    args = parser.parse_args()
    
    # Run on each dataset
    all_results = {}
    
    for dataset_name in args.datasets:
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"Processing dataset: {dataset_name}")
        LOGGER.info(f"{'='*60}")
        
        # Get dataset path
        adata_path = Path(DATASET_PATHS[dataset_name])
        if not adata_path.exists():
            LOGGER.error(f"Dataset file not found: {adata_path}")
            LOGGER.error(f"Skipping {dataset_name}")
            continue
        
        # Prepare splits
        split_config_path = args.output_base / f"{dataset_name}_split_seed{args.seed}.json"
        LOGGER.info(f"Preparing splits...")
        prepare_perturbation_splits(
            adata_path=adata_path,
            dataset_name=dataset_name,
            output_path=split_config_path,
            seed=args.seed,
        )
        
        # Run baselines
        output_dir = args.output_base / f"{dataset_name}_reproduced"
        LOGGER.info(f"Running baselines...")
        results_df = run_all_baselines(
            adata_path=adata_path,
            split_config_path=split_config_path,
            output_dir=output_dir,
            pca_dim=args.pca_dim,
            ridge_penalty=args.ridge_penalty,
            seed=args.seed,
        )
        
        all_results[dataset_name] = results_df
        LOGGER.info(f"Completed {dataset_name}: {len(results_df)} baselines")
    
    # Create summary
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("Summary")
    LOGGER.info(f"{'='*60}")
    
    summary_path = args.output_base / "all_datasets_summary.csv"
    summary_rows = []
    for dataset_name, results_df in all_results.items():
        for _, row in results_df.iterrows():
            summary_rows.append({
                "dataset": dataset_name,
                "baseline": row["baseline"],
                "mean_pearson_r": row["mean_pearson_r"],
                "mean_l2": row.get("mean_l2", None),
                "n_test_perturbations": row["n_test_perturbations"],
            })
    
    import pandas as pd
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    LOGGER.info(f"Saved summary to {summary_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("All Datasets Summary")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())


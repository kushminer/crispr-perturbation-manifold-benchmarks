#!/usr/bin/env python3
"""
CLI entry point for running all baseline models.

Usage:
    python -m goal_2_baselines.run_all \
        --adata_path data/gears_pert_data/adamson/perturb_processed.h5ad \
        --split_config results/split_config.json \
        --output_dir results/baselines
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .baseline_runner import run_all_baselines
from .baseline_types import BaselineType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run all baseline models for reproducibility"
    )
    parser.add_argument(
        "--adata_path",
        type=Path,
        required=True,
        help="Path to perturb_processed.h5ad file",
    )
    parser.add_argument(
        "--split_config",
        type=Path,
        required=True,
        help="Path to train/test/val split JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save results",
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
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=[bt.value for bt in BaselineType],
        help="Specific baselines to run (default: all 8)",
    )
    parser.add_argument(
        "--use_paper_implementation",
        action="store_true",
        help="Use paper's validated Python implementation (default: False)",
    )
    
    args = parser.parse_args()
    
    # Convert baseline names to BaselineType enums
    baseline_types = None
    if args.baselines:
        baseline_types = [BaselineType(bt) for bt in args.baselines]
    
    # Run baselines
    results_df = run_all_baselines(
        adata_path=args.adata_path,
        split_config_path=args.split_config,
        output_dir=args.output_dir,
        baseline_types=baseline_types,
        pca_dim=args.pca_dim,
        ridge_penalty=args.ridge_penalty,
        seed=args.seed,
        use_paper_implementation=args.use_paper_implementation,
    )
    
    LOGGER.info("Baseline reproduction complete!")
    LOGGER.info(f"\n{results_df.to_string()}")
    
    return 0


if __name__ == "__main__":
    exit(main())


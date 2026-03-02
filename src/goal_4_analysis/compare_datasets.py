#!/usr/bin/env python3
"""
Compare evaluation results across multiple datasets.

This script generates cross-dataset comparison reports and visualizations.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from shared.comparison import generate_comparison_report
from shared.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("compare_datasets")


def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation results across datasets"
    )
    parser.add_argument(
        "--configs",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to config YAML files for each dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/comparison"),
        help="Output directory for comparison results",
    )
    parser.add_argument(
        "--dataset-names",
        type=str,
        nargs="+",
        default=None,
        help="Optional dataset names (default: from config names)",
    )
    
    args = parser.parse_args()
    
    # Load configs and get result directories
    results_dirs = {}
    
    for i, config_path in enumerate(args.configs):
        config = load_config(config_path)
        
        # Use provided name or derive from config
        if args.dataset_names and i < len(args.dataset_names):
            dataset_name = args.dataset_names[i]
        else:
            dataset_name = config.dataset.name
        
        results_dirs[dataset_name] = config.output_root
    
    LOGGER.info("Comparing %d datasets:", len(results_dirs))
    for name, path in results_dirs.items():
        LOGGER.info("  %s: %s", name, path)
    
    # Generate comparison report
    report_path = generate_comparison_report(
        results_dirs=results_dirs,
        output_dir=args.output_dir,
    )
    
    LOGGER.info("✅ Comparison complete!")
    LOGGER.info("Report: %s", report_path)
    LOGGER.info("Figures: %s/fig_*.png", args.output_dir)
    LOGGER.info("Tables: %s/*_comparison.csv", args.output_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


#!/usr/bin/env python3
"""
Organize analysis results from Goal 3 (LSFT and Functional Class Holdout) 
into Goal 4 (Statistical Analysis) directory structure.

This script reorganizes analysis results to follow a clear structure:
- goal_4_analysis/lsft/{dataset}/ - LSFT analysis results
- goal_4_analysis/functional_class_holdout/{dataset}/ - LOGO analysis results
- goal_4_analysis/functional_class_holdout/aggregate/ - Cross-dataset LOGO analysis
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def organize_lsft_analyses(
    source_dir: Path,
    target_dir: Path,
    datasets: list[str] = ["adamson", "k562", "rpe1"],
) -> None:
    """Organize LSFT analysis results into goal_4_analysis/lsft/."""
    LOGGER.info("Organizing LSFT analyses...")
    
    target_base = target_dir / "lsft"
    target_base.mkdir(parents=True, exist_ok=True)
    
    for dataset in datasets:
        source_analysis = source_dir / f"{dataset}_analysis"
        target_dataset = target_base / dataset
        
        if not source_analysis.exists():
            LOGGER.warning(f"Source LSFT analysis not found: {source_analysis}")
            continue
        
        LOGGER.info(f"  Organizing {dataset} LSFT analysis...")
        
        # Copy all files from source to target
        if target_dataset.exists():
            shutil.rmtree(target_dataset)
        shutil.copytree(source_analysis, target_dataset)
        
        LOGGER.info(f"    Copied to {target_dataset}")


def organize_functional_class_holdout_analyses(
    source_dir: Path,
    target_dir: Path,
    datasets: list[str] = ["adamson", "replogle_k562", "replogle_rpe1"],
) -> None:
    """Organize functional class holdout (LOGO) analysis results into goal_4_analysis/functional_class_holdout/."""
    LOGGER.info("Organizing functional class holdout analyses...")
    
    target_base = target_dir / "functional_class_holdout"
    target_base.mkdir(parents=True, exist_ok=True)
    
    # Map source dataset names to target names
    dataset_map = {
        "adamson": "adamson",
        "replogle_k562": "replogle_k562",
        "replogle_rpe1": "replogle_rpe1",
    }
    
    for source_ds in datasets:
        target_ds = dataset_map.get(source_ds, source_ds)
        source_analysis = source_dir / source_ds
        target_dataset = target_base / target_ds
        
        if not source_analysis.exists():
            LOGGER.warning(f"Source LOGO analysis not found: {source_analysis}")
            continue
        
        LOGGER.info(f"  Organizing {target_ds} LOGO analysis...")
        
        # Copy comparison files and reports
        target_dataset.mkdir(parents=True, exist_ok=True)
        
        # Copy specific analysis files (comparison reports, plots, CSVs)
        for pattern in ["*comparison*.csv", "*comparison*.md", "*comparison*.png", "*.csv"]:
            for file in source_analysis.glob(pattern):
                if file.is_file():
                    shutil.copy2(file, target_dataset / file.name)
                    LOGGER.info(f"    Copied {file.name}")
        
        # Copy aggregate analysis if it exists
        source_aggregate = source_dir / "aggregate"
        if source_aggregate.exists():
            target_aggregate = target_base / "aggregate"
            if target_aggregate.exists():
                shutil.rmtree(target_aggregate)
            shutil.copytree(source_aggregate, target_aggregate)
            LOGGER.info(f"    Copied aggregate analysis")


def main():
    parser = argparse.ArgumentParser(
        description="Organize analysis results from Goal 3 into Goal 4 structure"
    )
    parser.add_argument(
        "--lsft_source",
        type=Path,
        default=Path("results/goal_3_prediction/lsft"),
        help="Source directory for LSFT analyses",
    )
    parser.add_argument(
        "--logo_source",
        type=Path,
        default=Path("results/goal_3_prediction/functional_class_holdout"),
        help="Source directory for LOGO analyses",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("results/goal_4_analysis"),
        help="Target directory for organized analyses",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["adamson", "k562", "rpe1"],
        help="Datasets to organize for LSFT",
    )
    parser.add_argument(
        "--logo_datasets",
        nargs="+",
        default=["adamson", "replogle_k562", "replogle_rpe1"],
        help="Datasets to organize for LOGO",
    )
    
    args = parser.parse_args()
    
    args.target.mkdir(parents=True, exist_ok=True)
    
    # Organize LSFT analyses
    organize_lsft_analyses(args.lsft_source, args.target, args.datasets)
    
    # Organize functional class holdout analyses
    organize_functional_class_holdout_analyses(
        args.logo_source, args.target, args.logo_datasets
    )
    
    LOGGER.info("✅ Analysis organization complete!")
    LOGGER.info(f"Results organized in: {args.target}")
    
    return 0


if __name__ == "__main__":
    exit(main())


#!/usr/bin/env python3
"""
Run similarity analysis for all datasets (Adamson, K562, RPE1) and create aggregate report.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# Dataset configurations
DATASETS = {
    "adamson": {
        "adata_path": "../paper/benchmark/data/gears_pert_data/adamson/perturb_processed.h5ad",
        "split_config": "results/goal_2_baselines/splits/adamson_split_seed1.json",
        "baseline_results": "results/goal_2_baselines/adamson_reproduced/baseline_results_reproduced.csv",
        "output_base": "results/similarity_adamson",
    },
    "k562": {
        "adata_path": "/Users/samuelminer/Documents/classes/nih_research/data_replogle_k562_essential/perturb_processed.h5ad",
        "split_config": "results/goal_2_baselines/splits/replogle_k562_essential_split_seed1.json",
        "baseline_results": "results/goal_2_baselines/replogle_k562_essential_reproduced/baseline_results_reproduced.csv",
        "output_base": "results/similarity_k562",
    },
    "rpe1": {
        "adata_path": "/Users/samuelminer/Documents/classes/nih_research/data_replogle_rpe1_essential/perturb_processed.h5ad",
        "split_config": "results/goal_2_baselines/splits/replogle_rpe1_essential_split_seed1.json",
        "baseline_results": "results/goal_2_baselines/replogle_rpe1_essential_reproduced/baseline_results_reproduced.csv",
        "output_base": "results/similarity_rpe1",
    },
}


def run_de_matrix_similarity(dataset_name: str, config: dict, k: int = 5, seed: int = 1) -> bool:
    """Run DE matrix similarity analysis."""
    LOGGER.info(f"Running DE matrix similarity for {dataset_name}")
    
    cmd = [
        "python", "-m", "similarity.de_matrix_similarity",
        "--adata_path", config["adata_path"],
        "--split_config", config["split_config"],
        "--baseline_results", config["baseline_results"],
        "--output_dir", f"{config['output_base']}/de_matrix_similarity",
        "--k", str(k),
        "--seed", str(seed),
    ]
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent, capture_output=True, text=True)
        if result.returncode != 0:
            LOGGER.error(f"DE matrix similarity failed for {dataset_name}: {result.stderr}")
            return False
        LOGGER.info(f"DE matrix similarity completed for {dataset_name}")
        return True
    except Exception as e:
        LOGGER.error(f"Error running DE matrix similarity for {dataset_name}: {e}")
        return False


def run_embedding_similarity(dataset_name: str, config: dict, k: int = 5, seed: int = 1) -> bool:
    """Run embedding similarity analysis."""
    LOGGER.info(f"Running embedding similarity for {dataset_name}")
    
    cmd = [
        "python", "-m", "similarity.embedding_similarity",
        "--adata_path", config["adata_path"],
        "--split_config", config["split_config"],
        "--baselines", "lpm_selftrained", "lpm_k562PertEmb", "lpm_gearsPertEmb", "lpm_rpe1PertEmb",
        "--output_dir", f"{config['output_base']}/embedding_similarity",
        "--k", str(k),
        "--pca_dim", "10",
        "--ridge_penalty", "0.1",
        "--seed", str(seed),
    ]
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent, capture_output=True, text=True)
        if result.returncode != 0:
            LOGGER.error(f"Embedding similarity failed for {dataset_name}: {result.stderr}")
            return False
        LOGGER.info(f"Embedding similarity completed for {dataset_name}")
        return True
    except Exception as e:
        LOGGER.error(f"Error running embedding similarity for {dataset_name}: {e}")
        return False


def run_comprehensive_report(dataset_name: str, config: dict) -> bool:
    """Run comprehensive report generation."""
    LOGGER.info(f"Generating comprehensive report for {dataset_name}")
    
    cmd = [
        "python", "-m", "similarity.create_comprehensive_report",
        "--embedding_similarity", f"{config['output_base']}/embedding_similarity/embedding_similarity_all_baselines.csv",
        "--de_matrix_similarity", f"{config['output_base']}/de_matrix_similarity/de_matrix_similarity_results.csv",
        "--embedding_regression", f"{config['output_base']}/embedding_similarity/embedding_regression_analysis_all_baselines.csv",
        "--de_matrix_regression", f"{config['output_base']}/de_matrix_similarity/de_matrix_regression_analysis.csv",
        "--output_dir", f"{config['output_base']}/comprehensive",
    ]
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent, capture_output=True, text=True)
        if result.returncode != 0:
            LOGGER.error(f"Comprehensive report failed for {dataset_name}: {result.stderr}")
            return False
        LOGGER.info(f"Comprehensive report completed for {dataset_name}")
        return True
    except Exception as e:
        LOGGER.error(f"Error generating comprehensive report for {dataset_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run similarity analysis for all datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["adamson", "k562", "rpe1", "all"],
        default=["all"],
        help="Datasets to analyze (default: all)",
    )
    parser.add_argument(
        "--skip_de_matrix",
        action="store_true",
        help="Skip DE matrix similarity analysis",
    )
    parser.add_argument(
        "--skip_embedding",
        action="store_true",
        help="Skip embedding similarity analysis",
    )
    parser.add_argument(
        "--skip_reports",
        action="store_true",
        help="Skip comprehensive report generation",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of top similarities to average (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: 1)",
    )
    
    args = parser.parse_args()
    
    # Determine which datasets to run
    if "all" in args.datasets:
        datasets_to_run = ["adamson", "k562", "rpe1"]
    else:
        datasets_to_run = args.datasets
    
    LOGGER.info(f"Running similarity analysis for datasets: {datasets_to_run}")
    
    results = {}
    
    for dataset_name in datasets_to_run:
        if dataset_name not in DATASETS:
            LOGGER.warning(f"Unknown dataset: {dataset_name}, skipping")
            continue
        
        config = DATASETS[dataset_name]
        results[dataset_name] = {}
        
        # Run DE matrix similarity
        if not args.skip_de_matrix:
            results[dataset_name]["de_matrix"] = run_de_matrix_similarity(
                dataset_name, config, k=args.k, seed=args.seed
            )
        else:
            results[dataset_name]["de_matrix"] = True  # Skip
        
        # Run embedding similarity
        if not args.skip_embedding:
            results[dataset_name]["embedding"] = run_embedding_similarity(
                dataset_name, config, k=args.k, seed=args.seed
            )
        else:
            results[dataset_name]["embedding"] = True  # Skip
        
        # Generate comprehensive report
        if not args.skip_reports:
            results[dataset_name]["report"] = run_comprehensive_report(dataset_name, config)
        else:
            results[dataset_name]["report"] = True  # Skip
    
    # Summary
    LOGGER.info("\n" + "="*60)
    LOGGER.info("SIMILARITY ANALYSIS SUMMARY")
    LOGGER.info("="*60)
    
    for dataset_name, result in results.items():
        LOGGER.info(f"\n{dataset_name.upper()}:")
        LOGGER.info(f"  DE Matrix Similarity: {'✅' if result.get('de_matrix') else '❌'}")
        LOGGER.info(f"  Embedding Similarity: {'✅' if result.get('embedding') else '❌'}")
        LOGGER.info(f"  Comprehensive Report: {'✅' if result.get('report') else '❌'}")
    
    LOGGER.info("\n" + "="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())



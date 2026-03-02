#!/usr/bin/env python3
"""
Compare paper's Python implementation vs R implementation.

Excludes random embeddings/inputs from comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


# Baseline configurations (excluding random)
BASELINE_CONFIGS = {
    "lpm_selftrained": {
        "gene_embedding": "training_data",
        "pert_embedding": "training_data",
    },
    "lpm_k562PertEmb": {
        "gene_embedding": "training_data",
        "pert_embedding": "results/replogle_k562_pert_emb_pca10_seed1.tsv",
    },
    "lpm_rpe1PertEmb": {
        "gene_embedding": "training_data",
        "pert_embedding": "results/replogle_rpe1_pert_emb_pca10_seed1.tsv",
    },
}


def run_paper_python_script(
    dataset_name: str,
    test_train_config_id: str,
    working_dir: Path,
    result_id: str,
    gene_embedding: str,
    pert_embedding: str,
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
) -> bool:
    """Run paper's Python baseline script."""
    py_script_path = Path(__file__).parent.parent.parent.parent / "paper" / "benchmark" / "src" / "run_linear_pretrained_model.py"
    
    # Resolve to absolute paths
    working_dir = working_dir.resolve()
    py_script_path = py_script_path.resolve()
    
    # Check if Python script exists (may not be in fresh repository)
    if not py_script_path.exists():
        LOGGER.warning(
            f"Python script not found: {py_script_path}\n"
            f"This script may not be in the original repository.\n"
            f"R script exists and can be used instead: {py_script_path.parent / 'run_linear_pretrained_model.R'}"
        )
        return False
    
    # Resolve embedding paths relative to working_dir
    if pert_embedding.endswith(".tsv"):
        pert_embedding_path = working_dir / pert_embedding
        if not pert_embedding_path.exists():
            repo_root = Path(__file__).resolve().parents[2]
            local_results = repo_root / "results" / pert_embedding.split("/")[-1]
            if local_results.exists():
                pert_embedding = str(local_results.resolve())
            else:
                LOGGER.warning(f"Embedding file not found: {pert_embedding_path}")
                return False
        else:
            pert_embedding = str(pert_embedding_path.resolve())
    
    cmd = [
        sys.executable,
        str(py_script_path),
        "--dataset_name", dataset_name,
        "--test_train_config_id", test_train_config_id,
        "--pca_dim", str(pca_dim),
        "--ridge_penalty", str(ridge_penalty),
        "--seed", str(seed),
        "--gene_embedding", gene_embedding,
        "--pert_embedding", pert_embedding,
        "--working_dir", str(working_dir),
        "--result_id", result_id,
    ]
    
    LOGGER.info(f"Running Python: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(working_dir))
    
    if result.returncode != 0:
        LOGGER.error(f"Python script failed: {result.stderr}")
        return False
    
    return True


def run_r_script(
    dataset_name: str,
    test_train_config_id: str,
    working_dir: Path,
    result_id: str,
    gene_embedding: str,
    pert_embedding: str,
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
) -> bool:
    """Run R baseline script."""
    r_script_path = Path(__file__).parent.parent.parent.parent / "paper" / "benchmark" / "src" / "run_linear_pretrained_model.R"
    
    # Resolve to absolute paths
    working_dir = working_dir.resolve()
    r_script_path = r_script_path.resolve()
    
    # Resolve embedding paths relative to working_dir
    if pert_embedding.endswith(".tsv"):
        pert_embedding_path = working_dir / pert_embedding
        if not pert_embedding_path.exists():
            repo_root = Path(__file__).resolve().parents[2]
            local_results = repo_root / "results" / pert_embedding.split("/")[-1]
            if local_results.exists():
                pert_embedding = str(local_results.resolve())
            else:
                LOGGER.warning(f"Embedding file not found: {pert_embedding_path}")
                return False
        else:
            pert_embedding = str(pert_embedding_path.resolve())
    
    cmd = [
        "Rscript",
        "--no-restore",
        str(r_script_path),
        "--dataset_name", dataset_name,
        "--test_train_config_id", test_train_config_id,
        "--pca_dim", str(pca_dim),
        "--ridge_penalty", str(ridge_penalty),
        "--seed", str(seed),
        "--gene_embedding", gene_embedding,
        "--pert_embedding", pert_embedding,
        "--working_dir", str(working_dir),
        "--result_id", result_id,
    ]
    
    LOGGER.info(f"Running R: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(working_dir))
    
    if result.returncode != 0:
        LOGGER.error(f"R script failed: {result.stderr}")
        return False
    
    return True


def load_predictions(result_dir: Path) -> Tuple[Dict[str, List[float]], List[str]]:
    """Load predictions from JSON files."""
    predictions_path = result_dir / "all_predictions.json"
    gene_names_path = result_dir / "gene_names.json"
    
    if not predictions_path.exists() or not gene_names_path.exists():
        raise FileNotFoundError(f"Results not found in {result_dir}")
    
    with open(predictions_path) as f:
        predictions = json.load(f)
    
    with open(gene_names_path) as f:
        gene_names = json.load(f)
    
    return predictions, gene_names


def compare_predictions(
    r_predictions: Dict[str, List[float]],
    py_predictions: Dict[str, List[float]],
    test_perturbations: List[str],
    tolerance: float = 0.01,
) -> Dict:
    """Compare R and Python predictions."""
    comparison = {
        "perturbations": [],
        "pearson_r": [],
        "l2": [],
        "max_diff": [],
        "mean_abs_diff": [],
        "within_tolerance": [],
    }
    
    for pert in test_perturbations:
        # Both use clean_condition (without +ctrl) as keys
        clean_pert_name = pert.replace("+ctrl", "")
        
        if clean_pert_name not in r_predictions:
            LOGGER.warning(f"R prediction not found for {clean_pert_name}")
            continue
        
        if clean_pert_name not in py_predictions:
            LOGGER.warning(f"Python prediction not found for {clean_pert_name}")
            continue
        
        # Convert to arrays, handling NA values
        r_pred_list = r_predictions[clean_pert_name]
        py_pred_list = py_predictions[clean_pert_name]
        
        # Filter out NA values
        r_pred_list = [x for x in r_pred_list if x != 'NA' and x is not None and not (isinstance(x, float) and np.isnan(x))]
        py_pred_list = [x for x in py_pred_list if x != 'NA' and x is not None and not (isinstance(x, float) and np.isnan(x))]
        
        r_pred = np.array(r_pred_list, dtype=np.float64)
        py_pred = np.array(py_pred_list, dtype=np.float64)
        
        if len(r_pred) != len(py_pred):
            LOGGER.warning(f"Length mismatch for {pert}: R={len(r_pred)}, Python={len(py_pred)}")
            continue
        
        if len(r_pred) == 0:
            LOGGER.warning(f"No valid predictions for {pert}")
            continue
        
        # Compute metrics
        from shared.metrics import compute_metrics
        metrics = compute_metrics(r_pred, py_pred)
        
        max_diff = np.max(np.abs(r_pred - py_pred))
        mean_abs_diff = np.mean(np.abs(r_pred - py_pred))
        within_tol = max_diff <= tolerance
        
        comparison["perturbations"].append(pert)
        comparison["pearson_r"].append(metrics["pearson_r"])
        comparison["l2"].append(metrics["l2"])
        comparison["max_diff"].append(max_diff)
        comparison["mean_abs_diff"].append(mean_abs_diff)
        comparison["within_tolerance"].append(within_tol)
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Compare paper's Python vs R implementation"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="adamson",
        help="Dataset name (default: adamson)",
    )
    parser.add_argument(
        "--working_dir",
        type=str,
        default="../paper/benchmark",
        help="Working directory (default: ../paper/benchmark)",
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
        "--tolerance",
        type=float,
        default=0.01,
        help="Numerical tolerance for agreement (default: 0.01)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("validation/paper_python_vs_r"),
        help="Output directory (default: validation/paper_python_vs_r)",
    )
    
    args = parser.parse_args()
    
    # Resolve working_dir to absolute path
    working_dir = Path(args.working_dir).resolve()
    
    # Prepare split config
    test_train_config_id = f"seed_{args.seed}_{args.dataset_name}_split"
    split_config_path = Path(__file__).parent.parent.parent / "results" / f"{args.dataset_name}_split_seed{args.seed}.json"
    
    # If split config doesn't exist, create it
    if not split_config_path.exists():
        LOGGER.info(f"Creating split config: {split_config_path}")
        from goal_2_baselines.split_logic import prepare_perturbation_splits
        
        if args.dataset_name == "adamson":
            adata_path = Path(__file__).parent.parent.parent.parent / "paper" / "benchmark" / "data" / "gears_pert_data" / "adamson" / "perturb_processed.h5ad"
        else:
            LOGGER.error(f"Dataset path not configured: {args.dataset_name}")
            return 1
        
        prepare_perturbation_splits(
            adata_path=adata_path,
            dataset_name=args.dataset_name,
            output_path=split_config_path,
            seed=args.seed,
        )
    
    # Copy split config to working_dir
    r_split_config_path = working_dir / "results" / test_train_config_id
    r_split_config_path.parent.mkdir(parents=True, exist_ok=True)
    
    import shutil
    shutil.copy(split_config_path, r_split_config_path)
    LOGGER.info(f"Copied split config to {r_split_config_path}")
    
    # Load split config to get test perturbations
    from goal_2_baselines.split_logic import load_split_config
    split_config = load_split_config(split_config_path)
    all_test_perts = split_config.get("test", [])
    
    LOGGER.info(f"Testing {len(all_test_perts)} test perturbations")
    
    # Run validation for each baseline
    all_results = []
    
    for baseline_name, config in BASELINE_CONFIGS.items():
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"Validating baseline: {baseline_name}")
        LOGGER.info(f"{'='*60}")
        
        # Run Python
        result_id_py = f"py_{baseline_name}_seed{args.seed}"
        py_result_dir = working_dir / "results" / result_id_py
        
        LOGGER.info("Running Python baseline...")
        if not run_paper_python_script(
            dataset_name=args.dataset_name,
            test_train_config_id=test_train_config_id,
            working_dir=working_dir,
            result_id=result_id_py,
            gene_embedding=config["gene_embedding"],
            pert_embedding=config["pert_embedding"],
            pca_dim=args.pca_dim,
            ridge_penalty=args.ridge_penalty,
            seed=args.seed,
        ):
            LOGGER.error(f"Failed to run Python baseline: {baseline_name}")
            continue
        
        # Load Python results
        try:
            py_predictions, py_gene_names = load_predictions(py_result_dir)
        except Exception as e:
            LOGGER.error(f"Failed to load Python results: {e}")
            continue
        
        # Run R
        result_id_r = f"r_{baseline_name}_seed{args.seed}"
        r_result_dir = working_dir / "results" / result_id_r
        
        LOGGER.info("Running R baseline...")
        if not run_r_script(
            dataset_name=args.dataset_name,
            test_train_config_id=test_train_config_id,
            working_dir=working_dir,
            result_id=result_id_r,
            gene_embedding=config["gene_embedding"],
            pert_embedding=config["pert_embedding"],
            pca_dim=args.pca_dim,
            ridge_penalty=args.ridge_penalty,
            seed=args.seed,
        ):
            LOGGER.error(f"Failed to run R baseline: {baseline_name}")
            continue
        
        # Load R results
        try:
            r_predictions, r_gene_names = load_predictions(r_result_dir)
        except Exception as e:
            LOGGER.error(f"Failed to load R results: {e}")
            continue
        
        # Compare on all test perturbations
        comparison = compare_predictions(
            r_predictions=r_predictions,
            py_predictions=py_predictions,
            test_perturbations=all_test_perts,
            tolerance=args.tolerance,
        )
        
        # Aggregate results
        if comparison["perturbations"]:
            mean_r = np.mean(comparison["pearson_r"])
            mean_l2 = np.mean(comparison["l2"])
            max_diff = np.max(comparison["max_diff"])
            mean_abs_diff = np.mean(comparison["mean_abs_diff"])
            within_tol_count = sum(comparison["within_tolerance"])
            total_count = len(comparison["perturbations"])
            
            all_results.append({
                "baseline": baseline_name,
                "mean_pearson_r": mean_r,
                "mean_l2": mean_l2,
                "max_diff": max_diff,
                "mean_abs_diff": mean_abs_diff,
                "within_tolerance": f"{within_tol_count}/{total_count}",
                "n_test_perturbations": total_count,
            })
            
            LOGGER.info(f"Comparison results:")
            LOGGER.info(f"  Mean Pearson r: {mean_r:.6f}")
            LOGGER.info(f"  Mean L2: {mean_l2:.6f}")
            LOGGER.info(f"  Max difference: {max_diff:.6f}")
            LOGGER.info(f"  Mean absolute difference: {mean_abs_diff:.6f}")
            LOGGER.info(f"  Within tolerance: {within_tol_count}/{total_count}")
    
    # Save summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary_path = output_dir / "paper_python_vs_r_comparison.csv"
        results_df.to_csv(summary_path, index=False)
        LOGGER.info(f"\nSaved comparison summary to {summary_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("Paper Python vs R Comparison Summary")
        print("=" * 80)
        print(results_df.to_string(index=False))
        print("=" * 80)
        
        # Check if all within tolerance
        all_within_tol = all(
            int(r["within_tolerance"].split("/")[0]) == int(r["within_tolerance"].split("/")[1])
            for r in all_results
        )
        
        if all_within_tol:
            print("\n✅ All baselines within tolerance!")
            return 0
        else:
            print("\n⚠️  Some baselines outside tolerance")
            return 1
    else:
        LOGGER.error("No results to compare")
        return 1


if __name__ == "__main__":
    exit(main())

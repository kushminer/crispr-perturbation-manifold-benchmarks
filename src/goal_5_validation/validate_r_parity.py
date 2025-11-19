#!/usr/bin/env python3
"""
Validate Python baseline results against R implementation.

Runs a random subset of baselines and test perturbations to validate
that Python and R implementations produce matching results.

Usage:
    python -m goal_2_baselines.validate_r_parity \
        --dataset_name adamson \
        --working_dir ../paper/benchmark \
        --n_baselines 4 \
        --n_test_perturbations 5 \
        --tolerance 0.01
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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


# Baseline configurations for R script
BASELINE_CONFIGS = {
    "lpm_selftrained": {
        "gene_embedding": "training_data",
        "pert_embedding": "training_data",
    },
    "lpm_randomPertEmb": {
        "gene_embedding": "training_data",
        "pert_embedding": "random",
    },
    "lpm_randomGeneEmb": {
        "gene_embedding": "random",
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


def run_r_baseline(
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
            # Try evaluation_framework results
            eval_framework_results = Path(__file__).parent.parent.parent.parent / "evaluation_framework" / "results" / pert_embedding.split("/")[-1]
            if eval_framework_results.exists():
                pert_embedding = str(eval_framework_results.resolve())
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


def load_r_results(result_dir: Path) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Load R results from JSON files."""
    predictions_path = result_dir / "all_predictions.json"
    gene_names_path = result_dir / "gene_names.json"
    
    if not predictions_path.exists() or not gene_names_path.exists():
        raise FileNotFoundError(f"R results not found in {result_dir}")
    
    with open(predictions_path) as f:
        predictions = json.load(f)
    
    with open(gene_names_path) as f:
        gene_names = json.load(f)
    
    # Convert to numpy array (perturbations × genes)
    pred_matrix = np.array([predictions[pert] for pert in predictions.keys()])
    
    return predictions, gene_names


def run_python_baseline(
    dataset_name: str,
    split_config_path: Path,
    baseline_name: str,
    output_dir: Path,
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
) -> bool:
    """Run Python baseline."""
    from goal_2_baselines.baseline_types import BaselineType, get_baseline_config
    from goal_2_baselines.baseline_runner import run_single_baseline
    import anndata as ad
    
    # Map baseline name to BaselineType
    baseline_map = {
        "lpm_selftrained": BaselineType.SELFTRAINED,
        "lpm_randomPertEmb": BaselineType.RANDOM_PERT_EMB,
        "lpm_randomGeneEmb": BaselineType.RANDOM_GENE_EMB,
        "lpm_k562PertEmb": BaselineType.K562_PERT_EMB,
        "lpm_rpe1PertEmb": BaselineType.RPE1_PERT_EMB,
    }
    
    if baseline_name not in baseline_map:
        LOGGER.error(f"Unknown baseline: {baseline_name}")
        return False
    
    baseline_type = baseline_map[baseline_name]
    
    # Load data
    if dataset_name == "adamson":
        adata_path = Path(__file__).parent.parent.parent.parent / "paper" / "benchmark" / "data" / "gears_pert_data" / "adamson" / "perturb_processed.h5ad"
    else:
        LOGGER.error(f"Dataset path not configured: {dataset_name}")
        return False
    
    adata = ad.read_h5ad(adata_path)
    
    # Load splits
    from goal_2_baselines.split_logic import load_split_config
    split_config = load_split_config(split_config_path)
    
    # Compute Y matrix
    from goal_2_baselines.baseline_runner import compute_pseudobulk_expression_changes
    Y_df, split_labels = compute_pseudobulk_expression_changes(adata, split_config, seed)
    
    # Split Y
    train_perts = split_labels.get("train", [])
    test_perts = split_labels.get("test", [])
    
    Y_train = Y_df[train_perts]
    Y_test = Y_df[test_perts] if test_perts else pd.DataFrame()
    
    # Get config
    config = get_baseline_config(
        baseline_type,
        pca_dim=pca_dim,
        ridge_penalty=ridge_penalty,
        seed=seed,
    )
    
    # Get gene name mapping
    gene_name_mapping = None
    if "gene_name" in adata.var.columns:
        gene_name_mapping = dict(zip(adata.var_names, adata.var["gene_name"]))
    
    # Run baseline
    result = run_single_baseline(
        Y_train=Y_train,
        Y_test=Y_test,
        config=config,
        gene_names=Y_df.index.tolist(),
        gene_name_mapping=gene_name_mapping,
    )
    
    # Save predictions
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert predictions to format matching R
    # R saves as: {perturbation_name: [gene1_pred, gene2_pred, ...]}
    # pred is (genes, perturbations), so we transpose to (perturbations, genes)
    predictions = {}
    if "predictions" in result and result["predictions"].size > 0:
        # result["predictions"] is (genes, test_perturbations)
        pred_matrix = result["predictions"]  # genes × perts
        # Get test perturbation names from Y_test columns
        test_pert_names = Y_test.columns.tolist()
        # R uses clean_condition (without +ctrl) for keys
        for i, pert_name in enumerate(test_pert_names):
            # Extract predictions for this perturbation (all genes)
            # R uses clean_condition (without +ctrl) as key
            clean_pert_name = pert_name.replace("+ctrl", "")
            predictions[clean_pert_name] = pred_matrix[:, i].tolist()
    
    # Get gene names (may have been filtered for common genes)
    # Use gene_names from result if available, otherwise from Y_df
    if "gene_names" in result:
        gene_names = result["gene_names"]
    else:
        gene_names = Y_df.index.tolist()
    
    # Save as JSON (matching R format)
    with open(output_dir / "all_predictions.json", "w") as f:
        json.dump(predictions, f)
    
    with open(output_dir / "gene_names.json", "w") as f:
        json.dump(gene_names, f)
    
    return True


def compare_predictions(
    r_predictions: Dict[str, List[float]],
    py_predictions: Dict[str, List[float]],
    test_perturbations: List[str],
    tolerance: float = 0.01,
) -> Dict:
    """Compare R and Python predictions.
    
    Note: R uses clean_condition (without +ctrl), Python uses condition (with +ctrl).
    We map Python condition names to R clean_condition names.
    """
    comparison = {
        "perturbations": [],
        "pearson_r": [],
        "l2": [],
        "max_diff": [],
        "within_tolerance": [],
    }
    
    for pert in test_perturbations:
        # Both R and Python use clean_condition (without +ctrl) as keys
        clean_pert_name = pert.replace("+ctrl", "")
        
        if clean_pert_name not in r_predictions:
            LOGGER.warning(f"R prediction not found for {clean_pert_name}")
            continue
        
        if clean_pert_name not in py_predictions:
            LOGGER.warning(f"Python prediction not found for {clean_pert_name}")
            continue
        
        r_pred = np.array(r_predictions[clean_pert_name])
        py_pred = np.array(py_predictions[clean_pert_name])
        
        if len(r_pred) != len(py_pred):
            LOGGER.warning(f"Length mismatch for {pert}: R={len(r_pred)}, Python={len(py_pred)}")
            continue
        
        # Compute metrics
        from shared.metrics import compute_metrics
        metrics = compute_metrics(r_pred, py_pred)
        
        max_diff = np.max(np.abs(r_pred - py_pred))
        within_tol = max_diff <= tolerance
        
        comparison["perturbations"].append(pert)
        comparison["pearson_r"].append(metrics["pearson_r"])
        comparison["l2"].append(metrics["l2"])
        comparison["max_diff"].append(max_diff)
        comparison["within_tolerance"].append(within_tol)
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Validate Python baselines against R implementation"
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
        help="Working directory for R script (default: ../paper/benchmark)",
    )
    parser.add_argument(
        "--n_baselines",
        type=int,
        default=4,
        help="Number of baselines to test (default: 4)",
    )
    parser.add_argument(
        "--n_test_perturbations",
        type=int,
        default=5,
        help="Number of test perturbations to compare (default: 5)",
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
        default=Path("validation/r_parity"),
        help="Output directory for validation results (default: validation/r_parity)",
    )
    
    args = parser.parse_args()
    
    # Resolve working_dir to absolute path
    working_dir = Path(args.working_dir).resolve()
    args.working_dir = working_dir
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Select random subset of baselines
    available_baselines = list(BASELINE_CONFIGS.keys())
    n_to_test = min(args.n_baselines, len(available_baselines))
    selected_baselines = random.sample(available_baselines, n_to_test)
    
    LOGGER.info(f"Selected {n_to_test} baselines: {selected_baselines}")
    
    # Prepare split config
    # For R, we need the split config in the working_dir/results/
    # For Python, we can use our existing split config
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
    
    # Copy split config to working_dir for R
    r_split_config_path = working_dir / "results" / test_train_config_id
    r_split_config_path.parent.mkdir(parents=True, exist_ok=True)
    
    import shutil
    shutil.copy(split_config_path, r_split_config_path)
    LOGGER.info(f"Copied split config to {r_split_config_path}")
    
    # Load split config to get test perturbations
    from goal_2_baselines.split_logic import load_split_config
    split_config = load_split_config(split_config_path)
    all_test_perts = split_config.get("test", [])
    
    # Select random subset of test perturbations
    n_test = min(args.n_test_perturbations, len(all_test_perts))
    selected_test_perts = random.sample(all_test_perts, n_test)
    
    LOGGER.info(f"Selected {n_test} test perturbations: {selected_test_perts[:5]}...")
    
    # Run validation for each baseline
    all_results = []
    
    for baseline_name in selected_baselines:
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"Validating baseline: {baseline_name}")
        LOGGER.info(f"{'='*60}")
        
        config = BASELINE_CONFIGS[baseline_name]
        
        # Run R
        result_id_r = f"r_{baseline_name}_seed{args.seed}"
        r_result_dir = args.working_dir / "results" / result_id_r
        
        LOGGER.info("Running R baseline...")
        if not run_r_baseline(
            dataset_name=args.dataset_name,
            test_train_config_id=test_train_config_id,
            working_dir=args.working_dir,
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
            r_predictions, r_gene_names = load_r_results(r_result_dir)
        except Exception as e:
            LOGGER.error(f"Failed to load R results: {e}")
            continue
        
        # Run Python
        result_id_py = f"py_{baseline_name}_seed{args.seed}"
        py_result_dir = args.output_dir / result_id_py
        
        LOGGER.info("Running Python baseline...")
        if not run_python_baseline(
            dataset_name=args.dataset_name,
            split_config_path=split_config_path,
            baseline_name=baseline_name,
            output_dir=py_result_dir,
            pca_dim=args.pca_dim,
            ridge_penalty=args.ridge_penalty,
            seed=args.seed,
        ):
            LOGGER.error(f"Failed to run Python baseline: {baseline_name}")
            continue
        
        # Load Python results
        try:
            py_predictions, py_gene_names = load_r_results(py_result_dir)
        except Exception as e:
            LOGGER.error(f"Failed to load Python results: {e}")
            continue
        
        # Compare on selected test perturbations
        # Note: R uses clean_condition (without +ctrl), Python uses condition (with +ctrl)
        # We need to map between them
        comparison = compare_predictions(
            r_predictions=r_predictions,
            py_predictions=py_predictions,
            test_perturbations=selected_test_perts,
            tolerance=args.tolerance,
        )
        
        # Aggregate results
        if comparison["perturbations"]:
            mean_r = np.mean(comparison["pearson_r"])
            mean_l2 = np.mean(comparison["l2"])
            max_diff = np.max(comparison["max_diff"])
            within_tol_count = sum(comparison["within_tolerance"])
            total_count = len(comparison["perturbations"])
            
            all_results.append({
                "baseline": baseline_name,
                "mean_pearson_r": mean_r,
                "mean_l2": mean_l2,
                "max_diff": max_diff,
                "within_tolerance": f"{within_tol_count}/{total_count}",
                "n_test_perturbations": total_count,
            })
            
            LOGGER.info(f"Comparison results:")
            LOGGER.info(f"  Mean Pearson r: {mean_r:.6f}")
            LOGGER.info(f"  Mean L2: {mean_l2:.6f}")
            LOGGER.info(f"  Max difference: {max_diff:.6f}")
            LOGGER.info(f"  Within tolerance: {within_tol_count}/{total_count}")
    
    # Save summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary_path = output_dir / "r_parity_validation.csv"
        results_df.to_csv(summary_path, index=False)
        LOGGER.info(f"\nSaved validation summary to {summary_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("R vs Python Validation Summary")
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


"""
LOGO (Leave-One-GO-Out) evaluation for functional-class holdout.

This module implements LOGO evaluation that isolates specific functional classes
(e.g., Transcription genes) as the test set and trains on all other classes.

The key hypothesis: If scGPT embeddings cannot outperform random embeddings
on a biologically meaningful holdout (Transcription), then the embedding space
lacks semantic structure for extrapolation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import anndata as ad
import pandas as pd

from shared.io import load_annotations
from goal_2_baselines.baseline_runner import (
    compute_pseudobulk_expression_changes,
    run_single_baseline,
)
from goal_2_baselines.baseline_types import BaselineConfig, BaselineType, get_baseline_config

LOGGER = logging.getLogger(__name__)


@dataclass
class LogoResult:
    """Result for a single perturbation in LOGO evaluation."""
    perturbation: str
    baseline: str
    class_name: str
    pearson_r: float
    l2: float
    split_type: str = "functional_class_holdout"


def run_logo_evaluation(
    adata_path: Path,
    annotation_path: Path,
    dataset_name: str,
    output_dir: Path,
    class_name: str = "Transcription",
    baseline_types: Optional[List[BaselineType]] = None,
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
) -> pd.DataFrame:
    """
    Run LOGO evaluation isolating a functional class as test set.
    
    This is a stronger test than random splits because it evaluates biological
    extrapolation: Can the model predict Transcription genes using only non-Transcription
    training data?
    
    Args:
        adata_path: Path to perturb_processed.h5ad
        annotation_path: Path to functional class annotations TSV (target, class columns)
        dataset_name: Name of dataset (adamson, replogle_k562_essential, etc.)
        output_dir: Directory to save results
        class_name: Functional class to hold out as test set (default: "Transcription")
        baseline_types: List of baseline types to run (None = all 8 + mean_response)
        pca_dim: PCA dimension
        ridge_penalty: Ridge penalty
        seed: Random seed
    
    Returns:
        DataFrame with results (perturbation, baseline, class, pearson_r, l2, split_type)
    """
    LOGGER.info("=" * 60)
    LOGGER.info("LOGO EVALUATION: Functional Class Holdout")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Dataset: {dataset_name}")
    LOGGER.info(f"Holdout class: {class_name}")
    LOGGER.info(f"Output directory: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    LOGGER.info(f"Loading annotations from {annotation_path}")
    annotations = load_annotations(annotation_path)
    annotations["target"] = annotations["target"].astype(str)
    LOGGER.info(f"Loaded {len(annotations)} annotations")
    
    # Identify holdout class perturbations
    holdout_targets = annotations.loc[
        annotations["class"] == class_name, "target"
    ].unique().tolist()
    LOGGER.info(f"Found {len(holdout_targets)} perturbations in class '{class_name}'")
    
    if len(holdout_targets) == 0:
        raise ValueError(
            f"No perturbations found for class '{class_name}'. "
            f"Available classes: {annotations['class'].unique().tolist()}"
        )
    
    # Load data
    LOGGER.info(f"Loading data from {adata_path}")
    adata = ad.read_h5ad(adata_path)
    
    # Compute Y matrix (pseudobulk expression changes)
    # We need a dummy split config for compute_pseudobulk_expression_changes
    # It uses the split config to filter conditions, but we'll use all conditions
    # and then split by functional class
    all_conditions = sorted(adata.obs["condition"].unique().tolist())
    dummy_split_config = {
        "train": all_conditions,  # Will be re-split by functional class
        "test": [],
        "val": [],
    }
    
    LOGGER.info("Computing pseudobulk expression changes")
    Y_df, _ = compute_pseudobulk_expression_changes(adata, dummy_split_config, seed)
    LOGGER.info(f"Y matrix shape: {Y_df.shape} (genes × perturbations)")
    
    # Create functional class split
    # Train = all non-holdout class perturbations
    # Test = holdout class perturbations only
    available_targets = set(Y_df.columns)
    holdout_targets_available = [t for t in holdout_targets if t in available_targets]
    train_targets = [t for t in Y_df.columns if t not in holdout_targets_available]
    
    LOGGER.info(f"Train perturbations (non-{class_name}): {len(train_targets)}")
    LOGGER.info(f"Test perturbations ({class_name}): {len(holdout_targets_available)}")
    
    if len(train_targets) < 2:
        raise ValueError(
            f"Insufficient training data: {len(train_targets)} perturbations. "
            f"Need at least 2 for model training."
        )
    
    if len(holdout_targets_available) == 0:
        raise ValueError(
            f"No test perturbations available for class '{class_name}'. "
            f"Check that annotation targets match perturbation names in data."
        )
    
    # Split Y matrix
    Y_train = Y_df[train_targets]
    Y_test = Y_df[holdout_targets_available]
    
    LOGGER.info(f"Y_train shape: {Y_train.shape}")
    LOGGER.info(f"Y_test shape: {Y_test.shape}")
    
    # Default to all 8 baselines + mean_response
    if baseline_types is None:
        baseline_types = [
            BaselineType.SELFTRAINED,
            BaselineType.RANDOM_PERT_EMB,
            BaselineType.RANDOM_GENE_EMB,
            BaselineType.SCGPT_GENE_EMB,
            BaselineType.SCFOUNDATION_GENE_EMB,
            BaselineType.GEARS_PERT_EMB,
            BaselineType.K562_PERT_EMB,
            BaselineType.RPE1_PERT_EMB,
            BaselineType.MEAN_RESPONSE,
        ]
    
    LOGGER.info(f"Running {len(baseline_types)} baselines")
    
    # Get gene names (these are var_names, e.g., Ensembl IDs)
    gene_names = Y_df.index.tolist()
    
    # Get gene_name mapping if available (for alignment with embeddings that use gene symbols)
    gene_name_mapping = None
    if "gene_name" in adata.var.columns:
        gene_name_mapping = dict(zip(adata.var_names, adata.var["gene_name"]))
        LOGGER.info(f"Using gene name mapping for alignment")
    
    # Run each baseline
    all_results: List[LogoResult] = []
    
    for baseline_type in baseline_types:
        LOGGER.info("-" * 60)
        LOGGER.info(f"Running baseline: {baseline_type.value}")
        LOGGER.info("-" * 60)
        
        # Run baseline
        try:
            # Handle mean-response baseline separately
            if baseline_type == BaselineType.MEAN_RESPONSE:
                from goal_2_baselines.baseline_runner import run_mean_response_baseline
                result = run_mean_response_baseline(
                    Y_train=Y_train,
                    Y_test=Y_test,
                )
            else:
                # Get baseline configuration
                config = get_baseline_config(
                    baseline_type=baseline_type,
                    pca_dim=pca_dim,
                    ridge_penalty=ridge_penalty,
                    seed=seed,
                )
                
                result = run_single_baseline(
                    Y_train=Y_train,
                    Y_test=Y_test,
                    config=config,
                    gene_names=gene_names,
                    gene_name_mapping=gene_name_mapping,
                )
            
            # Extract metrics (dict mapping perturbation -> metrics dict)
            metrics_dict = result.get("metrics", {})
            test_perts = Y_test.columns.tolist()
            
            # Handle per-perturbation results
            # metrics_dict is {perturbation_name: {"pearson_r": float, "l2": float}, ...}
            for pert in test_perts:
                if pert in metrics_dict:
                    pert_metrics = metrics_dict[pert]
                    all_results.append(
                        LogoResult(
                            perturbation=pert,
                            baseline=baseline_type.value,
                            class_name=class_name,
                            pearson_r=pert_metrics.get("pearson_r", float("nan")),
                            l2=pert_metrics.get("l2", float("nan")),
                        )
                    )
                else:
                    # If metrics not available for this perturbation, skip
                    LOGGER.warning(
                        f"  No metrics available for perturbation {pert} "
                        f"in baseline {baseline_type.value}"
                    )
            
            # Compute mean r for logging
            pert_metrics = result.get("metrics", {})
            if pert_metrics:
                mean_r = sum(m.get("pearson_r", 0.0) for m in pert_metrics.values()) / len(pert_metrics)
                LOGGER.info(f"  Completed: {baseline_type.value} (mean r={mean_r:.4f}, n={len(pert_metrics)})")
            else:
                LOGGER.warning(f"  Completed: {baseline_type.value} (no metrics available)")
            
        except Exception as e:
            LOGGER.error(f"  Failed: {baseline_type.value}: {e}", exc_info=True)
            # Continue with other baselines
    
    # Convert to DataFrame
    results_df = pd.DataFrame([
        {
            "perturbation": r.perturbation,
            "baseline": r.baseline,
            "class": r.class_name,
            "pearson_r": r.pearson_r,
            "l2": r.l2,
            "split_type": r.split_type,
        }
        for r in all_results
    ])
    
    # Save results
    output_csv = output_dir / f"logo_{dataset_name}_{class_name.lower()}_results.csv"
    results_df.to_csv(output_csv, index=False)
    LOGGER.info(f"\nSaved results to {output_csv}")
    
    # Summary statistics
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("LOGO EVALUATION SUMMARY")
    LOGGER.info("=" * 60)
    
    if len(results_df) > 0:
        summary = results_df.groupby("baseline").agg({
            "pearson_r": ["mean", "std", "count"],
            "l2": ["mean", "std"],
        })
        LOGGER.info(f"\n{summary}")
        
        # Key comparison: scGPT vs Random
        scgpt_r = results_df[results_df["baseline"] == BaselineType.SCGPT_GENE_EMB.value]["pearson_r"].mean()
        random_r = results_df[results_df["baseline"] == BaselineType.RANDOM_GENE_EMB.value]["pearson_r"].mean()
        
        LOGGER.info("\n" + "-" * 60)
        LOGGER.info("KEY COMPARISON: scGPT vs Random Gene Embeddings")
        LOGGER.info("-" * 60)
        LOGGER.info(f"scGPT (lpm_scgptGeneEmb): mean r = {scgpt_r:.4f}")
        LOGGER.info(f"Random (lpm_randomGeneEmb): mean r = {random_r:.4f}")
        LOGGER.info(f"Difference: {scgpt_r - random_r:.4f}")
        
        if abs(scgpt_r - random_r) < 0.01:
            LOGGER.warning(
                "⚠️  scGPT performs similarly to Random embeddings. "
                "This suggests the embedding space lacks semantic structure for extrapolation."
            )
    
    LOGGER.info("\n" + "=" * 60)
    
    return results_df


def main():
    """CLI entry point for LOGO evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run LOGO evaluation (Functional Class Holdout)"
    )
    parser.add_argument(
        "--adata_path",
        type=Path,
        required=True,
        help="Path to perturb_processed.h5ad file",
    )
    parser.add_argument(
        "--annotation_path",
        type=Path,
        required=True,
        help="Path to functional class annotations TSV (target, class columns)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name (adamson, replogle_k562_essential, etc.)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save results",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default="Transcription",
        help="Functional class to hold out as test set (default: Transcription)",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=[bt.value for bt in BaselineType],
        help="Specific baselines to run (default: all 8 + mean_response)",
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
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Convert baseline names to BaselineType enums
    baseline_types = None
    if args.baselines:
        baseline_types = [BaselineType(bt) for bt in args.baselines]
    
    # Run LOGO evaluation
    results_df = run_logo_evaluation(
        adata_path=args.adata_path,
        annotation_path=args.annotation_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        class_name=args.class_name,
        baseline_types=baseline_types,
        pca_dim=args.pca_dim,
        ridge_penalty=args.ridge_penalty,
        seed=args.seed,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())

"""
LOGO evaluation with resampling support.

Extends LOGO evaluation to include:
1. Bootstrap confidence intervals for summary metrics
2. Paired baseline comparisons with permutation tests
3. Standardized output format (JSONL/Parquet)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from goal_3_prediction.functional_class_holdout.logo import run_logo_evaluation
from goal_3_prediction.lsft.compare_baselines_resampling import (
    compare_all_baseline_pairs,
    save_baseline_comparisons,
)
from stats.bootstrapping import bootstrap_mean_ci

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def standardize_logo_output(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize LOGO output for resampling.
    
    Ensures required fields are present:
    - perturbation
    - baseline (or baseline_type)
    - pearson_r
    - l2
    - class_name (or class)
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Original LOGO results DataFrame
    
    Returns
    -------
    pd.DataFrame
        Standardized DataFrame
    """
    standardized = results_df.copy()
    
    # Normalize column names
    if "baseline" in standardized.columns:
        standardized["baseline_type"] = standardized["baseline"]
    elif "baseline_type" not in standardized.columns:
        raise ValueError("Missing 'baseline' or 'baseline_type' column")
    
    if "class" in standardized.columns:
        standardized["class_name"] = standardized["class"]
    elif "class_name" not in standardized.columns:
        raise ValueError("Missing 'class' or 'class_name' column")
    
    # Ensure required fields exist
    required_fields = [
        "perturbation",
        "baseline_type",
        "class_name",
        "pearson_r",
        "l2",
    ]
    
    missing_fields = [f for f in required_fields if f not in standardized.columns]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    return standardized


def compute_logo_summary_with_cis(
    standardized_df: pd.DataFrame,
    n_boot: int = 1000,
    random_state: Optional[int] = None,
    alpha: float = 0.05,
) -> Dict:
    """
    Compute LOGO summary statistics with bootstrap confidence intervals.
    
    Parameters
    ----------
    standardized_df : pd.DataFrame
        Standardized LOGO results
    n_boot : int
        Number of bootstrap samples
    random_state : int or None
        Random seed
    alpha : float
        Significance level (1 - confidence level)
    
    Returns
    -------
    Dict
        Summary statistics with bootstrap CIs, grouped by baseline
    """
    summary = {}
    
    # Group by baseline_type
    grouped = standardized_df.groupby("baseline_type")
    
    for baseline, group_df in grouped:
        # Get per-perturbation metrics
        pearson_r_values = group_df["pearson_r"].dropna().values
        l2_values = group_df["l2"].dropna().values
        
        # Compute bootstrap CIs
        if len(pearson_r_values) > 0:
            pearson_mean, pearson_ci_lower, pearson_ci_upper = bootstrap_mean_ci(
                pearson_r_values, n_boot=n_boot, alpha=alpha, random_state=random_state
            )
        else:
            pearson_mean = pearson_ci_lower = pearson_ci_upper = np.nan
        
        if len(l2_values) > 0:
            l2_mean, l2_ci_lower, l2_ci_upper = bootstrap_mean_ci(
                l2_values, n_boot=n_boot, alpha=alpha, random_state=random_state
            )
        else:
            l2_mean = l2_ci_lower = l2_ci_upper = np.nan
        
        # Store summary
        summary[baseline] = {
            "baseline_type": baseline,
            "n_perturbations": len(group_df),
            "pearson_r": {
                "mean": float(pearson_mean),
                "ci_lower": float(pearson_ci_lower),
                "ci_upper": float(pearson_ci_upper),
                "std": float(np.std(pearson_r_values)) if len(pearson_r_values) > 0 else np.nan,
            },
            "l2": {
                "mean": float(l2_mean),
                "ci_lower": float(l2_ci_lower),
                "ci_upper": float(l2_ci_upper),
                "std": float(np.std(l2_values)) if len(l2_values) > 0 else np.nan,
            },
            "n_boot": n_boot,
            "alpha": alpha,
        }
    
    return summary


def save_standardized_logo_results(
    standardized_df: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    class_name: str,
    format: str = "both",  # "csv", "jsonl", "parquet", or "both"
) -> Dict[str, Path]:
    """
    Save standardized LOGO results in multiple formats.
    
    Parameters
    ----------
    standardized_df : pd.DataFrame
        Standardized LOGO results
    output_dir : Path
        Output directory
    dataset_name : str
        Dataset name
    class_name : str
        Functional class name
    format : str
        Output format: "csv", "jsonl", "parquet", or "both"
    
    Returns
    -------
    Dict[str, Path]
        Dictionary mapping format names to output file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    
    base_name = f"logo_{dataset_name}_{class_name}_standardized"
    
    # Save CSV (always, for backward compatibility)
    csv_path = output_dir / f"{base_name}.csv"
    standardized_df.to_csv(csv_path, index=False)
    output_paths["csv"] = csv_path
    LOGGER.info(f"Saved standardized CSV to {csv_path}")
    
    if format in ["jsonl", "both"]:
        # Save JSONL (one JSON object per line)
        jsonl_path = output_dir / f"{base_name}.jsonl"
        with open(jsonl_path, "w") as f:
            for _, row in standardized_df.iterrows():
                # Convert row to dict, handle NaN
                row_dict = row.to_dict()
                # Replace NaN with None for JSON serialization
                row_dict = {k: (None if pd.isna(v) else v) for k, v in row_dict.items()}
                f.write(json.dumps(row_dict) + "\n")
        output_paths["jsonl"] = jsonl_path
        LOGGER.info(f"Saved standardized JSONL to {jsonl_path}")
    
    if format in ["parquet", "both"]:
        # Save Parquet (efficient binary format)
        try:
            parquet_path = output_dir / f"{base_name}.parquet"
            standardized_df.to_parquet(parquet_path, index=False)
            output_paths["parquet"] = parquet_path
            LOGGER.info(f"Saved standardized Parquet to {parquet_path}")
        except ImportError:
            LOGGER.warning("pyarrow not installed, skipping Parquet output. Install with: pip install pyarrow")
    
    return output_paths


def run_logo_with_resampling(
    adata_path: Path,
    annotation_path: Path,
    dataset_name: str,
    output_dir: Path,
    class_name: str = "Transcription",
    baseline_types: Optional[List] = None,
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
    n_boot: int = 1000,
    n_perm: int = 10000,
    output_format: str = "both",
    skip_comparisons: bool = False,
) -> Dict:
    """
    Run LOGO evaluation with resampling support.
    
    This function:
    1. Runs standard LOGO evaluation
    2. Standardizes output format
    3. Saves standardized results (CSV, JSONL, Parquet)
    4. Computes bootstrap CIs for summary metrics
    5. Performs paired baseline comparisons (optional)
    
    Parameters
    ----------
    adata_path : Path
        Path to adata file
    annotation_path : Path
        Path to functional class annotations
    dataset_name : str
        Dataset name
    output_dir : Path
        Output directory
    class_name : str
        Functional class to hold out
    baseline_types : Optional[List]
        List of baseline types (None = all)
    pca_dim : int
        PCA dimension
    ridge_penalty : float
        Ridge penalty
    seed : int
        Random seed
    n_boot : int
        Number of bootstrap samples for CIs
    n_perm : int
        Number of permutations for comparison tests
    output_format : str
        Output format: "csv", "jsonl", "parquet", or "both"
    skip_comparisons : bool
        Skip baseline comparisons (requires multiple baselines)
    
    Returns
    -------
    Dict
        Dictionary containing:
        - results_df: Standardized results DataFrame
        - summary: Summary statistics with bootstrap CIs
        - output_paths: Paths to saved files
    """
    LOGGER.info(f"Running LOGO evaluation with resampling support for {dataset_name}")
    LOGGER.info(f"Holdout class: {class_name}")
    
    # Run standard LOGO evaluation
    results_df = run_logo_evaluation(
        adata_path=adata_path,
        annotation_path=annotation_path,
        dataset_name=dataset_name,
        output_dir=output_dir,
        class_name=class_name,
        baseline_types=baseline_types,
        pca_dim=pca_dim,
        ridge_penalty=ridge_penalty,
        seed=seed,
    )
    
    # Standardize output
    standardized_df = standardize_logo_output(results_df)
    
    # Save standardized results
    output_paths = save_standardized_logo_results(
        standardized_df=standardized_df,
        output_dir=output_dir,
        dataset_name=dataset_name,
        class_name=class_name,
        format=output_format,
    )
    
    # Compute summary statistics with bootstrap CIs
    summary = compute_logo_summary_with_cis(
        standardized_df=standardized_df,
        n_boot=n_boot,
        random_state=seed,
    )
    
    # Save summary
    summary_path = output_dir / f"logo_{dataset_name}_{class_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info(f"Saved summary with CIs to {summary_path}")
    
    # Baseline comparisons (optional)
    comparison_df = None
    if not skip_comparisons:
        LOGGER.info("Computing baseline comparisons...")
        try:
            comparison_df = compare_all_baseline_pairs(
                results_df=standardized_df,
                metrics=["pearson_r", "l2"],
                top_pcts=None,  # LOGO doesn't use top_pct
                n_perm=n_perm,
                n_boot=n_boot,
                random_state=seed,
            )
            comparison_path = output_dir / f"logo_{dataset_name}_{class_name}_baseline_comparisons"
            save_baseline_comparisons(comparison_df, comparison_path, format="both")
            LOGGER.info(f"Saved baseline comparisons to {comparison_path}")
        except Exception as e:
            LOGGER.warning(f"Baseline comparisons failed: {e}")
    
    return {
        "results_df": standardized_df,
        "summary": summary,
        "output_paths": output_paths,
        "summary_path": summary_path,
        "comparison_df": comparison_df,
    }


def main():
    """CLI entry point for LOGO evaluation with resampling."""
    import argparse
    from goal_2_baselines.baseline_types import BaselineType
    
    parser = argparse.ArgumentParser(
        description="Run LOGO evaluation with resampling support"
    )
    parser.add_argument("--adata_path", type=Path, required=True, help="Path to adata file")
    parser.add_argument("--annotation_path", type=Path, required=True, help="Path to annotations TSV")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--class_name", type=str, default="Transcription", help="Holdout class")
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=[bt.value for bt in BaselineType],
        help="Specific baselines to run (default: all)",
    )
    parser.add_argument("--pca_dim", type=int, default=10, help="PCA dimension")
    parser.add_argument("--ridge_penalty", type=float, default=0.1, help="Ridge penalty")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--n_boot", type=int, default=1000, help="Number of bootstrap samples")
    parser.add_argument("--n_perm", type=int, default=10000, help="Number of permutations")
    parser.add_argument(
        "--skip_comparisons",
        action="store_true",
        help="Skip baseline comparisons",
    )
    
    args = parser.parse_args()
    
    # Convert baseline names to BaselineType enums
    baseline_types = None
    if args.baselines:
        baseline_types = [BaselineType(bt) for bt in args.baselines]
    
    # Run LOGO with resampling
    results = run_logo_with_resampling(
        adata_path=args.adata_path,
        annotation_path=args.annotation_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        class_name=args.class_name,
        baseline_types=baseline_types,
        pca_dim=args.pca_dim,
        ridge_penalty=args.ridge_penalty,
        seed=args.seed,
        n_boot=args.n_boot,
        n_perm=args.n_perm,
        skip_comparisons=args.skip_comparisons,
    )
    
    LOGGER.info("\n✓ LOGO evaluation with resampling complete!")
    LOGGER.info(f"✓ Results saved to: {results['output_paths']}")
    LOGGER.info(f"✓ Summary with CIs saved to: {results['summary_path']}")
    if results['comparison_df'] is not None:
        LOGGER.info("✓ Baseline comparisons complete")


if __name__ == "__main__":
    main()

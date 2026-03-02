"""
LSFT evaluation with resampling support (v2).

This module extends the original LSFT evaluation to:
1. Emit standardized per-perturbation metrics (JSONL/Parquet)
2. Compute bootstrap confidence intervals for summary metrics
3. Support resampling-based statistical analysis
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from goal_3_prediction.lsft.lsft import evaluate_lsft
from stats.bootstrapping import bootstrap_mean_ci

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def standardize_lsft_output(
    results_df: pd.DataFrame,
    total_train_size: int,
) -> pd.DataFrame:
    """
    Standardize LSFT output to include all required fields for resampling.
    
    Adds standardized fields:
    - pearson_r: Local Pearson r performance
    - l2: Local L2 performance
    - hardness: Top-K cosine similarity (mean similarity to filtered training set)
    - embedding_similarity: Same as hardness (for consistency)
    - split_fraction: Fraction of training data used (local_train_size / total_train_size)
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Original LSFT results DataFrame
    total_train_size : int
        Total number of training perturbations
    
    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with additional fields
    """
    standardized = results_df.copy()
    
    # Add standardized field names
    standardized["pearson_r"] = standardized["performance_local_pearson_r"]
    standardized["l2"] = standardized["performance_local_l2"]
    
    # Hardness is the mean similarity to filtered training perturbations
    standardized["hardness"] = standardized["local_mean_similarity"]
    standardized["embedding_similarity"] = standardized["local_mean_similarity"]
    
    # Split fraction = fraction of training data used
    standardized["split_fraction"] = standardized["local_train_size"] / total_train_size
    
    # Ensure required fields exist
    required_fields = [
        "test_perturbation",
        "baseline_type",
        "top_pct",
        "pearson_r",
        "l2",
        "hardness",
        "embedding_similarity",
        "split_fraction",
    ]
    
    # Check all required fields are present
    missing_fields = [f for f in required_fields if f not in standardized.columns]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    return standardized


def save_standardized_lsft_results(
    standardized_df: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    baseline_type: str,
    format: str = "both",  # "csv", "jsonl", "parquet", or "both"
) -> Dict[str, Path]:
    """
    Save standardized LSFT results in multiple formats.
    
    Parameters
    ----------
    standardized_df : pd.DataFrame
        Standardized LSFT results
    output_dir : Path
        Output directory
    dataset_name : str
        Dataset name
    baseline_type : str
        Baseline type identifier
    format : str
        Output format: "csv", "jsonl", "parquet", or "both"
    
    Returns
    -------
    Dict[str, Path]
        Dictionary mapping format names to output file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    
    base_name = f"lsft_{dataset_name}_{baseline_type}_standardized"
    
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


def evaluate_lsft_with_resampling(
    adata_path: Path,
    split_config_path: Path,
    baseline_type,
    dataset_name: str,
    output_dir: Path,
    top_pcts: List[float] = [0.01, 0.05, 0.10],
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
    n_boot: int = 1000,
    output_format: str = "both",
) -> Dict:
    """
    Evaluate LSFT with resampling support.
    
    This function:
    1. Runs standard LSFT evaluation
    2. Standardizes output format
    3. Saves standardized results (CSV, JSONL, Parquet)
    4. Computes bootstrap CIs for summary metrics
    
    Parameters
    ----------
    adata_path : Path
        Path to adata file
    split_config_path : Path
        Path to split config JSON
    baseline_type
        Baseline type (BaselineType enum)
    dataset_name : str
        Dataset name
    output_dir : Path
        Output directory
    top_pcts : List[float]
        List of top percentages to try
    pca_dim : int
        PCA dimension
    ridge_penalty : float
        Ridge penalty
    seed : int
        Random seed
    n_boot : int
        Number of bootstrap samples for CIs
    output_format : str
        Output format: "csv", "jsonl", "parquet", or "both"
    
    Returns
    -------
    Dict
        Dictionary containing:
        - results_df: Standardized results DataFrame
        - summary: Summary statistics with bootstrap CIs
        - output_paths: Paths to saved files
    """
    LOGGER.info(f"Running LSFT evaluation with resampling support for {dataset_name}")
    
    # Run standard LSFT evaluation
    results_df = evaluate_lsft(
        adata_path=adata_path,
        split_config_path=split_config_path,
        baseline_type=baseline_type,
        dataset_name=dataset_name,
        output_dir=output_dir,
        top_pcts=top_pcts,
        pca_dim=pca_dim,
        ridge_penalty=ridge_penalty,
        seed=seed,
    )
    
    # Get total training size (needed for split_fraction)
    # Reload split config to get actual total train size
    from goal_2_baselines.split_logic import load_split_config
    split_config = load_split_config(split_config_path)
    total_train_size = len(split_config.get("train", []))
    
    # Standardize output
    standardized_df = standardize_lsft_output(results_df, total_train_size)
    
    # Save standardized results
    output_paths = save_standardized_lsft_results(
        standardized_df=standardized_df,
        output_dir=output_dir,
        dataset_name=dataset_name,
        baseline_type=baseline_type.value,
        format=output_format,
    )
    
    # Compute summary statistics with bootstrap CIs
    summary = compute_lsft_summary_with_cis(
        standardized_df=standardized_df,
        n_boot=n_boot,
        random_state=seed,
    )
    
    # Save summary
    summary_path = output_dir / f"lsft_{dataset_name}_{baseline_type.value}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info(f"Saved summary with CIs to {summary_path}")
    
    return {
        "results_df": standardized_df,
        "summary": summary,
        "output_paths": output_paths,
        "summary_path": summary_path,
    }


def compute_lsft_summary_with_cis(
    standardized_df: pd.DataFrame,
    n_boot: int = 1000,
    random_state: Optional[int] = None,
    alpha: float = 0.05,
) -> Dict:
    """
    Compute LSFT summary statistics with bootstrap confidence intervals.
    
    Parameters
    ----------
    standardized_df : pd.DataFrame
        Standardized LSFT results
    n_boot : int
        Number of bootstrap samples
    random_state : int or None
        Random seed
    alpha : float
        Significance level (1 - confidence level)
    
    Returns
    -------
    Dict
        Summary statistics with bootstrap CIs, grouped by baseline_type and top_pct
    """
    summary = {}
    
    # Group by baseline_type and top_pct
    grouped = standardized_df.groupby(["baseline_type", "top_pct"])
    
    for (baseline, top_pct), group_df in grouped:
        key = f"{baseline}_top{top_pct*100:.0f}pct"
        
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
        summary[key] = {
            "baseline_type": baseline,
            "top_pct": top_pct,
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
        
        # Additional statistics
        summary[key]["hardness"] = {
            "mean": float(group_df["hardness"].mean()),
            "std": float(group_df["hardness"].std()),
        }
        summary[key]["split_fraction"] = {
            "mean": float(group_df["split_fraction"].mean()),
            "std": float(group_df["split_fraction"].std()),
        }
    
    return summary


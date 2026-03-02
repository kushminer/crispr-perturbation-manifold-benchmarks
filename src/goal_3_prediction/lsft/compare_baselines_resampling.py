"""
Paired baseline comparisons with permutation tests and bootstrap CIs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from stats.bootstrapping import bootstrap_mean_ci
from stats.permutation import paired_permutation_test

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def compare_baselines_with_resampling(
    results_df: pd.DataFrame,
    baseline1: str,
    baseline2: str,
    metric: str = "pearson_r",
    top_pct: Optional[float] = None,
    n_perm: int = 10000,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> Dict:
    """
    Compare two baselines using paired permutation test and bootstrap CI.
    
    For each test perturbation, computes delta = baseline1 - baseline2,
    then performs:
    1. Permutation test on deltas (two-sided)
    2. Bootstrap CI on mean delta
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Standardized LSFT results DataFrame
    baseline1 : str
        First baseline type identifier
    baseline2 : str
        Second baseline type identifier
    metric : str
        Metric to compare: "pearson_r" or "l2"
    top_pct : float or None
        If provided, filter to specific top_pct
    n_perm : int
        Number of permutations for permutation test
    n_boot : int
        Number of bootstrap samples for CI
    alpha : float
        Significance level
    random_state : int or None
        Random seed
    
    Returns
    -------
    Dict
        Comparison results with p-value, mean delta, CI, etc.
    """
    # Filter to relevant baselines and top_pct (if specified)
    filtered_df = results_df[
        (results_df["baseline_type"].isin([baseline1, baseline2]))
    ].copy()
    
    if top_pct is not None:
        filtered_df = filtered_df[filtered_df["top_pct"] == top_pct]
    
    if len(filtered_df) == 0:
        raise ValueError(f"No data found for baselines {baseline1} and {baseline2}")
    
    # Pivot to get paired comparisons (one row per perturbation and top_pct)
    pivot_df = filtered_df.pivot_table(
        index=["test_perturbation", "top_pct"],
        columns="baseline_type",
        values=metric,
        aggfunc="first",
    )
    
    # Check if both baselines are present
    if baseline1 not in pivot_df.columns or baseline2 not in pivot_df.columns:
        missing = [b for b in [baseline1, baseline2] if b not in pivot_df.columns]
        raise ValueError(f"Missing baselines in data: {missing}")
    
    # Compute deltas per perturbation
    deltas = (pivot_df[baseline1] - pivot_df[baseline2]).dropna().values
    
    if len(deltas) == 0:
        raise ValueError("No valid paired comparisons found")
    
    # Compute permutation test
    mean_delta, p_value = paired_permutation_test(
        deltas,
        n_perm=n_perm,
        alternative="two-sided",
        random_state=random_state,
    )
    
    # Compute bootstrap CI on mean delta
    mean_delta_ci, ci_lower, ci_upper = bootstrap_mean_ci(
        deltas,
        n_boot=n_boot,
        alpha=alpha,
        random_state=random_state,
    )
    
    # Additional statistics
    baseline1_mean = pivot_df[baseline1].mean()
    baseline2_mean = pivot_df[baseline2].mean()
    
    # Bootstrap CIs for individual baselines
    baseline1_values = pivot_df[baseline1].dropna().values
    baseline2_values = pivot_df[baseline2].dropna().values
    
    if len(baseline1_values) > 0:
        b1_mean, b1_ci_lower, b1_ci_upper = bootstrap_mean_ci(
            baseline1_values, n_boot=n_boot, alpha=alpha, random_state=random_state
        )
    else:
        b1_mean = b1_ci_lower = b1_ci_upper = np.nan
    
    if len(baseline2_values) > 0:
        b2_mean, b2_ci_lower, b2_ci_upper = bootstrap_mean_ci(
            baseline2_values, n_boot=n_boot, alpha=alpha, random_state=random_state
        )
    else:
        b2_mean = b2_ci_lower = b2_ci_upper = np.nan
    
    return {
        "baseline1": baseline1,
        "baseline2": baseline2,
        "metric": metric,
        "top_pct": top_pct,
        "n_pairs": len(deltas),
        "baseline1_mean": float(baseline1_mean),
        "baseline1_ci_lower": float(b1_ci_lower),
        "baseline1_ci_upper": float(b1_ci_upper),
        "baseline2_mean": float(baseline2_mean),
        "baseline2_ci_lower": float(b2_ci_lower),
        "baseline2_ci_upper": float(b2_ci_upper),
        "mean_delta": float(mean_delta_ci),
        "delta_ci_lower": float(ci_lower),
        "delta_ci_upper": float(ci_upper),
        "p_value": float(p_value),
        "n_perm": n_perm,
        "n_boot": n_boot,
        "alpha": alpha,
    }


def compare_all_baseline_pairs(
    results_df: pd.DataFrame,
    baseline_types: Optional[List[str]] = None,
    metrics: List[str] = ["pearson_r", "l2"],
    top_pcts: Optional[List[float]] = None,
    n_perm: int = 10000,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compare all pairs of baselines with permutation tests and bootstrap CIs.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Standardized LSFT results DataFrame
    baseline_types : List[str] or None
        If provided, only compare these baselines. Otherwise, compare all.
    metrics : List[str]
        Metrics to compare
    top_pcts : List[float] or None
        If provided, compare for each top_pct separately. Otherwise, aggregate.
    n_perm : int
        Number of permutations
    n_boot : int
        Number of bootstrap samples
    alpha : float
        Significance level
    random_state : int or None
        Random seed
    
    Returns
    -------
    pd.DataFrame
        Comparison results for all pairs
    """
    if baseline_types is None:
        baseline_types = sorted(results_df["baseline_type"].unique())
    
    if len(baseline_types) < 2:
        raise ValueError("Need at least 2 baselines to compare")
    
    if top_pcts is None:
        top_pcts = [None]  # Aggregate across all top_pcts
    
    all_comparisons = []
    
    # Compare all pairs
    for i, baseline1 in enumerate(baseline_types):
        for baseline2 in baseline_types[i + 1 :]:
            for metric in metrics:
                for top_pct in top_pcts:
                    try:
                        comparison = compare_baselines_with_resampling(
                            results_df=results_df,
                            baseline1=baseline1,
                            baseline2=baseline2,
                            metric=metric,
                            top_pct=top_pct,
                            n_perm=n_perm,
                            n_boot=n_boot,
                            alpha=alpha,
                            random_state=random_state,
                        )
                        all_comparisons.append(comparison)
                    except Exception as e:
                        LOGGER.warning(
                            f"Failed to compare {baseline1} vs {baseline2} "
                            f"(metric={metric}, top_pct={top_pct}): {e}"
                        )
    
    if not all_comparisons:
        raise ValueError("No valid comparisons found")
    
    comparison_df = pd.DataFrame(all_comparisons)
    return comparison_df


def save_baseline_comparisons(
    comparison_df: pd.DataFrame,
    output_path: Path,
    format: str = "both",  # "csv", "json", or "both"
):
    """
    Save baseline comparison results.
    
    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison results DataFrame
    output_path : Path
        Output file path (without extension)
    format : str
        Output format: "csv", "json", or "both"
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format in ["csv", "both"]:
        csv_path = output_path.with_suffix(".csv")
        comparison_df.to_csv(csv_path, index=False)
        LOGGER.info(f"Saved baseline comparisons to {csv_path}")
    
    if format in ["json", "both"]:
        json_path = output_path.with_suffix(".json")
        # Convert DataFrame to list of dicts for JSON
        comparison_dict = comparison_df.to_dict(orient="records")
        with open(json_path, "w") as f:
            json.dump(comparison_dict, f, indent=2)
        LOGGER.info(f"Saved baseline comparisons to {json_path}")

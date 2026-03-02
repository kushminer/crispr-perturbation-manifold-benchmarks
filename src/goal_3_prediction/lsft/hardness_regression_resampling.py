"""
Hardness-performance regression with bootstrapped slopes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress

from stats.bootstrapping import bootstrap_mean_ci

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def fit_hardness_performance_regression(
    hardness: np.ndarray,
    performance: np.ndarray,
) -> Dict:
    """
    Fit linear regression: performance = slope * hardness + intercept.
    
    Parameters
    ----------
    hardness : np.ndarray
        Hardness values (x-axis)
    performance : np.ndarray
        Performance values (y-axis, e.g., Pearson r)
    
    Returns
    -------
    Dict
        Regression results: slope, intercept, r, r_squared, p_value, stderr
    """
    # Remove NaN pairs
    valid_mask = ~(np.isnan(hardness) | np.isnan(performance))
    hardness_clean = hardness[valid_mask]
    performance_clean = performance[valid_mask]
    
    if len(hardness_clean) < 2:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r": np.nan,
            "r_squared": np.nan,
            "p_value": np.nan,
            "stderr": np.nan,
            "n_points": len(hardness_clean),
        }
    
    # Fit regression
    slope, intercept, r, p_value, stderr = linregress(hardness_clean, performance_clean)
    r_squared = r ** 2
    
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r": float(r),
        "r_squared": float(r_squared),
        "p_value": float(p_value),
        "stderr": float(stderr),
        "n_points": len(hardness_clean),
    }


def bootstrap_hardness_regression(
    hardness: np.ndarray,
    performance: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> Dict:
    """
    Bootstrap hardness-performance regression to get CI on slope, r, R².
    
    Parameters
    ----------
    hardness : np.ndarray
        Hardness values
    performance : np.ndarray
        Performance values
    n_boot : int
        Number of bootstrap samples
    alpha : float
        Significance level
    random_state : int or None
        Random seed
    
    Returns
    -------
    Dict
        Regression results with bootstrap CIs
    """
    # Remove NaN pairs
    valid_mask = ~(np.isnan(hardness) | np.isnan(performance))
    hardness_clean = hardness[valid_mask]
    performance_clean = performance[valid_mask]
    
    if len(hardness_clean) < 2:
        return {
            "slope": np.nan,
            "slope_ci_lower": np.nan,
            "slope_ci_upper": np.nan,
            "intercept": np.nan,
            "r": np.nan,
            "r_ci_lower": np.nan,
            "r_ci_upper": np.nan,
            "r_squared": np.nan,
            "r_squared_ci_lower": np.nan,
            "r_squared_ci_upper": np.nan,
            "p_value": np.nan,
            "n_points": len(hardness_clean),
            "n_boot": n_boot,
        }
    
    # Fit regression on original data
    original_reg = fit_hardness_performance_regression(hardness_clean, performance_clean)
    
    # Bootstrap resampling
    rng = np.random.default_rng(random_state)
    n = len(hardness_clean)
    
    boot_slopes = []
    boot_rs = []
    boot_r_squareds = []
    
    for _ in range(n_boot):
        # Resample with replacement
        boot_indices = rng.choice(n, size=n, replace=True)
        boot_hardness = hardness_clean[boot_indices]
        boot_performance = performance_clean[boot_indices]
        
        # Fit regression on bootstrap sample
        boot_reg = fit_hardness_performance_regression(boot_hardness, boot_performance)
        
        if not np.isnan(boot_reg["slope"]):
            boot_slopes.append(boot_reg["slope"])
            boot_rs.append(boot_reg["r"])
            boot_r_squareds.append(boot_reg["r_squared"])
    
    if len(boot_slopes) == 0:
        LOGGER.warning("All bootstrap regressions failed, returning original without CIs")
        return {
            **original_reg,
            "slope_ci_lower": np.nan,
            "slope_ci_upper": np.nan,
            "r_ci_lower": np.nan,
            "r_ci_upper": np.nan,
            "r_squared_ci_lower": np.nan,
            "r_squared_ci_upper": np.nan,
            "n_boot": n_boot,
        }
    
    # Compute percentile-based CIs
    percentiles = (100 * alpha / 2, 100 * (1 - alpha / 2))
    
    slope_ci_lower, slope_ci_upper = np.percentile(boot_slopes, percentiles)
    r_ci_lower, r_ci_upper = np.percentile(boot_rs, percentiles)
    r_squared_ci_lower, r_squared_ci_upper = np.percentile(boot_r_squareds, percentiles)
    
    return {
        "slope": original_reg["slope"],
        "slope_ci_lower": float(slope_ci_lower),
        "slope_ci_upper": float(slope_ci_upper),
        "intercept": original_reg["intercept"],
        "r": original_reg["r"],
        "r_ci_lower": float(r_ci_lower),
        "r_ci_upper": float(r_ci_upper),
        "r_squared": original_reg["r_squared"],
        "r_squared_ci_lower": float(r_squared_ci_lower),
        "r_squared_ci_upper": float(r_squared_ci_upper),
        "p_value": original_reg["p_value"],
        "n_points": original_reg["n_points"],
        "n_boot": n_boot,
        "alpha": alpha,
    }


def compute_hardness_regressions_for_lsft(
    results_df: pd.DataFrame,
    performance_metric: str = "pearson_r",
    hardness_metric: str = "hardness",
    baseline_types: Optional[List[str]] = None,
    top_pcts: Optional[List[float]] = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute hardness-performance regressions for all baseline/top_pct combinations.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Standardized LSFT results DataFrame
    performance_metric : str
        Performance metric to use (e.g., "pearson_r", "l2")
    hardness_metric : str
        Hardness metric to use (e.g., "hardness", "embedding_similarity")
    baseline_types : List[str] or None
        If provided, only compute for these baselines
    top_pcts : List[float] or None
        If provided, only compute for these top_pcts. Otherwise, use all.
    n_boot : int
        Number of bootstrap samples
    alpha : float
        Significance level
    random_state : int or None
        Random seed
    
    Returns
    -------
    pd.DataFrame
        Regression results with bootstrap CIs for each baseline/top_pct
    """
    if baseline_types is None:
        baseline_types = sorted(results_df["baseline_type"].unique())
    
    if top_pcts is None:
        top_pcts = sorted(results_df["top_pct"].unique())
    
    all_regressions = []
    
    for baseline_type in baseline_types:
        for top_pct in top_pcts:
            # Filter data
            subset = results_df[
                (results_df["baseline_type"] == baseline_type)
                & (results_df["top_pct"] == top_pct)
            ]
            
            if len(subset) < 2:
                LOGGER.warning(
                    f"Insufficient data for {baseline_type} at {top_pct*100}%: {len(subset)} points"
                )
                continue
            
            # Get hardness and performance
            hardness = subset[hardness_metric].values
            performance = subset[performance_metric].values
            
            # Compute bootstrapped regression
            regression = bootstrap_hardness_regression(
                hardness=hardness,
                performance=performance,
                n_boot=n_boot,
                alpha=alpha,
                random_state=random_state,
            )
            
            # Add metadata
            regression["baseline_type"] = baseline_type
            regression["top_pct"] = top_pct
            regression["performance_metric"] = performance_metric
            regression["hardness_metric"] = hardness_metric
            
            all_regressions.append(regression)
    
    if not all_regressions:
        raise ValueError("No valid regressions computed")
    
    regression_df = pd.DataFrame(all_regressions)
    return regression_df


def save_hardness_regressions(
    regression_df: pd.DataFrame,
    output_path: Path,
    format: str = "both",  # "csv", "json", or "both"
):
    """
    Save hardness regression results.
    
    Parameters
    ----------
    regression_df : pd.DataFrame
        Regression results DataFrame
    output_path : Path
        Output file path (without extension)
    format : str
        Output format: "csv", "json", or "both"
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format in ["csv", "both"]:
        csv_path = output_path.with_suffix(".csv")
        regression_df.to_csv(csv_path, index=False)
        LOGGER.info(f"Saved hardness regressions to {csv_path}")
    
    if format in ["json", "both"]:
        json_path = output_path.with_suffix(".json")
        regression_dict = regression_df.to_dict(orient="records")
        with open(json_path, "w") as f:
            json.dump(regression_dict, f, indent=2)
        LOGGER.info(f"Saved hardness regressions to {json_path}")

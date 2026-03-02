#!/usr/bin/env python3
"""
Noise Injection & Lipschitz Estimation: Robustness Analysis

This module implements Epic 3 of the Manifold Law Diagnostic Suite.

Goal: Measure robustness of local interpolation under controlled noise.
Estimate Lipschitz constant of LSFT prediction function.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from goal_2_baselines.baseline_types import BaselineType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def inject_noise(
    data: np.ndarray,
    noise_type: str = "gaussian",
    noise_level: float = 0.1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Inject noise into data.
    
    Args:
        data: Input data array
        noise_type: "gaussian" or "dropout"
        noise_level: For gaussian: std dev (σ), for dropout: probability (p)
        seed: Random seed
        
    Returns:
        Noisy data array
    """
    rng = np.random.default_rng(seed)
    
    if noise_type == "gaussian":
        noise = rng.normal(0, noise_level, size=data.shape)
        return data + noise
    elif noise_type == "dropout":
        mask = rng.random(data.shape) > noise_level
        return data * mask
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def estimate_lipschitz_constant(
    noise_levels: np.ndarray,
    r_values: np.ndarray,
) -> float:
    """
    Estimate Lipschitz constant from noise sensitivity curve.
    
    L ≈ max |Δprediction| / σ
    
    Args:
        noise_levels: Array of noise levels (σ values)
        r_values: Array of Pearson r values at each noise level
        
    Returns:
        Estimated Lipschitz constant
    """
    if len(noise_levels) < 2:
        return np.nan
    
    # Compute sensitivity (drop in r per unit noise)
    r_drop = r_values[0] - r_values  # Drop from baseline (no noise)
    sensitivity = np.abs(np.diff(r_drop)) / np.diff(noise_levels)
    
    # Lipschitz constant is maximum sensitivity
    L = np.max(sensitivity) if len(sensitivity) > 0 else np.nan
    return float(L)


def run_noise_injection_analysis(
    noise_results_df: pd.DataFrame,
    output_dir: Path = Path("."),
) -> Dict:
    """
    Analyze noise injection results to compute sensitivity curves and Lipschitz constants.
    
    Args:
        noise_results_df: DataFrame with columns: k, noise_level, mean_r, mean_l2
                         MUST include noise_level=0 baseline
        output_dir: Output directory
        
    Returns:
        Dictionary with analysis results including Lipschitz constants
    """
    LOGGER.info("Analyzing noise injection results...")
    
    # Verify baseline (noise_level=0) exists
    baseline_data = noise_results_df[noise_results_df['noise_level'] == 0.0]
    if len(baseline_data) == 0:
        LOGGER.error("No baseline (noise_level=0) found! Cannot compute sensitivity curves.")
        raise ValueError("noise_results_df must include noise_level=0 baseline")
    
    # Group by k for analysis
    analysis_results = []
    
    for k in noise_results_df['k'].unique():
        k_data = noise_results_df[noise_results_df['k'] == k].sort_values('noise_level')
        
        # Get baseline
        baseline = k_data[k_data['noise_level'] == 0.0]
        if len(baseline) == 0:
            LOGGER.warning(f"No baseline for k={k}, skipping")
            continue
        
        r_baseline = baseline['mean_r'].iloc[0]
        l2_baseline = baseline['mean_l2'].iloc[0]
        
        # Get noisy conditions (exclude baseline)
        noisy_data = k_data[k_data['noise_level'] > 0].dropna(subset=['mean_r'])
        
        if len(noisy_data) == 0:
            LOGGER.warning(f"No noisy data for k={k}, skipping")
            continue
        
        noise_levels = noisy_data['noise_level'].values
        r_values = noisy_data['mean_r'].values
        
        # Compute sensitivity: Δr(σ) = r(σ) - r(0)
        delta_r = r_values - r_baseline
        
        # Compute Lipschitz constant: L = max |r(σ) - r(0)| / σ
        abs_sensitivity = np.abs(delta_r) / noise_levels
        lipschitz_constant = np.max(abs_sensitivity) if len(abs_sensitivity) > 0 else np.nan
        
        # Compute mean sensitivity
        mean_sensitivity = np.mean(abs_sensitivity) if len(abs_sensitivity) > 0 else np.nan
        
        analysis_results.append({
            "k": k,
            "r_baseline": r_baseline,
            "l2_baseline": l2_baseline,
            "lipschitz_constant": lipschitz_constant,
            "mean_sensitivity": mean_sensitivity,
            "max_delta_r": np.max(np.abs(delta_r)) if len(delta_r) > 0 else np.nan,
            "n_noise_levels": len(noisy_data),
        })
    
    analysis_df = pd.DataFrame(analysis_results)
    
    # Save analysis
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "noise_sensitivity_analysis.csv"
    analysis_df.to_csv(analysis_path, index=False)
    LOGGER.info(f"Saved noise sensitivity analysis to {analysis_path}")
    
    return {
        "analysis": analysis_df,
        "summary": {
            "mean_lipschitz": analysis_df['lipschitz_constant'].mean() if len(analysis_df) > 0 else np.nan,
            "max_lipschitz": analysis_df['lipschitz_constant'].max() if len(analysis_df) > 0 else np.nan,
        }
    }


def analyze_noise_injection_results(
    noise_results_df: pd.DataFrame,
    output_dir: Path = Path("."),
) -> Dict:
    """
    Analyze noise injection results to compute sensitivity curves and Lipschitz constants.
    
    Args:
        noise_results_df: DataFrame with columns: k, noise_level, mean_r, mean_l2
                         MUST include noise_level=0 baseline
        output_dir: Output directory
        
    Returns:
        Dictionary with analysis results including Lipschitz constants
    """
    LOGGER.info("Analyzing noise injection results...")
    
    # Verify baseline (noise_level=0) exists
    baseline_data = noise_results_df[noise_results_df['noise_level'] == 0.0]
    if len(baseline_data) == 0:
        LOGGER.error("No baseline (noise_level=0) found! Cannot compute sensitivity curves.")
        raise ValueError("noise_results_df must include noise_level=0 baseline")
    
    # Group by k for analysis
    analysis_results = []
    
    for k in noise_results_df['k'].unique():
        k_data = noise_results_df[noise_results_df['k'] == k].sort_values('noise_level')
        
        # Get baseline
        baseline = k_data[k_data['noise_level'] == 0.0]
        if len(baseline) == 0:
            LOGGER.warning(f"No baseline for k={k}, skipping")
            continue
        
        r_baseline = baseline['mean_r'].iloc[0]
        l2_baseline = baseline['mean_l2'].iloc[0]
        
        # Get noisy conditions (exclude baseline and NaN values)
        noisy_data = k_data[(k_data['noise_level'] > 0) & (k_data['mean_r'].notna())]
        
        if len(noisy_data) == 0:
            LOGGER.warning(f"No noisy data for k={k}, skipping")
            continue
        
        noise_levels = noisy_data['noise_level'].values
        r_values = noisy_data['mean_r'].values
        
        # Compute sensitivity: Δr(σ) = r(σ) - r(0)
        delta_r = r_values - r_baseline
        
        # Compute Lipschitz constant: L = max |r(σ) - r(0)| / σ
        abs_sensitivity = np.abs(delta_r) / noise_levels
        lipschitz_constant = np.max(abs_sensitivity) if len(abs_sensitivity) > 0 else np.nan
        
        # Compute mean sensitivity
        mean_sensitivity = np.mean(abs_sensitivity) if len(abs_sensitivity) > 0 else np.nan
        
        analysis_results.append({
            "k": k,
            "r_baseline": r_baseline,
            "l2_baseline": l2_baseline,
            "lipschitz_constant": lipschitz_constant,
            "mean_sensitivity": mean_sensitivity,
            "max_delta_r": np.max(np.abs(delta_r)) if len(delta_r) > 0 else np.nan,
            "n_noise_levels": len(noisy_data),
        })
    
    analysis_df = pd.DataFrame(analysis_results)
    
    # Save analysis
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "noise_sensitivity_analysis.csv"
    analysis_df.to_csv(analysis_path, index=False)
    LOGGER.info(f"Saved noise sensitivity analysis to {analysis_path}")
    
    return {
        "analysis": analysis_df,
        "summary": {
            "mean_lipschitz": analysis_df['lipschitz_constant'].mean() if len(analysis_df) > 0 else np.nan,
            "max_lipschitz": analysis_df['lipschitz_constant'].max() if len(analysis_df) > 0 else np.nan,
        }
    }


if __name__ == "__main__":
    # Placeholder main - would integrate with LSFT evaluation
    pass


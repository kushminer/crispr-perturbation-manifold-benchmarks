#!/usr/bin/env python3
"""
Curvature Sweep: Local Neighborhood Size Sensitivity Analysis

This module implements Epic 1 of the Manifold Law Diagnostic Suite.

Goal: Quantify how Pearson r changes as local neighborhood size increases
(k ∈ {3, 5, 10, 20, 50, 100}). Detect the classic "U-shaped" curve indicating
smooth manifolds (best at small k; degradation as k grows).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from goal_2_baselines.baseline_types import BaselineType
from goal_3_prediction.lsft.lsft_k_sweep import evaluate_lsft_with_k_list

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def compute_curvature_index(
    k_values: np.ndarray,
    r_values: np.ndarray,
) -> Dict[str, float]:
    """
    Compute curvature index from r vs k curve.
    
    Estimates effective manifold curvature: C = d²r/dk²
    Detects U-shaped curve (negative curvature at small k, positive at large k).
    
    Args:
        k_values: Array of k values (neighborhood sizes)
        r_values: Array of Pearson r values
        
    Returns:
        Dictionary with curvature metrics
    """
    # Sort by k
    sort_idx = np.argsort(k_values)
    k_sorted = k_values[sort_idx]
    r_sorted = r_values[sort_idx]
    
    # Compute first derivative (dr/dk) using finite differences
    if len(k_sorted) < 2:
        return {"curvature_index": np.nan, "is_u_shaped": False}
    
    dr_dk = np.diff(r_sorted) / np.diff(k_sorted)
    
    # Compute second derivative (d²r/dk²)
    if len(dr_dk) < 2:
        return {"curvature_index": np.nan, "is_u_shaped": False}
    
    d2r_dk2 = np.diff(dr_dk) / np.diff(k_sorted[1:])
    
    # Curvature index: mean of second derivative
    curvature_index = np.mean(d2r_dk2)
    
    # Detect U-shape: negative at small k, positive at large k
    # Or: peak at intermediate k (first derivative crosses zero)
    has_zero_crossing = np.any(np.diff(np.sign(dr_dk))) if len(dr_dk) > 1 else False
    
    # Alternative: check if there's a clear peak (r increases then decreases)
    if len(r_sorted) >= 3:
        # Find peak: max r in the middle range (not at endpoints)
        mid_start = len(r_sorted) // 4
        mid_end = 3 * len(r_sorted) // 4
        mid_r = r_sorted[mid_start:mid_end]
        if len(mid_r) > 0:
            peak_idx = mid_start + np.argmax(mid_r)
            is_u_shaped = (peak_idx > 0) and (peak_idx < len(r_sorted) - 1)
        else:
            is_u_shaped = False
    else:
        is_u_shaped = False
    
    return {
        "curvature_index": float(curvature_index),
        "is_u_shaped": bool(is_u_shaped),
        "mean_d2r_dk2": float(np.mean(d2r_dk2)) if len(d2r_dk2) > 0 else np.nan,
        "min_k_r": float(r_sorted[0]),
        "max_k_r": float(r_sorted[-1]),
        "peak_k": float(k_sorted[np.argmax(r_sorted)]) if len(r_sorted) > 0 else np.nan,
        "peak_r": float(np.max(r_sorted)) if len(r_sorted) > 0 else np.nan,
    }


def run_curvature_sweep(
    adata_path: Path,
    split_config_path: Path,
    baseline_type: BaselineType,
    dataset_name: str,
    output_dir: Path,
    k_list: List[int] = [3, 5, 10, 20, 50, 100],
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
) -> pd.DataFrame:
    """
    Run curvature sweep: evaluate LSFT for multiple k values.
    
    Args:
        adata_path: Path to adata file
        split_config_path: Path to split config JSON
        baseline_type: Baseline type
        dataset_name: Dataset name
        output_dir: Output directory
        k_list: List of exact k values (neighborhood sizes) to test
        pca_dim: PCA dimension
        ridge_penalty: Ridge penalty
        seed: Random seed
        
    Returns:
        DataFrame with results for each test perturbation and k value
    """
    LOGGER.info(f"Running curvature sweep for {dataset_name}")
    LOGGER.info(f"Baseline: {baseline_type.value}")
    LOGGER.info(f"K values: {k_list}")
    
    # Run LSFT with k_list
    results_df = evaluate_lsft_with_k_list(
        adata_path=adata_path,
        split_config_path=split_config_path,
        baseline_type=baseline_type,
        dataset_name=dataset_name,
        output_dir=output_dir,
        k_list=k_list,
        pca_dim=pca_dim,
        ridge_penalty=ridge_penalty,
        seed=seed,
    )
    
    return results_df


def analyze_curvature_results(
    results_df: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    baseline_type: str,
) -> Dict:
    """
    Analyze curvature sweep results and generate plots.
    
    Args:
        results_df: DataFrame with columns: test_perturbation, k, pearson_r, l2, etc.
        output_dir: Output directory for plots
        dataset_name: Dataset name
        baseline_type: Baseline type name
        
    Returns:
        Dictionary with summary statistics
    """
    LOGGER.info("Analyzing curvature sweep results...")
    
    # Check if results_df is empty or missing required columns
    if len(results_df) == 0:
        LOGGER.error("Results DataFrame is empty - cannot analyze")
        raise ValueError("Results DataFrame is empty")
    
    if "k" not in results_df.columns:
        LOGGER.error(f"Results DataFrame missing 'k' column. Columns: {results_df.columns.tolist()}")
        raise ValueError(f"Results DataFrame must have 'k' column. Got columns: {results_df.columns.tolist()}")
    
    if "pearson_r" not in results_df.columns:
        LOGGER.error(f"Results DataFrame missing 'pearson_r' column. Columns: {results_df.columns.tolist()}")
        raise ValueError(f"Results DataFrame must have 'pearson_r' column. Got columns: {results_df.columns.tolist()}")
    
    # Aggregate across test perturbations
    summary = results_df.groupby("k").agg({
        "pearson_r": ["mean", "std", "count"],
        "l2": ["mean", "std"],
    }).reset_index()
    
    summary.columns = ["k", "mean_r", "std_r", "n_perturbations", "mean_l2", "std_l2"]
    
    # Compute curvature index
    k_values = summary["k"].values
    r_values = summary["mean_r"].values
    
    curvature_metrics = compute_curvature_index(k_values, r_values)
    
    # Save summary
    summary_path = output_dir / f"curvature_sweep_summary_{dataset_name}_{baseline_type}.csv"
    summary.to_csv(summary_path, index=False)
    LOGGER.info(f"Saved summary to {summary_path}")
    
    # Generate plots
    plot_path = output_dir / f"curvature_sweep_r_vs_k_{dataset_name}_{baseline_type}.png"
    _plot_curvature_sweep(summary, plot_path, dataset_name, baseline_type)
    
    plot_path_l2 = output_dir / f"curvature_sweep_l2_vs_k_{dataset_name}_{baseline_type}.png"
    _plot_curvature_sweep_l2(summary, plot_path_l2, dataset_name, baseline_type)
    
    return {
        "summary": summary,
        "curvature_metrics": curvature_metrics,
        "summary_path": summary_path,
        "plot_path": plot_path,
    }


def _plot_curvature_sweep(
    summary: pd.DataFrame,
    output_path: Path,
    dataset_name: str,
    baseline_type: str,
):
    """Plot Pearson r vs k."""
    plt.figure(figsize=(10, 6))
    
    k = summary["k"].values
    mean_r = summary["mean_r"].values
    std_r = summary["std_r"].values
    
    plt.errorbar(k, mean_r, yerr=std_r, fmt='o-', linewidth=2, markersize=8, capsize=5)
    plt.xlabel('Neighborhood Size (k)', fontsize=12)
    plt.ylabel('Pearson r (mean ± std)', fontsize=12)
    plt.title(f'Curvature Sweep: r vs k\n{dataset_name} - {baseline_type}', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    # Add annotation for peak
    peak_idx = np.argmax(mean_r)
    plt.annotate(f'Peak: k={k[peak_idx]}, r={mean_r[peak_idx]:.3f}',
                xy=(k[peak_idx], mean_r[peak_idx]),
                xytext=(k[peak_idx]*1.5, mean_r[peak_idx]+0.05),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    LOGGER.info(f"Saved plot to {output_path}")


def _plot_curvature_sweep_l2(
    summary: pd.DataFrame,
    output_path: Path,
    dataset_name: str,
    baseline_type: str,
):
    """Plot L2 vs k."""
    plt.figure(figsize=(10, 6))
    
    k = summary["k"].values
    mean_l2 = summary["mean_l2"].values
    std_l2 = summary["std_l2"].values
    
    plt.errorbar(k, mean_l2, yerr=std_l2, fmt='s--', linewidth=2, markersize=8, capsize=5, color='orange')
    plt.xlabel('Neighborhood Size (k)', fontsize=12)
    plt.ylabel('L2 Distance (mean ± std)', fontsize=12)
    plt.title(f'Curvature Sweep: L2 vs k\n{dataset_name} - {baseline_type}', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    LOGGER.info(f"Saved plot to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Curvature Sweep Analysis")
    parser.add_argument("--adata_path", type=Path, required=True)
    parser.add_argument("--split_config", type=Path, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--baseline_type", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--k_list", type=int, nargs="+", default=[3, 5, 10, 20, 50, 100])
    parser.add_argument("--pca_dim", type=int, default=10)
    parser.add_argument("--ridge_penalty", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    
    args = parser.parse_args()
    
    baseline_type = BaselineType(args.baseline_type)
    
    results_df = run_curvature_sweep(
        adata_path=args.adata_path,
        split_config_path=args.split_config,
        baseline_type=baseline_type,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        k_list=args.k_list,
        pca_dim=args.pca_dim,
        ridge_penalty=args.ridge_penalty,
        seed=args.seed,
    )
    
    # Analyze results
    analysis = analyze_curvature_results(
        results_df=results_df,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        baseline_type=baseline_type.value,
    )
    
    print(f"\nCurvature Metrics:")
    print(analysis["curvature_metrics"])
    print(f"\nSummary saved to: {analysis['summary_path']}")
    print(f"Plot saved to: {analysis['plot_path']}")


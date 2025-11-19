"""
Enhanced visualizations with CI overlays for LSFT resampling results (Issue 10).

This module creates:
1. Beeswarm plots with per-perturbation points + mean + CI bars
2. Hardness curves with regression line + bootstrapped CI bands
3. Baseline comparison plots with delta distribution + significance markers
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def load_lsft_summary(summary_path: Path) -> Dict:
    """Load LSFT summary with CIs from JSON file."""
    with open(summary_path, "r") as f:
        return json.load(f)


def create_beeswarm_with_ci(
    results_df: pd.DataFrame,
    summary: Dict,
    output_path: Path,
    metric: str = "pearson_r",
    baseline_type: Optional[str] = None,
    top_pct: Optional[float] = None,
    figsize: tuple = (10, 6),
):
    """
    Create beeswarm plot with per-perturbation points and mean + CI bar.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Standardized LSFT results DataFrame
    summary : Dict
        Summary statistics with CIs (from compute_lsft_summary_with_cis)
    metric : str
        Metric to plot: "pearson_r" or "l2"
    output_path : Path
        Output file path
    baseline_type : str or None
        If provided, filter to this baseline
    top_pct : float or None
        If provided, filter to this top_pct
    figsize : tuple
        Figure size
    """
    # Filter data
    plot_df = results_df.copy()
    if baseline_type:
        plot_df = plot_df[plot_df["baseline_type"] == baseline_type]
    if top_pct is not None:
        plot_df = plot_df[plot_df["top_pct"] == top_pct]
    
    if len(plot_df) == 0:
        LOGGER.warning(f"No data to plot (baseline={baseline_type}, top_pct={top_pct})")
        return
    
    # Get metric values
    values = plot_df[metric].dropna().values
    
    # Get summary statistics
    key = f"{baseline_type}_top{top_pct*100:.0f}pct" if baseline_type and top_pct is not None else None
    if key and key in summary:
        mean_val = summary[key][metric]["mean"]
        ci_lower = summary[key][metric]["ci_lower"]
        ci_upper = summary[key][metric]["ci_upper"]
    else:
        # Fallback to computing from data
        mean_val = np.mean(values)
        ci_lower = mean_val
        ci_upper = mean_val
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Beeswarm plot (jittered points)
    y_pos = 0
    x_jittered = np.random.normal(y_pos, 0.05, size=len(values))
    ax.scatter(x_jittered, values, alpha=0.5, s=20, color="gray")
    
    # Mean line
    ax.axhline(mean_val, color="red", linestyle="--", linewidth=2, label="Mean")
    
    # CI bar
    ax.errorbar(
        y_pos,
        mean_val,
        yerr=[[mean_val - ci_lower], [ci_upper - mean_val]],
        fmt="o",
        color="red",
        markersize=10,
        capsize=10,
        capthick=2,
        label="95% CI",
    )
    
    # Labels
    title = f"Beeswarm Plot: {metric}"
    if baseline_type:
        title += f" ({baseline_type})"
    if top_pct is not None:
        title += f" (Top {top_pct*100:.0f}%)"
    
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_xlabel("")
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"Saved beeswarm plot to {output_path}")


def create_hardness_curve_with_ci(
    results_df: pd.DataFrame,
    regression_results: pd.DataFrame,
    output_path: Path,
    metric: str = "pearson_r",
    hardness_metric: str = "hardness",
    baseline_type: Optional[str] = None,
    top_pct: Optional[float] = None,
    figsize: tuple = (10, 6),
):
    """
    Create hardness-performance curve with regression line + bootstrapped CI bands.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Standardized LSFT results DataFrame
    regression_results : pd.DataFrame
        Hardness regression results with CIs (from bootstrap_hardness_regression)
    metric : str
        Performance metric: "pearson_r" or "l2"
    hardness_metric : str
        Hardness metric: "hardness" or "embedding_similarity"
    output_path : Path
        Output file path
    baseline_type : str or None
        If provided, filter to this baseline
    top_pct : float or None
        If provided, filter to this top_pct
    figsize : tuple
        Figure size
    """
    # Filter data
    plot_df = results_df.copy()
    if baseline_type:
        plot_df = plot_df[plot_df["baseline_type"] == baseline_type]
    if top_pct is not None:
        plot_df = plot_df[plot_df["top_pct"] == top_pct]
    
    if len(plot_df) == 0:
        LOGGER.warning(f"No data to plot (baseline={baseline_type}, top_pct={top_pct})")
        return
    
    # Get regression results
    reg_filtered = regression_results.copy()
    if baseline_type:
        reg_filtered = reg_filtered[reg_filtered["baseline_type"] == baseline_type]
    if top_pct is not None:
        reg_filtered = reg_filtered[reg_filtered["top_pct"] == top_pct]
    
    if len(reg_filtered) == 0:
        LOGGER.warning("No regression results found, plotting without CI bands")
        has_ci = False
    else:
        has_ci = True
        reg_row = reg_filtered.iloc[0]
        slope = reg_row["slope"]
        intercept = reg_row["intercept"]
        slope_ci_lower = reg_row.get("slope_ci_lower", slope)
        slope_ci_upper = reg_row.get("slope_ci_upper", slope)
    
    # Get data
    hardness = plot_df[hardness_metric].values
    performance = plot_df[metric].values
    
    # Remove NaN pairs
    valid_mask = ~(np.isnan(hardness) | np.isnan(performance))
    hardness_clean = hardness[valid_mask]
    performance_clean = performance[valid_mask]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(hardness_clean, performance_clean, alpha=0.6, s=50, color="gray")
    
    # Regression line
    if has_ci:
        x_min, x_max = hardness_clean.min(), hardness_clean.max()
        x_range = np.linspace(x_min, x_max, 100)
        
        # Main regression line
        y_line = slope * x_range + intercept
        ax.plot(x_range, y_line, color="red", linewidth=2, label="Regression")
        
        # CI bands (using slope CIs - simplified)
        # In practice, you'd bootstrap the full regression for each x
        # For now, we show the uncertainty in slope
        y_upper = slope_ci_upper * x_range + intercept
        y_lower = slope_ci_lower * x_range + intercept
        ax.fill_between(
            x_range,
            y_lower,
            y_upper,
            alpha=0.2,
            color="red",
            label="95% CI Band",
        )
    
    # Labels
    title = f"Hardness-Performance: {metric} vs {hardness_metric}"
    if baseline_type:
        title += f" ({baseline_type})"
    if top_pct is not None:
        title += f" (Top {top_pct*100:.0f}%)"
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(hardness_metric, fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"Saved hardness curve to {output_path}")


def create_baseline_comparison_with_significance(
    comparison_df: pd.DataFrame,
    output_path: Path,
    metric: str = "pearson_r",
    top_pct: Optional[float] = None,
    figsize: tuple = (12, 8),
):
    """
    Create baseline comparison plot with delta distribution + significance markers.
    
    Parameters
    ----------
    comparison_df : pd.DataFrame
        Baseline comparison results (from compare_all_baseline_pairs)
    metric : str
        Metric to plot: "pearson_r" or "l2"
    output_path : Path
        Output file path
    top_pct : float or None
        If provided, filter to this top_pct
    figsize : tuple
        Figure size
    """
    # Filter data
    plot_df = comparison_df.copy()
    plot_df = plot_df[plot_df["metric"] == metric]
    if top_pct is not None:
        plot_df = plot_df[plot_df["top_pct"] == top_pct]
    
    if len(plot_df) == 0:
        LOGGER.warning(f"No comparison data to plot (metric={metric}, top_pct={top_pct})")
        return
    
    # Create comparison pairs
    pairs = []
    for _, row in plot_df.iterrows():
        pairs.append({
            "baseline1": row["baseline1"],
            "baseline2": row["baseline2"],
            "mean_delta": row["mean_delta"],
            "ci_lower": row["delta_ci_lower"],
            "ci_upper": row["delta_ci_upper"],
            "p_value": row["p_value"],
        })
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot deltas with CIs
    y_pos = np.arange(len(pairs))
    deltas = [p["mean_delta"] for p in pairs]
    ci_lowers = [p["mean_delta"] - p["ci_lower"] for p in pairs]
    ci_uppers = [p["ci_upper"] - p["mean_delta"] for p in pairs]
    p_values = [p["p_value"] for p in pairs]
    
    # Error bars
    ax.errorbar(
        deltas,
        y_pos,
        xerr=[ci_lowers, ci_uppers],
        fmt="o",
        color="blue",
        markersize=8,
        capsize=5,
        capthick=2,
    )
    
    # Significance markers
    for i, p_val in enumerate(p_values):
        if p_val < 0.001:
            marker = "***"
        elif p_val < 0.01:
            marker = "**"
        elif p_val < 0.05:
            marker = "*"
        else:
            marker = "ns"
        
        # Place marker above CI
        x_pos = deltas[i] + ci_uppers[i] + 0.01
        ax.text(x_pos, i, marker, fontsize=10, va="center")
    
    # Zero line
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    
    # Labels
    pair_labels = [f"{p['baseline1']} vs {p['baseline2']}" for p in pairs]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pair_labels)
    ax.set_xlabel(f"Mean Delta ({metric})", fontsize=12)
    ax.set_ylabel("Baseline Pair", fontsize=12)
    
    title = f"Baseline Comparison: {metric}"
    if top_pct is not None:
        title += f" (Top {top_pct*100:.0f}%)"
    ax.set_title(title, fontsize=14)
    
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"Saved baseline comparison plot to {output_path}")


def create_all_lsft_visualizations_with_ci(
    results_df: pd.DataFrame,
    summary_path: Path,
    output_dir: Path,
    regression_results_path: Optional[Path] = None,
    comparison_results_path: Optional[Path] = None,
    dataset_name: str = "adamson",
    baseline_type: Optional[str] = None,
    top_pcts: List[float] = [0.01, 0.05, 0.10],
):
    """
    Create all enhanced visualizations with CI overlays.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Standardized LSFT results DataFrame
    summary_path : Path
        Path to summary JSON file (with CIs)
    regression_results_path : Path or None
        Path to hardness regression results CSV (optional)
    comparison_results_path : Path or None
        Path to baseline comparison results CSV (optional)
    output_dir : Path
        Output directory
    dataset_name : str
        Dataset name
    baseline_type : str or None
        If provided, filter to this baseline
    top_pcts : List[float]
        Top percentages to plot
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load summary
    summary = load_lsft_summary(summary_path)
    
    # Load regression results if available
    regression_df = None
    if regression_results_path and regression_results_path.exists():
        regression_df = pd.read_csv(regression_results_path)
    
    # Load comparison results if available
    comparison_df = None
    if comparison_results_path and comparison_results_path.exists():
        comparison_df = pd.read_csv(comparison_results_path)
    
    # Create visualizations for each top_pct
    for top_pct in top_pcts:
        # Beeswarm plots
        for metric in ["pearson_r", "l2"]:
            output_path = output_dir / f"beeswarm_{dataset_name}_{metric}_top{top_pct*100:.0f}pct.png"
            create_beeswarm_with_ci(
                results_df=results_df,
                summary=summary,
                metric=metric,
                output_path=output_path,
                baseline_type=baseline_type,
                top_pct=top_pct,
            )
        
        # Hardness curves
        if regression_df is not None:
            for metric in ["pearson_r", "l2"]:
                output_path = output_dir / f"hardness_{dataset_name}_{metric}_top{top_pct*100:.0f}pct.png"
                create_hardness_curve_with_ci(
                    results_df=results_df,
                    regression_results=regression_df,
                    metric=metric,
                    hardness_metric="hardness",
                    output_path=output_path,
                    baseline_type=baseline_type,
                    top_pct=top_pct,
                )
    
    # Baseline comparison plot
    if comparison_df is not None:
        for metric in ["pearson_r", "l2"]:
            output_path = output_dir / f"baseline_comparison_{dataset_name}_{metric}.png"
            create_baseline_comparison_with_significance(
                comparison_df=comparison_df,
                metric=metric,
                output_path=output_path,
                top_pct=None,  # Aggregate across all top_pcts
            )
    
    LOGGER.info(f"All visualizations saved to {output_dir}")


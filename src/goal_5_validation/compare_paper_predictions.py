#!/usr/bin/env python3
"""
Compare our baseline runner outputs to paper's single_perturbation_results_predictions.Rds.

This script loads the paper's RDS file and compares it to our baseline predictions,
producing a comprehensive statistical report.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr

# Try to import rpy2 for reading RDS files
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    RDS_AVAILABLE = True
except ImportError:
    RDS_AVAILABLE = False
    logging.warning("rpy2 not available. Will try alternative methods to read RDS files.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


def load_rds_file(rds_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load RDS file and extract predictions.
    
    The RDS file contains a list of baseline runs, each with:
    - id: job ID
    - name: baseline name (e.g., "adamson-1-scgpt")
    - perturbation: list of perturbation names
    - prediction: list of named vectors (gene predictions for each perturbation)
    
    Args:
        rds_path: Path to .Rds file
    
    Returns:
        Dictionary mapping baseline_name -> {perturbation_name -> prediction_array}
    """
    if not RDS_AVAILABLE:
        raise ImportError("rpy2 is required to read RDS files. Install with: pip install rpy2")
    
    LOGGER.info(f"Loading RDS file: {rds_path}")
    
    # Read RDS file
    base = importr("base")
    rds_data = base.readRDS(str(rds_path))
    
    # Convert to Python list
    r_list = list(rds_data)
    LOGGER.info(f"Loaded RDS: {len(r_list)} baseline runs")
    
    # Extract predictions
    all_predictions = {}
    
    for run in r_list:
        # Extract run information
        run_name = str(run.rx2("name")[0])  # e.g., "adamson-1-scgpt"
        perturbations = list(run.rx2("perturbation"))
        predictions = run.rx2("prediction")
        
        # Parse baseline name from run_name (format: "dataset-split-baseline")
        # e.g., "adamson-1-scgpt" -> baseline = "scgpt"
        parts = run_name.split("-")
        if len(parts) >= 3:
            baseline_name = "-".join(parts[2:])  # Handle multi-part baseline names
        else:
            baseline_name = run_name
        
        # Map to our baseline naming convention
        baseline_map = {
            "scgpt": "lpm_scgptGeneEmb",
            "scfoundation": "lpm_scFoundationGeneEmb",
            "selftrained": "lpm_selftrained",
            "gears": "lpm_gearsPertEmb",
            "k562": "lpm_k562PertEmb",
            "rpe1": "lpm_rpe1PertEmb",
            "randompert": "lpm_randomPertEmb",
            "randomgene": "lpm_randomGeneEmb",
        }
        
        # Try to match baseline name
        matched_baseline = None
        for key, value in baseline_map.items():
            if key in baseline_name.lower():
                matched_baseline = value
                break
        
        if matched_baseline is None:
            LOGGER.warning(f"Could not match baseline name: {baseline_name}, using as-is")
            matched_baseline = baseline_name
        
        # Extract predictions for each perturbation
        if matched_baseline not in all_predictions:
            all_predictions[matched_baseline] = {}
        
        # Convert R predictions to numpy arrays
        # predictions is a list where each element corresponds to a perturbation
        for i, pert_name in enumerate(perturbations):
            try:
                # Access prediction by index (predictions is a list)
                pert_pred = predictions[i]
                
                # Check if it's NULL
                if str(pert_pred) == "NULL" or pert_pred is None:
                    LOGGER.warning(f"Skipping NULL prediction for {pert_name}")
                    continue
                
                # Convert named vector to numpy array
                # Handle both named vectors and regular vectors
                if hasattr(pert_pred, "names"):
                    # Named vector - extract values
                    pred_array = np.array(list(pert_pred))
                else:
                    # Regular vector
                    pred_array = np.array(list(pert_pred))
                
                all_predictions[matched_baseline][pert_name] = pred_array
            except Exception as e:
                LOGGER.warning(f"Error extracting prediction for {pert_name}: {e}")
                continue
    
    LOGGER.info(f"Extracted predictions for {len(all_predictions)} baselines")
    for baseline, perts in all_predictions.items():
        LOGGER.info(f"  {baseline}: {len(perts)} perturbations")
    
    return all_predictions


def load_our_predictions(
    predictions_dir: Optional[Path] = None,
    adata_path: Optional[Path] = None,
    split_config_path: Optional[Path] = None,
    baseline_type: Optional[str] = None,
    output_dir: Optional[Path] = None,
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Load our baseline predictions from saved files or compute them.
    
    Args:
        predictions_dir: Directory containing saved predictions (e.g., results/goal_2_baselines/adamson_reproduced/{baseline_type}/)
        adata_path: Path to perturb_processed.h5ad (if predictions not saved)
        split_config_path: Path to split config JSON (if predictions not saved)
        baseline_type: Baseline type name (if predictions not saved)
        output_dir: Directory where predictions might be saved (if predictions not saved)
        pca_dim: PCA dimension (if computing)
        ridge_penalty: Ridge penalty (if computing)
        seed: Random seed (if computing)
    
    Returns:
        Dictionary mapping perturbation names to prediction arrays
    """
    import json
    
    # Try to load from saved files first
    if predictions_dir is not None and predictions_dir.exists():
        predictions_path = predictions_dir / "predictions.json"
        gene_names_path = predictions_dir / "gene_names.json"
        
        if predictions_path.exists() and gene_names_path.exists():
            LOGGER.info(f"Loading saved predictions from {predictions_dir}")
            with open(predictions_path) as f:
                predictions_dict = json.load(f)
            
            # Convert to numpy arrays
            predictions = {pert: np.array(pred) for pert, pred in predictions_dict.items()}
            LOGGER.info(f"Loaded {len(predictions)} saved predictions")
            return predictions
    
    # Fall back to computing predictions
    if adata_path is None or split_config_path is None or baseline_type is None:
        raise ValueError("Either predictions_dir must exist, or adata_path, split_config_path, and baseline_type must be provided")
    
    LOGGER.info(f"Computing predictions for baseline: {baseline_type}")
    
    from goal_2_baselines.baseline_runner import run_single_baseline, compute_pseudobulk_expression_changes
    from goal_2_baselines.baseline_types import BaselineType, get_baseline_config
    from goal_2_baselines.split_logic import load_split_config
    
    # Load data
    adata = ad.read_h5ad(adata_path)
    split_config = load_split_config(split_config_path)
    
    # Compute Y matrix
    Y_df, split_labels = compute_pseudobulk_expression_changes(adata, split_config, seed)
    
    # Split Y
    train_perts = split_labels.get("train", [])
    test_perts = split_labels.get("test", [])
    
    Y_train = Y_df[train_perts]
    Y_test = Y_df[test_perts] if test_perts else pd.DataFrame()
    
    # Get gene names
    gene_names = Y_df.index.tolist()
    
    # Get gene name mapping
    gene_name_mapping = None
    if "gene_name" in adata.var.columns:
        gene_name_mapping = dict(zip(adata.var_names, adata.var["gene_name"]))
    
    # Map baseline name to BaselineType
    baseline_map = {
        "lpm_selftrained": BaselineType.SELFTRAINED,
        "lpm_randomPertEmb": BaselineType.RANDOM_PERT_EMB,
        "lpm_randomGeneEmb": BaselineType.RANDOM_GENE_EMB,
        "lpm_scgptGeneEmb": BaselineType.SCGPT_GENE_EMB,
        "lpm_scFoundationGeneEmb": BaselineType.SCFOUNDATION_GENE_EMB,
        "lpm_gearsPertEmb": BaselineType.GEARS_PERT_EMB,
        "lpm_k562PertEmb": BaselineType.K562_PERT_EMB,
        "lpm_rpe1PertEmb": BaselineType.RPE1_PERT_EMB,
    }
    
    if baseline_type not in baseline_map:
        raise ValueError(f"Unknown baseline type: {baseline_type}")
    
    baseline_type_enum = baseline_map[baseline_type]
    
    # Get config
    config = get_baseline_config(
        baseline_type_enum,
        pca_dim=pca_dim,
        ridge_penalty=ridge_penalty,
        seed=seed,
    )
    
    # Run baseline
    result = run_single_baseline(
        Y_train=Y_train,
        Y_test=Y_test,
        config=config,
        gene_names=gene_names,
        gene_name_mapping=gene_name_mapping,
        adata_path=adata_path,
        split_config=split_config,
    )
    
    # Extract predictions
    predictions = {}
    if "predictions" in result and result["predictions"].size > 0:
        Y_pred = result["predictions"]  # genes × test_perturbations
        test_pert_names = Y_test.columns.tolist() if not Y_test.empty else []
        
        for i, pert_name in enumerate(test_pert_names):
            clean_pert_name = pert_name.replace("+ctrl", "")
            predictions[clean_pert_name] = Y_pred[:, i]
    
    LOGGER.info(f"Computed {len(predictions)} predictions")
    
    return predictions


def compare_predictions(
    paper_predictions: Dict[str, np.ndarray],
    our_predictions: Dict[str, np.ndarray],
    gene_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compare paper predictions to our predictions.
    
    Args:
        paper_predictions: Dictionary mapping perturbation names to prediction arrays
        our_predictions: Dictionary mapping perturbation names to prediction arrays
        gene_names: Optional list of gene names for alignment
    
    Returns:
        DataFrame with comparison statistics
    """
    LOGGER.info("Comparing predictions")
    
    # Find common perturbations
    common_perts = set(paper_predictions.keys()) & set(our_predictions.keys())
    LOGGER.info(f"Found {len(common_perts)} common perturbations")
    
    if len(common_perts) == 0:
        raise ValueError("No common perturbations found between paper and our predictions")
    
    results = []
    
    for pert_name in common_perts:
        paper_pred = paper_predictions[pert_name]
        our_pred = our_predictions[pert_name]
        
        # Ensure same length
        min_len = min(len(paper_pred), len(our_pred))
        paper_pred = paper_pred[:min_len]
        our_pred = our_pred[:min_len]
        
        # Remove NaN values
        valid_mask = ~(np.isnan(paper_pred) | np.isnan(our_pred))
        if valid_mask.sum() < 2:
            continue
        
        paper_clean = paper_pred[valid_mask]
        our_clean = our_pred[valid_mask]
        
        # Compute statistics
        pearson_r, pearson_p = pearsonr(paper_clean, our_clean)
        
        try:
            spearman_rho, spearman_p = spearmanr(paper_clean, our_clean)
        except (ValueError, stats.ConstantInputWarning):
            spearman_rho = np.nan
            spearman_p = np.nan
        
        # Mean squared error
        mse = np.mean((paper_clean - our_clean) ** 2)
        rmse = np.sqrt(mse)
        
        # Mean absolute error
        mae = np.mean(np.abs(paper_clean - our_clean))
        
        # R²
        ss_res = np.sum((paper_clean - our_clean) ** 2)
        ss_tot = np.sum((paper_clean - np.mean(paper_clean)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        
        # Mean and std
        paper_mean = np.mean(paper_clean)
        our_mean = np.mean(our_clean)
        paper_std = np.std(paper_clean)
        our_std = np.std(our_clean)
        
        results.append({
            "perturbation": pert_name,
            "n_genes": int(valid_mask.sum()),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(spearman_p),
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "paper_mean": float(paper_mean),
            "our_mean": float(our_mean),
            "paper_std": float(paper_std),
            "our_std": float(our_std),
            "mean_diff": float(our_mean - paper_mean),
            "std_diff": float(our_std - paper_std),
        })
    
    comparison_df = pd.DataFrame(results)
    LOGGER.info(f"Computed comparison statistics for {len(comparison_df)} perturbations")
    
    return comparison_df


def create_comparison_visualizations(
    comparison_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create visualization comparing paper and our predictions."""
    LOGGER.info("Creating comparison visualizations")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Pearson correlation distribution
    ax = axes[0, 0]
    ax.hist(comparison_df["pearson_r"], bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Pearson Correlation (r)")
    ax.set_ylabel("Frequency")
    ax.set_title("A. Pearson Correlation Distribution")
    ax.axvline(comparison_df["pearson_r"].mean(), color="red", linestyle="--", 
               label=f"Mean: {comparison_df['pearson_r'].mean():.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. R² distribution
    ax = axes[0, 1]
    ax.hist(comparison_df["r2"], bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("R²")
    ax.set_ylabel("Frequency")
    ax.set_title("B. R² Distribution")
    ax.axvline(comparison_df["r2"].mean(), color="red", linestyle="--",
               label=f"Mean: {comparison_df['r2'].mean():.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. RMSE distribution
    ax = axes[0, 2]
    ax.hist(comparison_df["rmse"], bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Frequency")
    ax.set_title("C. RMSE Distribution")
    ax.axvline(comparison_df["rmse"].mean(), color="red", linestyle="--",
               label=f"Mean: {comparison_df['rmse'].mean():.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Scatter: Paper mean vs Our mean
    ax = axes[1, 0]
    ax.scatter(comparison_df["paper_mean"], comparison_df["our_mean"], alpha=0.6)
    # Add diagonal line
    min_val = min(comparison_df["paper_mean"].min(), comparison_df["our_mean"].min())
    max_val = max(comparison_df["paper_mean"].max(), comparison_df["our_mean"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8, label="y=x")
    ax.set_xlabel("Paper Mean Prediction")
    ax.set_ylabel("Our Mean Prediction")
    ax.set_title("D. Mean Prediction Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Scatter: Paper std vs Our std
    ax = axes[1, 1]
    ax.scatter(comparison_df["paper_std"], comparison_df["our_std"], alpha=0.6)
    min_val = min(comparison_df["paper_std"].min(), comparison_df["our_std"].min())
    max_val = max(comparison_df["paper_std"].max(), comparison_df["our_std"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8, label="y=x")
    ax.set_xlabel("Paper Std Prediction")
    ax.set_ylabel("Our Std Prediction")
    ax.set_title("E. Std Prediction Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Correlation vs R²
    ax = axes[1, 2]
    ax.scatter(comparison_df["pearson_r"], comparison_df["r2"], alpha=0.6)
    ax.set_xlabel("Pearson Correlation (r)")
    ax.set_ylabel("R²")
    ax.set_title("F. Correlation vs R²")
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Paper vs Our Baseline Predictions: Statistical Comparison", 
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    LOGGER.info(f"Saved comparison visualizations to {output_path}")
    plt.close()


def generate_statistical_report(
    comparison_df: pd.DataFrame,
    baseline_type: str,
    dataset_name: str,
    output_path: Path,
) -> None:
    """Generate comprehensive statistical report."""
    LOGGER.info("Generating statistical report")
    
    report = []
    report.append("# Paper vs Our Baseline Predictions: Statistical Comparison Report\n\n")
    report.append(f"**Date:** 2025-11-17  \n")
    report.append(f"**Baseline:** {baseline_type}  \n")
    report.append(f"**Dataset:** {dataset_name}  \n\n")
    
    report.append("---\n\n")
    report.append("## Executive Summary\n\n")
    
    n_perturbations = len(comparison_df)
    mean_r = comparison_df["pearson_r"].mean()
    mean_r2 = comparison_df["r2"].mean()
    mean_rmse = comparison_df["rmse"].mean()
    mean_mae = comparison_df["mae"].mean()
    
    report.append(f"- **Number of perturbations compared:** {n_perturbations}\n")
    report.append(f"- **Mean Pearson correlation:** {mean_r:.4f}\n")
    report.append(f"- **Mean R²:** {mean_r2:.4f}\n")
    report.append(f"- **Mean RMSE:** {mean_rmse:.4f}\n")
    report.append(f"- **Mean MAE:** {mean_mae:.4f}\n\n")
    
    # Count high correlations
    high_corr = (comparison_df["pearson_r"] > 0.95).sum()
    medium_corr = ((comparison_df["pearson_r"] > 0.90) & (comparison_df["pearson_r"] <= 0.95)).sum()
    low_corr = (comparison_df["pearson_r"] <= 0.90).sum()
    
    report.append(f"- **High correlation (r > 0.95):** {high_corr} ({high_corr/n_perturbations*100:.1f}%)\n")
    report.append(f"- **Medium correlation (0.90 < r ≤ 0.95):** {medium_corr} ({medium_corr/n_perturbations*100:.1f}%)\n")
    report.append(f"- **Low correlation (r ≤ 0.90):** {low_corr} ({low_corr/n_perturbations*100:.1f}%)\n\n")
    
    report.append("### Key Finding\n\n")
    if mean_r > 0.95:
        report.append(f"**Excellent agreement** between paper and our predictions (mean r = {mean_r:.4f}). ")
        report.append("Our implementation closely matches the paper's results.\n\n")
    elif mean_r > 0.90:
        report.append(f"**Good agreement** between paper and our predictions (mean r = {mean_r:.4f}). ")
        report.append("Our implementation shows strong similarity to the paper's results.\n\n")
    else:
        report.append(f"**Moderate agreement** between paper and our predictions (mean r = {mean_r:.4f}). ")
        report.append("Some discrepancies exist that may require investigation.\n\n")
    
    report.append("---\n\n")
    report.append("## 1. Descriptive Statistics\n\n")
    
    report.append("### 1.1 Correlation Statistics\n\n")
    report.append("| Statistic | Pearson r | Spearman ρ | R² |\n")
    report.append("|-----------|-----------|------------|----|\n")
    report.append(f"| Mean | {comparison_df['pearson_r'].mean():.4f} | {comparison_df['spearman_rho'].mean():.4f} | {comparison_df['r2'].mean():.4f} |\n")
    report.append(f"| Std | {comparison_df['pearson_r'].std():.4f} | {comparison_df['spearman_rho'].std():.4f} | {comparison_df['r2'].std():.4f} |\n")
    report.append(f"| Min | {comparison_df['pearson_r'].min():.4f} | {comparison_df['spearman_rho'].min():.4f} | {comparison_df['r2'].min():.4f} |\n")
    report.append(f"| Max | {comparison_df['pearson_r'].max():.4f} | {comparison_df['spearman_rho'].max():.4f} | {comparison_df['r2'].max():.4f} |\n\n")
    
    report.append("### 1.2 Error Statistics\n\n")
    report.append("| Statistic | RMSE | MAE | MSE |\n")
    report.append("|----------|------|-----|-----|\n")
    report.append(f"| Mean | {comparison_df['rmse'].mean():.4f} | {comparison_df['mae'].mean():.4f} | {comparison_df['mse'].mean():.4f} |\n")
    report.append(f"| Std | {comparison_df['rmse'].std():.4f} | {comparison_df['mae'].std():.4f} | {comparison_df['mse'].std():.4f} |\n")
    report.append(f"| Min | {comparison_df['rmse'].min():.4f} | {comparison_df['mae'].min():.4f} | {comparison_df['mse'].min():.4f} |\n")
    report.append(f"| Max | {comparison_df['rmse'].max():.4f} | {comparison_df['mae'].max():.4f} | {comparison_df['mse'].max():.4f} |\n\n")
    
    report.append("### 1.3 Prediction Statistics\n\n")
    report.append("| Statistic | Paper Mean | Our Mean | Paper Std | Our Std |\n")
    report.append("|----------|------------|----------|-----------|----------|\n")
    report.append(f"| Mean | {comparison_df['paper_mean'].mean():.4f} | {comparison_df['our_mean'].mean():.4f} | "
                 f"{comparison_df['paper_std'].mean():.4f} | {comparison_df['our_std'].mean():.4f} |\n")
    report.append(f"| Mean Difference | - | {comparison_df['mean_diff'].mean():.4f} | - | {comparison_df['std_diff'].mean():.4f} |\n\n")
    
    report.append("---\n\n")
    report.append("## 2. Per-Perturbation Analysis\n\n")
    
    # Top and bottom correlations
    top_corr = comparison_df.nlargest(5, "pearson_r")
    bottom_corr = comparison_df.nsmallest(5, "pearson_r")
    
    report.append("### 2.1 Highest Correlations\n\n")
    report.append("| Perturbation | Pearson r | R² | RMSE |\n")
    report.append("|-------------|-----------|----|------|\n")
    for _, row in top_corr.iterrows():
        report.append(f"| {row['perturbation']} | {row['pearson_r']:.4f} | {row['r2']:.4f} | {row['rmse']:.4f} |\n")
    
    report.append("\n### 2.2 Lowest Correlations\n\n")
    report.append("| Perturbation | Pearson r | R² | RMSE |\n")
    report.append("|-------------|-----------|----|------|\n")
    for _, row in bottom_corr.iterrows():
        report.append(f"| {row['perturbation']} | {row['pearson_r']:.4f} | {row['r2']:.4f} | {row['rmse']:.4f} |\n")
    
    report.append("\n---\n\n")
    report.append("## 3. Statistical Tests\n\n")
    
    # Test if mean correlation is significantly different from 1.0
    t_stat, p_value = stats.ttest_1samp(comparison_df["pearson_r"], 1.0)
    report.append(f"### 3.1 Test: Mean Correlation = 1.0\n\n")
    report.append(f"- **t-statistic:** {t_stat:.4f}\n")
    report.append(f"- **p-value:** {p_value:.4e}\n")
    if p_value < 0.05:
        report.append(f"- **Conclusion:** Mean correlation is significantly different from 1.0 (p < 0.05)\n\n")
    else:
        report.append(f"- **Conclusion:** Mean correlation is not significantly different from 1.0 (p ≥ 0.05)\n\n")
    
    # Test if mean difference is significantly different from 0
    t_stat, p_value = stats.ttest_1samp(comparison_df["mean_diff"], 0.0)
    report.append(f"### 3.2 Test: Mean Prediction Difference = 0\n\n")
    report.append(f"- **t-statistic:** {t_stat:.4f}\n")
    report.append(f"- **p-value:** {p_value:.4e}\n")
    if p_value < 0.05:
        report.append(f"- **Conclusion:** Mean prediction difference is significantly different from 0 (p < 0.05)\n\n")
    else:
        report.append(f"- **Conclusion:** Mean prediction difference is not significantly different from 0 (p ≥ 0.05)\n\n")
    
    report.append("---\n\n")
    report.append("## 4. Conclusions\n\n")
    
    if mean_r > 0.95:
        report.append("1. **Excellent agreement:** Our predictions show excellent agreement with the paper's predictions, ")
        report.append("indicating successful reproduction of the baseline methodology.\n\n")
    elif mean_r > 0.90:
        report.append("1. **Good agreement:** Our predictions show good agreement with the paper's predictions, ")
        report.append("with minor discrepancies that may be due to implementation details or numerical precision.\n\n")
    else:
        report.append("1. **Moderate agreement:** Our predictions show moderate agreement with the paper's predictions. ")
        report.append("Further investigation may be needed to identify sources of discrepancy.\n\n")
    
    report.append("2. **Error metrics:** RMSE and MAE values provide additional context for prediction differences.\n\n")
    report.append("3. **Per-perturbation variability:** Some perturbations show higher agreement than others, ")
    report.append("suggesting potential perturbation-specific factors.\n\n")
    
    report.append("---\n\n")
    report.append("**Report Generated:** 2025-11-17  \n")
    report.append("**Analysis Script:** `src/baselines/compare_paper_predictions.py`\n")
    
    with open(output_path, "w") as f:
        f.write("".join(report))
    
    LOGGER.info(f"Saved statistical report to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare our baseline predictions to paper's RDS file"
    )
    parser.add_argument(
        "--rds_path",
        type=Path,
        required=True,
        help="Path to single_perturbation_results_predictions.Rds",
    )
    parser.add_argument(
        "--adata_path",
        type=Path,
        required=True,
        help="Path to perturb_processed.h5ad",
    )
    parser.add_argument(
        "--split_config",
        type=Path,
        required=True,
        help="Path to split config JSON",
    )
    parser.add_argument(
        "--baseline_type",
        type=str,
        required=True,
        choices=[
            "lpm_selftrained",
            "lpm_randomPertEmb",
            "lpm_randomGeneEmb",
            "lpm_scgptGeneEmb",
            "lpm_scFoundationGeneEmb",
            "lpm_gearsPertEmb",
            "lpm_k562PertEmb",
            "lpm_rpe1PertEmb",
        ],
        help="Baseline type to compare",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/paper_comparison"),
        help="Output directory for comparison results",
    )
    parser.add_argument(
        "--predictions_dir",
        type=Path,
        default=None,
        help="Directory containing saved predictions (e.g., results/goal_2_baselines/adamson_reproduced/{baseline_type}/). If provided, will load from here instead of computing.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="unknown",
        help="Dataset name for report",
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
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load paper predictions
    paper_predictions_all = load_rds_file(args.rds_path)
    
    # Check if our baseline is in paper predictions
    if args.baseline_type not in paper_predictions_all:
        LOGGER.error(f"Baseline {args.baseline_type} not found in paper predictions.")
        LOGGER.info(f"Available baselines in paper: {list(paper_predictions_all.keys())}")
        return 1
    
    paper_predictions = paper_predictions_all[args.baseline_type]
    LOGGER.info(f"Found {len(paper_predictions)} perturbations in paper predictions for {args.baseline_type}")
    
    # Load our predictions (from saved files if available, otherwise compute)
    our_predictions = load_our_predictions(
        predictions_dir=args.predictions_dir,
        adata_path=args.adata_path if args.predictions_dir is None else None,
        split_config_path=args.split_config if args.predictions_dir is None else None,
        baseline_type=args.baseline_type if args.predictions_dir is None else None,
        output_dir=args.output_dir if args.predictions_dir is None else None,
        pca_dim=args.pca_dim,
        ridge_penalty=args.ridge_penalty,
        seed=args.seed,
    )
    
    LOGGER.info(f"Found {len(our_predictions)} perturbations in our predictions")
    
    # Compare predictions
    comparison_df = compare_predictions(paper_predictions, our_predictions)
    
    # Save comparison results
    comparison_df.to_csv(args.output_dir / "comparison_results.csv", index=False)
    LOGGER.info(f"Saved comparison results to {args.output_dir / 'comparison_results.csv'}")
    
    # Create visualizations
    create_comparison_visualizations(
        comparison_df,
        args.output_dir / "fig_comparison.png",
    )
    
    # Generate report
    generate_statistical_report(
        comparison_df,
        args.baseline_type,
        args.dataset_name,
        args.output_dir / "STATISTICAL_COMPARISON_REPORT.md",
    )
    
    return 0


if __name__ == "__main__":
    exit(main())


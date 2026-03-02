"""
Engine parity verification for LSFT point estimates.

Ensures that the resampling-enabled v2 engine produces identical point estimates
to the v1 engine (only adds CIs and resampling features).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from goal_3_prediction.lsft.lsft import evaluate_lsft
from goal_3_prediction.lsft.lsft_resampling import evaluate_lsft_with_resampling

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def verify_lsft_parity(
    adata_path: Path,
    split_config_path: Path,
    baseline_type,
    dataset_name: str,
    output_dir: Path,
    top_pcts: list[float] = [0.01, 0.05, 0.10],
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
    tolerance: float = 1e-6,
) -> Dict:
    """
    Verify that v1 and v2 engines produce identical point estimates.
    
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
    top_pcts : list[float]
        Top percentages to test
    pca_dim : int
        PCA dimension
    ridge_penalty : float
        Ridge penalty
    seed : int
        Random seed
    tolerance : float
        Tolerance for numerical comparison
    
    Returns
    -------
    Dict
        Parity verification results
    """
    LOGGER.info("=" * 60)
    LOGGER.info("LSFT ENGINE PARITY VERIFICATION")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Dataset: {dataset_name}")
    LOGGER.info(f"Baseline: {baseline_type.value}")
    LOGGER.info(f"Testing v1 vs v2 point estimates...")
    
    # Run v1 engine (original)
    LOGGER.info("\nRunning v1 engine (original)...")
    v1_results = evaluate_lsft(
        adata_path=adata_path,
        split_config_path=split_config_path,
        baseline_type=baseline_type,
        dataset_name=dataset_name,
        output_dir=output_dir / "v1",
        top_pcts=top_pcts,
        pca_dim=pca_dim,
        ridge_penalty=ridge_penalty,
        seed=seed,
    )
    
    # Run v2 engine (resampling-enabled)
    LOGGER.info("\nRunning v2 engine (resampling-enabled)...")
    v2_results = evaluate_lsft_with_resampling(
        adata_path=adata_path,
        split_config_path=split_config_path,
        baseline_type=baseline_type,
        dataset_name=dataset_name,
        output_dir=output_dir / "v2",
        top_pcts=top_pcts,
        pca_dim=pca_dim,
        ridge_penalty=ridge_penalty,
        seed=seed,
        n_boot=100,  # Minimal bootstrap for parity test
        output_format="csv",  # CSV only for parity
    )
    
    v2_results_df = v2_results["results_df"]
    
    # Compare point estimates
    LOGGER.info("\nComparing point estimates...")
    
    # Merge on test_perturbation and top_pct
    comparison = pd.merge(
        v1_results,
        v2_results_df,
        on=["test_perturbation", "top_pct"],
        suffixes=("_v1", "_v2"),
    )
    
    # Compare key metrics
    metrics_to_compare = [
        "performance_local_pearson_r",
        "performance_local_l2",
        "performance_baseline_pearson_r",
        "performance_baseline_l2",
        "improvement_pearson_r",
        "improvement_l2",
    ]
    
    parity_results = {
        "passed": True,
        "differences": {},
        "max_differences": {},
        "comparison": comparison,
    }
    
    for metric in metrics_to_compare:
        v1_col = f"{metric}_v1"
        v2_col = f"{metric}_v2"
        
        if v1_col not in comparison.columns or v2_col not in comparison.columns:
            LOGGER.warning(f"Metric {metric} not found in comparison")
            continue
        
        # Compare
        v1_values = comparison[v1_col].values
        v2_values = comparison[v2_col].values
        
        # Handle NaN
        valid_mask = ~(np.isnan(v1_values) | np.isnan(v2_values))
        v1_valid = v1_values[valid_mask]
        v2_valid = v2_values[valid_mask]
        
        if len(v1_valid) == 0:
            LOGGER.warning(f"No valid values for {metric}")
            continue
        
        # Compute differences
        differences = np.abs(v1_valid - v2_valid)
        max_diff = np.max(differences)
        mean_diff = np.mean(differences)
        
        parity_results["differences"][metric] = {
            "mean": float(mean_diff),
            "max": float(max_diff),
            "n_compared": len(v1_valid),
        }
        
        # Check if within tolerance
        if max_diff > tolerance:
            LOGGER.warning(
                f"⚠️  {metric}: Max difference = {max_diff:.2e} > tolerance = {tolerance:.2e}"
            )
            parity_results["passed"] = False
        else:
            LOGGER.info(f"✓ {metric}: Max difference = {max_diff:.2e} (within tolerance)")
    
    # Summary statistics
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("PARITY VERIFICATION SUMMARY")
    LOGGER.info("=" * 60)
    
    if parity_results["passed"]:
        LOGGER.info("✅ PARITY VERIFIED: v1 and v2 produce identical point estimates")
    else:
        LOGGER.warning("⚠️  PARITY FAILED: v1 and v2 produce different point estimates")
    
    # Print max differences
    LOGGER.info("\nMaximum differences:")
    for metric, diff_info in parity_results["differences"].items():
        LOGGER.info(f"  {metric}: {diff_info['max']:.2e} (mean: {diff_info['mean']:.2e})")
    
    return parity_results


def save_parity_report(
    parity_results: Dict,
    output_path: Path,
):
    """
    Save parity verification report.
    
    Parameters
    ----------
    parity_results : Dict
        Parity verification results
    output_path : Path
        Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("# LSFT Engine Parity Verification Report\n\n")
        f.write("## Summary\n\n")
        
        if parity_results["passed"]:
            f.write("✅ **PARITY VERIFIED**: v1 and v2 produce identical point estimates.\n\n")
        else:
            f.write("⚠️  **PARITY FAILED**: v1 and v2 produce different point estimates.\n\n")
        
        f.write("## Differences\n\n")
        f.write("| Metric | Max Difference | Mean Difference | N Compared |\n")
        f.write("|--------|----------------|-----------------|------------|\n")
        
        for metric, diff_info in parity_results["differences"].items():
            f.write(
                f"| {metric} | {diff_info['max']:.2e} | "
                f"{diff_info['mean']:.2e} | {diff_info['n_compared']} |\n"
            )
        
        f.write("\n## Notes\n\n")
        f.write("- Point estimates (means, correlations) must be identical between v1 and v2.\n")
        f.write("- v2 only adds confidence intervals and resampling features.\n")
        f.write("- Numerical differences may occur due to floating-point precision.\n")
    
    LOGGER.info(f"Saved parity report to {output_path}")


def main():
    """CLI entry point for parity verification."""
    import argparse
    from goal_2_baselines.baseline_types import BaselineType
    
    parser = argparse.ArgumentParser(
        description="Verify LSFT engine parity (v1 vs v2)"
    )
    parser.add_argument("--adata_path", type=Path, required=True, help="Path to adata file")
    parser.add_argument("--split_config", type=Path, required=True, help="Path to split config JSON")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--baseline_type",
        type=str,
        required=True,
        help="Baseline type (e.g., lpm_selftrained)",
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--top_pcts",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.10],
        help="Top percentages to test",
    )
    parser.add_argument("--pca_dim", type=int, default=10, help="PCA dimension")
    parser.add_argument("--ridge_penalty", type=float, default=0.1, help="Ridge penalty")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for numerical comparison",
    )
    
    args = parser.parse_args()
    
    # Parse baseline type
    try:
        baseline_type = BaselineType(args.baseline_type)
    except ValueError:
        raise ValueError(f"Unknown baseline type: {args.baseline_type}")
    
    # Run parity verification
    parity_results = verify_lsft_parity(
        adata_path=args.adata_path,
        split_config_path=args.split_config,
        baseline_type=baseline_type,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        top_pcts=args.top_pcts,
        pca_dim=args.pca_dim,
        ridge_penalty=args.ridge_penalty,
        seed=args.seed,
        tolerance=args.tolerance,
    )
    
    # Save report
    report_path = args.output_dir / f"parity_report_{args.dataset_name}_{baseline_type.value}.md"
    save_parity_report(parity_results, report_path)
    
    # Exit code based on parity status
    if parity_results["passed"]:
        LOGGER.info("\n✅ Parity verification PASSED")
        return 0
    else:
        LOGGER.warning("\n⚠️  Parity verification FAILED")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

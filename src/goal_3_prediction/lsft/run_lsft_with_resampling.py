#!/usr/bin/env python3
"""
Main entry point for running LSFT evaluation with resampling support.

This script:
1. Runs LSFT evaluation
2. Standardizes output (Issue 5)
3. Computes bootstrap CIs for summaries (Issue 6)
4. Performs paired baseline comparisons (Issue 7)
5. Computes hardness-performance regressions (Issue 8)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from goal_2_baselines.baseline_types import BaselineType
from goal_3_prediction.lsft.compare_baselines_resampling import (
    compare_all_baseline_pairs,
    save_baseline_comparisons,
)
from goal_3_prediction.lsft.hardness_regression_resampling import (
    compute_hardness_regressions_for_lsft,
    save_hardness_regressions,
)
from goal_3_prediction.lsft.lsft_resampling import (
    evaluate_lsft_with_resampling,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run LSFT evaluation with resampling support (Sprint 11)"
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
        help="Top percentages to try",
    )
    parser.add_argument("--pca_dim", type=int, default=10, help="PCA dimension")
    parser.add_argument("--ridge_penalty", type=float, default=0.1, help="Ridge penalty")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--n_boot", type=int, default=1000, help="Number of bootstrap samples")
    parser.add_argument("--n_perm", type=int, default=10000, help="Number of permutations")
    parser.add_argument(
        "--skip_comparisons",
        action="store_true",
        help="Skip baseline comparisons (requires multiple baselines)",
    )
    parser.add_argument(
        "--skip_regressions",
        action="store_true",
        help="Skip hardness-performance regressions",
    )

    args = parser.parse_args()

    # Parse baseline type
    try:
        baseline_type = BaselineType(args.baseline_type)
    except ValueError:
        raise ValueError(f"Unknown baseline type: {args.baseline_type}")

    LOGGER.info(f"Running LSFT with resampling for {args.dataset_name}")
    LOGGER.info(f"Baseline: {baseline_type.value}")

    # Run LSFT evaluation with resampling
    results = evaluate_lsft_with_resampling(
        adata_path=args.adata_path,
        split_config_path=args.split_config,
        baseline_type=baseline_type,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        top_pcts=args.top_pcts,
        pca_dim=args.pca_dim,
        ridge_penalty=args.ridge_penalty,
        seed=args.seed,
        n_boot=args.n_boot,
        output_format="both",
    )

    results_df = results["results_df"]
    summary = results["summary"]

    LOGGER.info(f"\n✓ LSFT evaluation complete")
    LOGGER.info(f"✓ Results saved to: {results['output_paths']}")
    LOGGER.info(f"✓ Summary with CIs saved to: {results['summary_path']}")

    # Baseline comparisons (Issue 7)
    if not args.skip_comparisons:
        LOGGER.info("\nComputing baseline comparisons...")
        try:
            comparison_df = compare_all_baseline_pairs(
                results_df=results_df,
                metrics=["pearson_r", "l2"],
                top_pcts=args.top_pcts,
                n_perm=args.n_perm,
                n_boot=args.n_boot,
                random_state=args.seed,
            )
            comparison_path = args.output_dir / f"lsft_{args.dataset_name}_baseline_comparisons"
            save_baseline_comparisons(comparison_df, comparison_path, format="both")
            LOGGER.info(f"✓ Baseline comparisons saved to: {comparison_path}")
        except Exception as e:
            LOGGER.warning(f"Baseline comparisons failed (may need multiple baselines): {e}")

    # Hardness-performance regressions (Issue 8)
    if not args.skip_regressions:
        LOGGER.info("\nComputing hardness-performance regressions...")
        try:
            regression_df = compute_hardness_regressions_for_lsft(
                results_df=results_df,
                performance_metric="pearson_r",
                hardness_metric="hardness",
                baseline_types=[baseline_type.value],
                top_pcts=args.top_pcts,
                n_boot=args.n_boot,
                random_state=args.seed,
            )
            regression_path = args.output_dir / f"lsft_{args.dataset_name}_{baseline_type.value}_hardness_regressions"
            save_hardness_regressions(regression_df, regression_path, format="both")
            LOGGER.info(f"✓ Hardness regressions saved to: {regression_path}")
        except Exception as e:
            LOGGER.warning(f"Hardness regressions failed: {e}")

    LOGGER.info("\n✓ All resampling analyses complete!")


if __name__ == "__main__":
    main()


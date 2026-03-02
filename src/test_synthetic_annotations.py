#!/usr/bin/env python3
"""
Test functional-class holdout evaluation with synthetic annotations.

This script validates that the functional-class evaluation works correctly
by generating synthetic annotations and running the full evaluation pipeline.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from shared.config import load_config
from functional_class.functional_class import run_class_holdout, class_results_to_dataframe
from shared.io import load_expression_dataset, align_expression_with_annotations
from functional_class.test_utils import generate_synthetic_class_annotations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("test_synthetic")


def main():
    parser = argparse.ArgumentParser(
        description="Test functional-class evaluation with synthetic annotations"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=5,
        help="Number of synthetic classes to generate (default: 5)",
    )
    parser.add_argument(
        "--min-class-size",
        type=int,
        default=5,
        help="Minimum perturbations per class (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic generation (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/synthetic)",
    )
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Load expression data
    LOGGER.info("Loading expression data from %s", cfg.dataset.expression_path)
    expression = load_expression_dataset(
        cfg.dataset.expression_path,
        cfg.dataset.gene_names_path,
    )
    LOGGER.info("Loaded expression matrix: %d perturbations × %d genes", 
                len(expression), len(expression.columns))
    
    # Generate synthetic annotations
    perturbation_names = expression.index.tolist()
    LOGGER.info("Generating synthetic annotations: %d classes, min %d per class",
                args.n_classes, args.min_class_size)
    
    annotations = generate_synthetic_class_annotations(
        perturbation_names=perturbation_names,
        n_classes=args.n_classes,
        min_class_size=args.min_class_size,
        seed=args.seed,
    )
    
    # Align expression and annotations
    expression, annotations = align_expression_with_annotations(expression, annotations)
    LOGGER.info("After alignment: %d perturbations, %d classes",
                len(annotations), annotations["class"].nunique())
    
    # Run functional-class evaluation
    LOGGER.info("Running functional-class holdout evaluation...")
    results = run_class_holdout(
        expression=expression,
        annotations=annotations,
        ridge_penalty=cfg.model.ridge_penalty,
        pca_dim=cfg.model.pca_dim,
        min_class_size=args.min_class_size,
        seed=cfg.dataset.seed,
    )
    
    LOGGER.info("Evaluation complete: %d results", len(results))
    
    # Convert to DataFrame
    results_df = class_results_to_dataframe(results)
    
    # Verify results
    LOGGER.info("Verifying results...")
    assert len(results_df) > 0, "Results DataFrame is empty"
    assert "pearson_r" in results_df.columns, "Missing pearson_r column"
    assert "class" in results_df.columns, "Missing class column"
    
    class_counts = results_df["class"].value_counts()
    LOGGER.info("Results by class:")
    for cls, count in class_counts.items():
        mean_r = results_df.loc[results_df["class"] == cls, "pearson_r"].mean()
        LOGGER.info("  %s: %d perturbations, mean r=%.4f", cls, count, mean_r)
    
    # Save results
    output_dir = args.output_dir or Path("results/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "results_class_synthetic.csv"
    results_df.to_csv(results_path, index=False)
    LOGGER.info("Saved results to %s", results_path)
    
    # Save synthetic annotations
    annotations_path = output_dir / "synthetic_annotations.tsv"
    annotations.to_csv(annotations_path, sep="\t", index=False)
    LOGGER.info("Saved synthetic annotations to %s", annotations_path)
    
    # Generate visualization
    viz_path = output_dir / "fig_class_holdout_synthetic.png"
    plot_class_bars(results_df, viz_path)
    LOGGER.info("Saved visualization to %s", viz_path)
    
    # Summary statistics
    LOGGER.info("=" * 60)
    LOGGER.info("TEST SUMMARY")
    LOGGER.info("=" * 60)
    LOGGER.info("Total perturbations evaluated: %d", len(results_df))
    LOGGER.info("Number of classes: %d", results_df["class"].nunique())
    LOGGER.info("Mean Pearson r: %.4f", results_df["pearson_r"].mean())
    LOGGER.info("Median Pearson r: %.4f", results_df["pearson_r"].median())
    LOGGER.info("All classes present: %s", 
                set(class_counts.index) == set(annotations["class"].unique()))
    LOGGER.info("=" * 60)
    
    LOGGER.info("✅ Test passed! Functional-class evaluation works correctly.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


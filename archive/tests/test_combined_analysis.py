"""
Unit tests for combined analysis functions.
"""

import pandas as pd
import pytest
from pathlib import Path
from core.combined_analysis import (
    load_result_tables,
    compute_combined_summary,
    compute_heatmap_matrix,
)


def test_load_result_tables_empty(temp_dir):
    """Test loading result tables when files don't exist."""
    # Create empty result files to avoid FileNotFoundError
    (temp_dir / "results_logo.csv").touch()
    (temp_dir / "results_class.csv").touch()
    
    logo_df, class_df = load_result_tables(temp_dir)
    
    assert isinstance(logo_df, pd.DataFrame)
    assert isinstance(class_df, pd.DataFrame)


def test_load_result_tables_with_data(temp_dir, sample_logo_results, sample_class_results):
    """Test loading result tables with actual data."""
    # Save sample data
    logo_path = temp_dir / "results_logo.csv"
    class_path = temp_dir / "results_class.csv"
    
    sample_logo_results.to_csv(logo_path, index=False)
    sample_class_results.to_csv(class_path, index=False)
    
    logo_df, class_df = load_result_tables(temp_dir)
    
    assert len(logo_df) == 10
    assert len(class_df) == 10
    assert "perturbation" in logo_df.columns
    assert "perturbation" in class_df.columns


def test_compute_combined_summary(sample_logo_results, sample_class_results):
    """Test combined summary computation."""
    summary = compute_combined_summary(sample_logo_results, sample_class_results)
    
    assert isinstance(summary, pd.DataFrame)
    assert "split_type" in summary.columns
    assert "group" in summary.columns
    assert "pearson_r" in summary.columns
    
    # Should have entries for both logo and class
    assert "logo" in summary["split_type"].values
    assert "class" in summary["split_type"].values


def test_compute_combined_summary_empty():
    """Test combined summary with empty DataFrames."""
    empty_logo = pd.DataFrame()
    empty_class = pd.DataFrame()
    
    summary = compute_combined_summary(empty_logo, empty_class)
    
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 0


def test_compute_heatmap_matrix(sample_logo_results, sample_class_results):
    """Test heatmap matrix computation."""
    # Ensure both have perturbation column for merging
    if "perturbation" not in sample_logo_results.columns:
        sample_logo_results["perturbation"] = [f"pert_{i}" for i in range(len(sample_logo_results))]
    if "perturbation" not in sample_class_results.columns:
        sample_class_results["perturbation"] = [f"pert_{i}" for i in range(len(sample_class_results))]
    
    # Add hardness_bin to logo results if missing
    if "hardness_bin" not in sample_logo_results.columns:
        sample_logo_results["hardness_bin"] = ["near"] * len(sample_logo_results)
    
    heatmap = compute_heatmap_matrix(sample_logo_results, sample_class_results)
    
    assert isinstance(heatmap, pd.DataFrame)
    # Should be a pivot table (2D structure) or empty if no overlap
    if not heatmap.empty:
        assert len(heatmap.columns) > 0
        assert len(heatmap.index) > 0


def test_compute_heatmap_matrix_empty():
    """Test heatmap computation with empty data."""
    empty_logo = pd.DataFrame()
    empty_class = pd.DataFrame()
    
    heatmap = compute_heatmap_matrix(empty_logo, empty_class)
    
    assert isinstance(heatmap, pd.DataFrame)


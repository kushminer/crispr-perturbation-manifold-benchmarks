"""
Unit tests for LOGO + hardness evaluation.
"""

import numpy as np
import pandas as pd
import pytest
from core.logo_hardness import (
    run_logo_evaluation,
    compute_similarity_matrix,
    assign_hardness_bins,
    results_to_dataframe,
)


def test_compute_similarity_matrix(synthetic_expression_matrix):
    """Test similarity matrix computation."""
    similarity = compute_similarity_matrix(synthetic_expression_matrix)
    
    assert isinstance(similarity, pd.DataFrame)
    assert similarity.shape[0] == similarity.shape[1]
    assert similarity.shape[0] == len(synthetic_expression_matrix)
    # Diagonal should be 1.0 (self-similarity)
    np.testing.assert_array_almost_equal(
        np.diag(similarity.values), np.ones(len(similarity))
    )
    # Should be symmetric
    np.testing.assert_array_almost_equal(
        similarity.values, similarity.values.T
    )


def test_assign_hardness_bins():
    """Test hardness bin assignment."""
    n = 30
    similarities = pd.Series(np.random.rand(n), index=[f"pert_{i}" for i in range(n)])
    
    # assign_hardness_bins returns a single string, not a Series
    # Test that it returns valid bin names
    bin_name = assign_hardness_bins(similarities, quantiles=[0.33, 0.66])
    
    assert bin_name in {"near", "mid", "far", "unknown"}


def test_run_logo_evaluation_basic(synthetic_expression_matrix):
    """Test basic LOGO evaluation."""
    results = run_logo_evaluation(
        expression=synthetic_expression_matrix,
        hardness_bins=[0.33, 0.66],
        alpha=0.1,
        pca_dim=10,
        block_clusters=False,
        cluster_size=None,
        seed=42,
    )
    
    assert len(results) == len(synthetic_expression_matrix)
    assert all(hasattr(r, "perturbation") for r in results)
    assert all(hasattr(r, "hardness_bin") for r in results)
    assert all(hasattr(r, "metrics") for r in results)
    assert all("pearson_r" in r.metrics for r in results)


def test_run_logo_evaluation_with_clustering(synthetic_expression_matrix):
    """Test LOGO evaluation with cluster blocking."""
    results = run_logo_evaluation(
        expression=synthetic_expression_matrix,
        hardness_bins=[0.33, 0.66],
        alpha=0.1,
        pca_dim=10,
        block_clusters=True,
        cluster_size=5,
        seed=42,
    )
    
    assert len(results) == len(synthetic_expression_matrix)
    assert all(hasattr(r, "cluster_blocked") for r in results)
    assert any(r.cluster_blocked for r in results)  # At least some should be blocked


def test_results_to_dataframe(sample_logo_results):
    """Test conversion of LOGO results to DataFrame."""
    # Create mock results
    from core.logo_hardness import LogoResult
    
    results = [
        LogoResult(
            perturbation=f"pert_{i}",
            hardness_bin=["near", "mid", "far"][i % 3],
            cluster_blocked=False,
            metrics={"pearson_r": 0.5, "mse": 0.1, "mae": 0.2},
        )
        for i in range(10)
    ]
    
    df = results_to_dataframe(results)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert "perturbation" in df.columns
    assert "hardness_bin" in df.columns
    assert "pearson_r" in df.columns
    assert "mse" in df.columns


def test_run_logo_evaluation_small_dataset():
    """Test LOGO evaluation with very small dataset."""
    # Create very small dataset
    expression = pd.DataFrame(
        np.random.randn(5, 10),
        index=[f"pert_{i}" for i in range(5)],
        columns=[f"gene_{i}" for i in range(10)],
    )
    
    results = run_logo_evaluation(
        expression=expression,
        hardness_bins=[0.5],  # Only one split
        alpha=0.1,
        pca_dim=3,  # Small PCA dim
        block_clusters=False,
        cluster_size=None,
        seed=42,
    )
    
    assert len(results) == 5


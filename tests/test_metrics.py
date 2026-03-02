"""
Unit tests for metrics computation.
"""

import numpy as np
import pytest
from shared.metrics import compute_metrics


def test_compute_metrics_perfect_correlation():
    """Test metrics with perfect correlation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    metrics = compute_metrics(y_true, y_pred)
    
    assert metrics["pearson_r"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["spearman_rho"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["mse"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-6)


def test_compute_metrics_negative_correlation():
    """Test metrics with negative correlation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    
    metrics = compute_metrics(y_true, y_pred)
    
    assert metrics["pearson_r"] == pytest.approx(-1.0, abs=1e-6)
    assert metrics["mse"] > 0
    assert metrics["mae"] > 0


def test_compute_metrics_no_correlation():
    """Test metrics with no correlation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([5.0, 1.0, 4.0, 2.0, 3.0])  # Random order
    
    metrics = compute_metrics(y_true, y_pred)
    
    assert abs(metrics["pearson_r"]) < 0.5
    assert metrics["mse"] > 0
    assert metrics["mae"] > 0


def test_compute_metrics_constant_values():
    """Test metrics with constant values."""
    y_true = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    y_pred = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    
    metrics = compute_metrics(y_true, y_pred)
    
    # Pearson r should be NaN for constant values
    assert np.isnan(metrics["pearson_r"]) or abs(metrics["pearson_r"]) < 1e-6
    assert metrics["mse"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["mae"] == pytest.approx(1.0, abs=1e-6)


def test_compute_metrics_different_lengths():
    """Test that metrics raise error for different lengths."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    
    with pytest.raises(ValueError):
        compute_metrics(y_true, y_pred)


def test_compute_metrics_with_nan():
    """Test metrics computation handles NaN values."""
    y_true = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    metrics = compute_metrics(y_true, y_pred)
    
    # Should compute metrics on non-NaN values
    assert "pearson_r" in metrics
    assert "mse" in metrics
    assert "mae" in metrics


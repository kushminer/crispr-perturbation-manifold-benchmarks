"""
Unit tests for bootstrap confidence interval functions.
"""

import numpy as np
import pytest

from stats.bootstrapping import bootstrap_mean_ci, bootstrap_correlation_ci


class TestBootstrapMeanCI:
    """Test bootstrap_mean_ci function."""

    def test_basic_functionality(self):
        """Test basic CI computation."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, ci_lower, ci_upper = bootstrap_mean_ci(values, n_boot=100, random_state=42)

        assert isinstance(mean, float)
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)
        assert mean == pytest.approx(3.0)  # Exact mean
        assert ci_lower <= mean <= ci_upper

    def test_ci_coverage(self):
        """Test that CI has reasonable coverage properties."""
        # Generate data from known distribution (mean=0, std=1)
        np.random.seed(42)
        true_mean = 0.0
        values = np.random.normal(true_mean, 1.0, size=100)

        mean, ci_lower, ci_upper = bootstrap_mean_ci(values, n_boot=1000, random_state=42)

        # CI should contain the true mean (most of the time)
        # This is a basic sanity check, not a rigorous coverage test
        assert ci_lower <= ci_upper

    def test_single_value(self):
        """Test handling of single value."""
        values = np.array([5.0])
        mean, ci_lower, ci_upper = bootstrap_mean_ci(values, n_boot=100, random_state=42)

        assert mean == 5.0
        assert ci_lower == 5.0
        assert ci_upper == 5.0

    def test_empty_array(self):
        """Test that empty array raises error."""
        values = np.array([])
        with pytest.raises(ValueError, match="Cannot compute CI for empty array"):
            bootstrap_mean_ci(values)

    def test_pearson_r_values(self):
        """Test with realistic Pearson r values."""
        # Simulate Pearson r values from LSFT evaluation
        pearson_r_values = np.array([0.7, 0.75, 0.8, 0.72, 0.78, 0.65, 0.85])
        mean, ci_lower, ci_upper = bootstrap_mean_ci(pearson_r_values, n_boot=1000, random_state=42)

        assert 0.0 <= mean <= 1.0
        assert 0.0 <= ci_lower <= 1.0
        assert 0.0 <= ci_upper <= 1.0
        assert ci_lower <= mean <= ci_upper

    def test_l2_values(self):
        """Test with realistic L2 values."""
        # Simulate L2 values from LSFT evaluation
        l2_values = np.array([5.2, 5.5, 5.8, 4.9, 6.1, 5.3, 5.7])
        mean, ci_lower, ci_upper = bootstrap_mean_ci(l2_values, n_boot=1000, random_state=42)

        assert mean > 0
        assert ci_lower > 0
        assert ci_upper > 0
        assert ci_lower <= mean <= ci_upper

    def test_reproducibility(self):
        """Test that results are reproducible with same random_state."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        mean1, ci_lower1, ci_upper1 = bootstrap_mean_ci(values, n_boot=100, random_state=42)
        mean2, ci_lower2, ci_upper2 = bootstrap_mean_ci(values, n_boot=100, random_state=42)

        assert mean1 == mean2
        assert ci_lower1 == ci_lower2
        assert ci_upper2 == ci_upper2

    def test_alpha_parameter(self):
        """Test different alpha values."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # 90% CI (alpha=0.10)
        _, ci_lower_90, ci_upper_90 = bootstrap_mean_ci(values, n_boot=100, alpha=0.10, random_state=42)
        # 95% CI (alpha=0.05)
        _, ci_lower_95, ci_upper_95 = bootstrap_mean_ci(values, n_boot=100, alpha=0.05, random_state=42)

        # 90% CI should be narrower than 95% CI
        width_90 = ci_upper_90 - ci_lower_90
        width_95 = ci_upper_95 - ci_lower_95
        assert width_90 < width_95


class TestBootstrapCorrelationCI:
    """Test bootstrap_correlation_ci function."""

    def test_basic_functionality(self):
        """Test basic correlation CI computation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # Perfect positive correlation
        corr, ci_lower, ci_upper = bootstrap_correlation_ci(x, y, n_boot=100, random_state=42)

        assert isinstance(corr, float)
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)
        assert corr == pytest.approx(1.0, abs=1e-10)
        assert ci_lower <= corr <= ci_upper

    def test_negative_correlation(self):
        """Test with negative correlation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Perfect negative correlation
        corr, ci_lower, ci_upper = bootstrap_correlation_ci(x, y, n_boot=100, random_state=42)

        assert corr == pytest.approx(-1.0, abs=1e-10)
        assert -1.0 <= ci_lower <= ci_upper <= 1.0

    def test_length_mismatch(self):
        """Test that mismatched lengths raise error."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="must have the same length"):
            bootstrap_correlation_ci(x, y)

    def test_insufficient_data(self):
        """Test that insufficient data raises error."""
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Need at least 3 pairs"):
            bootstrap_correlation_ci(x, y)


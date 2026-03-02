"""
Nonparametric bootstrap confidence intervals for LSFT evaluation metrics.

This module implements bootstrap methods to compute confidence intervals
for mean Pearson r and mean L2 across LSFT test perturbations.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def bootstrap_mean_ci(
    values: np.ndarray | list[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for the mean of a sample.

    Uses percentile bootstrap method: resample with replacement, compute mean
    for each bootstrap sample, then use percentiles of bootstrap distribution
    as confidence interval bounds.

    Parameters
    ----------
    values : array-like
        Sample values (e.g., Pearson r or L2 per perturbation)
    n_boot : int, default=1000
        Number of bootstrap samples
    alpha : float, default=0.05
        Significance level (1 - confidence level). For 95% CI, use alpha=0.05
    random_state : int or None, default=None
        Random seed for reproducibility

    Returns
    -------
    mean : float
        Sample mean (point estimate)
    ci_lower : float
        Lower bound of confidence interval
    ci_upper : float
        Upper bound of confidence interval

    Examples
    --------
    >>> values = np.array([0.7, 0.75, 0.8, 0.72, 0.78])
    >>> mean, ci_lower, ci_upper = bootstrap_mean_ci(values, n_boot=1000)
    >>> print(f"Mean: {mean:.3f}, 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    """
    values = np.asarray(values)
    n = len(values)

    if n == 0:
        raise ValueError("Cannot compute CI for empty array")
    if n == 1:
        LOGGER.warning("Only one value provided, CI will be degenerate")
        mean = float(values[0])
        return mean, mean, mean

    # Set random seed for reproducibility
    rng = np.random.default_rng(random_state)

    # Compute sample mean (point estimate)
    mean = float(np.mean(values))

    # Generate bootstrap samples
    bootstrap_means = []
    for _ in range(n_boot):
        # Resample with replacement
        bootstrap_sample = rng.choice(values, size=n, replace=True)
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)

    bootstrap_means = np.array(bootstrap_means)

    # Compute percentile-based CI
    # For 95% CI (alpha=0.05): use 2.5th and 97.5th percentiles
    # For (1-alpha)*100% CI: use (alpha/2)th and (1-alpha/2)th percentiles
    percentiles = (100 * alpha / 2, 100 * (1 - alpha / 2))
    ci_lower, ci_upper = np.percentile(bootstrap_means, percentiles)

    return mean, float(ci_lower), float(ci_upper)


def bootstrap_correlation_ci(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for Pearson correlation coefficient.

    Resamples pairs (x_i, y_i) with replacement and computes correlation
    for each bootstrap sample.

    Parameters
    ----------
    x : array-like
        First variable
    y : array-like
        Second variable (same length as x)
    n_boot : int, default=1000
        Number of bootstrap samples
    alpha : float, default=0.05
        Significance level
    random_state : int or None, default=None
        Random seed for reproducibility

    Returns
    -------
    corr : float
        Sample correlation (point estimate)
    ci_lower : float
        Lower bound of confidence interval
    ci_upper : float
        Upper bound of confidence interval
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 pairs to compute correlation")

    rng = np.random.default_rng(random_state)

    # Compute sample correlation (point estimate)
    corr = float(np.corrcoef(x, y)[0, 1])

    # Generate bootstrap samples
    bootstrap_corrs = []
    indices = np.arange(n)
    for _ in range(n_boot):
        # Resample indices with replacement
        boot_indices = rng.choice(indices, size=n, replace=True)
        boot_x = x[boot_indices]
        boot_y = y[boot_indices]
        boot_corr = np.corrcoef(boot_x, boot_y)[0, 1]
        # Handle NaN (e.g., if all x or all y are the same in bootstrap sample)
        if not np.isnan(boot_corr):
            bootstrap_corrs.append(boot_corr)

    if len(bootstrap_corrs) == 0:
        LOGGER.warning("All bootstrap correlations were NaN, returning degenerate CI")
        return corr, corr, corr

    bootstrap_corrs = np.array(bootstrap_corrs)

    # Compute percentile-based CI
    percentiles = (100 * alpha / 2, 100 * (1 - alpha / 2))
    ci_lower, ci_upper = np.percentile(bootstrap_corrs, percentiles)

    return corr, float(ci_lower), float(ci_upper)


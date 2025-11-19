"""
Paired permutation tests for baseline comparisons in LSFT evaluation.

This module implements paired sign-flip permutation tests to compare
baseline performance per LSFT perturbation.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def paired_permutation_test(
    deltas: np.ndarray | list[float],
    n_perm: int = 10000,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    random_state: int | None = None,
) -> tuple[float, float]:
    """
    Perform paired permutation test using sign-flip permutations.

    For comparing two baselines on paired data (e.g., same perturbations),
    we compute deltas = A - B for each pair, then test whether mean(delta)
    is significantly different from zero.

    The permutation test flips signs of deltas randomly and recomputes
    the test statistic (mean) for each permutation.

    Parameters
    ----------
    deltas : array-like
        Paired differences (e.g., baseline_A - baseline_B for each perturbation)
    n_perm : int, default=10000
        Number of permutations
    alternative : {"two-sided", "greater", "less"}, default="two-sided"
        Alternative hypothesis:
        - "two-sided": mean(delta) != 0
        - "greater": mean(delta) > 0 (A better than B)
        - "less": mean(delta) < 0 (B better than A)
    random_state : int or None, default=None
        Random seed for reproducibility

    Returns
    -------
    mean_delta : float
        Observed mean of deltas (point estimate)
    p_value : float
        Permutation p-value

    Examples
    --------
    >>> deltas = np.array([0.05, 0.10, -0.02, 0.08, 0.12])
    >>> mean_delta, p_value = paired_permutation_test(deltas, n_perm=1000)
    >>> print(f"Mean delta: {mean_delta:.3f}, p-value: {p_value:.4f}")
    """
    deltas = np.asarray(deltas)
    n = len(deltas)

    if n == 0:
        raise ValueError("Cannot compute p-value for empty array")
    if n == 1:
        LOGGER.warning("Only one pair provided, p-value will be degenerate")
        mean_delta = float(deltas[0])
        # With one observation, p-value can only be 1.0 (two-sided) or 0.5 (one-sided)
        if alternative == "two-sided":
            return mean_delta, 1.0
        else:
            return mean_delta, 0.5

    # Compute observed mean delta
    mean_delta = float(np.mean(deltas))

    # Set random seed for reproducibility
    rng = np.random.default_rng(random_state)

    # Generate permutations
    # For each permutation, flip signs randomly
    permuted_means = []
    for _ in range(n_perm):
        # Random sign flips (multiply each delta by random ±1)
        signs = rng.choice([-1, 1], size=n, replace=True)
        permuted_deltas = deltas * signs
        permuted_mean = np.mean(permuted_deltas)
        permuted_means.append(permuted_mean)

    permuted_means = np.array(permuted_means)

    # Compute p-value based on alternative hypothesis
    if alternative == "two-sided":
        # Count permutations where |mean| >= |observed_mean|
        abs_observed = abs(mean_delta)
        abs_permuted = np.abs(permuted_means)
        n_extreme = np.sum(abs_permuted >= abs_observed)
        p_value = (n_extreme + 1) / (n_perm + 1)  # Add 1 for observed value
    elif alternative == "greater":
        # Count permutations where mean >= observed_mean
        n_extreme = np.sum(permuted_means >= mean_delta)
        p_value = (n_extreme + 1) / (n_perm + 1)
    elif alternative == "less":
        # Count permutations where mean <= observed_mean
        n_extreme = np.sum(permuted_means <= mean_delta)
        p_value = (n_extreme + 1) / (n_perm + 1)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return mean_delta, float(p_value)


def paired_permutation_test_two_sample(
    group1: np.ndarray | list[float],
    group2: np.ndarray | list[float],
    n_perm: int = 10000,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    random_state: int | None = None,
) -> tuple[float, float]:
    """
    Perform paired permutation test for two groups (wrapper for convenience).

    Computes deltas = group1 - group2, then calls paired_permutation_test.

    Parameters
    ----------
    group1 : array-like
        First group values (e.g., baseline A performance per perturbation)
    group2 : array-like
        Second group values (e.g., baseline B performance per perturbation)
    n_perm : int, default=10000
        Number of permutations
    alternative : {"two-sided", "greater", "less"}, default="two-sided"
        Alternative hypothesis
    random_state : int or None, default=None
        Random seed for reproducibility

    Returns
    -------
    mean_delta : float
        Observed mean of (group1 - group2)
    p_value : float
        Permutation p-value
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    if len(group1) != len(group2):
        raise ValueError("group1 and group2 must have the same length")

    deltas = group1 - group2
    return paired_permutation_test(deltas, n_perm=n_perm, alternative=alternative, random_state=random_state)


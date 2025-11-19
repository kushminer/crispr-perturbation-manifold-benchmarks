"""
Statistical resampling utilities for evaluation framework.

This module provides bootstrap confidence intervals and permutation tests
for LSFT (Local Similarity-Filtered Training) evaluation.

Functions:
    bootstrap_mean_ci: Compute bootstrap confidence intervals for mean metrics
    paired_permutation_test: Perform paired permutation test for baseline comparisons
"""

from .bootstrapping import bootstrap_mean_ci
from .permutation import paired_permutation_test

__all__ = ["bootstrap_mean_ci", "paired_permutation_test"]


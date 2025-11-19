"""
Local Similarity-Filtered Training (LSFT) module.

This module implements the LSFT approach that filters training perturbations
by similarity to test perturbations and retrains models on filtered subsets.
"""

from .lsft import evaluate_lsft

__all__ = ["evaluate_lsft"]


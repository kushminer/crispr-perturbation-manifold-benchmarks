"""
Goal 2: Baseline Reproduction

This module contains code for reproducing all 9 baseline models (8 linear + mean-response)
from the Nature (2024) paper.
"""

from .baseline_runner import (
    compute_pseudobulk_expression_changes,
    construct_gene_embeddings,
    construct_pert_embeddings,
    run_single_baseline,
)
from .baseline_types import BaselineConfig, BaselineType, get_baseline_config
from .split_logic import load_split_config, prepare_perturbation_splits

__all__ = [
    "compute_pseudobulk_expression_changes",
    "construct_gene_embeddings",
    "construct_pert_embeddings",
    "run_single_baseline",
    "BaselineConfig",
    "BaselineType",
    "get_baseline_config",
    "load_split_config",
    "prepare_perturbation_splits",
]


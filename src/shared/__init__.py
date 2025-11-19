"""
Shared evaluation framework modules.

This module contains essential utilities used across all goals:
- Linear model solving (Y = A·K·B)
- Performance metrics
- Data I/O
- Configuration management
- Validation utilities
- Embedding parity validation
- Comparison utilities
"""

from .linear_model import solve_y_axb
from .metrics import compute_metrics
from .io import load_expression_matrix, load_annotations, load_expression_dataset, align_expression_with_annotations
from .config import ExperimentConfig, load_config

# Optional imports - may not exist in all configurations
try:
    from .validation import run_full_validation
except ImportError:
    run_full_validation = None

try:
    from .embedding_parity import run_embedding_parity
    validate_embeddings = run_embedding_parity  # Alias for backwards compatibility
except (ImportError, AttributeError):
    validate_embeddings = None

__all__ = [
    "solve_y_axb",
    "compute_metrics",
    "load_expression_matrix",
    "load_annotations",
    "load_expression_dataset",
    "align_expression_with_annotations",
    "ExperimentConfig",
    "load_config",
]
if run_full_validation is not None:
    __all__.append("run_full_validation")
if validate_embeddings is not None:
    __all__.append("validate_embeddings")


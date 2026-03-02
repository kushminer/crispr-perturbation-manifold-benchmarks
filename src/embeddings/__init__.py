"""
Embedding loaders and utilities.
"""

from . import registry  # noqa: F401
from .base import EmbeddingResult  # noqa: F401

# Ensure built-in loaders register themselves
from . import gears_go_perturbation  # noqa: F401
from . import pca_perturbation  # noqa: F401
from . import scgpt_gene  # noqa: F401
from . import scfoundation_gene  # noqa: F401

__all__ = ["registry", "EmbeddingResult"]

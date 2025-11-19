"""
Goal 1: Cosine Similarity Investigation

This module provides two types of similarity analysis:
1. DE Matrix Similarity: Computes similarity on pseudobulk expression changes (Y matrix)
2. Embedding Similarity: Computes similarity in baseline-specific embedding spaces (B matrices)
"""

from .de_matrix_similarity import (
    compute_similarity_statistics as compute_de_similarity_statistics,
    load_baseline_performance,
    load_pseudobulk_matrix,
    run_similarity_analysis as run_de_matrix_similarity_analysis,
)

from .embedding_similarity import (
    compute_embedding_similarity_statistics,
    run_embedding_similarity_analysis,
)

__all__ = [
    # DE Matrix similarity (expression space)
    "compute_de_similarity_statistics",
    "load_baseline_performance",
    "load_pseudobulk_matrix",
    "run_de_matrix_similarity_analysis",
    # Embedding similarity (baseline-specific)
    "compute_embedding_similarity_statistics",
    "run_embedding_similarity_analysis",
]


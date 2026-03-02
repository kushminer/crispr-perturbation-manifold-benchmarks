"""
Loader for GEARS GO-based perturbation embeddings (Python translation of
`extract_pert_embedding_from_gears.R`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh

from .base import EmbeddingResult
from .registry import register


def _build_adjacency_matrix(df: pd.DataFrame, symmetric: bool = True) -> sparse.csr_matrix:
    genes = (
        pd.concat([df["source"], df["target"]])
        .drop_duplicates()
        .tolist()
    )
    gene_index: Dict[str, int] = {g: idx for idx, g in enumerate(genes)}

    rows = df["source"].map(gene_index).to_numpy()
    cols = df["target"].map(gene_index).to_numpy()
    weights = df["importance"].to_numpy()

    mat = sparse.coo_matrix((weights, (rows, cols)), shape=(len(genes), len(genes)))
    if symmetric:
        mat = mat + mat.T
    return mat.tocsr(), genes


def _spectral_embedding(adj: sparse.csr_matrix, n_components: int) -> np.ndarray:
    # Add a tiny ridge to avoid singular matrices.
    diag = sparse.diags(np.full(adj.shape[0], 1e-6))
    sym = adj + diag

    k = min(n_components, sym.shape[0] - 1)
    vals, vecs = eigsh(sym, k=k, which="LA")
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    scales = np.sqrt(np.clip(vals[order], a_min=0, a_max=None))
    return (vecs * scales).T


@register("gears_go")
def load_gears_go_embedding(
    source_csv: Path,
    n_components: int = 10,
    min_edges: Optional[int] = None,
) -> EmbeddingResult:
    """
    Compute perturbation embeddings from a GO similarity graph.

    Parameters
    ----------
    source_csv:
        Path to GO similarity CSV (must include columns: source, target, importance).
    n_components:
        Number of spectral components to retain (default: 10).
    min_edges:
        Optional edge-count threshold. If provided, raises ValueError when the subset
        contains fewer than `min_edges` entries.
    """

    csv_path = Path(source_csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"GO similarity CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"source", "target", "importance"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in GO CSV: {missing}")

    if min_edges is not None and len(df) < min_edges:
        raise ValueError(
            f"GO subset has {len(df)} edges, fewer than required minimum {min_edges}."
        )

    adj, genes = _build_adjacency_matrix(df)
    embedding = _spectral_embedding(adj, n_components=n_components)

    dim_labels = [f"component_{i+1}" for i in range(embedding.shape[0])]
    return EmbeddingResult(
        values=embedding,
        item_labels=genes,
        dim_labels=dim_labels,
        metadata={
            "source_csv": str(csv_path),
            "n_components": n_components,
            "n_edges": int(len(df)),
            "n_genes": len(genes),
        },
    )


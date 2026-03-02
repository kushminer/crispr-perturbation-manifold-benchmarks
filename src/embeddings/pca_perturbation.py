"""
Loader for PCA-based perturbation embeddings (translation of
`extract_pert_embedding_pca.R`).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA

from .base import EmbeddingResult
from .registry import register


def _clean_condition(obs) -> np.ndarray:
    if "clean_condition" in obs:
        return obs["clean_condition"].astype(str).to_numpy()
    if "condition" not in obs:
        raise ValueError("AnnData obs is missing 'condition' column.")
    cond = obs["condition"].astype(str)
    cleaned = cond.str.replace("+ctrl", "", regex=False)
    return cleaned.to_numpy()


def _pseudobulk_matrix(adata: ad.AnnData) -> Tuple[np.ndarray, List[str]]:
    obs = adata.obs.copy()
    clean_cond = _clean_condition(obs)
    obs["_key"] = (
        obs["condition"].astype(str) + "|" + clean_cond
    )
    unique_keys = obs["_key"].unique().tolist()

    aggregated = []
    clean_labels = []
    for key in unique_keys:
        idx = np.where(obs["_key"] == key)[0]
        if len(idx) == 0:
            continue
        data = adata.X[idx]
        if sparse.issparse(data):
            mean_vec = np.asarray(data.mean(axis=0)).ravel()
        else:
            mean_vec = data.mean(axis=0)
        aggregated.append(np.asarray(mean_vec).ravel())
        clean_labels.append(clean_cond[idx[0]])

    matrix = np.vstack(aggregated)
    return matrix, clean_labels


@register("pca_perturbation")
def load_pca_perturbation_embedding(
    adata_path: Path,
    n_components: int = 10,
    seed: int = 1,
) -> EmbeddingResult:
    """
    Compute perturbation embeddings via PCA on pseudobulk expression profiles.

    Parameters
    ----------
    adata_path:
        Path to `perturb_processed.h5ad` (or subset).
    n_components:
        Number of PCA components (default: 10).
    seed:
        Random seed for PCA reproducibility.
    """
    path = Path(adata_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"h5ad file not found: {path}")

    rng = np.random.default_rng(seed)
    adata = ad.read_h5ad(path)

    matrix, labels = _pseudobulk_matrix(adata)
    if matrix.shape[0] == 0:
        raise ValueError("Pseudobulk produced zero perturbations.")

    X_for_pca = matrix  # perturbations x genes
    max_components = min(n_components, X_for_pca.shape[0], X_for_pca.shape[1])
    if max_components < 1:
        raise ValueError("Not enough data to compute PCA components.")

    pca = PCA(n_components=max_components, random_state=seed)
    scores = pca.fit_transform(X_for_pca)  # perturbations x components
    embedding = scores.T  # components x perturbations

    dim_labels = [f"PC{i+1}" for i in range(embedding.shape[0])]
    return EmbeddingResult(
        values=embedding,
        item_labels=labels,
        dim_labels=dim_labels,
        metadata={
            "adata_path": str(path),
            "n_components": int(embedding.shape[0]),
            "n_genes": int(matrix.shape[1]),
            "n_perturbations": int(len(labels)),
        },
    )


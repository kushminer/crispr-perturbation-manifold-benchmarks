"""
Utilities for fitting and applying the linear perturbation model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class LinearModel:
    gene_embeddings: np.ndarray  # genes × d_g
    pert_embeddings: np.ndarray  # train_perts × d_p
    coef: np.ndarray  # d_g × d_p
    center: np.ndarray  # genes vector
    gene_pca: PCA
    pert_pca: PCA


def solve_y_axb(
    Y: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    ridge_penalty: float,
) -> Dict[str, np.ndarray]:
    """
    Solve Y = A K B for K with ridge penalties on both sides.
    """
    if Y.shape[0] != A.shape[0]:
        raise ValueError("Row dimension mismatch between Y and A.")
    if Y.shape[1] != B.shape[1]:
        raise ValueError("Column dimension mismatch between Y and B.")

    AtA = A.T @ A + ridge_penalty * np.eye(A.shape[1])
    BBt = B @ B.T + ridge_penalty * np.eye(B.shape[0])

    tmp = np.linalg.solve(AtA, A.T @ Y @ B.T)
    K = np.linalg.solve(BBt, tmp.T).T
    return {"K": K}


def fit_linear_model(
    train_matrix: np.ndarray,
    pca_dim: int,
    ridge_penalty: float,
    seed: int = 1,
) -> LinearModel:
    """
    Fit the linear perturbation model on the training matrix (perturbations × genes).
    """
    rng = np.random.default_rng(seed)
    n_perts, n_genes = train_matrix.shape
    pca_dim_gene = min(pca_dim, n_genes, n_perts)
    pca_dim_pert = min(pca_dim, n_perts)

    gene_pca = PCA(n_components=pca_dim_gene, random_state=seed)
    gene_embeddings = gene_pca.fit_transform(train_matrix.T)  # genes × dim_g

    pert_pca = PCA(n_components=pca_dim_pert, random_state=seed)
    pert_embeddings = pert_pca.fit_transform(train_matrix)  # perts × dim_p

    Y = train_matrix.T  # genes × perts
    A = gene_embeddings
    B = pert_embeddings.T

    solution = solve_y_axb(Y, A=A, B=B, ridge_penalty=ridge_penalty)
    K = solution["K"]

    center = gene_embeddings.mean(axis=0)
    return LinearModel(
        gene_embeddings=gene_embeddings,
        pert_embeddings=pert_embeddings,
        coef=K,
        center=center,
        gene_pca=gene_pca,
        pert_pca=pert_pca,
    )


def predict_perturbation(
    model: LinearModel,
    pert_vector: np.ndarray,
) -> np.ndarray:
    """
    Predict gene expression for a single perturbation vector.
    """
    pert_emb = model.pert_pca.transform(pert_vector.reshape(1, -1))  # 1 × dim_p
    gene_basis = model.gene_embeddings
    pred_centered = gene_basis @ model.coef @ pert_emb.T
    pred = pred_centered[:, 0]
    return pred


"""
Paper's Python implementation of run_linear_pretrained_model.

This is the validated Python translation of run_linear_pretrained_model.R
that matches the R implementation with Pearson r ≥ 0.999.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.linalg import solve
from sklearn.decomposition import PCA

LOGGER = logging.getLogger(__name__)


def solve_y_axb(Y, A=None, B=None, A_ridge=0.01, B_ridge=0.01):
    """
    Solve Y = A * K * B using ridge regression.
    
    This matches the R implementation exactly.
    
    Parameters
    ----------
    Y : array-like
        Target matrix (genes × perturbations)
    A : array-like or None
        Gene embeddings (genes × embedding_dim)
    B : array-like or None
        Perturbation embeddings (embedding_dim × perturbations)
    A_ridge : float
        Ridge penalty for A
    B_ridge : float
        Ridge penalty for B
        
    Returns
    -------
    dict
        Dictionary with 'K' (coefficient matrix) and 'center' (row means of Y)
    """
    Y = np.asarray(Y)
    center = np.mean(Y, axis=1, keepdims=True)
    Y = Y - center
    
    if A is not None and B is not None:
        A = np.asarray(A)
        B = np.asarray(B)
        assert Y.shape[0] == A.shape[0], f"Y rows ({Y.shape[0]}) != A rows ({A.shape[0]})"
        assert Y.shape[1] == B.shape[1], f"Y cols ({Y.shape[1]}) != B cols ({B.shape[1]})"
        
        # Solve: K = (A^T A + λI)^(-1) A^T Y B^T (B B^T + λI)^(-1)
        AtA_ridge = A.T @ A + A_ridge * np.eye(A.shape[1])
        BBt_ridge = B @ B.T + B_ridge * np.eye(B.shape[0])
        
        tmp = solve(AtA_ridge, A.T @ Y @ B.T, assume_a='sym')
        tmp = solve(BBt_ridge, tmp.T, assume_a='sym').T
        
    elif B is None:
        A = np.asarray(A)
        AtA_ridge = A.T @ A + A_ridge * np.eye(A.shape[1])
        tmp = solve(AtA_ridge, A.T @ Y, assume_a='sym')
        
    elif A is None:
        B = np.asarray(B)
        BBt_ridge = B @ B.T + B_ridge * np.eye(B.shape[0])
        tmp = (Y @ B.T @ solve(BBt_ridge, np.eye(B.shape[0]), assume_a='sym')).T
        
    else:
        raise ValueError("Either A or B must be non-null")
    
    tmp = np.nan_to_num(tmp, nan=0.0, posinf=0.0, neginf=0.0)
    
    return {'K': tmp, 'center': center.flatten()}


def pseudobulk_adata(adata: ad.AnnData, group_by: List[str]) -> ad.AnnData:
    """
    Pseudobulk anndata object by grouping variables.
    Takes mean expression across cells in each group.
    
    Parameters
    ----------
    adata : AnnData
        Input AnnData object
    group_by : list of str
        Column names in adata.obs to group by
        
    Returns
    -------
    AnnData
        Pseudobulked AnnData object
    """
    # Create grouping key
    adata.obs['_group_key'] = adata.obs[group_by].apply(
        lambda x: '|'.join(x.astype(str)), axis=1
    )
    
    # Get unique groups
    unique_groups = sorted(adata.obs['_group_key'].unique())
    
    # Aggregate expression for each group
    X_list = []
    obs_list = []
    
    for group in unique_groups:
        mask = adata.obs['_group_key'] == group
        group_data = adata[mask]
        
        # Mean expression across cells in group
        if sparse.issparse(group_data.X):
            X_mean = np.array(group_data.X.mean(axis=0)).flatten()
        else:
            X_mean = group_data.X.mean(axis=0)
            if X_mean.ndim > 1:
                X_mean = X_mean.flatten()
        
        X_list.append(X_mean)
        
        # Get metadata (should be same for all cells in group)
        obs_dict = group_data.obs.iloc[0].to_dict()
        obs_list.append(obs_dict)
    
    # Create new AnnData
    X_pbulk = np.vstack(X_list)
    obs_pbulk = pd.DataFrame(obs_list)
    
    pbulk_adata = ad.AnnData(
        X=X_pbulk,
        obs=obs_pbulk,
        var=adata.var.copy()
    )
    
    # Clean up temporary column
    if '_group_key' in pbulk_adata.obs.columns:
        pbulk_adata.obs = pbulk_adata.obs.drop(columns=['_group_key'])
    
    return pbulk_adata


def run_paper_baseline(
    adata_path: Path,
    split_config: Dict[str, List[str]],
    gene_embedding: str = "training_data",
    pert_embedding: str = "training_data",
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
    pert_embedding_path: Optional[Path] = None,
) -> Dict:
    """
    Run baseline using paper's validated Python implementation.
    
    This function wraps the paper's implementation to match our interface.
    
    Parameters
    ----------
    adata_path : Path
        Path to perturb_processed.h5ad
    split_config : Dict[str, List[str]]
        Train/test/val split configuration
    gene_embedding : str
        Gene embedding source ("training_data", "identity", "zero", "random", or path to TSV)
    pert_embedding : str
        Perturbation embedding source ("training_data", "identity", "zero", "random", or path to TSV)
    pca_dim : int
        PCA dimension
    ridge_penalty : float
        Ridge penalty
    seed : int
        Random seed
    pert_embedding_path : Optional[Path]
        Path to precomputed perturbation embedding file (for cross-dataset baselines)
    
    Returns
    -------
    Dict
        Dictionary with predictions, metrics, and other results
    """
    np.random.seed(seed)
    
    # Load data
    LOGGER.info(f"Loading data from {adata_path}")
    adata = sc.read_h5ad(adata_path)
    
    # Ensure ctrl is in training set
    if "ctrl" not in split_config.get("train", []):
        split_config["train"] = split_config.get("train", []) + ["ctrl"]
    
    # Only keep valid conditions
    valid_conditions = []
    for conditions in split_config.values():
        valid_conditions.extend(conditions)
    valid_conditions = list(set(valid_conditions))
    
    adata = adata[adata.obs['condition'].isin(valid_conditions)].copy()
    
    # Clean up condition names
    adata.obs['condition'] = adata.obs['condition'].astype('category')
    adata.obs['clean_condition'] = adata.obs['condition'].astype(str).str.replace(r'\+ctrl', '', regex=True)
    
    # Add training split labels
    training_map = {}
    for split_name, conditions in split_config.items():
        for cond in conditions:
            training_map[cond] = split_name
    
    adata.obs['training'] = adata.obs['condition'].map(training_map)
    
    # Set gene names as index if available
    if 'gene_name' in adata.var.columns:
        adata.var_names = adata.var['gene_name'].values
        adata.var = adata.var.set_index('gene_name', drop=False)
    
    # Compute baseline (mean control expression)
    ctrl_mask = adata.obs['condition'] == 'ctrl'
    if sparse.issparse(adata.X):
        baseline = np.array(adata[ctrl_mask].X.mean(axis=0)).flatten()
    else:
        baseline = adata[ctrl_mask].X.mean(axis=0)
    
    # Pseudobulk everything
    psce = pseudobulk_adata(adata, group_by=['condition', 'clean_condition', 'training'])
    
    # Compute expression change
    if sparse.issparse(psce.X):
        psce.X = psce.X.toarray()
    psce.layers['change'] = psce.X - baseline[np.newaxis, :]
    
    # Get training data
    train_mask = psce.obs['training'] == 'train'
    train_data = psce[train_mask].copy()
    
    # Get gene embeddings
    # Gene embeddings are: genes × embedding_dim
    if gene_embedding == "training_data":
        # PCA on training data: genes are features (columns), samples are perturbations (rows)
        # So we do PCA on the transpose: genes × samples -> genes × pca_dim
        pca = PCA(n_components=pca_dim, random_state=seed)
        gene_emb_train_only = pca.fit_transform(train_data.X.T)  # genes × pca_dim
        gene_emb_df = pd.DataFrame(gene_emb_train_only, index=train_data.var_names)
        # Reindex to include all genes (fill missing with zeros)
        gene_emb_df = gene_emb_df.reindex(psce.var_names, fill_value=0.0)
        gene_emb = gene_emb_df.values  # genes × pca_dim
        
    elif gene_embedding == "identity":
        gene_emb = np.eye(len(psce.var_names))
        pca_dim = len(psce.var_names)  # Update dimension for identity
        
    elif gene_embedding == "zero":
        gene_emb = np.zeros((len(psce.var_names), pca_dim))
        
    elif gene_embedding == "random":
        np.random.seed(seed)
        gene_emb = np.random.randn(len(psce.var_names), pca_dim)
        
    else:
        # Load from file
        gene_emb_df = pd.read_csv(gene_embedding, sep='\t', index_col=0)
        # R stores as embedding_dim × genes, so transpose to genes × embedding_dim
        if gene_emb_df.shape[0] < gene_emb_df.shape[1]:
            gene_emb_df = gene_emb_df.T
        # Reindex to match psce.var_names
        gene_emb_df = gene_emb_df.reindex(psce.var_names, fill_value=0.0)
        gene_emb = gene_emb_df.values
        pca_dim = gene_emb.shape[1]
    
    # Get perturbation embeddings
    # Perturbation embeddings are: embedding_dim × perturbations
    if pert_embedding == "training_data":
        # PCA on training data: perturbations are samples (rows), genes are features (columns)
        pca = PCA(n_components=pca_dim, random_state=seed)
        pert_emb = pca.fit_transform(train_data.X)  # perturbations × pca_dim
        pert_emb = pert_emb.T  # pca_dim × perturbations
        pert_emb_dict = {cond: pert_emb[:, i] 
                        for i, cond in enumerate(train_data.obs['clean_condition'].values)}
        
    elif pert_embedding == "identity":
        pert_emb_dict = {cond: np.eye(len(psce.obs))[i] 
                        for i, cond in enumerate(psce.obs['clean_condition'].values)}
        pca_dim = len(psce.obs)
        
    elif pert_embedding == "zero":
        pert_emb_dict = {cond: np.zeros(pca_dim) 
                        for cond in psce.obs['clean_condition'].values}
        
    elif pert_embedding == "random":
        np.random.seed(seed)
        pert_emb_dict = {cond: np.random.randn(pca_dim) 
                        for cond in psce.obs['clean_condition'].values}
        
    else:
        # Load from file (use pert_embedding_path if provided, otherwise use pert_embedding as path)
        emb_path = pert_embedding_path if pert_embedding_path else pert_embedding
        pert_emb_df = pd.read_csv(emb_path, sep='\t', index_col=0)
        # Format: embedding_dim × perturbations (columns are perturbations)
        pert_emb_dict = {col: pert_emb_df[col].values 
                        if col in pert_emb_df.columns else np.zeros(pert_emb_df.shape[0])
                        for col in psce.obs['clean_condition'].values}
        pca_dim = pert_emb_df.shape[0]
    
    # Ensure ctrl is in perturbation embeddings (set to zero vector)
    if 'ctrl' not in pert_emb_dict:
        pert_emb_dict['ctrl'] = np.zeros(pca_dim)
    
    # Match embeddings to training data
    train_pert_names = train_data.obs['clean_condition'].values
    train_gene_names = train_data.var_names.values
    
    # Find which perturbations in embedding are in training data
    pert_in_emb = [p for p in train_pert_names if p in pert_emb_dict]
    if len(pert_in_emb) <= 1:
        raise ValueError("Too few matches between clean_conditions and pert_embedding")
    
    # Find which genes in embedding are in training data
    gene_in_emb = [g for g in train_gene_names if g in psce.var_names]
    if len(gene_in_emb) <= 1:
        raise ValueError("Too few matches between gene names and gene_embedding")
    
    # Get matched embeddings for training
    gene_indices = [i for i, g in enumerate(psce.var_names) if g in gene_in_emb]
    gene_emb_train = gene_emb[gene_indices, :]  # matched_genes × embedding_dim
    
    # Get training data expression changes: Y is genes × perturbations
    # Find training perturbations that are in the embedding
    train_pert_idx = []
    train_pert_cond = []
    for i, cond in enumerate(psce.obs['clean_condition'].values):
        if cond in pert_in_emb and psce.obs['training'].values[i] == 'train':
            train_pert_idx.append(i)
            train_pert_cond.append(cond)
    
    # Get Y matrix: genes × perturbations
    Y = psce.layers['change'][np.ix_(train_pert_idx, gene_indices)].T  # genes × perturbations
    
    # Reorder pert_emb_train to match Y columns
    pert_emb_train_ordered = np.array([pert_emb_dict[p] for p in train_pert_cond]).T
    
    # Ensure dimensions match
    assert Y.shape[0] == len(gene_indices), f"Y rows {Y.shape[0]} != genes {len(gene_indices)}"
    assert Y.shape[1] == len(train_pert_cond), f"Y cols {Y.shape[1]} != train perts {len(train_pert_cond)}"
    assert gene_emb_train.shape[0] == len(gene_indices), "Gene embedding dimension mismatch"
    assert pert_emb_train_ordered.shape[1] == len(train_pert_cond), "Pert embedding dimension mismatch"
    assert gene_emb_train.shape[1] == pert_emb_train_ordered.shape[0], "Embedding dimension mismatch between gene and pert embeddings"
    
    # Solve for coefficients: Y = A * K * B
    # Y: genes × perturbations
    # A: genes × embedding_dim (gene_emb_train)
    # K: embedding_dim × embedding_dim (to be solved)
    # B: embedding_dim × perturbations (pert_emb_train_ordered)
    coefs = solve_y_axb(Y=Y, A=gene_emb_train, B=pert_emb_train_ordered,
                        A_ridge=ridge_penalty, B_ridge=ridge_penalty)
    
    # Get embeddings for all perturbations
    all_pert_names = psce.obs['clean_condition'].values
    pert_emb_all = np.array([pert_emb_dict.get(p, np.zeros(pca_dim)) 
                            for p in all_pert_names]).T  # embedding_dim × n_all_pert
    
    # Make predictions for all perturbations
    # pred = A * K * B + center + baseline
    # gene_emb_train: matched_genes × embedding_dim
    # coefs['K']: embedding_dim × embedding_dim
    # pert_emb_all: embedding_dim × n_all_pert
    pred = (gene_emb_train @ coefs['K'] @ pert_emb_all + 
            coefs['center'][:, np.newaxis] + 
            baseline[gene_indices, np.newaxis])
    
    # pred is now: matched_genes × n_all_pert
    
    # Extract test predictions
    test_mask = psce.obs['training'] == 'test'
    test_indices = np.where(test_mask)[0]
    
    if len(test_indices) > 0:
        # Get test predictions: matched_genes × n_test
        Y_pred_test = pred[:, test_indices]
        
        # Get test true values
        Y_test_true = psce.layers['change'][test_indices, :][:, gene_indices].T  # matched_genes × n_test
        
        # Compute metrics for each test perturbation
        from shared.metrics import compute_metrics
        
        test_pert_names = psce.obs['clean_condition'].values[test_indices]
        metrics = {}
        for i, pert_name in enumerate(test_pert_names):
            y_true = Y_test_true[:, i]
            y_pred = Y_pred_test[:, i]
            metrics[pert_name] = compute_metrics(y_true, y_pred)
    else:
        Y_pred_test = np.array([]).reshape(pred.shape[0], 0)
        metrics = {}
    
    return {
        "predictions": Y_pred_test,
        "metrics": metrics,
        "K": coefs['K'],
        "A": gene_emb_train,
        "B_train": pert_emb_train_ordered,
        "gene_names": [psce.var_names[i] for i in gene_indices],
        "baseline_type": "paper_implementation",
    }


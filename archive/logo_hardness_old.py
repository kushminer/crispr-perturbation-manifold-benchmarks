"""
Leave-one-gene-out (LOGO) evaluation with hardness bins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from .linear_model import fit_linear_model, predict_perturbation
from .metrics import compute_metrics

LOGGER = logging.getLogger(__name__)


@dataclass
class LogoResult:
    perturbation: str
    hardness_bin: str
    cluster_blocked: bool
    metrics: Dict[str, float]
    split_type: str = "logo"


def results_to_dataframe(results: List[LogoResult]) -> pd.DataFrame:
    records = []
    for res in results:
        row = {
            "perturbation": res.perturbation,
            "hardness_bin": res.hardness_bin,
            "cluster_blocked": res.cluster_blocked,
            "split_type": res.split_type,
        }
        row.update(res.metrics)
        records.append(row)
    return pd.DataFrame(records)


def compute_similarity_matrix(expression: pd.DataFrame) -> pd.DataFrame:
    sims = cosine_similarity(expression.values)
    return pd.DataFrame(sims, index=expression.index, columns=expression.index)


def compute_hardness_value(
    similarities: pd.Series,
    method: str = "mean",
    k_farthest: int = 10,
) -> float:
    """
    Compute hardness value using different metrics.
    
    Args:
        similarities: Series of similarity values to other perturbations
        method: Method to use ("mean", "min", "median", "k_farthest")
        k_farthest: Number of farthest neighbors for k_farthest method
        
    Returns:
        Hardness value (lower = harder/more dissimilar)
    """
    sims = similarities.dropna()
    if sims.empty:
        return np.nan
    
    if method == "mean":
        return sims.mean()
    elif method == "min":
        return sims.min()
    elif method == "median":
        return sims.median()
    elif method == "k_farthest":
        if len(sims) < k_farthest:
            return sims.min()
        # Get k smallest similarities (most dissimilar)
        k_smallest = sims.nsmallest(k_farthest)
        return k_smallest.mean()
    else:
        raise ValueError(f"Unknown hardness method: {method}. Choose from: mean, min, median, k_farthest")


def assign_hardness_bins(
    similarities: pd.Series,
    quantiles: Iterable[float],
    method: str = "mean",
    k_farthest: int = 10,
) -> str:
    """
    Assign hardness bin based on similarity distribution.
    
    Args:
        similarities: Series of similarity values to other perturbations
        quantiles: Quantile thresholds for binning (e.g., [0.33, 0.66])
        method: Method to compute hardness ("mean", "min", "median", "k_farthest")
        k_farthest: Number of farthest neighbors for k_farthest method
        
    Returns:
        Hardness bin name: "near", "mid", "far", or "unknown"
    """
    sims = similarities.dropna()
    if sims.empty:
        return "unknown"

    # Compute hardness value using selected method
    value = compute_hardness_value(sims, method=method, k_farthest=k_farthest)
    
    if np.isnan(value):
        return "unknown"

    # Compute quantile thresholds from all similarity values
    bins = sorted(float(q) for q in quantiles)
    thresholds = [sims.quantile(q) for q in bins]

    if not thresholds:
        return "near"
    if value <= thresholds[0]:
        return "near"
    if len(thresholds) == 1 or value <= thresholds[1]:
        return "mid"
    return "far"


def build_cluster_blocks(expression: pd.DataFrame, n_clusters: int, seed: int) -> Dict[str, int]:
    reducer = PCA(n_components=min(10, expression.shape[1], expression.shape[0]), random_state=seed)
    reduced = reducer.fit_transform(expression.values)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(reduced)
    return {pert: int(label) for pert, label in zip(expression.index, labels)}


def solve_linear_model(
    train_matrix: np.ndarray,
    train_targets: np.ndarray,
    alpha: float,
    center: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    y = train_targets
    X = train_matrix
    if center:
        y_mean = y.mean(axis=0, keepdims=True)
        y_centered = y - y_mean
    else:
        y_mean = np.zeros((1, y.shape[1]))
        y_centered = y

    Xt = X.T
    gram = Xt @ X + alpha * np.eye(X.shape[1])
    coef = np.linalg.solve(gram, Xt @ y_centered)
    return coef, y_mean


def evaluate_logo_fold(
    expression: pd.DataFrame,
    holdout_idx: int,
    alpha: float,
    pca_dim: int,
    seed: int,
    similarity_matrix: pd.DataFrame,
    hardness_bins: Iterable[float],
    cluster_map: Dict[str, int] | None = None,
    blocked: bool = False,
    hardness_method: str = "mean",
    k_farthest: int = 10,
    hardness_thresholds: List[float] | None = None,
) -> LogoResult:
    holdout_name = expression.index[holdout_idx]

    train_mask = np.ones(expression.shape[0], dtype=bool)
    train_mask[holdout_idx] = False

    if blocked and cluster_map:
        target_cluster = cluster_map[holdout_name]
        for i, pert in enumerate(expression.index):
            if cluster_map[pert] == target_cluster:
                train_mask[i] = False

    train_data = expression.iloc[train_mask]
    test_vector = expression.iloc[holdout_idx].to_numpy()

    metrics = {"pearson_r": np.nan, "spearman_rho": np.nan, "mse": np.nan, "mae": np.nan}

    if train_data.empty:
        LOGGER.warning("Skipping %s: no training data left after blocking.", holdout_name)
    else:
        train_matrix = train_data.to_numpy()
        model = fit_linear_model(
            train_matrix=train_matrix,
            pca_dim=min(pca_dim, train_matrix.shape[0], train_matrix.shape[1]),
            ridge_penalty=alpha,
            seed=seed,
        )
        pred = predict_perturbation(model, test_vector)
        metrics = compute_metrics(test_vector, pred)

    # Use pre-computed thresholds if provided, otherwise compute on-the-fly
    if hardness_thresholds is not None:
        similarities = similarity_matrix.loc[holdout_name].drop(holdout_name)
        hardness_val = compute_hardness_value(
            similarities, method=hardness_method, k_farthest=k_farthest
        )
        if np.isnan(hardness_val):
            hardness = "unknown"
        elif hardness_val <= hardness_thresholds[0]:
            hardness = "near"
        elif len(hardness_thresholds) == 1 or hardness_val <= hardness_thresholds[1]:
            hardness = "mid"
        else:
            hardness = "far"
    else:
        hardness = assign_hardness_bins(
            similarity_matrix.loc[holdout_name].drop(holdout_name),
            hardness_bins,
            method=hardness_method,
            k_farthest=k_farthest,
        )
    cluster_blocked = bool(blocked and cluster_map)
    return LogoResult(perturbation=holdout_name, hardness_bin=hardness, cluster_blocked=cluster_blocked, metrics=metrics)


def run_logo_evaluation(
    expression: pd.DataFrame,
    hardness_bins: Iterable[float],
    alpha: float,
    pca_dim: int,
    block_clusters: bool = False,
    cluster_size: int | None = None,
    seed: int = 1,
    hardness_method: str = "mean",
    k_farthest: int = 10,
) -> List[LogoResult]:
    similarity_matrix = compute_similarity_matrix(expression)
    cluster_map = None
    if block_clusters and cluster_size:
        n_clusters = max(2, expression.shape[0] // cluster_size)
        cluster_map = build_cluster_blocks(expression, n_clusters, seed)
        LOGGER.info("Built cluster map with %d clusters for blocking.", n_clusters)

    # First pass: compute all hardness values
    hardness_values = {}
    for idx in range(expression.shape[0]):
        holdout_name = expression.index[idx]
        similarities = similarity_matrix.loc[holdout_name].drop(holdout_name)
        hardness_val = compute_hardness_value(
            similarities, method=hardness_method, k_farthest=k_farthest
        )
        hardness_values[holdout_name] = hardness_val
    
    # Compute quantile thresholds from hardness values (not similarities)
    hardness_series = pd.Series(hardness_values)
    bins = sorted(float(q) for q in hardness_bins)
    hardness_thresholds = [hardness_series.quantile(q) for q in bins] if len(bins) > 0 else []
    
    LOGGER.info(
        "Hardness method: %s, thresholds: %s (from %d values)",
        hardness_method,
        hardness_thresholds,
        len(hardness_values),
    )

    # Second pass: run evaluations and assign bins using pre-computed thresholds
    results: List[LogoResult] = []
    for idx in range(expression.shape[0]):
        holdout_name = expression.index[idx]
        
        # Assign bin using pre-computed thresholds
        hardness_val = hardness_values[holdout_name]
        if np.isnan(hardness_val) or not hardness_thresholds:
            hardness_bin = "unknown"
        elif hardness_val <= hardness_thresholds[0]:
            hardness_bin = "near"
        elif len(hardness_thresholds) == 1 or hardness_val <= hardness_thresholds[1]:
            hardness_bin = "mid"
        else:
            hardness_bin = "far"
        
        # Run evaluation
        train_mask = np.ones(expression.shape[0], dtype=bool)
        train_mask[idx] = False

        if block_clusters and cluster_map:
            target_cluster = cluster_map[holdout_name]
            for i, pert in enumerate(expression.index):
                if cluster_map[pert] == target_cluster:
                    train_mask[i] = False

        train_data = expression.iloc[train_mask]
        test_vector = expression.iloc[idx].to_numpy()

        metrics = {"pearson_r": np.nan, "spearman_rho": np.nan, "mse": np.nan, "mae": np.nan}

        if train_data.empty:
            LOGGER.warning("Skipping %s: no training data left after blocking.", holdout_name)
        else:
            train_matrix = train_data.to_numpy()
            model = fit_linear_model(
                train_matrix=train_matrix,
                pca_dim=min(pca_dim, train_matrix.shape[0], train_matrix.shape[1]),
                ridge_penalty=alpha,
            seed=seed,
            )
            pred = predict_perturbation(model, test_vector)
            metrics = compute_metrics(test_vector, pred)

        cluster_blocked = bool(block_clusters and cluster_map)
        results.append(
            LogoResult(
                perturbation=holdout_name,
                hardness_bin=hardness_bin,
                cluster_blocked=cluster_blocked,
                metrics=metrics,
        )
        )
    return results


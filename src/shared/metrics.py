"""
Metric computations used across evaluation tasks.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _mask_finite(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & np.isfinite(b)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute Pearson's r, Spearman's rho, mean squared error, mean absolute error, and L2 (Euclidean distance).
    
    L2 is computed as: sqrt(sum((y_true - y_pred)^2))
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    mask = _mask_finite(y_true, y_pred)
    if not np.any(mask):
        return {k: np.nan for k in ("pearson_r", "spearman_rho", "mse", "mae", "l2")}

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    pearson = pearsonr(y_true, y_pred)[0] if y_true.size > 1 else np.nan
    spearman = spearmanr(y_true, y_pred)[0] if y_true.size > 1 else np.nan
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    l2 = float(np.sqrt(np.sum((y_true - y_pred) ** 2)))  # Euclidean distance

    return {
        "pearson_r": float(pearson),
        "spearman_rho": float(spearman),
        "mse": mse,
        "mae": mae,
        "l2": l2,
    }


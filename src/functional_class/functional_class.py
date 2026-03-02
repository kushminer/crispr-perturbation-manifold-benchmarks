"""
Functional-class holdout evaluation.

This module implements multi-class holdout evaluation, where each functional class
is held out from training in turn and evaluated separately. This is different from
LOGO (Leave-One-GO-Out) evaluation, which isolates a specific functional class
(e.g., Transcription) as the test set for all baselines.

Differences:
- Multi-class holdout (this module): Iterates over all functional classes, holding
  out each class one at a time. Useful for comprehensive evaluation across all classes.
  
- LOGO evaluation (goal_3_prediction.functional_class_holdout.logo): Holds out a
  single specified functional class (e.g., Transcription) as the test set and trains
  on all other classes. Runs all 8 baselines for comparison (scGPT vs Random, etc.).
  Designed to test biological extrapolation for specific classes.

For LOGO evaluation with all baselines, use:
    goal_3_prediction.functional_class_holdout.logo.run_logo_evaluation()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from shared.linear_model import fit_linear_model, predict_perturbation, solve_y_axb
from shared.metrics import compute_metrics
from shared.io import load_annotations

LOGGER = logging.getLogger(__name__)


@dataclass
class ClassHoldoutResult:
    perturbation: str
    functional_class: str
    metrics: Dict[str, float]
    split_type: str = "class_holdout"


def run_class_holdout(
    expression: pd.DataFrame,
    annotations: pd.DataFrame,
    ridge_penalty: float,
    pca_dim: int,
    min_class_size: int = 3,
    seed: int = 1,
) -> List[ClassHoldoutResult]:
    """
    Evaluate multi-class holdout by removing each functional class from the training set.
    
    This function iterates over all functional classes that meet the minimum size
    threshold, holding out each class one at a time for evaluation. This is a
    comprehensive evaluation across all classes.
    
    For LOGO evaluation (isolating a single specific class with all baselines),
    use: goal_3_prediction.functional_class_holdout.logo.run_logo_evaluation()
    
    Args:
        expression: Expression matrix (perturbations × genes)
        annotations: DataFrame with 'target' and 'class' columns
        ridge_penalty: Ridge regression regularization
        pca_dim: PCA dimensionality
        min_class_size: Minimum perturbations per class to evaluate
        seed: Random seed
    
    Returns:
        List of ClassHoldoutResult objects, one per perturbation in held-out classes
    """
    annotations = annotations.copy()
    annotations["target"] = annotations["target"].astype(str)
    available_targets = set(expression.index)
    
    # Log class distribution
    class_counts = annotations.groupby("class")["target"].nunique()
    LOGGER.info("Found %d unique classes in annotations", len(class_counts))
    LOGGER.info("Class size distribution: min=%d, median=%.1f, max=%d", 
                class_counts.min(), class_counts.median(), class_counts.max())
    
    valid_classes = class_counts.loc[class_counts >= min_class_size].index
    LOGGER.info("Evaluating %d classes with size ≥ %d", len(valid_classes), min_class_size)
    
    if len(valid_classes) == 0:
        LOGGER.warning(
            "No classes meet minimum size threshold (%d). "
            "Consider lowering 'functional_min_class_size' in config or enriching annotations.",
            min_class_size
        )
        return []

    results: List[ClassHoldoutResult] = []
    total_classes = len(valid_classes)
    
    for idx, cls in enumerate(valid_classes, 1):
        LOGGER.info("Evaluating class %d/%d: %s", idx, total_classes, cls)
        
        holdout_targets = [
            t for t in annotations.loc[annotations["class"] == cls, "target"].unique()
            if t in available_targets
        ]
        train_targets = expression.index.difference(holdout_targets)

        LOGGER.info("  Holdout: %d perturbations, Training: %d perturbations", 
                   len(holdout_targets), len(train_targets))

        if len(train_targets) < 2:
            LOGGER.warning("  Skipping class %s due to insufficient training data.", cls)
            continue

        train_matrix = expression.loc[train_targets].to_numpy()
        model = fit_linear_model(
            train_matrix=train_matrix,
            pca_dim=min(pca_dim, train_matrix.shape[0], train_matrix.shape[1]),
            ridge_penalty=ridge_penalty,
            seed=seed,
        )

        class_results = []
        for perturbation in holdout_targets:
            vector = expression.loc[perturbation].to_numpy()
            prediction = predict_perturbation(model, vector)
            metrics = compute_metrics(vector, prediction)
            class_results.append(
                ClassHoldoutResult(
                    perturbation=perturbation,
                    functional_class=cls,
                    metrics=metrics,
                )
            )
        
        results.extend(class_results)
        
        # Log summary for this class
        if class_results:
            mean_r = sum(r.metrics["pearson_r"] for r in class_results) / len(class_results)
            LOGGER.info("  Completed: %d perturbations evaluated, mean r=%.4f", 
                       len(class_results), mean_r)

    LOGGER.info("Functional-class evaluation complete: %d total results across %d classes",
                len(results), len(valid_classes))
    
    return results


def class_results_to_dataframe(results: List[ClassHoldoutResult]) -> pd.DataFrame:
    rows = []
    for res in results:
        row = {
            "perturbation": res.perturbation,
            "class": res.functional_class,
            "split_type": res.split_type,
        }
        row.update(res.metrics)
        rows.append(row)
    return pd.DataFrame(rows)

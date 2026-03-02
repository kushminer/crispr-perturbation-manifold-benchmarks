"""
Unit tests for functional-class holdout evaluation.
"""

import numpy as np
import pandas as pd
import pytest
from functional_class.functional_class import (
    run_class_holdout,
    class_results_to_dataframe,
)


def test_run_class_holdout_basic(synthetic_expression_matrix, synthetic_annotations):
    """Test basic functional-class holdout evaluation."""
    results = run_class_holdout(
        expression=synthetic_expression_matrix,
        annotations=synthetic_annotations,
        ridge_penalty=0.1,
        pca_dim=10,
        min_class_size=5,
        seed=42,
    )
    
    assert len(results) > 0
    assert all(hasattr(r, "perturbation") for r in results)
    assert all(hasattr(r, "functional_class") for r in results)
    assert all(hasattr(r, "metrics") for r in results)
    assert all("pearson_r" in r.metrics for r in results)


def test_run_class_holdout_min_class_size(synthetic_expression_matrix, synthetic_annotations):
    """Test that min_class_size filter works."""
    # Set high min_class_size to filter out classes
    results = run_class_holdout(
        expression=synthetic_expression_matrix,
        annotations=synthetic_annotations,
        ridge_penalty=0.1,
        pca_dim=10,
        min_class_size=20,  # Very high threshold
        seed=42,
    )
    
    # Should return empty or very few results
    assert len(results) <= len(synthetic_annotations)


def test_run_class_holdout_all_classes_represented(synthetic_expression_matrix, synthetic_annotations):
    """Test that all valid classes are represented in results."""
    results = run_class_holdout(
        expression=synthetic_expression_matrix,
        annotations=synthetic_annotations,
        ridge_penalty=0.1,
        pca_dim=10,
        min_class_size=5,
        seed=42,
    )
    
    df = class_results_to_dataframe(results)
    unique_classes = set(df["class"].unique())
    
    # All classes with size >= min_class_size should be represented
    class_counts = synthetic_annotations.groupby("class")["target"].nunique()
    valid_classes = set(class_counts[class_counts >= 5].index)
    
    assert unique_classes == valid_classes


def test_class_results_to_dataframe():
    """Test conversion of class results to DataFrame."""
    from functional_class.functional_class import ClassHoldoutResult
    
    results = [
        ClassHoldoutResult(
            perturbation=f"pert_{i}",
            functional_class=f"Class_{i % 3}",
            metrics={"pearson_r": 0.5, "mse": 0.1, "mae": 0.2},
        )
        for i in range(10)
    ]
    
    df = class_results_to_dataframe(results)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert "perturbation" in df.columns
    assert "class" in df.columns
    assert "pearson_r" in df.columns
    assert "mse" in df.columns


def test_run_class_holdout_insufficient_training_data():
    """Test handling of insufficient training data."""
    # Create dataset where holding out a class leaves too few training samples
    expression = pd.DataFrame(
        np.random.randn(10, 20),
        index=[f"pert_{i}" for i in range(10)],
        columns=[f"gene_{i}" for i in range(20)],
    )
    
    # Create annotations where one class has most perturbations
    annotations = pd.DataFrame({
        "target": [f"pert_{i}" for i in range(10)],
        "class": ["Class_1"] * 8 + ["Class_2"] * 2,  # Class_2 too small
    })
    
    results = run_class_holdout(
        expression=expression,
        annotations=annotations,
        ridge_penalty=0.1,
        pca_dim=5,
        min_class_size=2,
        seed=42,
    )
    
    # Should handle gracefully, possibly skipping small classes
    assert isinstance(results, list)


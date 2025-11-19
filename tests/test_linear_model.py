"""
Unit tests for linear model fitting and prediction.
"""

import numpy as np
import pytest
from shared.linear_model import fit_linear_model, predict_perturbation


def test_fit_linear_model_basic():
    """Test basic linear model fitting."""
    # Create simple training data
    n_samples = 20
    n_genes = 50
    train_matrix = np.random.randn(n_samples, n_genes)
    
    model = fit_linear_model(
        train_matrix=train_matrix,
        pca_dim=10,
        ridge_penalty=0.1,
        seed=42,
    )
    
    assert hasattr(model, "gene_pca")
    assert hasattr(model, "pert_pca")
    assert hasattr(model, "coef")
    assert hasattr(model, "center")
    assert model.gene_embeddings.shape == (n_genes, 10)
    assert model.pert_embeddings.shape == (n_samples, 10)


def test_fit_linear_model_pca_dim_larger_than_data():
    """Test that PCA dim is capped at data dimensions."""
    n_samples = 5
    n_genes = 10
    train_matrix = np.random.randn(n_samples, n_genes)
    
    model = fit_linear_model(
        train_matrix=train_matrix,
        pca_dim=20,  # Larger than both dimensions
        ridge_penalty=0.1,
        seed=42,
    )
    
    # PCA dim should be capped at min(n_samples, n_genes)
    expected_dim = min(n_samples, n_genes)
    assert model.gene_embeddings.shape[1] <= expected_dim
    assert model.pert_embeddings.shape[1] <= expected_dim


def test_predict_perturbation():
    """Test perturbation prediction."""
    n_samples = 20
    n_genes = 50
    train_matrix = np.random.randn(n_samples, n_genes)
    
    model = fit_linear_model(
        train_matrix=train_matrix,
        pca_dim=10,
        ridge_penalty=0.1,
        seed=42,
    )
    
    # Create target vector
    target_vector = np.random.randn(n_genes)
    
    prediction = predict_perturbation(model, target_vector)
    
    assert prediction.shape == (n_genes,)
    assert not np.any(np.isnan(prediction))
    assert not np.any(np.isinf(prediction))


def test_predict_perturbation_shape_mismatch():
    """Test that shape mismatch raises error."""
    n_samples = 20
    n_genes = 50
    train_matrix = np.random.randn(n_samples, n_genes)
    
    model = fit_linear_model(
        train_matrix=train_matrix,
        pca_dim=10,
        ridge_penalty=0.1,
        seed=42,
    )
    
    # Wrong size target vector
    target_vector = np.random.randn(n_genes + 10)
    
    with pytest.raises((ValueError, AssertionError)):
        predict_perturbation(model, target_vector)


def test_fit_linear_model_reproducibility():
    """Test that model fitting is reproducible with same seed."""
    n_samples = 20
    n_genes = 50
    train_matrix = np.random.randn(n_samples, n_genes)
    
    model1 = fit_linear_model(
        train_matrix=train_matrix,
        pca_dim=10,
        ridge_penalty=0.1,
        seed=42,
    )
    
    model2 = fit_linear_model(
        train_matrix=train_matrix,
        pca_dim=10,
        ridge_penalty=0.1,
        seed=42,
    )
    
    # Check that models are identical
    np.testing.assert_array_almost_equal(model1.gene_embeddings, model2.gene_embeddings)
    np.testing.assert_array_almost_equal(model1.pert_embeddings, model2.pert_embeddings)
    np.testing.assert_array_almost_equal(model1.coef, model2.coef)


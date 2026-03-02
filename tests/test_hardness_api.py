#!/usr/bin/env python3
"""
Unit tests for hardness_api.py
"""

import numpy as np
import pytest

from goal_1_similarity.hardness_api import (
    compute_multiple_targets_similarity,
    compute_target_similarity,
)


def test_identical_target_train():
    """Test that identical target and train embeddings give similarity = 1, hardness = 0."""
    # Create a simple embedding
    embedding = np.array([1.0, 2.0, 3.0])
    
    # Same embedding in training set
    train_embeddings = embedding.reshape(1, -1)
    
    result = compute_target_similarity(embedding, train_embeddings)
    
    assert result["max_sim"] == pytest.approx(1.0, abs=1e-6)
    assert result["hardness_max"] == pytest.approx(0.0, abs=1e-6)
    assert result["mean_topk"][1] == pytest.approx(1.0, abs=1e-6)
    assert result["hardness_k"][1] == pytest.approx(0.0, abs=1e-6)


def test_orthogonal_vectors():
    """Test that orthogonal vectors give similarity = 0."""
    # Create orthogonal vectors
    target = np.array([1.0, 0.0, 0.0])
    train = np.array([[0.0, 1.0, 0.0]])  # Orthogonal to target
    
    result = compute_target_similarity(target, train)
    
    assert result["max_sim"] == pytest.approx(0.0, abs=1e-6)
    assert result["hardness_max"] == pytest.approx(1.0, abs=1e-6)


def test_opposite_vectors():
    """Test that opposite vectors give negative similarity (cosine similarity)."""
    target = np.array([1.0, 0.0, 0.0])
    train = np.array([[-1.0, 0.0, 0.0]])  # Opposite direction
    
    result = compute_target_similarity(target, train)
    
    assert result["max_sim"] == pytest.approx(-1.0, abs=1e-6)
    assert result["hardness_max"] == pytest.approx(2.0, abs=1e-6)  # 1 - (-1) = 2


def test_multiple_training_perturbations():
    """Test with multiple training perturbations."""
    target = np.array([1.0, 0.0, 0.0])
    train = np.array([
        [1.0, 0.0, 0.0],  # Identical
        [0.5, 0.5, 0.0],  # Somewhat similar
        [0.0, 1.0, 0.0],  # Orthogonal
    ])
    
    result = compute_target_similarity(target, train, k_values=[1, 2, 3])
    
    assert result["max_sim"] == pytest.approx(1.0, abs=1e-6)
    assert result["mean_topk"][1] == pytest.approx(1.0, abs=1e-6)
    # mean_topk[2] should be average of top 2, which includes the identical one
    assert result["mean_topk"][2] >= 0.5  # At least as good as average
    # mean_topk[3] might be lower than [2] if the third best is worse than average
    # Just check that all values are defined
    assert result["mean_topk"][3] is not None
    assert len(result["all_sims"]) == 3


def test_multiple_targets():
    """Test compute_multiple_targets_similarity."""
    # targets should be (n_targets, d) = (2, 3)
    targets = np.array([
        [1.0, 0.0, 0.0],  # target1
        [0.0, 1.0, 0.0],  # target2
    ])
    # train should be (n_train, d) = (3, 3)
    train = np.array([
        [1.0, 0.0, 0.0],  # train[0]
        [0.0, 1.0, 0.0],  # train[1]
        [0.0, 0.0, 1.0],  # train[2]
    ])
    target_names = ["target1", "target2"]
    
    results = compute_multiple_targets_similarity(targets, train, target_names, k_values=[1, 2])
    
    assert len(results) == 2
    assert "target1" in results
    assert "target2" in results
    
    # target1 should be most similar to train[0]
    assert results["target1"]["max_sim"] == pytest.approx(1.0, abs=1e-6)
    # target2 should be most similar to train[1]
    assert results["target2"]["max_sim"] == pytest.approx(1.0, abs=1e-6)


def test_different_shapes():
    """Test that function handles different input shapes correctly."""
    target = np.array([1.0, 2.0, 3.0])  # 1D
    train = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]])  # 2D
    
    result1 = compute_target_similarity(target, train)
    
    # Test with transposed (d, n_train) format
    train_T = train.T  # (3, 2)
    result2 = compute_target_similarity(target, train_T)
    
    # Results should be the same
    assert result1["max_sim"] == pytest.approx(result2["max_sim"], abs=1e-6)


def test_k_values_larger_than_train():
    """Test that k values larger than training set size are handled correctly."""
    target = np.array([1.0, 0.0, 0.0])
    train = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # Only 2 training perturbations
    
    result = compute_target_similarity(target, train, k_values=[1, 5, 10])
    
    # Should handle k=5, k=10 gracefully (use all available)
    assert result["mean_topk"][1] is not None
    assert result["mean_topk"][5] is not None
    assert result["mean_topk"][10] is not None
    # All should use mean of available perturbations
    assert result["mean_topk"][5] == result["mean_topk"][10]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

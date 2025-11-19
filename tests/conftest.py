"""
Pytest configuration and shared fixtures for unit tests.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile
import json


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def synthetic_expression_matrix(random_seed):
    """Generate synthetic expression matrix for testing."""
    n_perturbations = 50
    n_genes = 100
    
    # Generate random expression data
    data = np.random.randn(n_perturbations, n_genes)
    perturbations = [f"pert_{i}" for i in range(n_perturbations)]
    genes = [f"gene_{i}" for i in range(n_genes)]
    
    return pd.DataFrame(data, index=perturbations, columns=genes)


@pytest.fixture
def synthetic_annotations():
    """Generate synthetic functional class annotations."""
    n_perturbations = 50
    n_classes = 5
    
    # Create balanced classes
    annotations = []
    for i in range(n_perturbations):
        class_idx = i % n_classes
        annotations.append({
            "target": f"pert_{i}",
            "class": f"Class_{class_idx + 1}"
        })
    
    return pd.DataFrame(annotations)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_predictions_json(temp_dir):
    """Create sample predictions JSON file."""
    predictions = {
        "pert_0": [0.1, 0.2, 0.3, 0.4, 0.5],
        "pert_1": [0.2, 0.3, 0.4, 0.5, 0.6],
        "ctrl": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    gene_names = ["gene_0", "gene_1", "gene_2", "gene_3", "gene_4"]
    
    pred_path = temp_dir / "predictions.json"
    gene_path = temp_dir / "gene_names.json"
    
    with open(pred_path, "w") as f:
        json.dump(predictions, f)
    
    with open(gene_path, "w") as f:
        json.dump(gene_names, f)
    
    return pred_path, gene_path


@pytest.fixture
def sample_logo_results():
    """Create sample LOGO results DataFrame."""
    return pd.DataFrame({
        "perturbation": [f"pert_{i}" for i in range(10)],
        "hardness_bin": ["near"] * 3 + ["mid"] * 4 + ["far"] * 3,
        "cluster_blocked": [False] * 10,
        "pearson_r": np.random.rand(10),
        "spearman_rho": np.random.rand(10),
        "mse": np.random.rand(10),
        "mae": np.random.rand(10),
    })


@pytest.fixture
def sample_class_results():
    """Create sample functional-class results DataFrame."""
    return pd.DataFrame({
        "perturbation": [f"pert_{i}" for i in range(10)],
        "class": ["Class_1"] * 5 + ["Class_2"] * 5,
        "split_type": ["class_holdout"] * 10,
        "pearson_r": np.random.rand(10),
        "spearman_rho": np.random.rand(10),
        "mse": np.random.rand(10),
        "mae": np.random.rand(10),
    })


#!/usr/bin/env python3
"""
Test script to verify fresh environment setup.
Run this after: pip install -r requirements.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing core imports...")
    
    try:
        # Core dependencies
        import numpy as np
        import pandas as pd
        import scipy
        import sklearn
        import anndata as ad
        import yaml
        import matplotlib
        import seaborn as sns
        import umap
        import torch
        print("  ✓ Core dependencies imported")
    except ImportError as e:
        print(f"  ✗ Failed to import core dependency: {e}")
        return False
    
    try:
        # Core framework modules
        from core.config import ExperimentConfig, load_config
        from core.io import load_expression_dataset, load_annotations
        from core.metrics import compute_metrics
        from core.linear_model import solve_y_axb
        print("  ✓ Core framework modules imported")
    except ImportError as e:
        print(f"  ✗ Failed to import core module: {e}")
        return False
    
    try:
        # Embedding modules
        from embeddings.registry import get, load
        from embeddings.base import EmbeddingResult
        print("  ✓ Embedding modules imported")
    except ImportError as e:
        print(f"  ✗ Failed to import embedding module: {e}")
        return False
    
    try:
        # Evaluation modules
        from functional_class.functional_class import run_class_holdout
        print("  ✓ Evaluation modules imported")
    except ImportError as e:
        print(f"  ✗ Failed to import evaluation module: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality with synthetic data."""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        import pandas as pd
        from core.linear_model import solve_y_axb
        
        # Create synthetic data
        n_genes = 100
        n_perts = 50
        d = 10
        
        Y = np.random.randn(n_genes, n_perts)
        A = np.random.randn(n_genes, d)
        B = np.random.randn(d, n_perts)
        
        # Test solve_y_axb
        result = solve_y_axb(Y, A, B, ridge_penalty=0.1)
        assert "K" in result
        assert result["K"].shape == (d, d)
        print("  ✓ Linear model solver works")
        
        # Test metrics
        from core.metrics import compute_metrics
        y_true = np.random.randn(100)
        y_pred = y_true + 0.1 * np.random.randn(100)
        metrics = compute_metrics(y_true, y_pred)
        assert isinstance(metrics, dict)
        assert "pearson_r" in metrics
        print("  ✓ Metrics computation works")
        
        return True
    except Exception as e:
        print(f"  ✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embedding_registry():
    """Test embedding registry."""
    print("\nTesting embedding registry...")
    
    try:
        from embeddings.registry import get, load
        
        # Check that loaders are registered
        available = ["gears_go", "pca_perturbation", "scgpt_gene", "scfoundation_gene"]
        for name in available:
            loader = get(name)
            assert loader is not None, f"Loader {name} not found"
        print("  ✓ Embedding loaders registered")
        
        return True
    except Exception as e:
        print(f"  ✗ Embedding registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        from core.config import load_config
        from pathlib import Path
        
        # Check if a config file exists
        config_path = Path("configs/config_adamson.yaml")
        if config_path.exists():
            config = load_config(config_path)
            assert config is not None
            print("  ✓ Configuration loading works")
        else:
            print("  ⚠ Config file not found (skipping)")
        
        return True
    except Exception as e:
        print(f"  ✗ Configuration loading test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Fresh Environment Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test basic functionality
    results.append(("Basic Functionality", test_basic_functionality()))
    
    # Test embedding registry
    results.append(("Embedding Registry", test_embedding_registry()))
    
    # Test config loading
    results.append(("Configuration Loading", test_config_loading()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed! Fresh environment is working.")
        return 0
    else:
        print("✗ Some tests failed. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())


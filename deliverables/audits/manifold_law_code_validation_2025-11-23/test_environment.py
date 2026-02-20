#!/usr/bin/env python3
"""
Quick environment test to verify all imports and basic functionality work.
"""

import sys
from pathlib import Path

# Add src to path
base_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(base_dir / "src"))

def test_imports():
    """Test that all key modules can be imported."""
    print("Testing imports...")
    
    try:
        from goal_2_baselines.baseline_runner import run_all_baselines
        print("  ✓ goal_2_baselines.baseline_runner")
    except Exception as e:
        print(f"  ✗ goal_2_baselines.baseline_runner: {e}")
        return False
    
    try:
        from goal_3_prediction.lsft.lsft import evaluate_lsft
        print("  ✓ goal_3_prediction.lsft.lsft")
    except Exception as e:
        print(f"  ✗ goal_3_prediction.lsft.lsft: {e}")
        return False
    
    try:
        from goal_3_prediction.functional_class_holdout.logo import run_logo_evaluation
        print("  ✓ goal_3_prediction.functional_class_holdout.logo")
    except Exception as e:
        print(f"  ✗ goal_3_prediction.functional_class_holdout.logo: {e}")
        return False
    
    try:
        from shared.linear_model import solve_y_axb
        print("  ✓ shared.linear_model")
    except Exception as e:
        print(f"  ✗ shared.linear_model: {e}")
        return False
    
    try:
        from shared.metrics import compute_metrics
        print("  ✓ shared.metrics")
    except Exception as e:
        print(f"  ✗ shared.metrics: {e}")
        return False
    
    try:
        from stats.bootstrapping import bootstrap_mean_ci
        print("  ✓ stats.bootstrapping")
    except Exception as e:
        print(f"  ✗ stats.bootstrapping: {e}")
        return False
    
    try:
        from stats.permutation import paired_permutation_test
        print("  ✓ stats.permutation")
    except Exception as e:
        print(f"  ✗ stats.permutation: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of key functions."""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        from shared.linear_model import solve_y_axb
        
        # Create small toy matrices
        A = np.random.randn(10, 5)
        B_train = np.random.randn(5, 20)
        Y_train = A @ np.random.randn(5, 5) @ B_train
        
        # Test solving
        solution = solve_y_axb(Y_train, A, B_train, ridge_penalty=0.1)
        print(f"  ✓ solve_y_axb works (K shape: {solution['K'].shape})")
    except Exception as e:
        print(f"  ✗ solve_y_axb failed: {e}")
        return False
    
    try:
        from shared.metrics import compute_metrics
        
        # Test metrics
        pred = np.random.randn(100)
        truth = pred + 0.1 * np.random.randn(100)
        metrics = compute_metrics(truth, pred)
        print(f"  ✓ compute_metrics works (r: {metrics['pearson_r']:.3f})")
    except Exception as e:
        print(f"  ✗ compute_metrics failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 70)
    print("Environment Test")
    print("=" * 70)
    print()
    
    success = True
    success &= test_imports()
    success &= test_basic_functionality()
    
    print()
    print("=" * 70)
    if success:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed!")
        sys.exit(1)


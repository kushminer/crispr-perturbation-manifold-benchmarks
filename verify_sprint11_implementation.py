#!/usr/bin/env python3
"""
Verification script for Sprint 11 implementation.

This script verifies that all Sprint 11 modules are properly implemented
and can be imported and used correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_statistics_modules():
    """Test statistics modules (Issues 3-4)."""
    print("Testing statistics modules...")
    
    try:
        from stats.bootstrapping import bootstrap_mean_ci, bootstrap_correlation_ci
        from stats.permutation import paired_permutation_test
        import numpy as np
        
        # Test bootstrap
        values = np.array([0.7, 0.75, 0.8, 0.72, 0.78])
        mean, ci_l, ci_u = bootstrap_mean_ci(values, n_boot=100, random_state=42)
        assert ci_l <= mean <= ci_u, "Bootstrap CI check failed"
        print("  ✓ Bootstrap functions work")
        
        # Test permutation
        deltas = np.array([0.05, 0.10, -0.02, 0.08, 0.12])
        mean_d, p = paired_permutation_test(deltas, n_perm=100, random_state=42)
        assert 0 <= p <= 1, "Permutation test p-value check failed"
        print("  ✓ Permutation test functions work")
        
        return True
    except Exception as e:
        print(f"  ✗ Statistics modules failed: {e}")
        return False


def test_lsft_resampling_modules():
    """Test LSFT resampling modules (Issues 5-8)."""
    print("Testing LSFT resampling modules...")
    
    try:
        from goal_3_prediction.lsft.lsft_resampling import (
            standardize_lsft_output,
            compute_lsft_summary_with_cis,
        )
        from goal_3_prediction.lsft.compare_baselines_resampling import (
            compare_baselines_with_resampling,
        )
        from goal_3_prediction.lsft.hardness_regression_resampling import (
            bootstrap_hardness_regression,
        )
        import pandas as pd
        import numpy as np
        
        # Test standardization
        df = pd.DataFrame({
            'test_perturbation': ['p1', 'p2'],
            'baseline_type': ['A', 'A'],
            'top_pct': [0.01, 0.01],
            'performance_local_pearson_r': [0.7, 0.8],
            'performance_local_l2': [5.0, 4.5],
            'local_mean_similarity': [0.85, 0.9],
            'local_train_size': [10, 10],
        })
        std = standardize_lsft_output(df, total_train_size=100)
        assert 'pearson_r' in std.columns, "Standardization failed"
        print("  ✓ LSFT standardization works")
        
        # Test summary with CIs
        summary = compute_lsft_summary_with_cis(std, n_boot=100, random_state=42)
        assert len(summary) > 0, "Summary computation failed"
        print("  ✓ Summary with CIs works")
        
        # Test hardness regression
        hardness = np.array([0.7, 0.75, 0.8, 0.72, 0.78])
        performance = np.array([0.6, 0.65, 0.7, 0.62, 0.68])
        reg = bootstrap_hardness_regression(hardness, performance, n_boot=100, random_state=42)
        assert 'slope' in reg, "Hardness regression failed"
        print("  ✓ Hardness regression works")
        
        return True
    except Exception as e:
        print(f"  ✗ LSFT resampling modules failed: {e}")
        return False


def test_logo_resampling_modules():
    """Test LOGO resampling modules (Issue 9)."""
    print("Testing LOGO resampling modules...")
    
    try:
        from goal_3_prediction.functional_class_holdout.logo_resampling import (
            standardize_logo_output,
            compute_logo_summary_with_cis,
        )
        import pandas as pd
        import numpy as np
        
        # Test standardization
        df = pd.DataFrame({
            'perturbation': ['p1', 'p2', 'p3'],
            'baseline': ['A', 'A', 'A'],
            'class': ['Transcription', 'Transcription', 'Transcription'],
            'pearson_r': [0.6, 0.65, 0.7],
            'l2': [5.0, 4.5, 4.0],
        })
        std = standardize_logo_output(df)
        assert 'baseline_type' in std.columns, "LOGO standardization failed"
        print("  ✓ LOGO standardization works")
        
        # Test summary with CIs
        summary = compute_logo_summary_with_cis(std, n_boot=100, random_state=42)
        assert len(summary) > 0, "LOGO summary computation failed"
        print("  ✓ LOGO summary with CIs works")
        
        return True
    except Exception as e:
        print(f"  ✗ LOGO resampling modules failed: {e}")
        return False


def test_visualization_modules():
    """Test visualization modules (Issue 10)."""
    print("Testing visualization modules...")
    
    try:
        from goal_3_prediction.lsft.visualize_resampling import (
            create_beeswarm_with_ci,
            create_hardness_curve_with_ci,
            create_baseline_comparison_with_significance,
        )
        print("  ✓ Visualization functions can be imported")
        return True
    except Exception as e:
        print(f"  ✗ Visualization modules failed: {e}")
        return False


def test_parity_verification():
    """Test parity verification module (Issue 11)."""
    print("Testing parity verification...")
    
    try:
        from goal_3_prediction.lsft.verify_parity import verify_lsft_parity
        print("  ✓ Parity verification module can be imported")
        return True
    except Exception as e:
        print(f"  ✗ Parity verification failed: {e}")
        return False


def test_documentation_exists():
    """Test that documentation exists (Issue 12)."""
    print("Testing documentation...")
    
    try:
        doc_path = Path(__file__).parent / "docs" / "resampling.md"
        assert doc_path.exists(), f"Documentation not found: {doc_path}"
        print(f"  ✓ Documentation exists: {doc_path}")
        return True
    except Exception as e:
        print(f"  ✗ Documentation check failed: {e}")
        return False


def test_ci_workflow():
    """Test that CI workflow exists (Issue 2)."""
    print("Testing CI workflow...")
    
    try:
        ci_path = Path(__file__).parent / ".github" / "workflows" / "ci.yml"
        assert ci_path.exists(), f"CI workflow not found: {ci_path}"
        print(f"  ✓ CI workflow exists: {ci_path}")
        return True
    except Exception as e:
        print(f"  ✗ CI workflow check failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Sprint 11 Implementation Verification")
    print("=" * 60)
    print()
    
    tests = [
        ("Statistics Modules (Issues 3-4)", test_statistics_modules),
        ("LSFT Resampling Modules (Issues 5-8)", test_lsft_resampling_modules),
        ("LOGO Resampling Modules (Issue 9)", test_logo_resampling_modules),
        ("Visualization Modules (Issue 10)", test_visualization_modules),
        ("Parity Verification (Issue 11)", test_parity_verification),
        ("Documentation (Issue 12)", test_documentation_exists),
        ("CI Workflow (Issue 2)", test_ci_workflow),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
            print()
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            results.append((name, False))
            print()
    
    # Summary
    print("=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All Sprint 11 modules verified successfully!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


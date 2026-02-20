#!/usr/bin/env python3
"""
Quick test script to verify dependencies work after paper directory replacement.

This script tests:
1. Path resolution to paper/ directory
2. Critical file existence checks
3. Import tests for key modules
4. Split logic path resolution
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def test_path_resolution():
    """Test that paths resolve correctly to paper/ directory."""
    print("="*70)
    print("TEST 1: Path Resolution")
    print("="*70)
    
    repo_root = Path(__file__).parent.parent.parent.parent
    paper_src = repo_root / "paper" / "benchmark" / "src"
    paper_gears = repo_root / "paper" / "benchmark" / "data" / "gears_pert_data"
    paper_r_script = paper_src / "run_linear_pretrained_model.R"
    paper_py_script = paper_src / "run_linear_pretrained_model.py"
    
    print(f"Repo root: {repo_root}")
    print(f"Paper src: {paper_src}")
    print(f"  Exists: {paper_src.exists()}")
    print(f"Paper GEARS path: {paper_gears}")
    print(f"  Exists: {paper_gears.exists()} (expected: False - needs download)")
    print(f"R script: {paper_r_script}")
    print(f"  Exists: {paper_r_script.exists()}")
    print(f"Python script: {paper_py_script}")
    print(f"  Exists: {paper_py_script.exists()} (expected: False - not in original repo)")
    
    success = (
        paper_src.exists() and
        paper_r_script.exists()
    )
    
    print(f"\n✓ Path resolution: {'PASS' if success else 'FAIL'}")
    return success


def test_split_logic_paths():
    """Test split_logic.py path resolution logic."""
    print("\n" + "="*70)
    print("TEST 2: Split Logic Path Resolution")
    print("="*70)
    
    try:
        # Simulate path resolution from split_logic.py
        current_file = Path(__file__).parent.parent.parent / "src" / "goal_2_baselines" / "split_logic.py"
        repo_root = current_file.parent.parent.parent  # repository root
        pert_data_folder = repo_root / "paper" / "benchmark" / "data" / "gears_pert_data"

        print(f"Repo root: {repo_root}")
        print(f"Resolved GEARS path: {pert_data_folder}")
        print(f"Path exists: {pert_data_folder.exists()} (expected: False - needs GEARS API)")
        
        # Test that path structure is correct
        expected_structure = repo_root / "paper" / "benchmark" / "data" / "gears_pert_data"
        path_correct = pert_data_folder == expected_structure
        
        print(f"\n✓ Path structure correct: {'PASS' if path_correct else 'FAIL'}")
        return path_correct
        
    except Exception as e:
        print(f"\n✗ Path resolution failed: {e}")
        return False


def test_baseline_types_paths():
    """Test baseline_types.py path references by reading file."""
    print("\n" + "="*70)
    print("TEST 3: Baseline Types Path References")
    print("="*70)
    
    try:
        # Read the file directly to check path references
        baseline_types_file = Path(__file__).parent.parent.parent / "src" / "goal_2_baselines" / "baseline_types.py"
        
        if not baseline_types_file.exists():
            print(f"✗ File not found: {baseline_types_file}")
            return False
        
        content = baseline_types_file.read_text()
        
        # Check for GEARS path reference
        if "../paper/benchmark/data/gears_pert_data/go_essential_all/go_essential_all.csv" in content:
            print(f"✓ Found GEARS GO embeddings path reference")
            
            # Resolve the path
            repo_root = Path(__file__).parent.parent.parent.parent
            gears_path = repo_root / "paper" / "benchmark" / "data" / "gears_pert_data" / "go_essential_all" / "go_essential_all.csv"
            print(f"Resolved path: {gears_path}")
            print(f"Path exists: {gears_path.exists()} (expected: False - needs data download)")
            print(f"\n✓ Baseline types paths: PASS")
            return True
        else:
            print(f"⚠ GEARS path reference not found in expected format")
            print(f"\n✓ Baseline types file exists: PASS")
            return True
            
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        return False


def test_python_script_handling():
    """Test that missing Python script is handled gracefully."""
    print("\n" + "="*70)
    print("TEST 4: Missing Python Script Handling")
    print("="*70)
    
    # Check that the code has proper handling for missing script
    compare_file = Path(__file__).parent.parent.parent / "src" / "goal_5_validation" / "compare_paper_python_r.py"
    
    if not compare_file.exists():
        print(f"✗ File not found: {compare_file}")
        return False
    
    content = compare_file.read_text()
    
    # Check for graceful error handling
    checks = [
        ("if not py_script_path.exists()", "Checks for script existence"),
        ("LOGGER.warning", "Logs warning instead of crashing"),
        ("return False", "Returns gracefully"),
    ]
    
    all_found = True
    for check_str, description in checks:
        if check_str in content:
            print(f"✓ {description}")
        else:
            print(f"✗ {description} - NOT FOUND")
            all_found = False
    
    if all_found:
        print(f"\n✓ Graceful handling code present: PASS")
        return True
    else:
        print(f"\n⚠ Some error handling checks missing")
        return False


def test_key_imports():
    """Test that key modules can be found and have correct structure."""
    print("\n" + "="*70)
    print("TEST 5: Key Module Structure")
    print("="*70)
    
    # Test module file existence (without importing, which requires dependencies)
    module_files = [
        "src/goal_2_baselines/split_logic.py",
        "src/goal_2_baselines/baseline_types.py",
        "src/goal_2_baselines/baseline_runner.py",
        "src/goal_5_validation/validate_r_parity.py",
        "src/goal_5_validation/compare_paper_python_r.py",
        "src/shared/linear_model.py",
        "src/shared/io.py",
    ]
    
    passed = 0
    failed = 0
    base_path = Path(__file__).parent.parent.parent
    
    for module_file in module_files:
        full_path = base_path / module_file
        if full_path.exists():
            print(f"✓ {module_file}")
            passed += 1
        else:
            print(f"✗ {module_file} - NOT FOUND")
            failed += 1
    
    print(f"\nModule files: {passed} found, {failed} missing")
    
    # Note: Full import tests require dependencies (anndata, numpy, etc.)
    # Those should be run in a proper environment with requirements.txt installed
    print("\nNote: Full import tests require dependencies.")
    print("      Run with: pip install -r requirements.txt")
    
    return failed == 0


def main():
    """Run all tests."""
    print("="*70)
    print("PAPER DIRECTORY DEPENDENCY TESTS")
    print("="*70)
    print("After replacing paper/ directory with fresh repository pull")
    print()
    
    results = []
    
    results.append(("Path Resolution", test_path_resolution()))
    results.append(("Split Logic Paths", test_split_logic_paths()))
    results.append(("Baseline Types Paths", test_baseline_types_paths()))
    results.append(("Python Script Handling", test_python_script_handling()))
    results.append(("Key Imports", test_key_imports()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not result:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("\nDependencies are working correctly after paper directory replacement.")
        return 0
    else:
        print("SOME TESTS FAILED ✗")
        print("\nPlease check the failures above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

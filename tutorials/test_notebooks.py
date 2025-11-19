#!/usr/bin/env python3
"""
Test script for tutorial notebooks.

This script tests all tutorial notebooks by:
1. Validating notebook structure
2. Executing notebooks with synthetic data (option 2) to avoid input() prompts
3. Checking for runtime errors

Usage:
    python test_notebooks.py
"""

import json
import sys
from pathlib import Path
import subprocess
import tempfile
import shutil

NOTEBOOK_DIR = Path(__file__).parent
NOTEBOOKS = [
    "tutorial_goal_1_similarity.ipynb",
    "tutorial_goal_2_baselines.ipynb",
    "tutorial_goal_3_predictions.ipynb",
    "tutorial_goal_4_analysis.ipynb",
    "tutorial_goal_5_validation.ipynb",
]


def modify_input_calls(notebook_content, default_value="2"):
    """
    Modify input() calls in notebook to use default value for automated testing.
    
    Replaces: input("...") with: "2"  # synthetic data
    """
    modified = False
    for cell in notebook_content.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            if isinstance(source, str):
                source = source.split("\n")
            
            new_source = []
            for line in source:
                # Replace input() calls with default value
                if "input(" in line and "DATA SOURCE SELECTION" in "".join(source):
                    # Find the input() call and replace with default
                    import re
                    # Replace input(...) with default value
                    new_line = re.sub(
                        r'input\([^)]+\)',
                        f'"{default_value}"',
                        line
                    )
                    new_source.append(new_line)
                    if new_line != line:
                        modified = True
                else:
                    new_source.append(line)
            
            cell["source"] = "\n".join(new_source) if isinstance(cell.get("source", ""), str) else new_source
    
    return notebook_content, modified


def test_notebook_structure(notebook_path):
    """Validate notebook structure."""
    print(f"\n{'='*70}")
    print(f"Testing structure: {notebook_path.name}")
    print(f"{'='*70}")
    
    try:
        with open(notebook_path, 'r') as f:
            nb = json.load(f)
        
        # Check basic structure
        assert "cells" in nb, "Missing 'cells' key"
        assert len(nb["cells"]) > 0, "No cells found"
        
        # Count cell types
        code_cells = sum(1 for cell in nb["cells"] if cell.get("cell_type") == "code")
        markdown_cells = sum(1 for cell in nb["cells"] if cell.get("cell_type") == "markdown")
        
        print(f"  ✓ Notebook structure valid")
        print(f"  ✓ Total cells: {len(nb['cells'])}")
        print(f"  ✓ Code cells: {code_cells}")
        print(f"  ✓ Markdown cells: {markdown_cells}")
        
        # Check for data selection prompt
        has_data_selection = False
        for cell in nb["cells"]:
            if cell.get("cell_type") == "code":
                source = "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else cell.get("source", "")
                if "DATA SOURCE SELECTION" in source:
                    has_data_selection = True
                    break
        
        if has_data_selection:
            print(f"  ✓ Contains data selection prompt")
        else:
            print(f"  ⚠ Missing data selection prompt")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_notebook_execution(notebook_path, use_synthetic=True):
    """Test notebook execution with modified input() calls."""
    print(f"\n{'='*70}")
    print(f"Testing execution: {notebook_path.name}")
    print(f"{'='*70}")
    
    try:
        # Load notebook
        with open(notebook_path, 'r') as f:
            nb = json.load(f)
        
        # Modify input() calls for automated testing
        default_value = "2" if use_synthetic else "1"
        nb_modified, was_modified = modify_input_calls(nb, default_value)
        
        if was_modified:
            print(f"  ✓ Modified input() calls for automated testing (using option {default_value})")
        else:
            print(f"  ℹ No input() calls found to modify")
        
        # Create temporary notebook file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            json.dump(nb_modified, f, indent=1)
            temp_nb_path = f.name
        
        try:
            # Execute notebook using nbconvert
            print(f"  Executing notebook...")
            result = subprocess.run(
                [
                    sys.executable, "-m", "jupyter", "nbconvert",
                    "--to", "notebook",
                    "--execute",
                    "--ExecutePreprocessor.timeout=300",
                    "--ExecutePreprocessor.kernel_name=python3",
                    "--ExecutePreprocessor.allow_errors=False",
                    "--output", str(temp_nb_path),
                    "--output-dir", str(tempfile.gettempdir()),
                    str(temp_nb_path),
                ],
                capture_output=True,
                text=True,
                cwd=NOTEBOOK_DIR,
            )
            
            if result.returncode == 0:
                print(f"  ✓ Notebook executed successfully")
                return True
            else:
                print(f"  ✗ Execution failed:")
                print(f"    STDOUT: {result.stdout[:500]}")
                print(f"    STDERR: {result.stderr[:500]}")
                return False
                
        finally:
            # Clean up temp file
            try:
                Path(temp_nb_path).unlink()
            except:
                pass
                
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("TUTORIAL NOTEBOOKS TEST SUITE")
    print("="*70)
    
    results = {}
    
    # Test structure
    print("\n" + "="*70)
    print("PHASE 1: Structure Validation")
    print("="*70)
    for nb_name in NOTEBOOKS:
        nb_path = NOTEBOOK_DIR / nb_name
        if not nb_path.exists():
            print(f"\n✗ Notebook not found: {nb_name}")
            results[nb_name] = {"structure": False, "execution": False}
            continue
        
        results[nb_name] = {}
        results[nb_name]["structure"] = test_notebook_structure(nb_path)
    
    # Test execution (with synthetic data)
    print("\n" + "="*70)
    print("PHASE 2: Execution Testing (Synthetic Data Mode)")
    print("="*70)
    print("Note: This will execute notebooks with synthetic data (option 2)")
    print("      to avoid requiring user input or data downloads.")
    
    for nb_name in NOTEBOOKS:
        nb_path = NOTEBOOK_DIR / nb_name
        if not nb_path.exists():
            continue
        
        results[nb_name]["execution"] = test_notebook_execution(nb_path, use_synthetic=True)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for nb_name, test_results in results.items():
        structure_ok = test_results.get("structure", False)
        execution_ok = test_results.get("execution", False)
        
        status = "✓ PASS" if (structure_ok and execution_ok) else "✗ FAIL"
        print(f"{status}: {nb_name}")
        print(f"  Structure: {'✓' if structure_ok else '✗'}")
        print(f"  Execution: {'✓' if execution_ok else '✗'}")
        
        if not (structure_ok and execution_ok):
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        return 0
    else:
        print("SOME TESTS FAILED ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())


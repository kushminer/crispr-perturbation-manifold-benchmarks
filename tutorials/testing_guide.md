# Tutorial Notebooks Testing Guide

## Overview

All 5 tutorial notebooks have been updated with interactive data selection prompts. This guide explains how to test them.

## What Was Added

Each notebook now includes:
1. **Data Selection Prompt**: At the beginning, asks user to choose:
   - Option 1: Real data (from files or previous notebook outputs)
   - Option 2: Synthetic data (no downloads required)

2. **Automatic Output Detection**: Notebooks 2-5 automatically detect and load outputs from earlier tutorials if available

3. **Output Saving**: Real data runs save outputs to `tutorials/outputs/` for use in subsequent notebooks

## Structure Validation

All notebooks have been validated for:
- ✓ Valid JSON structure
- ✓ Contains data selection prompt
- ✓ Proper cell structure (code + markdown)

## Manual Testing Instructions

### Option A: Test with Synthetic Data (Recommended for Quick Testing)

1. Open each notebook in Jupyter
2. When prompted, enter `2` (for synthetic data)
3. Run all cells sequentially
4. Verify:
   - All cells execute without errors
   - Visualizations render correctly
   - Output values are reasonable

### Option B: Test with Real Data (Full Testing)

1. **Tutorial 1 (Similarity)**:
   - Enter `1` for real data
   - Verify it loads Adamson dataset
   - Check that Y matrix is saved to `tutorials/outputs/goal_1_Y_matrix.csv`

2. **Tutorial 2 (Baselines)**:
   - Enter `1` for real data
   - It should detect and load `goal_1_Y_matrix.csv` from Tutorial 1
   - Verify it runs all baselines
   - Check that outputs are saved to `tutorials/outputs/goal_2_*.csv`

3. **Tutorial 3 (Predictions)**:
   - Enter `1` for real data
   - It should detect and load `goal_2_Y_train.csv` and `goal_2_Y_test.csv`
   - Verify LSFT and LOGO sections run correctly
   - Check visualizations render

4. **Tutorial 4 (Analysis)**:
   - Enter `1` for real data
   - It should detect and load baseline results from Tutorial 2
   - Verify statistical tests run
   - Check visualizations render

5. **Tutorial 5 (Validation)**:
   - Enter `1` for real data
   - It should detect and load embeddings/results from previous tutorials
   - Verify parity comparisons run

## Automated Testing

A test script (`test_notebooks.py`) is provided that:
- Validates notebook structure
- Attempts execution with synthetic data (option 2)
- Reports any errors

To run:
```bash
cd tutorials
python test_notebooks.py
```

**Note**: Automated execution requires `jupyter nbconvert` and may have limitations with interactive prompts. Manual testing is recommended for full validation.

## Expected Behavior

### With Synthetic Data (Option 2):
- No file downloads required
- All notebooks should execute quickly
- Results should be consistent (seeded random data)
- Visualizations should render correctly

### With Real Data (Option 1):
- May require data files to be present
- If previous notebook outputs exist, they will be loaded automatically
- If files are missing, notebooks will fall back to synthetic data with a warning

## Known Issues / Notes

1. **Input() Prompts**: The `input()` function requires interactive execution. For automated testing, the test script modifies notebooks to use default values.

2. **File Dependencies**: Real data mode requires:
   - Tutorial 1: `paper/benchmark/data/gears_pert_data/adamson/perturb_processed.h5ad`
   - Tutorial 1: `results/goal_2_baselines/splits/adamson_split_seed1.json`
   - Tutorial 3: `data/annotations/adamson_functional_classes_enriched.tsv` (for LOGO section)

3. **Output Directory**: All outputs are saved to `tutorials/outputs/`. This directory is created automatically if it doesn't exist.

## Checklist for Manual Testing

- [ ] Tutorial 1: Runs with synthetic data (option 2)
- [ ] Tutorial 1: Runs with real data (option 1) - if files available
- [ ] Tutorial 1: Saves Y matrix output correctly
- [ ] Tutorial 2: Detects and loads Tutorial 1 output
- [ ] Tutorial 2: Runs all baselines successfully
- [ ] Tutorial 2: Saves Y_train and Y_test outputs
- [ ] Tutorial 3: Detects and loads Tutorial 2 outputs
- [ ] Tutorial 3: LSFT section runs correctly
- [ ] Tutorial 3: LOGO section runs correctly (with synthetic annotations if needed)
- [ ] Tutorial 3: Visualizations render
- [ ] Tutorial 4: Loads baseline results (synthetic or real)
- [ ] Tutorial 4: Statistical tests run correctly
- [ ] Tutorial 4: Visualizations render
- [ ] Tutorial 5: Loads embeddings/results (synthetic or real)
- [ ] Tutorial 5: Parity comparisons run correctly
- [ ] Tutorial 5: Visualizations render

## Troubleshooting

### Issue: `input()` prompt doesn't appear
- **Solution**: Make sure you're running the notebook in an interactive environment (Jupyter Lab/Notebook, VS Code with Jupyter extension, etc.)

### Issue: "Previous outputs not found" warning
- **Solution**: This is expected if you haven't run previous notebooks. The notebook will use synthetic data instead.

### Issue: FileNotFoundError for data files
- **Solution**: Either:
  1. Use synthetic data (option 2)
  2. Ensure data files are in the expected locations
  3. Notebooks will automatically fall back to synthetic data

### Issue: Import errors
- **Solution**: Make sure all required packages are installed:
  ```bash
  pip install numpy pandas matplotlib seaborn scikit-learn scipy anndata
  ```

## Summary

All notebooks are now structured with:
- ✓ Data selection prompts
- ✓ Automatic output detection
- ✓ Output saving for data chaining
- ✓ Fallback to synthetic data
- ✓ Valid JSON structure
- ✓ Proper error handling

Manual testing is recommended to verify full functionality, especially visualizations and interactive prompts.

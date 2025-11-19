# Baseline Results Cleanup Summary

**Date:** 2025-11-17  
**Action:** Cleaned up baseline results to retain only `baseline_runner` results

---

## Removed Directories

### Paper Implementation Results
- ✅ `results/baselines/adamson_paper_implementation/` - Removed (paper's implementation results)

### Old Result Directories
- ✅ `results/baselines/adamson/` - Removed (old per-baseline subdirectories)
- ✅ `results/baselines/replogle/` - Removed (old per-baseline subdirectories)
- ✅ `results/baselines/rpe1/` - Removed (old per-baseline subdirectories)

### Duplicate Directories
- ✅ `results/baselines/replogle_k562_reproduced/` - Removed (duplicate, keeping `replogle_k562_essential_reproduced`)

### Comparison Results
- ✅ `results/comparison_with_paper/` - Removed (contained paper_implementation comparisons)

---

## Retained Directories

### Baseline Runner Results (from `baseline_runner.py`)
- ✅ `results/baselines/adamson_reproduced/` - Contains `baseline_results_reproduced.csv`
- ✅ `results/baselines/replogle_k562_essential_reproduced/` - Contains `baseline_results_reproduced.csv`
- ✅ `results/baselines/replogle_rpe1_essential_reproduced/` - Contains `baseline_results_reproduced.csv`

### Summary Files
- ✅ `results/baselines/all_datasets_summary.csv` - Aggregated results across all datasets
- ✅ `results/baselines/*_split_seed1.json` - Split configuration files

---

## Updated Files

### Scripts
- ✅ `src/baselines/compare_with_paper_results.py` - Removed `paper_impl_results` parameter
- ✅ `data/paper_results/README.md` - Updated usage example

### Documentation
- ✅ `docs/PAPER_IMPLEMENTATION_INTEGRATION.md` - Updated to note results are not retained
- ✅ `docs/PAPER_IMPLEMENTATION_RESULTS.md` - Deleted (no longer needed)

---

## Current Baseline Results Structure

```
results/baselines/
├── adamson_reproduced/
│   └── baseline_results_reproduced.csv
├── replogle_k562_essential_reproduced/
│   └── baseline_results_reproduced.csv
├── replogle_rpe1_essential_reproduced/
│   └── baseline_results_reproduced.csv
├── all_datasets_summary.csv
├── adamson_split_seed1.json
├── replogle_k562_essential_split_seed1.json
└── replogle_rpe1_essential_split_seed1.json
```

---

## Notes

- **Paper's implementation** (`paper_implementation.py`) is still available in code for validation purposes
- **Results from paper's implementation** are not retained (only used for validation)
- **All production results** come from `baseline_runner.py` (our implementation)
- **Comparison scripts** now only compare our implementation vs paper's published results

---

**Last Updated:** 2025-11-17


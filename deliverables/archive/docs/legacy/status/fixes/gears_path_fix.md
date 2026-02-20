# GEARS Embedding Path Fix

## Problem
GEARS and self-trained PCA baselines were producing identical outputs because the GEARS CSV file path was incorrect, causing the embeddings to fail to load (or silently fallback).

## Root Cause
The path in `baseline_types.py` was:
```
../paper/benchmark/data/gears_pert_data/go_essential_all/go_essential_all.csv
```

But the actual file location is:
```
../linear_perturbation_prediction-Paper/paper/benchmark/data/gears_pert_data/go_essential_all/go_essential_all.csv
```

The path was missing the `linear_perturbation_prediction-Paper` directory.

## Fix Applied
Updated `src/goal_2_baselines/baseline_types.py` to use the correct path:
```python
"source_csv": "../linear_perturbation_prediction-Paper/paper/benchmark/data/gears_pert_data/go_essential_all/go_essential_all.csv",
```

## Verification
After the fix:
- ✅ GEARS CSV file is found and loaded successfully
- ✅ GEARS embeddings load: shape=(10, 9853), 9853 perturbations
- ✅ Found 64/68 common perturbations for Adamson dataset
- ✅ Embeddings are validated as different from training_data (max_diff=16.64)
- ✅ GEARS produces different results: r=0.207 vs self-trained r=0.396

## Results Comparison

### Adamson Dataset
- **Self-trained PCA**: r = 0.396
- **GEARS Pert Emb**: r = 0.207 (now different!)

The fix is complete and working correctly.


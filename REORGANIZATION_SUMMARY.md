# Repository Reorganization Summary

**Date:** 2025-11-18  
**Purpose:** Improve navigation and clarity by organizing code, results, and documentation according to the 5 core goals.

## Overview

The repository has been reorganized to clearly align with the 5 core goals:
1. **Investigate cosine similarity** of targets to embedding space
2. **Reproduce original baseline** results
3. **Make predictions** on original train/val/test split after filtering for cosine similarity
4. **Statistically analyze** the results
5. **Validate parity** for producing embeddings and 8 baseline scripts from original paper

## Key Changes

### Module Renaming

- `similarity/` → `goal_1_similarity/` (Goal 1)
- `baselines/` → `goal_2_baselines/` (Goal 2 - baseline reproduction only)
- `prediction/` → `goal_3_prediction/` (Goal 3)
- `core/` → `shared/` (Shared utilities)

### New Modules Created

- `goal_4_analysis/` (Goal 4 - statistical analysis)
- `goal_5_validation/` (Goal 5 - parity validation)
- `scripts/` (Consolidated utility scripts)
- `data/` (Data utilities and legacy scripts)

### Results Reorganization

All results are now organized by goal:

```
results/
├── goal_1_similarity/      # Similarity investigation results
├── goal_2_baselines/       # Baseline reproduction results + splits
├── goal_3_prediction/      # LSFT results
├── goal_4_analysis/        # Statistical analysis results
├── goal_5_validation/      # Parity validation reports
└── embeddings/             # Precomputed embeddings
```

### Documentation Reorganization

Documentation is now organized by goal:

```
docs/
├── goal_1_similarity/      # Similarity analysis documentation
├── goal_2_baselines/       # Baseline specifications and results
├── goal_5_validation/      # Parity validation documentation
└── shared/                 # Shared documentation (reproducibility, etc.)
```

## Import Changes

All imports have been updated:

**Before:**
```python
from similarity.embedding_similarity import ...
from baselines.baseline_runner import ...
from prediction.lsft import ...
from core.metrics import ...
```

**After:**
```python
from goal_1_similarity.embedding_similarity import ...
from goal_2_baselines.baseline_runner import ...
from goal_3_prediction.lsft import ...
from shared.metrics import ...
```

## Command Changes

All command-line entry points have been updated:

**Before:**
```bash
python -m similarity.embedding_similarity ...
python -m baselines.run_all ...
python -m prediction.lsft ...
```

**After:**
```bash
python -m goal_1_similarity.embedding_similarity ...
python -m goal_2_baselines.run_all ...
python -m goal_3_prediction.lsft ...
```

## Result Path Changes

Result paths have been updated:

**Before:**
```
results/baselines/adamson_reproduced/
results/embedding_similarity/
results/prediction/adamson/lsft/
results/analysis/
```

**After:**
```
results/goal_2_baselines/adamson_reproduced/
results/goal_1_similarity/embedding_similarity/
results/goal_3_prediction/adamson/lsft/
results/goal_4_analysis/
```

## Benefits

1. **Clear goal alignment** - Each module/directory clearly maps to a goal
2. **Better navigation** - Easier to find code/results for specific goals
3. **Reduced clutter** - Scattered scripts consolidated
4. **Consistent naming** - Goal-based prefixes make structure obvious
5. **Scalability** - Easy to add new goals or extend existing ones

## Migration Notes

- All imports have been automatically updated
- Result paths in scripts have been updated
- Documentation references have been updated
- Test paths have been updated

If you have any scripts or notebooks that reference the old structure, update them using the mappings above.

## See Also

- **Main README:** `README.md` - Updated with new structure and examples
- **Migration Guide:** `archive/MIGRATION_GUIDE.md` - Details on what was archived


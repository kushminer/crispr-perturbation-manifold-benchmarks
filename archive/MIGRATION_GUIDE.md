# Migration Guide: Repository Refinement

**Date:** 2025-11-18  
**Purpose:** Document what was archived and how to access archived functionality if needed.

## Overview

This repository was refined to focus on 5 core goals:
1. Investigate cosine similarity of targets to embedding space
2. Reproduce original baseline results
3. Make predictions on original train/val/test split after filtering for cosine similarity
4. Statistically analyze the results
5. Validate parity for producing embeddings and 8 baseline scripts from original paper

All code not directly supporting these goals has been archived.

## What Was Archived

### 1. Prediction Modules (Sprint 7 Extensions)

**Location:** `archive/prediction_modules/`

- `ensembles.py` - Similarity-weighted ensemble predictions
- `model_selection.py` - Similarity-based model selection
- `frontier_plots.py` - Hardness-calibrated performance curves
- `local_regression.py` - Local embedding filtering (separate from LSFT)

**Reason:** These were exploratory extensions from Sprint 7, not part of the core 5 goals. If needed, they can be re-integrated, but dependencies should be checked first.

**How to Use:**
```python
# If you need these modules, copy them back to src/prediction/
# Update imports from eval_framework to core
from core.metrics import compute_metrics
from core.linear_model import solve_y_axb
```

### 2. LOGO + Hardness Evaluation Framework

**Location:** `archive/eval_framework_logo/`

- `logo_hardness.py` - LOGO + hardness evaluation
- `combined_analysis.py` - Merges LOGO + class results
- `visualization.py` - LOGO-specific visualizations

**Reason:** LOGO + Hardness evaluation was a separate evaluation framework not aligned with the core 5 goals. Functional class evaluation remains in the core framework.

**How to Use:**
```python
# If you need LOGO evaluation, copy files back and update imports
from core.linear_model import solve_y_axb
from core.metrics import compute_metrics
from core.io import load_expression_dataset
# Note: You'll also need to restore the main.py task_logo function
```

### 3. Historical Documentation

**Location:** `archive/docs/`

**Archived:**
- `PHASE0_*.md` - Phase 0 cleanup documentation
- `PHASE1_*.md` - Phase 1 baseline documentation (except core specs)
- `SPRINT5_*.md` - Sprint 5 documentation
- `HARDNESS_METRIC_*.md` - Hardness metric documentation
- `ISSUE13_*.md` - Consolidated into `docs/baseline_specs.md`
- `project_management/` - Project management docs
- `status_reports/` - Status reports
- `IMPLEMENTATION_STATUS.md`, `TASK_LIST_NEXT_ITERATION.md`, `REORGANIZATION_NOTES.md`

**Reason:** Historical documentation from development phases. Core specifications are consolidated into `docs/baseline_specs.md`.

**Replacement:**
- Baseline specifications: See `docs/baseline_specs.md`
- Reproducibility: See `docs/REPRODUCIBILITY.md`
- API documentation: See individual module docstrings

### 4. Historical Results

**Location:** `archive/results/`

- `hardness/` - LOGO + hardness evaluation results (132MB)
- `paper_comparison/` - Paper comparison results (8.2MB)

**Reason:** Large intermediate results directories that can be regenerated if needed.

**Note:** Core results (baselines, similarity, prediction, analysis) remain in `results/`.

## Module Reorganization

### Changes

1. **`eval_framework/` → `core/`**
   - Renamed for clarity
   - Contains: `linear_model.py`, `metrics.py`, `io.py`, `config.py`, `validation.py`, `embedding_parity.py`, `comparison.py`

2. **`functional_class/` → Separate Module**
   - Moved from `eval_framework/` to `src/functional_class/`
   - Contains: `functional_class.py`, `class_mapping.py`, `test_utils.py`

3. **`logo_gene_similarity.py` → `lsft.py`**
   - Renamed for clarity (Local Similarity-Filtered Training)
   - Location: `src/prediction/lsft.py`

### Import Updates

All imports have been updated:
- `from eval_framework.X import Y` → `from core.X import Y`
- `from eval_framework.functional_class import X` → `from functional_class.functional_class import X`
- `from prediction.logo_gene_similarity import X` → `from prediction.lsft import X`

## Breaking Changes

### `main.py` CLI

The following tasks are no longer available (archived):
- `logo` - LOGO + hardness evaluation
- `combined` - Combined LOGO + class analysis
- `visualize` - LOGO-specific visualizations

**Still Available:**
- `class` - Functional class holdout (will be updated)
- `validate` - Validation suite
- `validate-embeddings` - Embedding parity validation

### Configuration Files

Config files in `configs/` may reference LOGO tasks. These will need updates if you want to use them.

## Checking Dependencies

If you re-integrate archived code, check for:

1. **Import updates:** Change `eval_framework` to `core` or `functional_class`
2. **Function signatures:** Some functions may have changed
3. **Data structures:** Ensure compatibility with current data formats

## Questions?

If you need archived functionality:
1. Check this guide first
2. Review the archived code in `archive/`
3. Update imports and dependencies
4. Test thoroughly before re-integrating


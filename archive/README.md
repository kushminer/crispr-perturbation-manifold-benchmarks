# Archive Directory

This directory contains deprecated code, documentation, and results that are no longer part of the core evaluation framework but are preserved for historical reference.

## Repository Refinement (2025-11-18)

The repository was refined to focus on 5 core goals:
1. Investigate cosine similarity of targets to embedding space
2. Reproduce original baseline results
3. Make predictions on original train/val/test split after filtering for cosine similarity
4. Statistically analyze the results
5. Validate parity for producing embeddings and 8 baseline scripts from original paper

All code not directly supporting these goals has been archived here.

## Contents

### `prediction_modules/`

**Date Archived:** 2025-11-18  
**Reason:** Exploratory Sprint 7 extensions not aligned with core goals.

**Files:**
- `ensembles.py` - Similarity-weighted ensemble predictions
- `model_selection.py` - Similarity-based model selection
- `frontier_plots.py` - Hardness-calibrated performance curves
- `local_regression.py` - Local embedding filtering (separate from LSFT)

**Note:** LSFT (Local Similarity-Filtered Training) remains in `src/prediction/lsft.py` as it's part of core Goal 3.

### `eval_framework_logo/`

**Date Archived:** 2025-11-18  
**Reason:** LOGO + Hardness evaluation framework not aligned with core goals.

**Files:**
- `logo_hardness.py` - LOGO + hardness evaluation
- `combined_analysis.py` - Merges LOGO + class results
- `visualization.py` - LOGO-specific visualizations

**Note:** Functional class evaluation remains in the core framework at `src/functional_class/`.

### `docs/`

**Date Archived:** 2025-11-18  
**Reason:** Historical documentation from development phases.

**Contents:**
- `PHASE0_*.md` - Phase 0 cleanup documentation
- `PHASE1_*.md` - Phase 1 baseline documentation
- `SPRINT5_*.md` - Sprint 5 documentation
- `HARDNESS_METRIC_*.md` - Hardness metric documentation
- `ISSUE13_*.md` - Consolidated into `docs/baseline_specs.md`
- `project_management/` - Project management docs
- `status_reports/` - Status reports
- `IMPLEMENTATION_STATUS.md`, `TASK_LIST_NEXT_ITERATION.md`, `REORGANIZATION_NOTES.md`

**Replacement:** Core specifications consolidated in `docs/baseline_specs.md`.

### `results/`

**Date Archived:** 2025-11-18  
**Reason:** Large intermediate results directories that can be regenerated if needed.

**Contents:**
- `hardness/` - LOGO + hardness evaluation results (132MB)
- `paper_comparison/` - Paper comparison results (8.2MB)

**Note:** Core results (baselines, similarity, prediction, analysis) remain in `results/`.

### `tests/`

**Date Archived:** 2025-11-18  
**Reason:** Tests for archived functionality.

**Files:**
- `test_logo_hardness.py` - Tests for LOGO evaluation
- `test_combined_analysis.py` - Tests for combined analysis

### `logo_old/`

**Date Archived:** Earlier  
**Reason:** Older version of LOGO implementation.

## Migration Guide

For detailed information about what was archived, why, and how to access archived functionality if needed, see:

**`MIGRATION_GUIDE.md`** - Complete migration documentation with:
- What was archived and why
- How to re-integrate archived code if needed
- Import update instructions
- Breaking changes documentation

## Archive Policy

Files in this directory are:
- ✅ Preserved for historical reference
- ✅ Available for review or re-integration if needed
- ❌ Not maintained or tested
- ❌ Not part of the active codebase

If you need archived functionality:
1. Check `MIGRATION_GUIDE.md` first
2. Review the archived code
3. Update imports (`eval_framework` → `core`)
4. Test thoroughly before re-integrating

---

**See Also:**
- Main README: `../README.md`
- Core goals: `../README.md#overview`
- Migration guide: `MIGRATION_GUIDE.md`


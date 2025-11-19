# Repository Restructuring Plan (Issue 0.4)

**Date:** 2025-11-14  
**Status:** 🔄 **PLANNING**

---

## Current Structure

```
evaluation_framework/
├── src/
│   ├── eval_framework/     # Mixed: logo, class, combined, utils
│   ├── embeddings/          # Embedding loaders
│   ├── legacy_scripts/     # Parity validation scripts
│   └── [various scripts]   # Utility scripts
├── configs/
├── data/
├── docs/
├── tests/
└── validation/
```

## Target Structure

```
evaluation_framework/
├── src/
│   ├── baselines/          # Linear baseline models
│   │   ├── __init__.py
│   │   ├── linear_model.py
│   │   └── linear_baseline.py
│   ├── eval_logo/          # LOGO + similarity evaluation
│   │   ├── __init__.py
│   │   ├── logo_hardness.py
│   │   └── similarity.py
│   ├── eval_class/          # Functional-class holdout
│   │   ├── __init__.py
│   │   ├── functional_class.py
│   │   └── class_mapping.py
│   ├── eval_combined/       # Combined analysis
│   │   ├── __init__.py
│   │   └── combined_analysis.py
│   ├── embeddings/          # Embedding loaders (keep as-is)
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── gears_go_perturbation.py
│   │   ├── pca_perturbation.py
│   │   ├── scgpt_gene.py
│   │   └── scfoundation_gene.py
│   ├── utils/               # Shared utilities
│   │   ├── __init__.py
│   │   ├── io.py
│   │   ├── metrics.py
│   │   ├── config.py
│   │   └── validation.py
│   ├── legacy_scripts/      # Parity validation (keep as-is)
│   └── main.py              # CLI entry point
├── configs/
├── data/
├── docs/
├── tests/
└── validation/
```

## Migration Plan

### Step 1: Create New Directory Structure
- [ ] Create `src/baselines/`
- [ ] Create `src/eval_logo/`
- [ ] Create `src/eval_class/`
- [ ] Create `src/eval_combined/`
- [ ] Create `src/utils/`

### Step 2: Move Files from `eval_framework/`

**To `baselines/`:**
- `linear_model.py` → `baselines/linear_model.py`

**To `eval_logo/`:**
- `logo_hardness.py` → `eval_logo/logo_hardness.py`

**To `eval_class/`:**
- `functional_class.py` → `eval_class/functional_class.py`
- `class_mapping.py` → `eval_class/class_mapping.py`

**To `eval_combined/`:**
- `combined_analysis.py` → `eval_combined/combined_analysis.py`
- `comparison.py` → `eval_combined/comparison.py` (if needed)

**To `utils/`:**
- `io.py` → `utils/io.py`
- `metrics.py` → `utils/metrics.py`
- `config.py` → `utils/config.py`
- `validation.py` → `utils/validation.py`
- `test_utils.py` → `utils/test_utils.py`

**Keep in place:**
- `embedding_parity.py` → Move to `validation/` or keep in `eval_framework/` temporarily
- `visualization.py` → Move to `utils/` or create `utils/visualization.py`

### Step 3: Update Imports

**Files to update:**
- `main.py` - Update all imports
- All test files in `tests/`
- All modules that import from `eval_framework`

**Import pattern changes:**
```python
# Old
from eval_framework.logo_hardness import ...
from eval_framework.functional_class import ...
from eval_framework.io import ...

# New
from eval_logo.logo_hardness import ...
from eval_class.functional_class import ...
from utils.io import ...
```

### Step 4: Update `__init__.py` Files

Create/update `__init__.py` in each new directory to expose public APIs.

### Step 5: Remove Old `eval_framework/` Directory

After all imports are updated and tests pass:
- [ ] Remove `src/eval_framework/` directory

### Step 6: Update Documentation

- [ ] Update `README.md` with new structure
- [ ] Update any docs that reference old paths
- [ ] Update import examples in documentation

## Alternative: Minimal Restructuring

If full restructuring is too disruptive, we could:

1. **Keep current structure** but add clear organization:
   - Keep `eval_framework/` but organize it better
   - Add clear module docstrings
   - Update README to explain structure

2. **Partial restructuring:**
   - Only move clearly separable modules
   - Keep related modules together
   - Minimize import changes

## Decision

**Recommendation:** Start with **Alternative 1 (Minimal Restructuring)** to minimize disruption, then consider full restructuring if needed.

**Rationale:**
- Current structure works
- Full restructuring requires extensive import updates
- Risk of breaking existing functionality
- Can always restructure later if needed

---

**Next Steps:**
1. Decide on restructuring approach (full vs minimal)
2. If minimal: Update documentation and add clear organization
3. If full: Execute migration plan step-by-step


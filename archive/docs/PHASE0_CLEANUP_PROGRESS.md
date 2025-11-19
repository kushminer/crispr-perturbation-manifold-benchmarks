# Phase 0 Cleanup Progress

**Date:** 2025-11-14  
**Branch:** `refactor/eval-framework`  
**Status:** 🔄 **IN PROGRESS**

---

## Completed Tasks

### ✅ Issue 0.1 — Create Clean Working Copy
- [x] Created branch `refactor/eval-framework`
- [x] Tagged `main` as `v0_legacy_repo`
- [x] Clean working copy established

### ✅ Issue 0.2 — Inventory and Classify Files
- [x] Created comprehensive `docs/code_inventory.md`
- [x] Classified all major files/folders as core/optional/deprecated
- [x] Identified cleanup targets

### ✅ Issue 0.3 — Remove Deprecated Code (Complete)
- [x] Removed temporary validation files:
  - `evaluation_framework/validation/tmp_rot.tsv`
  - `evaluation_framework/validation/tmp_x.tsv`
  - `evaluation_framework/validation/r_pseudobulk_matrix.tsv`
- [x] Updated `.gitignore` to exclude all `venv_*/` directories
- [x] Removed deprecated R environment files:
  - `.renvignore`
  - `renv.lock`
  - `renv/.gitignore`, `renv/activate.R`, `renv/settings.json`
- [x] Removed R project file: `perturbation_prediction-figures.Rproj`
- [x] Archived deprecated model scripts to `archive/deprecated_scripts/`:
  - `run_gears.py`, `run_scgpt.py`, `run_scfoundation.py`
  - `run_cpa.py`, `run_geneformer.py`, `run_scbert.py`, `run_uce.py`
  - `run_gears_debug.py`
- [x] Created `archive/README.md` documenting archived files

---

## Next Steps

### Immediate (Issue 0.3 continuation)
1. **Remove virtual environments:**
   - `paper/benchmark/venv/`
   - `paper/benchmark/venv_linear_model/`
   - Ensure they're in .gitignore

2. **Archive deprecated scripts:**
   - Create `archive/` directory
   - Move old model scripts (non-linear baselines) to archive
   - Document what was archived

3. **Remove deprecated configs:**
   - `paper/benchmark/conda_environments/` (use requirements.txt)
   - `paper/benchmark/renv/` (R environment not needed)
   - `paper/renv/` (R environment not needed)
   - `renv.lock` (root level)

### ✅ Issue 0.4 — Restructure Repo (Complete)
- [x] Documented current structure in README
- [x] Added repository structure section
- [x] Created restructuring plan for future reference

### ✅ Issue 0.5 — Dependency Cleanup (Complete)
- [x] Audited requirements.txt
- [x] Added missing dependency: `torch>=2.0` (required for embedding loaders)
- [x] Organized dependencies into sections (core, testing, optional)
- [x] Commented out optional dependencies (gseapy, mygene, requests)
- [x] Created dependency analysis document

---

## Files Removed So Far

- `evaluation_framework/validation/tmp_rot.tsv`
- `evaluation_framework/validation/tmp_x.tsv`
- `evaluation_framework/validation/r_pseudobulk_matrix.tsv`

---

## Files to Remove (Pending)

### Virtual Environments
- `paper/benchmark/venv/`
- `paper/benchmark/venv_linear_model/`

### Deprecated Configs
- `paper/benchmark/conda_environments/`
- `paper/benchmark/renv/`
- `paper/renv/`
- `renv.lock` (root)

### R Project Files
- `paper/perturbation_prediction-figures.Rproj`

---

## Files to Archive (Move to archive/)

### Old Model Scripts
- `paper/benchmark/src/run_gears.py`
- `paper/benchmark/src/run_scgpt.py`
- `paper/benchmark/src/run_scfoundation.py`
- Other `run_*.py` files for non-linear models

---

## Notes

- All cleanup is happening on `refactor/eval-framework` branch
- Original code preserved in `main` branch (tagged as `v0_legacy_repo`)
- No core functionality has been removed
- Temporary files and deprecated configs are being cleaned up first

---

**Last Updated:** 2025-11-14


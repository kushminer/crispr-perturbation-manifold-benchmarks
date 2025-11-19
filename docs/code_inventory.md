# Code Inventory & Classification

**Date:** 2025-11-14  
**Purpose:** Classify all code, configs, and notebooks as **core**, **optional**, or **deprecated** for Phase 0 cleanup.

**Classification:**
- **core** – Required for baselines, embeddings, or Adamson/Replogle preprocessing
- **optional** – Nice-to-have analyses, but not needed for upcoming work
- **deprecated** – Old experiments that won't be maintained

---

## Top-Level Structure

### `/evaluation_framework/` — **CORE** (Main evaluation framework)

This is the primary evaluation framework directory. Most content here is core.

#### `/evaluation_framework/src/` — **CORE**

| File/Directory | Classification | Notes |
|---------------|----------------|-------|
| `main.py` | **core** | CLI entry point for evaluation framework |
| `embeddings/` | **core** | Embedding loaders (scGPT, scFoundation, GEARS, PCA) |
| `eval_framework/` | **core** | LOGO, functional-class, combined analysis |
| `legacy_scripts/` | **core** | Legacy R/Python scripts for parity validation |
| `annotate_classes.py` | **optional** | Annotation utilities (may be useful) |
| `compare_datasets.py` | **optional** | Dataset comparison utilities |
| `compare_hardness_metrics.py` | **optional** | Hardness metric comparison |
| `export_python_reference_baseline.py` | **optional** | Reference baseline export |
| `generate_replogle_annotations.py` | **core** | Generate Replogle annotations |
| `generate_replogle_expression.py` | **core** | Generate Replogle expression data |
| `test_synthetic_annotations.py` | **optional** | Test utilities |

#### `/evaluation_framework/configs/` — **CORE**

| File/Directory | Classification | Notes |
|---------------|----------------|-------|
| `config_adamson.yaml` | **core** | Adamson dataset configuration |
| `config_replogle.yaml` | **core** | Replogle dataset configuration |
| `config_replogle_min.yaml` | **optional** | Minimal Replogle config (may be redundant) |
| `validation/` | **optional** | Validation configs/images |

#### `/evaluation_framework/data/` — **CORE**

| File/Directory | Classification | Notes |
|---------------|----------------|-------|
| `annotations/` | **core** | Functional class annotations |
| `gears_pert_data/` | **core** | GEARS perturbation data |
| `models/` | **core** | Model checkpoints (scGPT, scFoundation) |

#### `/evaluation_framework/docs/` — **CORE** (Documentation)

| File/Directory | Classification | Notes |
|---------------|----------------|-------|
| `EMBEDDING_CHECKPOINT_SOURCES.md` | **core** | Checkpoint provenance |
| `ISSUE13_*.md` | **core** | Issue #13 specifications |
| `SPRINT5_*.md` | **core** | Sprint 5 documentation |
| `PHASE0_REPO_CLEANUP_PLAN.md` | **core** | This cleanup plan |
| `code_inventory.md` | **core** | This file |
| `HARDNESS_METRIC_*.md` | **optional** | Hardness metric documentation |
| `project_management/` | **optional** | Project management docs |

#### `/evaluation_framework/tests/` — **CORE**

| File/Directory | Classification | Notes |
|---------------|----------------|-------|
| All test files | **core** | Unit tests for evaluation framework |

#### `/evaluation_framework/validation/` — **CORE**

| File/Directory | Classification | Notes |
|---------------|----------------|-------|
| `embedding_parity/` | **core** | Embedding parity validation |
| `embedding_parity_config.yaml` | **core** | Parity validation config |
| `embedding_parity_plots/` | **core** | Parity validation plots |
| `embedding_subsets/` | **core** | Subset data for parity |
| `legacy_runs/` | **core** | Legacy script outputs for comparison |
| `r_pseudobulk_matrix.tsv` | **deprecated** | Temporary validation file |
| `tmp_*.tsv` | **deprecated** | Temporary files (can delete) |

#### `/evaluation_framework/results/` — **OPTIONAL** (Generated results)

| File/Directory | Classification | Notes |
|---------------|----------------|-------|
| All results | **optional** | Generated results (can be regenerated) |

#### `/evaluation_framework/` (Root files)

| File | Classification | Notes |
|------|----------------|-------|
| `README.md` | **core** | Main README |
| `requirements.txt` | **core** | Python dependencies |
| `pytest.ini` | **core** | Test configuration |
| `GITHUB_ISSUES_TEMPLATE.md` | **optional** | Issue templates |
| `IMPLEMENTATION_STATUS.md` | **optional** | Status documentation |
| `TASK_LIST_NEXT_ITERATION.md` | **optional** | Task lists |
| `REORGANIZATION_NOTES.md` | **optional** | Reorganization notes |

---

### `/paper/` — **MIXED** (Original benchmark code)

This contains the original Nature benchmark code. Some is core (for reference/parity), some is deprecated.

#### `/paper/benchmark/src/` — **CORE** (Reference implementations)

| File/Directory | Classification | Notes |
|---------------|----------------|-------|
| `run_linear_pretrained_model.R` | **core** | Original R linear model (for parity) |
| `run_linear_pretrained_model.py` | **core** | Original Python linear model (for parity) |
| `extract_pert_embedding_pca.R` | **core** | Original PCA embedding script (for parity) |
| `extract_pert_embedding_from_gears.R` | **core** | Original GEARS embedding script (for parity) |
| `extract_gene_embedding_scgpt.py` | **core** | Original scGPT embedding script (for parity) |
| `extract_gene_embedding_scfoundation.py` | **core** | Original scFoundation embedding script (for parity) |
| `prepare_perturbation_data.py` | **core** | Data preparation script |
| `EXPLANATION_run_linear_pretrained_model.md` | **core** | Documentation |
| Other model scripts (gears, scgpt, etc.) | **deprecated** | Not needed for linear baselines |
| `run_*.py` (non-linear models) | **deprecated** | Deep learning models not needed |
| `*.ipynb` | **optional** | Notebooks (may be useful for reference) |

#### `/paper/benchmark/data/` — **CORE**

| File/Directory | Classification | Notes |
|---------------|----------------|-------|
| All data files | **core** | Original benchmark data |

#### `/paper/benchmark/submission/` — **OPTIONAL**

| File/Directory | Classification | Notes |
|---------------|----------------|-------|
| `run_perturbation_benchmark.R` | **optional** | Original benchmark runner (reference) |

#### `/paper/benchmark/conda_environments/` — **DEPRECATED**

| File/Directory | Classification | Notes |
|---------------|----------------|-------|
| All conda env files | **deprecated** | Not needed (use requirements.txt) |

#### `/paper/benchmark/renv/` — **DEPRECATED**

| File/Directory | Classification | Notes |
|---------------|----------------|-------|
| R environment files | **deprecated** | Not needed for Python framework |

#### `/paper/benchmark/venv*/` — **DEPRECATED**

| File/Directory | Classification | Notes |
|---------------|----------------|-------|
| Virtual environments | **deprecated** | Should not be in repo |

#### `/paper/notebooks/` — **OPTIONAL**

| File/Directory | Classification | Notes |
|---------------|----------------|-------|
| All notebooks | **optional** | Analysis notebooks (may be useful for reference) |

#### `/paper/plots/` — **OPTIONAL**

| File/Directory | Classification | Notes |
|---------------|----------------|-------|
| All plots | **optional** | Generated plots (can be regenerated) |

#### `/paper/source_data/` — **OPTIONAL**

| File/Directory | Classification | Notes |
|---------------|----------------|-------|
| All source data | **optional** | Source data files (may be needed for regeneration) |

#### `/paper/` (Root files)

| File | Classification | Notes |
|------|----------------|-------|
| `README.md` | **core** | Benchmark documentation |
| `LICENSE.md` | **core** | License file |
| `render_notebooks.sh` | **optional** | Notebook rendering script |
| `perturbation_prediction-figures.Rproj` | **deprecated** | R project file (not needed) |
| `renv/` | **deprecated** | R environment (not needed) |

---

### `/data/` — **CORE**

| File/Directory | Classification | Notes |
|---------------|----------------|-------|
| `annotations/` | **core** | Dataset annotations |

---

### `/reference_data/` — **OPTIONAL**

| File/Directory | Classification | Notes |
|---------------|----------------|-------|
| `synthetic_baseline/` | **optional** | Synthetic baseline data (may be useful) |

---

### `/illustrations/` — **OPTIONAL**

| File/Directory | Classification | Notes |
|---------------|----------------|-------|
| All illustration files | **optional** | Figure source files |

---

### Root Files

| File | Classification | Notes |
|------|----------------|-------|
| `README.md` | **core** | Main repository README |
| `copy_benchmark_project.sh` | **optional** | Utility script |
| `renv.lock` | **deprecated** | R environment lock (not needed) |

---

## Summary Statistics

### Core Files
- **Primary location:** `/evaluation_framework/` (most files)
- **Reference location:** `/paper/benchmark/src/` (original scripts for parity)
- **Data location:** `/data/`, `/paper/benchmark/data/`, `/evaluation_framework/data/`

### Optional Files
- Analysis notebooks
- Generated results/plots
- Documentation beyond core specs
- Comparison utilities

### Deprecated Files
- Old deep learning model scripts (not linear baselines)
- Conda/R environment files (use requirements.txt)
- Virtual environments in repo
- Temporary validation files
- R project files

---

## Cleanup Recommendations

### Safe to Delete
1. **Temporary files:**
   - `evaluation_framework/validation/tmp_*.tsv`
   - `evaluation_framework/validation/r_pseudobulk_matrix.tsv`

2. **Virtual environments:**
   - `paper/benchmark/venv/`
   - `paper/benchmark/venv_linear_model/`

3. **Environment configs (redundant):**
   - `paper/benchmark/conda_environments/` (use requirements.txt)
   - `paper/benchmark/renv/` (R environment not needed)
   - `paper/renv/` (R environment not needed)
   - `renv.lock` (root level)

4. **R project files:**
   - `paper/perturbation_prediction-figures.Rproj`

### Archive (Move to archive/)
1. **Old model scripts:**
   - `paper/benchmark/src/run_gears.py`
   - `paper/benchmark/src/run_scgpt.py`
   - `paper/benchmark/src/run_scfoundation.py`
   - `paper/benchmark/src/run_*.py` (non-linear models)

2. **Analysis notebooks (if not actively used):**
   - `paper/notebooks/` (may want to keep for reference)

3. **Generated plots:**
   - `paper/plots/` (can be regenerated)

### Keep (Core)
1. **All evaluation framework code** (`/evaluation_framework/src/`)
2. **All embedding loaders** (`/evaluation_framework/src/embeddings/`)
3. **All evaluation modules** (`/evaluation_framework/src/eval_framework/`)
4. **Original benchmark scripts** (`/paper/benchmark/src/` for parity)
5. **Data files** (all data directories)
6. **Configs** (`/evaluation_framework/configs/`)
7. **Tests** (`/evaluation_framework/tests/`)
8. **Documentation** (`/evaluation_framework/docs/` core files)

---

**Last Updated:** 2025-11-14  
**Status:** Inventory Complete — Ready for Cleanup


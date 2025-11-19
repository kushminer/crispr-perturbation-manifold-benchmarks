# Dependencies on `paper/` Directory

This document catalogs all dependencies that the evaluation framework has on the `paper/` directory.

## Background

The `paper/` directory is a copy of the original Nature Methods 2025 paper repository:
- **Paper**: [Ahlmann-Eltze et al. (2025)](https://www.nature.com/articles/s41592-025-02772-6)
- **Original Repository**: https://github.com/const-ae/linear_perturbation_prediction-Paper
- **Title**: "Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines"

The evaluation framework extends the paper's work with additional analysis goals (similarity analysis, LSFT, LOGO, etc.) and can optionally use the paper's code and data for compatibility and validation.

## Summary

**Status**: ⚠️ **Partial Dependencies** - The evaluation framework has some dependencies on `paper/` for:
1. **GEARS compatibility** (GEARS expects data in `paper/benchmark/data/gears_pert_data/`)
2. **Default paths** (can be overridden with command-line arguments)
3. **R parity validation** (needs access to paper's R scripts)

**Impact**: Most dependencies are **optional** or can be **overridden**. The framework can work independently if:
- Data is placed in the expected locations (via GEARS API or manual placement)
- Command-line arguments override default paths
- R validation is skipped (if not needed)

## Hard Dependencies (Code)

### 1. `split_logic.py` - GEARS Data Path

**File**: `src/goal_2_baselines/split_logic.py`  
**Lines**: 158-181

```python
# Tries to find paper/benchmark/data/gears_pert_data/
framework_root = current_file.parent.parent.parent.parent  # evaluation_framework/
repo_root = framework_root.parent  # repository root
pert_data_folder = repo_root / "paper" / "benchmark" / "data" / "gears_pert_data"
```

**Purpose**: GEARS expects data in this structure. This is a fallback when `pert_data_folder` is not specified.

**Override**: Pass `pert_data_folder` argument to `create_split_from_adata()` or `prepare_perturbation_splits()`

**Impact**: ⚠️ **Medium** - Only used when using GEARS for split generation without specifying a path.

### 2. `baseline_types.py` - GEARS GO Embeddings

**File**: `src/goal_2_baselines/baseline_types.py`  
**Lines**: 103

```python
pert_embedding_args={
    "source_csv": "../paper/benchmark/data/gears_pert_data/go_essential_all/go_essential_all.csv",
}
```

**Purpose**: Path to GEARS GO perturbation embeddings CSV file.

**Override**: The embedding loader can accept a different path via config or arguments.

**Impact**: ⚠️ **Low** - Only affects `lpm_gearsPertEmb` baseline. Other baselines don't need this.

### 3. `validate_r_parity.py` - R Working Directory

**File**: `src/goal_5_validation/validate_r_parity.py`  
**Lines**: 319

```python
default="../paper/benchmark",
help="Working directory for R script (default: ../paper/benchmark)",
```

**Purpose**: Default working directory for running R scripts from the paper.

**Override**: Use `--working_dir` command-line argument to specify different directory.

**Impact**: ⚠️ **Low** - Only needed for Goal 5 (R parity validation), which is optional.

### 4. `compare_paper_python_r.py` - R Working Directory

**File**: `src/goal_5_validation/compare_paper_python_r.py`  
**Lines**: 254

```python
default="../paper/benchmark",
help="Working directory (default: ../paper/benchmark)",
```

**Purpose**: Same as above - default working directory for R scripts.

**Override**: Use `--working_dir` command-line argument.

**Impact**: ⚠️ **Low** - Only needed for Goal 5 (R parity validation).

## Default Paths (Can Be Overridden)

These scripts use `paper/benchmark/data/gears_pert_data/` as **default paths** but accept command-line arguments to override them:

### 1. `run_all_datasets.py`

**File**: `src/goal_2_baselines/run_all_datasets.py`  
**Line**: 36

```python
"adamson": "../paper/benchmark/data/gears_pert_data/adamson/perturb_processed.h5ad",
```

**Override**: Use `--datasets` with custom paths or modify the `DATASET_PATHS` dictionary.

### 2. `run_all_datasets_similarity.py`

**File**: `src/goal_1_similarity/run_all_datasets_similarity.py`  
**Line**: 23

```python
"adata_path": "../paper/benchmark/data/gears_pert_data/adamson/perturb_processed.h5ad",
```

**Override**: Modify the `DATASETS` dictionary or pass custom paths.

### 3. `generate_replogle_expression.py`

**File**: `src/scripts/generate_replogle_expression.py`  
**Line**: 110

```python
default=Path("../../paper/benchmark/data/gears_pert_data"),
```

**Override**: Use `--gears_data_dir` command-line argument.

### 4. `build_embedding_subsets.py`

**File**: `src/embeddings/build_embedding_subsets.py`  
**Lines**: 24, 28

```python
REPO_ROOT / "paper/benchmark/data/gears_pert_data/go_essential_all/go_essential_all.csv"
REPO_ROOT / "paper/benchmark/data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad"
```

**Override**: Modify `REPO_ROOT` or paths in the script.

## Documentation/Examples

These are **documentation only** and don't affect code execution:

- **README.md**: Example commands use `../paper/benchmark/data/gears_pert_data/`
- **Tutorial notebooks**: Example paths reference `paper/benchmark/data/`
- **Shell scripts** (`run_lsft_all_datasets.sh`, `run_logo_all_datasets.sh`): Example paths
- **Documentation files**: Reference paper paths in examples

**Impact**: ✅ **None** - These are just examples for users.

## Making the Framework Independent

To make the evaluation framework fully independent of `paper/`:

### Option 1: Use GEARS API

GEARS will automatically download data to the expected location:
```python
from gears import PertData
pert_data = PertData("./paper/benchmark/data/gears_pert_data/")
pert_data.download_dataset("adamson")
```

This creates the expected structure even if `paper/` doesn't exist initially.

### Option 2: Override All Paths

Use command-line arguments or config files to specify custom paths:
```bash
python -m goal_2_baselines.baseline_runner \
  --adata_path /custom/path/to/data.h5ad \
  --split_config /custom/path/to/split.json
```

### Option 3: Use Synthetic Data

Tutorial notebooks support synthetic data generation (option 2) which doesn't require any external files.

### Option 4: Symlink/Copy Data

If you have data elsewhere, symlink or copy it to the expected `paper/benchmark/data/gears_pert_data/` structure.

## Recommendations

1. **For Most Users**: 
   - Use GEARS API - it will create the expected structure automatically
   - The `paper/` directory structure is just where GEARS expects data

2. **For Independent Deployment**:
   - Override all paths via command-line arguments
   - Or modify default paths in scripts
   - Skip Goal 5 (R validation) if R scripts aren't available

3. **For Tutorials**:
   - Use synthetic data (option 2) - no dependencies needed
   - Or download data using GEARS API first

## Summary Table

| Dependency | Location | Type | Override | Impact |
|------------|----------|------|----------|--------|
| GEARS data path | `split_logic.py` | Hard (fallback) | `pert_data_folder` arg | Medium |
| GO embeddings CSV | `baseline_types.py` | Hard | Embedding loader args | Low |
| R working dir | `validate_r_parity.py` | Default | `--working_dir` | Low |
| R working dir | `compare_paper_python_r.py` | Default | `--working_dir` | Low |
| Dataset defaults | `run_all_datasets.py` | Default | Modify dict/args | Low |
| Similarity defaults | `run_all_datasets_similarity.py` | Default | Modify dict | Low |
| Expression gen default | `generate_replogle_expression.py` | Default | `--gears_data_dir` | Low |
| Embedding subset paths | `build_embedding_subsets.py` | Hard | Modify script | Low |
| Documentation | Various READMEs | Example only | N/A | None |

## Conclusion

The evaluation framework has **minimal hard dependencies** on `paper/`. Most references are:
- ✅ **Default paths** that can be overridden
- ✅ **GEARS compatibility** (expected data structure)
- ✅ **Documentation examples** (no runtime impact)

The framework **can work independently** if you:
1. Override paths via command-line arguments
2. Use GEARS API to download data (creates expected structure)
3. Skip optional features (R validation, specific baselines)

**Recommendation**: Keep the `paper/` structure as the default for GEARS compatibility, but document that paths can be overridden for independent deployment.


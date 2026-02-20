# Code Validation Sprint - Overview

**Date:** 2025-11-23  
**Purpose:** Systematic validation of implementation correctness for global baseline, LOGO, and LSFT evaluations  
**Scope:** Validation only - no new methods or analyses

---

## Sprint Goal

Validate or refute the correctness of implementations for:
- Global baseline (Nature-style single-KO linear model)
- LOGO (Leave-One-GO-Class-Out)
- LSFT (1%, 5%, 10% top neighbors)

using PCA, scGPT, scFoundation, GEARS, random embeddings, and the 8 Nature baselines.

---

## Environment Setup

### Python Version
- **Tested with:** Python 3.10+
- **Recommended:** Python 3.10 or 3.11

### Dependency Management

Dependencies are pinned in `requirements.txt` with exact versions validated on 2025-11-23.

To set up a fresh environment:

```bash
cd crispr-perturbation-manifold-benchmarks

# Using pip
pip install -r requirements.txt

# Using conda (recommended)
conda create -n lpm-validation python=3.10
conda activate lpm-validation
pip install -r requirements.txt
```

### Key Dependencies

- **numpy==1.26.4** - Numerical operations
- **pandas==2.2.2** - Data manipulation
- **scipy==1.14.1** - Scientific computing
- **scikit-learn==1.5.1** - PCA and machine learning utilities
- **anndata==0.12.2** - Single-cell data structures
- **torch>=2.0** - PyTorch for embedding loaders
- **pytest>=7.0** - Testing framework

---

## Main Pipeline Commands

### Global Baseline (Goal 2)

**Entry Point:** `src/goal_2_baselines/baseline_runner.py`

**Command:**
```bash
cd crispr-perturbation-manifold-benchmarks
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

python -m goal_2_baselines.run_all_datasets \
    --output_dir results/goal_2_baselines
```

Or for a single dataset:
```bash
python -m goal_2_baselines.baseline_runner \
    --adata_path data/datasets/adamson/perturb_processed.h5ad \
    --split_config results/goal_2_baselines/splits/adamson_split_seed1.json \
    --output_dir results/goal_2_baselines/adamson_reproduced \
    --pca_dim 10 \
    --ridge_penalty 0.1 \
    --seed 1
```

### LOGO Evaluation (Goal 3)

**Entry Point:** `src/goal_3_prediction/functional_class_holdout/logo.py`

**Command:**
```bash
cd crispr-perturbation-manifold-benchmarks
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

python -m goal_3_prediction.functional_class_holdout.logo \
    --adata_path data/datasets/adamson/perturb_processed.h5ad \
    --annotation_path data/annotations/adamson_functional_classes_go.tsv \
    --dataset_name adamson \
    --output_dir results/goal_3_prediction/functional_class_holdout/adamson \
    --class_name Transcription \
    --pca_dim 10 \
    --ridge_penalty 0.1 \
    --seed 1
```

### LSFT Evaluation (Goal 3)

**Entry Point:** `src/goal_3_prediction/lsft/lsft.py`

**Command:**
```bash
cd crispr-perturbation-manifold-benchmarks
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

python -m goal_3_prediction.lsft.lsft \
    --adata_path data/datasets/adamson/perturb_processed.h5ad \
    --split_config results/goal_2_baselines/splits/adamson_split_seed1.json \
    --dataset_name adamson \
    --output_dir results/goal_3_prediction/lsft/adamson \
    --baseline_type lpm_selftrained \
    --top_pcts 0.01 0.05 0.10 \
    --pca_dim 10 \
    --ridge_penalty 0.1 \
    --seed 1
```

### LSFT with Resampling

**Entry Point:** `src/goal_3_prediction/lsft/run_lsft_with_resampling.py`

**Command:**
```bash
cd crispr-perturbation-manifold-benchmarks
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

python -m goal_3_prediction.lsft.run_lsft_with_resampling \
    --adata_path data/datasets/adamson/perturb_processed.h5ad \
    --split_config results/goal_2_baselines/splits/adamson_split_seed1.json \
    --dataset_name adamson \
    --baseline_type lpm_selftrained \
    --output_dir results/goal_3_prediction/lsft_resampling/adamson \
    --n_boot 1000 \
    --n_perm 10000 \
    --top_pcts 0.01 0.05 0.10
```

### LOGO with Resampling

**Entry Point:** `src/goal_3_prediction/functional_class_holdout/logo_resampling.py`

**Command:**
```bash
cd crispr-perturbation-manifold-benchmarks
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

python -m goal_3_prediction.functional_class_holdout.logo_resampling \
    --adata_path data/datasets/adamson/perturb_processed.h5ad \
    --annotation_path data/annotations/adamson_functional_classes_go.tsv \
    --dataset_name adamson \
    --output_dir results/goal_3_prediction/functional_class_holdout_resampling/adamson \
    --class_name Transcription \
    --n_boot 1000 \
    --n_perm 10000
```

---

## Shell Scripts for Batch Runs

### Run All Baselines (All Datasets)

```bash
cd crispr-perturbation-manifold-benchmarks
./run_all_datasets.py  # or equivalent script
```

### Run All LSFT Evaluations

```bash
cd crispr-perturbation-manifold-benchmarks
./run_lsft_resampling_all.sh
```

### Run All LOGO Evaluations

```bash
cd crispr-perturbation-manifold-benchmarks
./run_logo_resampling_all.sh
```

---

## Data Paths

### Dataset Files
- **Adamson:** `data/datasets/adamson/perturb_processed.h5ad` (or path specified in paper/)
- **K562:** `data/datasets/replogle_k562_essential/perturb_processed.h5ad`
- **RPE1:** `data/datasets/replogle_rpe1_essential/perturb_processed.h5ad`

### Split Configurations
- **Adamson:** `results/goal_2_baselines/splits/adamson_split_seed1.json`
- **K562:** `results/goal_2_baselines/splits/replogle_k562_essential_split_seed1.json`
- **RPE1:** `results/goal_2_baselines/splits/replogle_rpe1_essential_split_seed1.json`

### Functional Class Annotations
- **Adamson:** `data/annotations/adamson_functional_classes_go.tsv`
- **K562:** `data/annotations/replogle_k562_functional_classes_go.tsv` (or improved version)
- **RPE1:** `data/annotations/replogle_rpe1_functional_classes_go.tsv`

---

## Validation Phases

This validation sprint is organized into 7 phases:

1. **Phase 0:** Environment & Reproducibility (this file)
2. **Phase 1:** Data Pipeline & Splits Check (`01_data_pipeline_check.md`)
3. **Phase 2:** Embeddings & Features Check (`02_embedding_check.md`)
4. **Phase 3:** Linear Baseline Reproduction Check (`03_baseline_reproduction_check.md`)
5. **Phase 4:** LOGO Implementation Check (`04_LOGO_split_check.md`)
6. **Phase 5:** LSFT Implementation Check (`05_LSFT_implementation_check.md`)
7. **Phase 6:** Metrics, Bootstrap & Permutations Check (`06_metrics_and_resampling_check.md`)
8. **Phase 7:** Summary & Sign-Off (`07_summary_report.md`)

Each phase has:
- A markdown checklist file
- A validation notebook in `notebooks/`
- Logs and outputs in `logs/` and `figures/`

### Generating Validation Plots

To generate visual validation plots (recommended for publication):

```bash
cd crispr-perturbation-manifold-benchmarks
python deliverables/audits/manifold_law_code_validation_2025-11-23/gen_validation_plots.py
```

This generates 6 essential plots proving pipeline correctness:
1. `pca_explained_variance_train_vs_all.png` - PCA fit correctness
2. `split_overlap_check.png` - Train/test/val disjoint verification
3. `baseline_toy_truth_vs_pred.png` - Baseline correctness
4. `lsft_neighbor_counts_topK.png` - LSFT neighbor selection logic
5. `bootstrap_distribution_example.png` - Bootstrap CI correctness
6. `permutation_null_distribution.png` - Permutation test correctness

---

## Quick Test

To verify the environment is set up correctly:

```bash
cd crispr-perturbation-manifold-benchmarks
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Test imports
python -c "from goal_2_baselines.baseline_runner import run_all_baselines; print('✓ Imports work')"
python -c "from goal_3_prediction.lsft.lsft import evaluate_lsft; print('✓ LSFT imports work')"
python -c "from goal_3_prediction.functional_class_holdout.logo import run_logo_evaluation; print('✓ LOGO imports work')"
```

---

## Notes

- All commands assume you're in the `crispr-perturbation-manifold-benchmarks` directory
- `PYTHONPATH` must include `src/` directory for module imports
- Paths may need to be adjusted based on actual data locations
- Check `data/README.md` for data download instructions if datasets are missing

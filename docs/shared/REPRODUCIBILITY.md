# Baseline Reproduction - Reproducibility Documentation

**Date:** 2025-11-17  
**Status:** ✅ **Baseline Reproduction Complete**

---

## Overview

This document describes the baseline reproduction process for the linear perturbation prediction models, ensuring reproducibility and validation against the original R implementation.

---

## Baseline Models

We reproduce **9 baseline models** (8 linear + 1 mean-response):

1. **lpm_selftrained** - PCA on training genes and perturbations
2. **lpm_randomPertEmb** - PCA on genes, random perturbation embeddings
3. **lpm_randomGeneEmb** - Random gene embeddings, PCA on perturbations
4. **lpm_scgptGeneEmb** - scGPT gene embeddings, PCA on perturbations
5. **lpm_scFoundationGeneEmb** - scFoundation gene embeddings, PCA on perturbations
6. **lpm_gearsPertEmb** - PCA on genes, GEARS perturbation embeddings
7. **lpm_k562PertEmb** - PCA on genes, K562 cross-dataset perturbation embeddings
8. **lpm_rpe1PertEmb** - PCA on genes, RPE1 cross-dataset perturbation embeddings
9. **mean_response** - Mean expression baseline

---

## Core Model

All linear baselines use the same mathematical model:

**Y ≈ A × K × B**

Where:
- **Y** (genes × perturbations): Pseudobulk expression changes (target - control)
- **A** (genes × d): Gene embedding matrix
- **K** (d × d): Learned interaction matrix (via ridge regression)
- **B** (d × perturbations): Perturbation embedding matrix

**Ridge Regression:**
```
K = argmin ||Y - A K B||² + λ(||A||² + ||B||²)
```

---

## Implementation Details

### Data Processing

1. **Pseudobulk Expression Changes (Y)**
   - Compute mean expression per condition
   - Subtract control baseline: `Y = mean(perturbation) - mean(control)`
   - Same for all baselines

2. **Train/Test/Val Split**
   - Use original split logic from `prepare_perturbation_data.py`
   - Default: 70% train, 15% test, 15% val
   - Control always in training set
   - Seed: 1 (for reproducibility)

### Embedding Construction

#### Gene Embeddings (A)

- **Training Data PCA**: `PCA(n_components=10).fit_transform(Y_train)` → (genes × 10)
- **Random**: Gaussian random matrix (genes × 10)
- **scGPT/scFoundation**: Load pretrained embeddings, align to target genes

#### Perturbation Embeddings (B)

- **Training Data PCA**: `PCA(n_components=10).fit_transform(Y_train.T).T` → (10 × perturbations)
- **Random**: Gaussian random matrix (10 × perturbations)
- **GEARS**: Load GO similarity embeddings, align by gene symbol
- **K562/RPE1**: Fit PCA on source dataset, transform target perturbations

### Cross-Dataset Baselines

For `lpm_k562PertEmb` and `lpm_rpe1PertEmb`:

1. Load source dataset (K562 or RPE1)
2. Compute pseudobulk expression
3. Fit PCA on source perturbations
4. Align genes between source and target
5. Transform target perturbations using fitted PCA

**Key Point:** Same PCA transformation used for all target datasets.

---

## Parameters

**Default Parameters:**
- PCA dimension: 10
- Ridge penalty: 0.1
- Random seed: 1
- Train/Test/Val split: 70/15/15

**Dataset-Specific:**
- Adamson: 87 conditions → 61 train, 12 test, 14 val

---

## Running Baselines

### Command

```bash
cd evaluation_framework
PYTHONPATH=src python -m goal_2_baselines.run_all \
    --adata_path ../paper/benchmark/data/gears_pert_data/adamson/perturb_processed.h5ad \
    --split_config results/adamson_split_seed1.json \
    --output_dir results/baselines/adamson_reproduced \
    --pca_dim 10 \
    --ridge_penalty 0.1 \
    --seed 1
```

### Output

- `baseline_results_reproduced.csv` - Summary with mean Pearson r per baseline
- Individual baseline predictions (if saved)

---

## Validation Against R

### Comparison Script

```bash
python -m goal_2_baselines.validate_against_r \
    --python_results results/baselines/adamson_reproduced/baseline_results_reproduced.csv \
    --r_results path/to/r/results.csv \
    --output_dir validation/baseline_comparison \
    --tolerance 0.01
```

### Expected Agreement

- Mean Pearson r: Within ±0.01 of R results
- Individual perturbation predictions: High correlation (>0.99)

---

## Results (Adamson, seed=1)

| Baseline | Mean Pearson r | Mean L2 | Test Perturbations |
|----------|---------------|---------|-------------------|
| lpm_selftrained | 0.946 | 2.265 | 12 |
| lpm_rpe1PertEmb | 0.937 | 1.943 | 12 |
| lpm_k562PertEmb | 0.929 | 2.413 | 12 |
| lpm_scgptGeneEmb | 0.811 | 3.733 | 12 |
| lpm_scFoundationGeneEmb | 0.777 | 3.979 | 12 |
| lpm_gearsPertEmb | 0.748 | 4.307 | 12 |
| lpm_randomGeneEmb | 0.721 | 4.344 | 12 |
| mean_response | 0.720 | 4.350 | 12 |
| lpm_randomPertEmb | 0.707 | 4.536 | 12 |

**Metrics:**
- **Pearson r**: Correlation coefficient (higher is better, range: -1 to 1)
- **L2**: Euclidean distance (lower is better), L2 = sqrt(sum((y_true - y_pred)²))

---

## Dependencies

### Required

- `numpy>=1.24`
- `pandas>=2.0`
- `scipy>=1.11`
- `scikit-learn>=1.3`
- `anndata>=0.10`
- `torch>=2.0` (for scGPT/scFoundation)

### Optional

- `gears` (for RPE1 baseline via PertData, though we use direct path now)

---

## File Structure

```
evaluation_framework/
├── src/
│   └── baselines/
│       ├── baseline_types.py      # Baseline type definitions
│       ├── baseline_runner.py    # Core runner logic
│       ├── split_logic.py         # Split generation
│       ├── run_all.py             # CLI entry point
│       └── validate_against_r.py # Validation script
├── results/
│   └── baselines/
│       └── adamson_reproduced/
│           └── baseline_results_reproduced.csv
└── docs/
    └── REPRODUCIBILITY.md         # This file
```

---

## Known Issues / Limitations

1. **Gene Alignment**: Some genes may not match between datasets (handled with zero padding)
2. **Perturbation Alignment**: GEARS embeddings may not cover all perturbations (handled with zero padding)
3. **RPE1 Loading**: Now uses direct path instead of PertData (faster, more reliable)

---

## Reproducibility Checklist

- [x] All baselines implemented
- [x] Split logic matches original
- [x] Pseudobulk computation matches original
- [x] PCA implementation matches original
- [x] Ridge regression matches original
- [ ] Validated against R results (pending R results)
- [x] Results documented
- [x] Code documented

---

## References

- Original paper: [Nature paper reference]
- R implementation: `paper/benchmark/src/run_linear_pretrained_model.R`
- Python implementation: `evaluation_framework/src/baselines/`

---

**Last Updated:** 2025-11-17


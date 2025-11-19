# R vs Python Baseline Validation

## Overview

This document describes the validation of Python baseline implementations against the original R implementation from the Nature paper.

## Validation Script

**Location:** `src/baselines/validate_r_parity.py`

**Purpose:** Run a random subset of baselines and test perturbations to validate that Python and R implementations produce matching results.

## Usage

```bash
python -m goal_2_baselines.validate_r_parity \
    --dataset_name adamson \
    --working_dir ../paper/benchmark \
    --n_baselines 4 \
    --n_test_perturbations 5 \
    --seed 1 \
    --tolerance 0.01
```

### Parameters

- `--dataset_name`: Dataset name (default: adamson)
- `--working_dir`: Working directory for R script (default: ../paper/benchmark)
- `--n_baselines`: Number of baselines to test (default: 4)
- `--n_test_perturbations`: Number of test perturbations to compare (default: 5)
- `--pca_dim`: PCA dimension (default: 10)
- `--ridge_penalty`: Ridge penalty (default: 0.1)
- `--seed`: Random seed (default: 1)
- `--tolerance`: Numerical tolerance for agreement (default: 0.01)
- `--output_dir`: Output directory for validation results (default: validation/r_parity)

## Supported Baselines

The script validates the following baselines:

1. **lpm_selftrained**: PCA on training data for both gene and perturbation embeddings
2. **lpm_randomPertEmb**: PCA on training data for genes, random for perturbations
3. **lpm_randomGeneEmb**: Random for genes, PCA on training data for perturbations
4. **lpm_k562PertEmb**: PCA on training data for genes, precomputed K562 PCA for perturbations
5. **lpm_rpe1PertEmb**: PCA on training data for genes, precomputed RPE1 PCA for perturbations

## Validation Process

1. **Select Random Subset**: Randomly selects `n_baselines` baselines and `n_test_perturbations` test perturbations
2. **Run R Baseline**: Executes the R script with the same parameters
3. **Run Python Baseline**: Executes the Python baseline with the same parameters
4. **Compare Predictions**: Computes Pearson correlation, L2 distance, and max difference for each test perturbation
5. **Generate Report**: Saves a CSV summary with comparison metrics

## Results Format

The validation script generates:

- **`r_parity_validation.csv`**: Summary table with:
  - `baseline`: Baseline name
  - `mean_pearson_r`: Mean Pearson correlation across test perturbations
  - `mean_l2`: Mean L2 distance across test perturbations
  - `max_diff`: Maximum absolute difference across all predictions
  - `within_tolerance`: Number of perturbations within tolerance / total
  - `n_test_perturbations`: Number of test perturbations compared

## Current Status

### Initial Validation Results

**Date:** 2025-11-17

**Test Configuration:**
- Dataset: Adamson
- Seed: 1
- Baselines tested: 2 (lpm_randomPertEmb, lpm_selftrained)
- Test perturbations: 5

**Results:**

| Baseline | Mean Pearson r | Mean L2 | Max Diff | Within Tolerance |
|----------|----------------|---------|----------|------------------|
| lpm_randomPertEmb | 0.090 | 41.15 | 5.30 | 0/5 |
| lpm_selftrained | 0.128 | 41.01 | 5.33 | 0/5 |

### Observations

1. **Random Embeddings**: Low correlation (0.09) is expected for random embeddings due to stochasticity and potential differences in random number generation between R and Python.

2. **Self-Trained Baseline**: The correlation (0.13) is still relatively low, suggesting potential differences in:
   - PCA implementation (R uses `irlba::prcomp_irlba`, Python uses `sklearn.decomposition.PCA`)
   - Pseudobulking method
   - Numerical precision

3. **Next Steps**:
   - Investigate PCA implementation differences
   - Compare pseudobulking methods
   - Test with precomputed embeddings (K562/RPE1) for deterministic comparison
   - Adjust tolerance based on expected numerical differences

## Known Issues

1. **RPE1 Embedding Path**: The `lpm_rpe1PertEmb` baseline requires the precomputed RPE1 embedding file. Ensure the path is correctly configured.

2. **Random Seed Synchronization**: Random embeddings may not match exactly due to different random number generators in R and Python.

3. **PCA Implementation**: R uses `irlba::prcomp_irlba` (truncated SVD), while Python uses `sklearn.decomposition.PCA` (full SVD). This may cause small numerical differences.

## Acceptance Criteria

For a baseline to pass validation:

- **Pearson r ≥ 0.99**: For deterministic baselines (non-random embeddings)
- **Max difference ≤ tolerance**: For all test perturbations
- **Within tolerance count = total count**: All perturbations must be within tolerance

## Future Work

1. **Investigate PCA Differences**: Compare `irlba::prcomp_irlba` vs `sklearn.decomposition.PCA`
2. **Pseudobulking Validation**: Ensure pseudobulking methods match exactly
3. **Cross-Dataset Embeddings**: Validate K562 and RPE1 precomputed embeddings
4. **Statistical Testing**: Add significance tests for agreement
5. **Visualization**: Create plots comparing R vs Python predictions

---

**Last Updated:** 2025-11-17


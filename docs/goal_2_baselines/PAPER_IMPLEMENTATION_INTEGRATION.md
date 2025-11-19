# Paper Implementation Integration

## Overview

The paper's validated Python implementation (`run_linear_pretrained_model.py`) has been integrated into our evaluation framework. This implementation matches the R version with Pearson correlation ≥ 0.999, providing a validated reference implementation.

## Location

**Module:** `src/baselines/paper_implementation.py`

This module contains:
- `solve_y_axb()`: Ridge regression solver (matches R implementation)
- `pseudobulk_adata()`: Pseudobulking function (matches R implementation)
- `run_paper_baseline()`: Main function to run baselines using paper's implementation

## Integration

The paper's implementation is integrated into our baseline runner via the `use_paper_implementation` flag in `run_single_baseline()`.

### Usage

```python
from baselines.baseline_runner import run_single_baseline
from baselines.baseline_types import BaselineType, get_baseline_config

# Use paper's implementation
result = run_single_baseline(
    Y_train=Y_train,
    Y_test=Y_test,
    config=get_baseline_config(BaselineType.SELFTRAINED),
    gene_names=gene_names,
    use_paper_implementation=True,  # Use paper's implementation
    adata_path=adata_path,
    split_config=split_config,
)
```

## Supported Baselines

The paper's implementation supports:

1. **lpm_selftrained**: PCA on training data for both gene and perturbation embeddings
2. **lpm_randomPertEmb**: PCA on training data for genes, random for perturbations
3. **lpm_randomGeneEmb**: Random for genes, PCA on training data for perturbations
4. **lpm_k562PertEmb**: PCA on training data for genes, precomputed K562 PCA for perturbations
5. **lpm_rpe1PertEmb**: PCA on training data for genes, precomputed RPE1 PCA for perturbations

## Key Differences from Our Implementation

1. **PCA Implementation**: Uses `sklearn.decomposition.PCA` (same as our implementation)
2. **Solver**: Uses `scipy.linalg.solve` with `assume_a='sym'` (slightly different from our implementation)
3. **Pseudobulking**: Custom implementation matching R's `glmGamPoi::pseudobulk`
4. **Data Processing**: Matches R's exact data processing pipeline

## Validation

The paper's implementation has been validated against the R implementation:

| Baseline | Mean Pearson r | Mean L2 | Max Diff |
|----------|----------------|---------|----------|
| **lpm_selftrained** | **0.999030** | 1.429 | 0.766 |
| **lpm_k562PertEmb** | **0.999383** | 1.281 | 0.315 |

See `docs/PAPER_PYTHON_VS_R_COMPARISON.md` for full validation results.

## When to Use

- **Use paper's implementation** when:
  - You need exact parity with R implementation for validation
  - You're debugging differences between implementations
  - You want a reference implementation for comparison

- **Use our implementation** (default) when:
  - Running production baseline evaluations
  - You need additional baseline types (scGPT, scFoundation, GEARS)
  - You want more flexible embedding loading
  - You're developing new baselines

**Note:** Results from paper's implementation are not retained in the results directory. The paper's implementation is available for validation purposes only.

## Future Work

1. Add support for scGPT/scFoundation embeddings in paper's implementation
2. Add support for GEARS embeddings in paper's implementation
3. Create unified interface that automatically selects best implementation
4. Add performance benchmarks comparing both implementations

---

**Last Updated:** 2025-11-17


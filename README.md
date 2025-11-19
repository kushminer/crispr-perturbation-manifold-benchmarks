# Linear Perturbation Prediction Evaluation Framework (v2 - Resampling-Enabled)

## Overview

This is **version 2** of the evaluation framework, enhanced with **resampling-based statistical methods** for LSFT (Local Similarity-Filtered Training) evaluation. This version maintains **point-estimate parity** with v1 while adding:

- **Bootstrap confidence intervals** for mean performance metrics
- **Permutation tests** for baseline comparisons
- **Bootstrapped regression** for hardness-performance relationships
- **Enhanced visualizations** with uncertainty quantification

## Version Information

- **v1 Baseline**: Original evaluation framework (see original repository)
- **v2 Resampling-Enabled**: This repository (adds statistical resampling)

### Key Differences from v1

| Feature | v1 | v2 |
|---------|----|----|
| Point estimates | ✅ | ✅ (identical) |
| Bootstrap CIs | ❌ | ✅ |
| Permutation tests | ❌ | ✅ |
| Hardness regression CIs | ❌ | ✅ |
| Per-perturbation output | Basic | Standardized (JSONL/Parquet) |

**Note**: All point estimates (means, correlations, etc.) are **identical** between v1 and v2. v2 only adds confidence intervals and significance tests.

## Quick Start

### Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install additional resampling dependencies** (if not in requirements.txt):
   ```bash
   pip install scipy statsmodels  # For bootstrap and permutation tests
   ```

### Running LSFT with Resampling

```bash
# Run LSFT with bootstrap CIs
PYTHONPATH=src python -m goal_3_prediction.lsft.lsft \
    --adata_path path/to/data.h5ad \
    --split_config results/goal_2_baselines/splits/adamson_split_seed1.json \
    --baseline_type lpm_selftrained \
    --output_dir results/lsft_with_resampling/ \
    --n_boot 1000  # Number of bootstrap samples
```

## New Features (Sprint 11)

### 1. Bootstrap Confidence Intervals

All LSFT summary metrics now include confidence intervals:

```python
{
    "baseline": "lpm_selftrained",
    "mean_pearson_r": 0.75,
    "pearson_r_ci_lower": 0.72,
    "pearson_r_ci_upper": 0.78,
    "mean_l2": 5.5,
    "l2_ci_lower": 5.2,
    "l2_ci_upper": 5.8,
    "n_boot": 1000
}
```

### 2. Permutation Tests

Paired baseline comparisons include p-values:

```python
{
    "baseline1": "lpm_scgptGeneEmb",
    "baseline2": "lpm_randomGeneEmb",
    "mean_delta_pearson_r": 0.15,
    "delta_ci_lower": 0.12,
    "delta_ci_upper": 0.18,
    "p_value": 0.001,
    "n_perm": 10000
}
```

### 3. Hardness-Performance Regression

Hardness plots include bootstrapped CI bands for:
- Regression slope
- Correlation coefficient (r)
- R²

### 4. Standardized Per-Perturbation Output

LSFT now emits standardized per-perturbation files (JSONL/Parquet) with:
- `perturbation`: Perturbation name
- `pearson_r`: Pearson correlation
- `l2`: L2 distance
- `hardness`: Top-K cosine similarity
- `embedding_similarity`: Similarity in embedding space
- `split_fraction`: Training data fraction used

## Repository Structure

```
evaluation_framework_v2/
├── src/
│   ├── goal_1_similarity/   # Goal 1: Cosine similarity (unchanged)
│   ├── goal_2_baselines/    # Goal 2: Baseline reproduction (unchanged)
│   ├── goal_3_prediction/   # Goal 3: LSFT + LOGO (enhanced with resampling)
│   ├── goal_4_analysis/     # Goal 4: Statistical analysis (enhanced)
│   ├── goal_5_validation/   # Goal 5: Parity validation (unchanged)
│   ├── shared/              # Shared utilities (unchanged)
│   └── stats/               # NEW: Resampling utilities
│       ├── bootstrapping.py # Bootstrap CI functions
│       └── permutation.py   # Permutation test functions
├── configs/                 # Dataset configurations (unchanged)
├── data/                    # Data files (unchanged)
├── docs/                    # Documentation (enhanced)
│   └── resampling.md        # NEW: Resampling documentation
├── tests/                   # Unit tests (enhanced)
└── results/                 # Generated results
```

## Migration from v1

If you have results from v1:

1. **Point estimates are identical** - you can directly compare means, correlations, etc.
2. **New fields** - v2 outputs include CI fields that v1 doesn't have
3. **No breaking changes** - all v1 outputs are still valid and comparable

## Documentation

- **Resampling Guide**: `docs/resampling.md` - How to use resampling features
- **LSFT Documentation**: `docs/goal_3_prediction/` - LSFT implementation details
- **API Reference**: See docstrings in `src/stats/` modules

## Testing

```bash
# Run all tests
PYTHONPATH=src pytest tests/ -v

# Run resampling-specific tests
PYTHONPATH=src pytest tests/test_bootstrapping.py tests/test_permutation.py -v
```

## Citation

If using this resampling-enabled version, cite:

1. **Original Paper**: Ahlmann-Eltze et al. (2025) - Nature Methods 2025
2. **Original Repository**: https://github.com/const-ae/linear_perturbation_prediction-Paper
3. **This Framework**: Note that this extends the evaluation with resampling-based statistical methods

## Changelog

See `CHANGELOG.md` for detailed version history and Sprint 11 enhancements.

---

**Status**: 🚧 In Development (Sprint 11)

This is the resampling-enabled v2. For the original v1 baseline, see the original repository.


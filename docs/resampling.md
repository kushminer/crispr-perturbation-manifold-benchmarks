# Resampling Engine Documentation

**Sprint 11 - Resampling-Enabled LSFT Evaluation**

This document describes the resampling features added to the LSFT (Local Similarity-Filtered Training) evaluation framework.

## Overview

The resampling engine (v2) extends the original evaluation framework (v1) with statistical resampling methods:

- **Bootstrap confidence intervals** for mean performance metrics
- **Permutation tests** for baseline comparisons
- **Bootstrapped regression** for hardness-performance relationships
- **Enhanced visualizations** with uncertainty quantification

**Key Principle**: v2 maintains **point-estimate parity** with v1. All means, correlations, and point estimates are identical between v1 and v2. v2 only adds confidence intervals and significance tests.

## Quick Start

### Running LSFT with Resampling

```bash
PYTHONPATH=src python -m goal_3_prediction.lsft.run_lsft_with_resampling \
    --adata_path data/adamson_processed.h5ad \
    --split_config results/goal_2_baselines/splits/adamson_split_seed1.json \
    --dataset_name adamson \
    --baseline_type lpm_selftrained \
    --output_dir results/lsft_resampling/ \
    --n_boot 1000 \
    --n_perm 10000
```

### Running LOGO with Resampling

```bash
PYTHONPATH=src python -m goal_3_prediction.functional_class_holdout.logo_resampling \
    --adata_path data/adamson_processed.h5ad \
    --annotation_path data/annotations/adamson_annotations.tsv \
    --dataset_name adamson \
    --output_dir results/logo_resampling/ \
    --class_name Transcription \
    --n_boot 1000 \
    --n_perm 10000
```

## Features

### 1. Bootstrap Confidence Intervals

All summary metrics now include 95% confidence intervals computed using percentile bootstrap:

```python
{
    "pearson_r": {
        "mean": 0.75,
        "ci_lower": 0.72,
        "ci_upper": 0.78,
        "std": 0.05
    },
    "l2": {
        "mean": 5.5,
        "ci_lower": 5.2,
        "ci_upper": 5.8,
        "std": 0.3
    },
    "n_boot": 1000,
    "alpha": 0.05
}
```

**Function**: `stats.bootstrapping.bootstrap_mean_ci()`

### 2. Permutation Tests

Paired baseline comparisons include p-values from sign-flip permutation tests:

```python
{
    "baseline1": "lpm_scgptGeneEmb",
    "baseline2": "lpm_randomGeneEmb",
    "mean_delta": 0.15,
    "delta_ci_lower": 0.12,
    "delta_ci_upper": 0.18,
    "p_value": 0.001,
    "n_perm": 10000
}
```

**Function**: `stats.permutation.paired_permutation_test()`

### 3. Hardness-Performance Regression

Hardness plots include bootstrapped CI bands for regression statistics:

```python
{
    "slope": 1.25,
    "slope_ci_lower": 1.15,
    "slope_ci_upper": 1.35,
    "r": 0.85,
    "r_ci_lower": 0.80,
    "r_ci_upper": 0.90,
    "r_squared": 0.72,
    "r_squared_ci_lower": 0.65,
    "r_squared_ci_upper": 0.80,
    "p_value": 0.001,
    "n_boot": 1000
}
```

**Function**: `goal_3_prediction.lsft.hardness_regression_resampling.bootstrap_hardness_regression()`

### 4. Standardized Output Format

LSFT results are saved in multiple formats:

- **CSV**: Human-readable, backward compatible
- **JSONL**: Machine-readable, one JSON object per line
- **Parquet**: Efficient binary format for large datasets

**Fields**:
- `test_perturbation`: Perturbation name
- `baseline_type`: Baseline identifier
- `top_pct`: Top percentage used
- `pearson_r`: Local Pearson correlation
- `l2`: Local L2 distance
- `hardness`: Top-K cosine similarity (mean)
- `embedding_similarity`: Same as hardness
- `split_fraction`: Fraction of training data used

**Function**: `goal_3_prediction.lsft.lsft_resampling.standardize_lsft_output()`

## API Reference

### Statistics Module (`src/stats/`)

#### `bootstrap_mean_ci(values, n_boot=1000, alpha=0.05, random_state=None)`

Compute bootstrap confidence interval for mean.

**Parameters**:
- `values`: Array-like sample values
- `n_boot`: Number of bootstrap samples (default: 1000)
- `alpha`: Significance level (default: 0.05 for 95% CI)
- `random_state`: Random seed for reproducibility

**Returns**: `(mean, ci_lower, ci_upper)`

#### `paired_permutation_test(deltas, n_perm=10000, alternative="two-sided", random_state=None)`

Perform paired permutation test using sign-flip permutations.

**Parameters**:
- `deltas`: Array-like paired differences
- `n_perm`: Number of permutations (default: 10000)
- `alternative`: "two-sided", "greater", or "less"
- `random_state`: Random seed for reproducibility

**Returns**: `(mean_delta, p_value)`

### LSFT Resampling Module (`src/goal_3_prediction/lsft/lsft_resampling.py`)

#### `evaluate_lsft_with_resampling(...)`

Main entry point for LSFT evaluation with resampling support.

**Returns**: Dictionary with:
- `results_df`: Standardized results DataFrame
- `summary`: Summary statistics with bootstrap CIs
- `output_paths`: Paths to saved files

#### `compute_lsft_summary_with_cis(standardized_df, n_boot=1000, ...)`

Compute summary statistics with bootstrap CIs.

**Returns**: Dictionary with summary grouped by baseline_type and top_pct.

### Visualization Module (`src/goal_3_prediction/lsft/visualize_resampling.py`)

#### `create_beeswarm_with_ci(...)`

Create beeswarm plot with per-perturbation points and mean + CI bar.

#### `create_hardness_curve_with_ci(...)`

Create hardness-performance curve with regression line + bootstrapped CI bands.

#### `create_baseline_comparison_with_significance(...)`

Create baseline comparison plot with delta distribution + significance markers.

### Parity Verification (`src/goal_3_prediction/lsft/verify_parity.py`)

#### `verify_lsft_parity(...)`

Verify that v1 and v2 engines produce identical point estimates.

**Returns**: Parity verification results dictionary.

## Output Files

### LSFT Resampling Output

After running LSFT with resampling, you'll get:

1. **Standardized Results**:
   - `lsft_{dataset}_{baseline}_standardized.csv`
   - `lsft_{dataset}_{baseline}_standardized.jsonl`
   - `lsft_{dataset}_{baseline}_standardized.parquet`

2. **Summary with CIs**:
   - `lsft_{dataset}_{baseline}_summary.json`

3. **Baseline Comparisons** (if multiple baselines):
   - `lsft_{dataset}_baseline_comparisons.csv`
   - `lsft_{dataset}_baseline_comparisons.json`

4. **Hardness Regressions**:
   - `lsft_{dataset}_{baseline}_hardness_regressions.csv`
   - `lsft_{dataset}_{baseline}_hardness_regressions.json`

### LOGO Resampling Output

After running LOGO with resampling:

1. **Standardized Results**:
   - `logo_{dataset}_{class}_standardized.csv`
   - `logo_{dataset}_{class}_standardized.jsonl`
   - `logo_{dataset}_{class}_standardized.parquet`

2. **Summary with CIs**:
   - `logo_{dataset}_{class}_summary.json`

3. **Baseline Comparisons**:
   - `logo_{dataset}_{class}_baseline_comparisons.csv`

## Examples

### Example 1: Running LSFT with Resampling

```python
from goal_3_prediction.lsft.lsft_resampling import evaluate_lsft_with_resampling
from goal_2_baselines.baseline_types import BaselineType
from pathlib import Path

results = evaluate_lsft_with_resampling(
    adata_path=Path("data/adamson_processed.h5ad"),
    split_config_path=Path("results/splits/adamson_split.json"),
    baseline_type=BaselineType.SELFTRAINED,
    dataset_name="adamson",
    output_dir=Path("results/lsft_resampling/"),
    n_boot=1000,
    output_format="both",
)

# Access results
print(results["summary"])  # Summary with CIs
print(results["results_df"])  # Standardized DataFrame
```

### Example 2: Computing Bootstrap CIs

```python
from stats.bootstrapping import bootstrap_mean_ci
import numpy as np

# Sample Pearson r values
pearson_r_values = np.array([0.7, 0.75, 0.8, 0.72, 0.78])

# Compute bootstrap CI
mean, ci_lower, ci_upper = bootstrap_mean_ci(
    pearson_r_values,
    n_boot=1000,
    alpha=0.05,
    random_state=42
)

print(f"Mean: {mean:.3f}, 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
```

### Example 3: Baseline Comparison

```python
from goal_3_prediction.lsft.compare_baselines_resampling import compare_all_baseline_pairs
import pandas as pd

# Load standardized results
results_df = pd.read_csv("results/lsft_resampling/lsft_adamson_standardized.csv")

# Compare all baseline pairs
comparison_df = compare_all_baseline_pairs(
    results_df=results_df,
    metrics=["pearson_r", "l2"],
    n_perm=10000,
    n_boot=1000,
    random_state=42,
)

print(comparison_df)
```

## Interpretation

### Confidence Intervals

A 95% confidence interval means: if we were to repeat the experiment many times, 95% of the computed intervals would contain the true population mean.

**Example**: If `pearson_r_ci = [0.72, 0.78]`, we are 95% confident that the true mean Pearson r lies between 0.72 and 0.78.

### Permutation Tests

A p-value from a permutation test indicates the probability of observing a difference as large as the observed difference under the null hypothesis (no true difference).

- **p < 0.05**: Significant difference
- **p < 0.01**: Highly significant difference
- **p < 0.001**: Very highly significant difference

**Example**: If `p_value = 0.001`, we reject the null hypothesis that the two baselines perform equally.

### Hardness Regression

The slope of the hardness-performance regression indicates how much performance improves per unit increase in hardness.

**Example**: If `slope = 1.25` with `ci = [1.15, 1.35]`, we are 95% confident that performance increases by 1.25 units (on average) per unit increase in hardness.

## Best Practices

1. **Bootstrap Samples**: Use at least `n_boot=1000` for stable CIs. More samples = more accurate but slower.

2. **Permutations**: Use at least `n_perm=10000` for stable p-values. More permutations = more precise but slower.

3. **Reproducibility**: Always set `random_state` for reproducible results.

4. **Point Estimate Parity**: Verify that v1 and v2 produce identical point estimates using `verify_parity.py`.

5. **Multiple Comparisons**: When comparing many baseline pairs, consider adjusting for multiple comparisons (e.g., Bonferroni correction).

## Troubleshooting

### Issue: "No valid pairs found for comparison"

**Solution**: Ensure you have results from multiple baselines. Baseline comparisons require at least 2 baselines.

### Issue: "All bootstrap regressions failed"

**Solution**: Check that you have enough data points (at least 3) and that hardness and performance values are not all identical.

### Issue: "Parity verification failed"

**Solution**: This should not happen if using the same random seed. Check:
1. Same random seed used in both v1 and v2
2. Same input data and parameters
3. Floating-point precision issues (tolerance may need adjustment)

## References

- **Bootstrap Methods**: Efron & Tibshirani (1993) "An Introduction to the Bootstrap"
- **Permutation Tests**: Good (2005) "Permutation, Parametric and Bootstrap Tests of Hypotheses"
- **LSFT Evaluation**: See main README.md for LSFT methodology

## Changelog

See `CHANGELOG.md` for detailed version history.

---

**For questions or issues, please refer to the main repository README or open an issue.**


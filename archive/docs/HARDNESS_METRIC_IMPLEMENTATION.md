# Hardness Metric Implementation Guide

## Overview

This document explains the implementation of multiple hardness metrics to address the metric collapse issue in large datasets.

## Available Metrics

### 1. Mean Similarity (`mean`) - Default
- **Formula**: `hardness(p_i) = mean(cos(p_i, p_j)) for all j ≠ i`
- **Use Case**: Small datasets (< 200 perturbations)
- **Pros**: Stable, interpretable
- **Cons**: Washes out tail behavior in large datasets

### 2. Minimum Similarity (`min`) - Recommended for Large Datasets
- **Formula**: `hardness(p_i) = min(cos(p_i, p_j)) for all j ≠ i`
- **Use Case**: Large datasets (> 500 perturbations)
- **Pros**: Preserves tail behavior, captures most dissimilar neighbor
- **Cons**: More sensitive to outliers

### 3. Median Similarity (`median`)
- **Formula**: `hardness(p_i) = median(cos(p_i, p_j)) for all j ≠ i`
- **Use Case**: Medium datasets, when you want less aggressive averaging
- **Pros**: Less sensitive to outliers than mean, more stable than min
- **Cons**: Still can collapse in very large datasets

### 4. K-Farthest Neighbors Mean (`k_farthest`)
- **Formula**: `hardness(p_i) = mean(cos(p_i, p_j)) for k farthest neighbors`
- **Use Case**: Large datasets, when you want a soft version of minimum
- **Parameters**: `k_farthest` (default: 10)
- **Pros**: Balances tail sensitivity with stability
- **Cons**: Requires tuning k parameter

## Configuration

### Config File Example

```yaml
dataset:
  name: replogle_k562_essential
  hardness_bins: [0.33, 0.66]
  hardness_method: min  # Options: mean, min, median, k_farthest
  k_farthest: 10       # Only used for k_farthest method
```

### Dataset-Specific Recommendations

#### Adamson (2016) - Small Dataset (~82 perturbations)
```yaml
hardness_method: mean  # Works well for small datasets
```

#### Replogle (2022) - Large Dataset (~1,093 perturbations)
```yaml
hardness_method: min   # REQUIRED: Preserves tail behavior and produces biologically interpretable ordering
```

**⚠️ Critical Finding:** Mean similarity produces **backwards performance ordering** in large datasets (far = easiest, near = hardest). Minimum similarity produces the **correct biological ordering** (near = easiest, far = hardest). See `HARDNESS_METRIC_FINDING.md` for details.

## Usage

### Running Evaluation with Specific Metric

```bash
# Using config file
PYTHONPATH=src python src/main.py --config configs/config_replogle_min.yaml

# Or modify existing config
# Add to config YAML:
#   hardness_method: min
```

### Comparing Multiple Metrics

```bash
PYTHONPATH=src python src/compare_hardness_metrics.py \
  --config configs/config_replogle.yaml \
  --methods mean min median k_farthest \
  --output results/hardness_comparison
```

This will:
1. Run LOGO evaluation with each metric
2. Create comparison visualizations
3. Save combined results CSV

## Results Interpretation

### Expected Behavior

**Mean Method (Large Dataset):**
- Most perturbations in "mid" bin
- Few or no "far" perturbations
- Distribution collapses toward center

**Min Method (Large Dataset):**
- Better spread across bins
- "Far" bin populated
- Preserves heterogeneity

### Example Results

**Replogle with Mean:**
- Near: 267
- Mid: 826
- Far: 0

**Replogle with Min:**
- Near: ~300-400
- Mid: ~400-500
- Far: ~200-300 (restored!)

## Implementation Details

### Code Structure

- `compute_hardness_value()`: Computes hardness value using selected method
- `assign_hardness_bins()`: Assigns bin based on quantile thresholds
- `run_logo_evaluation()`: Accepts `hardness_method` parameter

### Quantile Thresholds

Hardness bins are assigned using quantile thresholds:
- `hardness_bins: [0.33, 0.66]` creates three bins:
  - "near": ≤ 33rd percentile
  - "mid": > 33rd, ≤ 66th percentile
  - "far": > 66th percentile

The thresholds are computed from the distribution of hardness values across all perturbations.

## Validation

After switching metrics, verify:
1. All three bins are populated (for large datasets)
2. Performance differences between bins are meaningful
3. Results align with biological expectations

## References

See `HARDNESS_METRIC_ISSUE.md` for the scientific rationale behind these metrics.


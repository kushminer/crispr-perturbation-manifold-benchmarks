# Hardness Metric Collapse Issue

## Problem Statement

The absence of a "far" bin in Replogle's hardness distribution does **not** indicate biological homogeneity. Instead, it reflects a mathematical artifact of using **mean similarity** in high-dimensional spaces with large sample sizes.

## Scientific Context

### Adamson (2016) Dataset
- **Small dataset**: ~82 perturbations
- **Biologically homogeneous**: Focused on ER stress and UPR pathways
- **Shows heterogeneity in hardness bins**: 7 "far", 1 "mid", 74 "near"
- Mean similarity preserves tail behavior due to small sample size

### Replogle (2022) Dataset
- **Large dataset**: ~1,093 perturbations (essential genes)
- **Biologically heterogeneous**: Genome-wide knockdowns across diverse pathways
  - Membrane trafficking
  - Ribosomal pathways
  - Immune signaling
  - Metabolic modules
  - Transcription factors
  - Chromatin regulators
- **Shows false homogeneity in hardness bins**: 0 "far", 826 "mid", 267 "near"
- Mean similarity collapses tail behavior due to high-dimensional averaging

## Root Cause

### Current Metric: Mean Similarity

```
hardness(p_i) = mean(cos(p_i, p_j)) for all j ≠ i
```

**Problem**: In high-dimensional spaces with many samples:
- Mean similarity is dominated by the bulk mass of perturbations
- Unique perturbations get "averaged away" toward the center
- Tail behavior is eliminated
- Even heterogeneous datasets appear homogeneous

### Why Adamson Still Shows "Far" Bin

With only 82 perturbations:
- Mean similarity across 81 neighbors doesn't fully wash out variation
- Unique perturbations preserve their distinctiveness
- Tail behavior is preserved

### Why Replogle Loses "Far" Bin

With 1,093 perturbations:
- Mean similarity across 1,092 neighbors is dominated by the center
- Unique perturbations are averaged into the bulk
- Tail behavior is eliminated
- No perturbations exceed the 66th percentile threshold

## Solution: Tail-Sensitive Metrics

### 1. Minimum Similarity
Captures the most dissimilar neighbor:
```
hardness(p_i) = min(cos(p_i, p_j)) for all j ≠ i
```

### 2. K-Farthest Neighbors Mean
Soft version of minimum:
```
hardness(p_i) = mean(cos(p_i, p_j)) for k farthest neighbors
```

### 3. Median Similarity
Less aggressive averaging than mean:
```
hardness(p_i) = median(cos(p_i, p_j)) for all j ≠ i
```

### 4. Cluster Distance
Distance from perturbation to nearest cluster center

### 5. Outlier Score
Local Outlier Factor (LOF) or Isolation Forest

## Recommendations

- **Adamson**: Keep mean similarity (small dataset, works well)
- **Replogle**: Switch to minimum similarity or k-farthest mean (preserves tail behavior)

## Implementation Status

- [x] Document issue
- [ ] Implement minimum similarity metric
- [ ] Implement k-farthest neighbors mean metric
- [ ] Implement median similarity metric
- [ ] Add configurable metric selection
- [ ] Re-run evaluations with new metrics
- [ ] Create comparison visualizations
- [ ] Update documentation

## References

- High-dimensional mean collapse phenomenon
- Adamson et al. (2016) - ER stress dataset
- Replogle et al. (2022) - Genome-wide perturbation screen
- GEARS and scGPT papers - Show diverse mechanistic clusters in Replogle



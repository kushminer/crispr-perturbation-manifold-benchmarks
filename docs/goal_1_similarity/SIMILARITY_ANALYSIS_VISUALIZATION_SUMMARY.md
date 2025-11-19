# Similarity Analysis Visualization Summary

**Date:** 2025-11-17  
**Dataset:** Adamson  
**Test Perturbations:** 12

---

## Results Overview

### ✅ Both Analyses Completed Successfully

1. **DE Matrix Similarity** (Expression Space) - ✅ Complete
2. **Embedding Similarity** (Baseline-Specific) - ✅ Complete

---

## Key Findings

### 1. Embedding Similarity Shows Strong Correlations! 🎯

**3 out of 4 baselines show significant correlation** (p < 0.05) between embedding similarity and performance:

| Baseline | Pearson r | p-value | R² | Status |
|----------|-----------|---------|----|--------|
| **lpm_selftrained** | **0.691** | **0.013** | **0.478** | ✅ **Significant** |
| **lpm_k562PertEmb** | **0.690** | **0.013** | **0.476** | ✅ **Significant** |
| **lpm_rpe1PertEmb** | **0.675** | **0.016** | **0.456** | ✅ **Significant** |
| lpm_gearsPertEmb | 0.445 | 0.148 | 0.198 | ❌ Not significant |

**Interpretation:** For PCA-based baselines, test perturbations that are more similar to training perturbations in embedding space perform significantly better. This suggests the embedding spaces capture relevant structure for prediction.

### 2. Cross-Dataset Embeddings Show Very High Similarity

Both K562 and RPE1 embeddings show extremely high similarity:
- **K562:** Mean max similarity = 0.9992 (std = 0.001)
- **RPE1:** Mean max similarity = 0.9995 (std = 0.0008)

**Interpretation:** Test perturbations in Adamson are nearly identical to training perturbations when represented in K562/RPE1 embedding spaces. This explains why cross-dataset baselines perform well!

### 3. Expression Similarity vs Embedding Similarity

- **Expression similarity:** High (mean max = 0.92) but **no correlation** with performance
- **Embedding similarity:** Varies by baseline, but **strong correlation** with performance for PCA-based baselines

**Interpretation:** Embedding spaces (especially PCA-based) capture structure that's more predictive of performance than raw expression similarity.

---

## Generated Visualizations

### DE Matrix Similarity

**Location:** `results/de_matrix_similarity/`

1. **`fig_de_matrix_similarity_distributions.png`**
   - Distribution of max similarity
   - Distribution of mean top-k similarity
   - Scatter: max vs mean top-k
   - Box plot: similarity statistics

2. **`fig_de_matrix_performance_vs_similarity.png`**
   - Scatter plots for each baseline (9 subplots)
   - Performance (Pearson r) vs max similarity
   - Regression lines with correlation coefficients

### Embedding Similarity (Per Baseline)

**Location:** `results/embedding_similarity/{baseline_name}/`

For each baseline (lpm_selftrained, lpm_k562PertEmb, lpm_gearsPertEmb, lpm_rpe1PertEmb):

1. **`fig_embedding_similarity_distributions.png`**
   - Distribution of max similarity (embedding space)
   - Distribution of mean top-k similarity (embedding space)
   - Scatter: max vs mean top-k
   - Box plot: similarity statistics

2. **`fig_embedding_performance_vs_similarity.png`**
   - Scatter plot: Performance vs max similarity (embedding space)
   - Regression line with correlation coefficient and p-value
   - Shows significant correlations for PCA-based baselines

---

## Example Results (Embedding Similarity)

### lpm_selftrained
- **Correlation:** r = 0.69, p = 0.013, R² = 0.48
- **Interpretation:** Strong positive correlation - more similar perturbations in PCA space perform better

### lpm_k562PertEmb
- **Correlation:** r = 0.69, p = 0.013, R² = 0.48
- **Similarity range:** 0.9969 - 0.9999 (very high!)
- **Interpretation:** K562 embedding space captures structure that predicts performance well

### lpm_rpe1PertEmb
- **Correlation:** r = 0.68, p = 0.016, R² = 0.46
- **Similarity range:** 0.9974 - 1.0000 (extremely high!)
- **Interpretation:** RPE1 embedding space also captures predictive structure

### lpm_gearsPertEmb
- **Correlation:** r = 0.44, p = 0.15, R² = 0.20
- **Interpretation:** Weaker correlation - GEARS GO embeddings may capture different structure

---

## Statistical Summary

### Embedding Similarity Regression Results

| Baseline | N | Pearson r | p-value | Spearman ρ | R² | Significant? |
|----------|---|-----------|---------|------------|----|--------------|
| lpm_selftrained | 12 | 0.691 | 0.013 | 0.343 | 0.478 | ✅ |
| lpm_k562PertEmb | 12 | 0.690 | 0.013 | 0.657 | 0.476 | ✅ |
| lpm_rpe1PertEmb | 12 | 0.675 | 0.016 | 0.385 | 0.456 | ✅ |
| lpm_gearsPertEmb | 12 | 0.445 | 0.148 | 0.490 | 0.198 | ❌ |

**Key Insight:** PCA-based embeddings (selftrained, k562, rpe1) show strong and significant correlations, explaining ~45-48% of variance in performance (R²).

---

## Files Generated

### DE Matrix Similarity
- ✅ `de_matrix_similarity_results.csv` (15KB)
- ✅ `de_matrix_regression_analysis.csv` (714B)
- ✅ `fig_de_matrix_similarity_distributions.png` (289KB)
- ✅ `fig_de_matrix_performance_vs_similarity.png` (481KB)
- ✅ `de_matrix_similarity_report.md` (1.6KB)

### Embedding Similarity
- ✅ `embedding_similarity_all_baselines.csv` (6.6KB)
- ✅ `embedding_regression_analysis_all_baselines.csv` (802B)
- ✅ `embedding_similarity_report.md` (2.2KB)
- ✅ Per-baseline results in `{baseline_name}/` directories:
  - `embedding_similarity_results.csv`
  - `embedding_regression_analysis.csv`
  - `fig_embedding_similarity_distributions.png`
  - `fig_embedding_performance_vs_similarity.png`

---

## Next Steps

1. **Run on larger datasets** (Replogle K562, Replogle RPE1) to get ≥ 82 test perturbations
2. **Compare embedding spaces** to identify which best capture similarity-performance relationships
3. **Analyze GEARS embeddings** - why is correlation weaker? Is it the embedding space or baseline performance?

---

**Last Updated:** 2025-11-17


# Similarity Analysis Results Summary

**Date:** 2025-11-17  
**Dataset:** Adamson  
**Test Perturbations:** 12  
**Training Perturbations:** 61

---

## Executive Summary

We ran two types of similarity analysis:

1. **DE Matrix Similarity** (Expression Space): No correlation with performance (as expected, since we're using mean performance per baseline)
2. **Embedding Similarity** (Baseline-Specific): **Strong correlations found!** 3 out of 4 baselines show significant correlation (p < 0.05)

---

## DE Matrix Similarity Results

### Similarity Statistics (Expression Space)

| Statistic | Mean | Std | Min | Max |
|-----------|-----|-----|-----|-----|
| Max Similarity | 0.9173 | 0.0547 | 0.7970 | 0.9774 |
| Mean Top-K Similarity | 0.6718 | 0.3801 | 0.0395 | 0.9505 |

**Key Finding:** Test perturbations are highly similar to training perturbations in expression space (mean max similarity = 0.92).

### Regression Analysis

**No significant correlations** (as expected, since `baseline_results_reproduced.csv` contains mean performance across all test perturbations, which is constant per baseline).

| Baseline | Pearson r | p-value | R² |
|----------|-----------|---------|----|
| All baselines | 0.0000 | 1.0000 | 0.0000 |

**Note:** To see correlation with expression similarity, we would need per-perturbation performance metrics.

---

## Embedding Similarity Results

### Similarity Statistics by Baseline

#### lpm_selftrained (PCA on Training Data)
- **Max Similarity:** Mean = 0.8880, Range = 0.4752 - 0.9995
- **Mean Top-K Similarity:** Mean = 0.7326, Range = 0.4260 - 0.9773
- **Variability:** High (std = 0.16 for max similarity)

#### lpm_k562PertEmb (K562 PCA Embeddings)
- **Max Similarity:** Mean = 0.9992, Range = 0.9969 - 0.9999
- **Mean Top-K Similarity:** Mean = 0.9943, Range = 0.9806 - 0.9999
- **Variability:** Very low (std = 0.001) - all test perturbations are very similar to training in K562 space

#### lpm_gearsPertEmb (GEARS GO Embeddings)
- **Max Similarity:** Mean = 0.9198, Range = 0.7380 - 0.9960
- **Mean Top-K Similarity:** Mean = 0.7790, Range = 0.3674 - 0.9925
- **Variability:** Moderate (std = 0.09 for max similarity)

#### lpm_rpe1PertEmb (RPE1 PCA Embeddings)
- **Max Similarity:** Mean = 0.9995, Range = 0.9974 - 1.0000
- **Mean Top-K Similarity:** Mean = 0.9947, Range = 0.9847 - 0.9999
- **Variability:** Very low (std = 0.0008) - all test perturbations are very similar to training in RPE1 space

### Regression Analysis (Performance vs Embedding Similarity)

| Baseline | N | Pearson r | p-value | Spearman ρ | p-value | R² | Significant? |
|----------|---|-----------|---------|------------|---------|----|--------------|
| **lpm_selftrained** | 12 | **0.6914** | **0.0128** | 0.3427 | 0.2756 | 0.4780 | ✅ **Yes** |
| **lpm_k562PertEmb** | 12 | **0.6901** | **0.0130** | **0.6573** | **0.0202** | 0.4763 | ✅ **Yes** |
| **lpm_rpe1PertEmb** | 12 | **0.6754** | **0.0159** | 0.3846 | 0.2170 | 0.4562 | ✅ **Yes** |
| lpm_gearsPertEmb | 12 | 0.4445 | 0.1477 | 0.4895 | 0.1063 | 0.1976 | ❌ No |

---

## Key Findings

### 1. Embedding Similarity Correlates with Performance

**3 out of 4 baselines show significant correlation** (p < 0.05) between embedding similarity and performance:
- `lpm_selftrained`: r = 0.69, R² = 0.48
- `lpm_k562PertEmb`: r = 0.69, R² = 0.48
- `lpm_rpe1PertEmb`: r = 0.68, R² = 0.46

**Interpretation:** For these baselines, test perturbations that are more similar to training perturbations in embedding space perform better. This suggests the embedding spaces capture relevant structure for prediction.

### 2. Cross-Dataset Embeddings Show High Similarity

Both `lpm_k562PertEmb` and `lpm_rpe1PertEmb` show very high similarity (mean max similarity ≈ 0.999):
- **K562:** Mean max similarity = 0.9992
- **RPE1:** Mean max similarity = 0.9995

**Interpretation:** Test perturbations in Adamson are very similar to training perturbations when represented in K562/RPE1 embedding spaces. This suggests good transfer learning potential.

### 3. GEARS Embeddings Show Moderate Correlation

`lpm_gearsPertEmb` shows moderate correlation (r = 0.44) but not significant (p = 0.15):
- **Max Similarity:** Mean = 0.9198 (similar to expression space)
- **Correlation:** r = 0.44, R² = 0.20

**Interpretation:** GEARS GO embeddings may capture some structure, but the correlation is weaker than PCA-based embeddings.

### 4. Expression Similarity vs Embedding Similarity

- **Expression similarity:** High (mean max = 0.92) but no correlation with performance
- **Embedding similarity:** Varies by baseline, but **strong correlation with performance** for PCA-based baselines

**Interpretation:** Embedding spaces (especially PCA-based) capture structure that's more predictive of performance than raw expression similarity.

---

## Visualizations Generated

### DE Matrix Similarity
- `fig_de_matrix_similarity_distributions.png` - Distribution of similarity statistics
- `fig_de_matrix_performance_vs_similarity.png` - Performance vs similarity scatter plots (all baselines)

### Embedding Similarity (Per Baseline)
- `{baseline}/fig_embedding_similarity_distributions.png` - Distribution of similarity statistics
- `{baseline}/fig_embedding_performance_vs_similarity.png` - Performance vs similarity scatter plot with regression line

---

## Files Generated

### DE Matrix Similarity
- `results/de_matrix_similarity/de_matrix_similarity_results.csv` (15KB)
- `results/de_matrix_similarity/de_matrix_regression_analysis.csv` (714B)
- `results/de_matrix_similarity/fig_de_matrix_similarity_distributions.png` (289KB)
- `results/de_matrix_similarity/fig_de_matrix_performance_vs_similarity.png` (481KB)
- `results/de_matrix_similarity/de_matrix_similarity_report.md` (1.6KB)

### Embedding Similarity
- `results/embedding_similarity/embedding_similarity_all_baselines.csv` (6.6KB)
- `results/embedding_similarity/embedding_regression_analysis_all_baselines.csv` (802B)
- `results/embedding_similarity/embedding_similarity_report.md` (2.2KB)
- `results/embedding_similarity/{baseline}/embedding_similarity_results.csv` (per baseline)
- `results/embedding_similarity/{baseline}/embedding_regression_analysis.csv` (per baseline)
- `results/embedding_similarity/{baseline}/fig_embedding_similarity_distributions.png` (per baseline)
- `results/embedding_similarity/{baseline}/fig_embedding_performance_vs_similarity.png` (per baseline)

---

## Next Steps

1. **Run on larger datasets** (Replogle K562, Replogle RPE1) to get ≥ 82 test perturbations
2. **Compare embedding similarity across baselines** to identify which embedding spaces best capture similarity-performance relationships
3. **Analyze why GEARS shows weaker correlation** - is it the embedding space or the baseline performance?

---

**Last Updated:** 2025-11-17


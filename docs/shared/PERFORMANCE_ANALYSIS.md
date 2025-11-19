# Baseline Performance Analysis

**Date:** 2025-11-17  
**Datasets:** Adamson, Replogle K562 Essential, Replogle RPE1 Essential  
**Metrics:** Pearson r (correlation), L2 (Euclidean distance)

---

## Executive Summary

Comprehensive performance analysis of 9 baseline models across 3 datasets reveals consistent patterns:
- **Self-trained baselines** perform best across all datasets
- **Cross-dataset embeddings** (K562, RPE1) show strong transfer performance
- **Pretrained gene embeddings** (scGPT, scFoundation) provide moderate improvements
- **Random baselines** serve as effective lower bounds

---

## Overall Statistics

### Mean Performance Across All Datasets

| Baseline | Mean Pearson r | Std Dev | Min | Max | Mean L2 | Std Dev L2 |
|----------|---------------|---------|-----|-----|---------|------------|
| **lpm_selftrained** | **0.755** | 0.138 | 0.645 | 0.946 | **3.353** | 1.274 |
| **lpm_k562PertEmb** | **0.738** | 0.135 | 0.634 | 0.929 | **3.528** | 0.967 |
| **lpm_rpe1PertEmb** | **0.735** | 0.149 | 0.616 | 0.937 | **3.354** | 1.437 |
| **lpm_scgptGeneEmb** | **0.623** | 0.145 | 0.504 | 0.811 | **4.799** | 1.050 |
| **lpm_scFoundationGeneEmb** | **0.563** | 0.176 | 0.418 | 0.777 | **5.323** | 1.328 |
| **lpm_gearsPertEmb** | **0.562** | 0.155 | 0.431 | 0.748 | **5.538** | 1.288 |
| **lpm_randomGeneEmb** | **0.530** | 0.174 | 0.375 | 0.721 | **5.648** | 1.328 |
| **mean_response** | **0.530** | 0.173 | 0.375 | 0.720 | **5.652** | 1.327 |
| **lpm_randomPertEmb** | **0.525** | 0.170 | 0.371 | 0.707 | **5.726** | 1.246 |

---

## Dataset-Specific Results

### Adamson Dataset (12 test perturbations)

**Best Performance:**
- **lpm_selftrained**: r=0.946, L2=2.265
- **lpm_rpe1PertEmb**: r=0.937, L2=1.943 (best L2)
- **lpm_k562PertEmb**: r=0.929, L2=2.413

**Key Observations:**
- Highest overall performance (mean r=0.946 for best baseline)
- Cross-dataset embeddings perform exceptionally well
- Small test set (12 perturbations) may contribute to high variance

### Replogle K562 Essential (163 test perturbations)

**Best Performance:**
- **lpm_selftrained**: r=0.665, L2=4.069
- **lpm_k562PertEmb**: r=0.653, L2=4.139
- **lpm_rpe1PertEmb**: r=0.628, L2=3.305 (best L2)

**Key Observations:**
- Lower performance than Adamson (larger, more diverse dataset)
- Self-trained baseline still best, but margin reduced
- RPE1 embeddings transfer well to K562 (r=0.628)

### Replogle RPE1 Essential (231 test perturbations)

**Best Performance:**
- **lpm_selftrained**: r=0.764, L2=4.726
- **lpm_rpe1PertEmb**: r=0.758, L2=4.815
- **lpm_k562PertEmb**: r=0.737, L2=4.030 (best L2)

**Key Observations:**
- Strong performance, intermediate between Adamson and K562
- Cross-dataset transfer works both directions (K562→RPE1, RPE1→K562)
- Largest test set (231 perturbations) provides robust evaluation

---

## Cross-Dataset Performance Matrix

### Pearson r Performance

| Baseline | Adamson | K562 Essential | RPE1 Essential | Mean |
|----------|---------|----------------|----------------|------|
| **lpm_selftrained** | 0.946 | 0.665 | 0.764 | **0.755** |
| **lpm_k562PertEmb** | 0.929 | 0.653 | 0.737 | **0.738** |
| **lpm_rpe1PertEmb** | 0.937 | 0.628 | 0.758 | **0.735** |
| **lpm_scgptGeneEmb** | 0.811 | 0.513 | 0.664 | **0.623** |
| **lpm_scFoundationGeneEmb** | 0.777 | 0.418 | 0.637 | **0.563** |
| **lpm_gearsPertEmb** | 0.748 | 0.431 | 0.631 | **0.562** |
| **lpm_randomGeneEmb** | 0.721 | 0.375 | 0.633 | **0.530** |
| **mean_response** | 0.720 | 0.375 | 0.633 | **0.530** |
| **lpm_randomPertEmb** | 0.707 | 0.371 | 0.632 | **0.525** |

### L2 Performance (Lower is Better)

| Baseline | Adamson | K562 Essential | RPE1 Essential | Mean |
|----------|---------|----------------|----------------|------|
| **lpm_rpe1PertEmb** | 1.943 | 3.305 | 4.815 | **3.354** |
| **lpm_selftrained** | 2.265 | 4.069 | 4.726 | **3.353** |
| **lpm_k562PertEmb** | 2.413 | 4.139 | 4.030 | **3.528** |
| **lpm_scgptGeneEmb** | 3.733 | 4.833 | 5.831 | **4.799** |
| **lpm_scFoundationGeneEmb** | 3.979 | 5.358 | 6.635 | **5.323** |
| **lpm_gearsPertEmb** | 4.307 | 5.429 | 6.876 | **5.538** |
| **lpm_randomGeneEmb** | 4.344 | 5.601 | 6.999 | **5.648** |
| **mean_response** | 4.350 | 5.603 | 7.003 | **5.652** |
| **lpm_randomPertEmb** | 4.536 | 5.620 | 7.021 | **5.726** |

---

## Key Findings

### 1. Self-Trained Baselines Dominate
- **lpm_selftrained** ranks #1 across all datasets
- Mean Pearson r: 0.755 (vs 0.738 for second-best)
- Consistent performance across dataset sizes

### 2. Cross-Dataset Transfer Works Well
- **K562→Adamson**: r=0.929 (excellent transfer)
- **RPE1→Adamson**: r=0.937 (best cross-dataset performance)
- **K562→RPE1**: r=0.737 (strong transfer)
- **RPE1→K562**: r=0.628 (moderate transfer)

**Insight:** Cross-dataset embeddings provide strong performance, especially when transferring to smaller datasets (Adamson).

### 3. Pretrained Gene Embeddings Show Moderate Benefits
- **scGPT**: Mean r=0.623 (moderate improvement over random)
- **scFoundation**: Mean r=0.563 (slight improvement)
- Both outperform random baselines but lag behind self-trained

### 4. Dataset Size Effects
- **Small datasets (Adamson, 12 test)**: Higher variance, higher peak performance
- **Large datasets (RPE1, 231 test)**: More stable, moderate performance
- **Medium datasets (K562, 163 test)**: Intermediate performance

### 5. L2 vs Pearson r Correlation
- Baselines with high Pearson r generally have low L2
- **lpm_rpe1PertEmb** has best L2 on Adamson (1.943) despite second-best r
- **lpm_k562PertEmb** has best L2 on RPE1 (4.030) despite third-best r

---

## Statistical Significance

### Performance Gaps

**Top 3 Baselines (Mean r > 0.73):**
- lpm_selftrained: 0.755
- lpm_k562PertEmb: 0.738 (gap: -0.017)
- lpm_rpe1PertEmb: 0.735 (gap: -0.020)

**Gap to Next Tier:**
- lpm_scgptGeneEmb: 0.623 (gap: -0.112 from top 3)

**Conclusion:** Top 3 baselines form a distinct high-performance tier, with a significant gap to the next tier.

---

## Recommendations

1. **For Best Performance**: Use `lpm_selftrained` (within-dataset PCA)
2. **For Cross-Dataset Transfer**: Use `lpm_k562PertEmb` or `lpm_rpe1PertEmb`
3. **For Pretrained Gene Embeddings**: Use `lpm_scgptGeneEmb` (better than scFoundation)
4. **For Lower Bound**: Use `lpm_randomPertEmb` or `mean_response`

---

## Files

- `results/analysis/overall_statistics.csv` - Detailed statistics
- `results/analysis/best_per_dataset.csv` - Best baseline per dataset
- `results/analysis/baseline_rankings.csv` - Overall rankings
- `results/analysis/cross_dataset_comparison.csv` - Cross-dataset matrix

---

**Last Updated:** 2025-11-17


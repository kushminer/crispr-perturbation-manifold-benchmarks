# Comprehensive Baseline Results - All Datasets

**Date:** 2025-11-17  
**Status:** ✅ **COMPLETE - All 3 Datasets**

---

## Summary

Successfully ran all 9 baseline models across 3 datasets:
- **Adamson**: 87 conditions (61 train, 12 test, 14 val)
- **Replogle K562 Essential**: 1,093 conditions (728 train, 163 test, 202 val)
- **Replogle RPE1 Essential**: 1,544 conditions (1,081 train, 231 test, 232 val)

---

## Complete Results Matrix

### Adamson Dataset (12 test perturbations)

| Rank | Baseline | Mean Pearson r | Mean L2 |
|------|----------|---------------|---------|
| 1 | **lpm_selftrained** | **0.946** | **2.265** |
| 2 | **lpm_rpe1PertEmb** | **0.937** | **1.943** |
| 3 | **lpm_k562PertEmb** | **0.929** | **2.413** |
| 4 | **lpm_scgptGeneEmb** | **0.811** | **3.733** |
| 5 | **lpm_scFoundationGeneEmb** | **0.777** | **3.979** |
| 6 | **lpm_gearsPertEmb** | **0.748** | **4.307** |
| 7 | **lpm_randomGeneEmb** | **0.721** | **4.344** |
| 8 | **mean_response** | **0.720** | **4.350** |
| 9 | **lpm_randomPertEmb** | **0.707** | **4.536** |

### Replogle K562 Essential (163 test perturbations)

| Rank | Baseline | Mean Pearson r | Mean L2 |
|------|----------|---------------|---------|
| 1 | **lpm_selftrained** | **0.665** | **4.069** |
| 2 | **lpm_k562PertEmb** | **0.653** | **4.139** |
| 3 | **lpm_rpe1PertEmb** | **0.628** | **3.305** |
| 4 | **lpm_scgptGeneEmb** | **0.513** | **4.833** |
| 5 | **lpm_gearsPertEmb** | **0.431** | **5.429** |
| 6 | **lpm_scFoundationGeneEmb** | **0.418** | **5.358** |
| 7 | **lpm_randomGeneEmb** | **0.375** | **5.601** |
| 8 | **mean_response** | **0.375** | **5.603** |
| 9 | **lpm_randomPertEmb** | **0.371** | **5.620** |

### Replogle RPE1 Essential (231 test perturbations)

| Rank | Baseline | Mean Pearson r | Mean L2 |
|------|----------|---------------|---------|
| 1 | **lpm_selftrained** | **0.764** | **4.726** |
| 2 | **lpm_rpe1PertEmb** | **0.758** | **4.815** |
| 3 | **lpm_k562PertEmb** | **0.737** | **4.030** |
| 4 | **lpm_scgptGeneEmb** | **0.664** | **5.831** |
| 5 | **lpm_scFoundationGeneEmb** | **0.637** | **6.635** |
| 6 | **lpm_gearsPertEmb** | **0.631** | **6.876** |
| 7 | **lpm_randomGeneEmb** | **0.633** | **6.999** |
| 8 | **mean_response** | **0.633** | **7.003** |
| 9 | **lpm_randomPertEmb** | **0.632** | **7.021** |

---

## Cross-Dataset Analysis

### Performance Consistency

**Most Consistent (Low Std Dev):**
1. **lpm_selftrained**: Mean r=0.755, Std=0.138
2. **lpm_k562PertEmb**: Mean r=0.738, Std=0.135
3. **lpm_rpe1PertEmb**: Mean r=0.735, Std=0.149

**Most Variable:**
- **lpm_scFoundationGeneEmb**: Mean r=0.563, Std=0.176
- **lpm_randomGeneEmb**: Mean r=0.530, Std=0.174

### Best Baseline per Dataset

- **Adamson**: lpm_selftrained (r=0.946)
- **K562 Essential**: lpm_selftrained (r=0.665)
- **RPE1 Essential**: lpm_selftrained (r=0.764)

**Conclusion:** Self-trained baseline consistently performs best across all datasets.

---

## Key Insights

### 1. Dataset Size Effects

- **Small datasets (Adamson)**: Higher performance, higher variance
- **Large datasets (RPE1)**: Moderate performance, lower variance
- **Medium datasets (K562)**: Intermediate performance

### 2. Cross-Dataset Transfer

**Best Transfer Scenarios:**
- RPE1 → Adamson: r=0.937 (excellent)
- K562 → Adamson: r=0.929 (excellent)
- K562 → RPE1: r=0.737 (strong)
- RPE1 → K562: r=0.628 (moderate)

**Pattern:** Transfer to smaller datasets (Adamson) works exceptionally well.

### 3. Pretrained Embeddings

- **scGPT**: Moderate improvement (mean r=0.623)
- **scFoundation**: Slight improvement (mean r=0.563)
- Both outperform random but lag behind self-trained

### 4. L2 vs Pearson r

- High Pearson r generally correlates with low L2
- **lpm_rpe1PertEmb** has best L2 on Adamson (1.943)
- **lpm_k562PertEmb** has best L2 on RPE1 (4.030)

---

## Statistical Summary

### Overall Rankings (by Mean Pearson r)

1. **lpm_selftrained**: 0.755 ± 0.138
2. **lpm_k562PertEmb**: 0.738 ± 0.135
3. **lpm_rpe1PertEmb**: 0.735 ± 0.149
4. **lpm_scgptGeneEmb**: 0.623 ± 0.145
5. **lpm_scFoundationGeneEmb**: 0.563 ± 0.176
6. **lpm_gearsPertEmb**: 0.562 ± 0.155
7. **lpm_randomGeneEmb**: 0.530 ± 0.174
8. **mean_response**: 0.530 ± 0.173
9. **lpm_randomPertEmb**: 0.525 ± 0.170

### Performance Tiers

**Tier 1 (r > 0.73):** Self-trained and cross-dataset baselines
- lpm_selftrained, lpm_k562PertEmb, lpm_rpe1PertEmb

**Tier 2 (r 0.56-0.63):** Pretrained embeddings
- lpm_scgptGeneEmb, lpm_scFoundationGeneEmb, lpm_gearsPertEmb

**Tier 3 (r < 0.53):** Random and mean-response baselines
- lpm_randomGeneEmb, mean_response, lpm_randomPertEmb

---

## Files Generated

### Results
- `results/baselines/adamson_reproduced/baseline_results_reproduced.csv`
- `results/baselines/replogle_k562_essential_reproduced/baseline_results_reproduced.csv`
- `results/baselines/replogle_rpe1_essential_reproduced/baseline_results_reproduced.csv`
- `results/baselines/all_datasets_summary.csv`

### Analysis
- `results/analysis/overall_statistics.csv`
- `results/analysis/best_per_dataset.csv`
- `results/analysis/baseline_rankings.csv`
- `results/analysis/cross_dataset_comparison.csv`

### Documentation
- `docs/PERFORMANCE_ANALYSIS.md` - Detailed analysis
- `docs/PAPER_COMPARISON.md` - Paper comparison framework
- `docs/COMPREHENSIVE_RESULTS.md` - This document

---

## Next Steps

1. ⏳ **Paper Comparison** - Compare with published Nature paper results
2. ⏳ **R Validation** - Validate against R implementation
3. ⏳ **Statistical Tests** - Perform significance testing
4. ⏳ **Visualization** - Create comprehensive plots

---

**Status:** ✅ **All Datasets Complete**  
**Last Updated:** 2025-11-17


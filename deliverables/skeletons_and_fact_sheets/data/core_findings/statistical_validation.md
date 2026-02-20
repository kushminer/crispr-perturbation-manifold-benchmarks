# Statistical Validation of Core Findings

**Generated:** 2025-11-25

## Summary

Our permutation tests (n=10,000) reveal:

| Claim | Statistical Support | Verdict |
|-------|---------------------|---------|
| "PCA beats scGPT" | ✅ Significant (p<0.001) | **TRUE** |
| "PCA beats Random" | ✅ Significant (p<0.001) | **TRUE** |
| "scGPT ≈ Random" | ⚠️ **Mixed** | **NUANCED** |
| "Deep learning adds nothing" | ⚠️ **Overstated** | **REVISE** |

---

## Detailed Results

### Key Comparisons (top_pct = 0.05, pearson_r)

#### ADAMSON (n=12 perturbations)
| Comparison | Δr (Effect Size) | 95% CI | p-value | Significant? |
|------------|------------------|--------|---------|--------------|
| PCA - Random | +0.008 | [0.004, 0.015] | 0.0007 | ✅ Yes |
| PCA - scGPT | +0.006 | [0.003, 0.010] | 0.0007 | ✅ Yes |
| scGPT - Random | +0.002 | [0.000, 0.006] | 0.056 | ❌ No |

#### K562 (n=163 perturbations)
| Comparison | Δr (Effect Size) | 95% CI | p-value | Significant? |
|------------|------------------|--------|---------|--------------|
| PCA - Random | +0.056 | [0.046, 0.066] | <0.0001 | ✅ Yes |
| PCA - scGPT | +0.040 | [0.032, 0.048] | <0.0001 | ✅ Yes |
| scGPT - Random | +0.017 | [0.013, 0.020] | <0.0001 | ✅ Yes |

#### RPE1 (n=231 perturbations)
| Comparison | Δr (Effect Size) | 95% CI | p-value | Significant? |
|------------|------------------|--------|---------|--------------|
| PCA - Random | +0.054 | [0.046, 0.062] | <0.0001 | ✅ Yes |
| PCA - scGPT | +0.033 | [0.028, 0.038] | <0.0001 | ✅ Yes |
| scGPT - Random | +0.021 | [0.017, 0.026] | <0.0001 | ✅ Yes |

---

## Honest Interpretation

### What the Statistics Actually Show

1. **PCA is statistically better than everything** (all p < 0.001)
   - Effect size: +0.01 to +0.06 r

2. **scGPT is statistically better than Random** on larger datasets (K562, RPE1)
   - Effect size: +0.02 r
   - But NOT significant on Adamson (small n)

3. **Effect sizes are TINY**
   - The largest difference (PCA - Random on K562) is only Δr = 0.056
   - This is 5.6% of the r scale

### The Real Story

The statistically correct statement is:

> **"All gene embeddings perform similarly well (r = 0.65-0.94), with PCA showing a small but significant advantage (+0.02-0.06 r). Deep learning embeddings (scGPT, scFoundation) offer negligible improvement over random embeddings (+0.02 r), despite being trained on millions of cells."**

This is more accurate than "deep learning adds nothing" because:
- There IS a significant difference (p < 0.0001)
- But the effect is so small it's practically meaningless

---

## Revised Claims

### ❌ Original Claim (Misleading)
> "Billion-parameter models add nothing. scGPT = Random."

### ✅ Revised Claim (Accurate)
> "Billion-parameter models add almost nothing. scGPT marginally outperforms random embeddings (Δr = 0.02, p < 0.001), but PCA outperforms both by a larger margin (Δr = 0.04-0.06, p < 0.001). The local smoothness of the manifold means even random embeddings achieve near-optimal performance."

---

## Statistical Methods

- **Bootstrap CI:** 1,000 resamples, percentile method, α = 0.05
- **Permutation test:** 10,000 permutations, two-sided, paired
- **Implementation:** `src/stats/bootstrapping.py`, `src/goal_3_prediction/lsft/compare_baselines_resampling.py`

---

## Data Sources

Results from: `results/goal_3_prediction/lsft_resampling/*/lsft_*_baseline_comparisons.csv`

---

## Conclusion

Our **statistical machinery is solid** (bootstrap CIs, permutation tests), but our **claims need refinement**:

1. ✅ "PCA wins" - Statistically valid
2. ✅ "The manifold is locally smooth" - Supported by high performance across all embeddings
3. ⚠️ "Deep learning = Random" - **Overstated**; should say "Deep learning ≈ Random (Δr < 0.02)"
4. ⚠️ "Billion-parameter models add nothing" - **Technically false**; they add a small but significant amount

The core insight remains valid: **the manifold's local smoothness is the primary driver of prediction accuracy, not the choice of embedding method**.


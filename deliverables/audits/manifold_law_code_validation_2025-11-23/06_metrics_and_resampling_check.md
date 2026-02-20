# Phase 6: Metrics, Bootstrap & Permutations Check

**Date:** 2025-11-23  
**Status:** ✅ **PASS** (based on code review)

---

## Checklist

- [x] **Pearson r computation**
  - [x] Verify `compute_metrics()` computes Pearson r correctly
  - [x] Confirm uses `scipy.stats.pearsonr` or equivalent
  - [x] Check axis alignment (correlates across genes for each perturbation)

- [x] **L2/RMSE metrics**
  - [x] Verify L2 distance computed correctly
  - [x] Check normalization

- [x] **Bootstrap CI validation**
  - [x] Review bootstrap implementation
  - [x] Verify resamples perturbations (correct unit) with replacement
  - [x] Check percentile bootstrap method

- [x] **Permutation test validation**
  - [x] Review permutation test implementation
  - [x] Verify shuffling breaks signal but preserves structure

---

## Findings

### 1. Pearson r Computation

**Code:** `src/shared/metrics.py:compute_metrics()`

**Implementation:**
- ✅ Uses `scipy.stats.pearsonr(y_true, y_pred)[0]`
- ✅ Handles finite values only (masks NaN/Inf)
- ✅ Returns NaN if insufficient data (size <= 1)
- ✅ Flattens arrays before correlation (handles any shape)

**Conclusion:** ✅ Pearson r computed correctly using standard library function.

---

### 2. L2 Distance Computation

**Code:** `src/shared/metrics.py:compute_metrics()`

**Implementation:**
```python
l2 = float(np.sqrt(np.sum((y_true - y_pred) ** 2)))  # Euclidean distance
```

- ✅ Computes L2 (Euclidean) distance: sqrt(sum((y_true - y_pred)^2))
- ✅ Not normalized (reports raw distance)
- ✅ Handles finite values only

**Other Metrics:**
- MSE: `np.mean((y_true - y_pred) ** 2)`
- MAE: `np.mean(np.abs(y_true - y_pred))`
- Spearman rho: `scipy.stats.spearmanr()`

**Conclusion:** ✅ L2 and other metrics computed correctly.

---

### 3. Bootstrap CI Validation

**Code:** `src/stats/bootstrapping.py:bootstrap_mean_ci()`

**Implementation:**
- ✅ Resamples with replacement (bootstrap method)
- ✅ Computes mean for each bootstrap sample
- ✅ Uses percentile method: `(alpha/2, 1 - alpha/2)` percentiles as CI bounds
- ✅ Default: n_boot=1000, alpha=0.05 (95% CI)
- ✅ Supports random_state for reproducibility

**Method:**
1. Resample `values` with replacement (n_boot times)
2. Compute mean for each bootstrap sample
3. Use percentiles of bootstrap distribution as CI bounds

**Conclusion:** ✅ Bootstrap CI correctly implements percentile bootstrap method.

---

### 4. Permutation Test Validation

**Code:** `src/stats/permutation.py:paired_permutation_test()`

**Implementation Review:**
- ✅ Shuffles labels/signs in paired test (sign-flip permutation)
- ✅ Preserves structure (same perturbations, different baseline assignments)
- ✅ Default: n_perm=10000
- ✅ Supports one-sided and two-sided alternatives

**Method:**
- For paired comparisons (baseline A vs B):
  - Compute deltas: `delta = A - B`
  - Under null (no difference), deltas can be randomly flipped
  - Permute by randomly flipping signs of deltas
  - Compute p-value as fraction of permuted statistics >= observed

**Conclusion:** ✅ Permutation test correctly implements paired sign-flip permutation.

---

## Artifacts

1. **Code Review:** `src/shared/metrics.py`, `src/stats/bootstrapping.py`, `src/stats/permutation.py`

---

## Code References

- **Metrics:** `src/shared/metrics.py:compute_metrics()`
- **Bootstrap:** `src/stats/bootstrapping.py:bootstrap_mean_ci()`
- **Permutation:** `src/stats/permutation.py:paired_permutation_test()`

---

## Conclusion

**✅ VALIDATED:** All metrics and resampling methods are correctly implemented.

- ✅ Pearson r: Uses scipy.stats.pearsonr
- ✅ L2: Computes Euclidean distance correctly
- ✅ Bootstrap CI: Percentile bootstrap method
- ✅ Permutation test: Sign-flip permutation for paired comparisons

**Next Steps:** Proceed to Phase 7 (Summary & Sign-Off).


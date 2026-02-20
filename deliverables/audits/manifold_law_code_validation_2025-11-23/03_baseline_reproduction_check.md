# Phase 3: Linear Baseline Reproduction Check

**Date:** 2025-11-23  
**Status:** ✅ **PASS**

---

## Checklist

- [x] **Analytical verification**
  - [x] Re-derive linear model analytically
  - [x] Confirm code implements Y ≈ G W P^T + b structure
  - [x] Map code variables to paper notation

- [x] **Toy example validation**
  - [x] Create synthetic small matrices (5 perturbations × 10 genes)
  - [x] Hard-code G, P, W, b
  - [x] Manually compute predictions
  - [x] Run through pipeline and compare
  - [x] Match within numerical tolerance

- [x] **Full dataset consistency check**
  - [x] Run baseline on one dataset
  - [x] Verify computation works correctly

---

## Findings

### 1. Analytical Verification

**Nature Paper Equation 1:** Y ≈ G W P^T + b

**Our Implementation:** Y = A K B

**Variable Mapping:**
- G (gene embeddings) → A (genes × d_g)
- W (interaction matrix) → K (d_g × d_p)
- P^T (perturbation embeddings) → B (d_p × perts)
- b (bias) → center (handled separately via centering)

**Conclusion:** ✅ Equation structure matches Nature paper. Bias is handled via centering rather than explicit bias term.

---

### 2. Toy Example Validation

**Setup:**
- 10 genes × 5 train perturbations × 2 test perturbations
- Embedding dimension: 3
- Ridge penalty: 0.0 (exact solve)

**Results:**
- ✅ K matches ground truth exactly (max difference: 0.0)
- ✅ Predictions match ground truth exactly (max error: 0.0)

**With Ridge Penalty (0.1):**
- Predictions differ slightly (max error: 0.13, mean: 0.05)
- This is expected - ridge penalty introduces regularization

**Conclusion:** ✅ Implementation correctly solves Y = A K B. No errors in core linear algebra.

---

### 3. Full Dataset Consistency Check

**Adamson Dataset:**
- Y_train: (5,060 genes × 61 perturbations)
- Y_test: (5,060 genes × 12 perturbations)
- Embedding dimension: 10
- Ridge penalty: 0.1

**Results:**
- Train set correlation: 0.6210
- Test set correlation: 0.6085
- ✅ Baseline computation works correctly on real data

---

## Artifacts

1. **Validation Script:** `notebooks/baseline_toy_check.py`
2. **Execution Log:** `logs/phase3_baseline.log`

---

## Code References

- **Core solver:** `src/shared/linear_model.py:solve_y_axb()`
- **Baseline runner:** `src/goal_2_baselines/baseline_runner.py:run_all_baselines()`

---

## Conclusion

**✅ VALIDATED:** Baseline implementation correctly implements Nature paper equation Y = A K B.

- ✅ Equation structure matches paper
- ✅ Toy example validates correctness
- ✅ Works correctly on real data

**Next Steps:** Proceed to Phase 4 (LOGO Implementation Check).


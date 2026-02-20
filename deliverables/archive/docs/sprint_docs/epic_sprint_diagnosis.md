# Manifold Law Diagnostic Suite - Sprint Diagnosis

**Date:** 2025-11-24  
**Status:** ⚠️ **INCOMPLETE - ROOT CAUSES IDENTIFIED**

---

## Executive Summary

The Manifold Law Diagnostic Suite sprint did NOT complete successfully. Root causes have been identified:

1. **Epic 2:** WRONG SCRIPT was used - `run_epic2_mechanism_ablation.py` has PLACEHOLDERS instead of actual ablation logic
2. **Epic 3:** Noise injection ran but Lipschitz summary wasn't aggregated properly
3. **Report generation:** Aggregation code has bugs (fragmented baseline names, missing columns)

---

## Issue Analysis

### Epic 1: Curvature Sweep ✅ COMPLETE
- **Files:** 48 (24 summary + 24 detailed k-sweep)
- **Status:** Complete with data for all baseline-dataset combinations

### Epic 2: Mechanism Ablation ⚠️ INCOMPLETE
- **Files:** 18 (expected 24)
- **Problem 1:** Missing 6 baseline-dataset combinations
- **Problem 2:** `ablated_pearson_r` and `ablated_l2` columns are EMPTY
  - The ablation logic didn't actually run
  - Only `original_pearson_r` and `original_l2` are populated
- **Impact:** Cannot compute Δr (delta_r) for functional alignment analysis

### Epic 3: Noise Injection ⚠️ PARTIALLY COMPLETE
- **Files:** 25 (24 main + 1 summary expected)
- **Problem:** No `lipschitz_constant` computed
  - Noise sensitivity curves exist
  - Lipschitz estimation didn't run
- **Impact:** Cannot show robustness metric in 5-epic grid

### Epic 4: Direction-Flip Probe ✅ MOSTLY COMPLETE
- **Files:** 25
- **Status:** Data exists but may need aggregation review
- **Note:** Flip rates appear very low (mostly 0.0%)

### Epic 5: Tangent Alignment ⚠️ INCOMPLETE
- **Files:** 25
- **Problem:** Many TAS (Tangent Alignment Score) values are missing or suspicious
- **Impact:** Cannot properly compute tangent alignment metric

---

## Root Causes (DETAILED)

### 1. Epic 2: WRONG SCRIPT USED ❌

**File:** `src/goal_3_prediction/lsft/run_epic2_mechanism_ablation.py`

**Problem:** Lines 86-90 explicitly set ablation values to NaN with comment "Placeholder":
```python
# Placeholder - full implementation needs extended lsft_k_sweep
"ablated_pearson_r": np.nan,
"ablated_l2": np.nan,
"delta_r": np.nan,
```

**Correct Script:** `src/goal_3_prediction/lsft/mechanism_ablation.py` 
- This file HAS the correct implementation that runs both original and ablated LSFT
- It was NOT used in the sprint execution

**Fix:** Use `mechanism_ablation.py` instead of `run_epic2_mechanism_ablation.py`

### 2. Epic 3: AGGREGATION MISSING ⚠️

The noise injection DID run and the Lipschitz estimation code EXISTS in:
- `src/goal_3_prediction/lsft/noise_injection.py` (function: `analyze_noise_injection_results`)

**Problem:** The Lipschitz summary was computed per-file but not aggregated across all baselines.

**Fix:** Run the aggregation step to combine Lipschitz constants into `epic3_lipschitz_summary.csv`

### 3. Epic 5: DATA EXISTS ✅

Looking at the data, Epic 5 DOES have TAS values:
```
test_perturbation,tangent_alignment_score
ZNF326,0.6854284638503981
```

**Problem:** Report aggregation isn't reading Epic 5 data correctly.

### 4. Report Aggregation Bugs 🐛

**Problem 1:** Baseline names fragmented
- Getting `k562_lpm_scgptGeneEmb` instead of `lpm_scgptGeneEmb`
- Caused by concatenating dataset prefix to baseline name

**Problem 2:** Cross-epic summary code has bug
- Looking for wrong column names or not merging properly

**Problem 3:** "results" row appearing
- Invalid data being included in baseline_summary.csv

---

## Baseline Summary Issues

The `baseline_summary.csv` has these problems:
1. **Fragmented naming:** `k562_lpm_scgptGeneEmb` instead of `lpm_scgptGeneEmb`
2. **Missing metrics:** Epic 3 Lipschitz mostly NaN, Epic 5 TAS mostly NaN
3. **Invalid row:** Contains a row called "results"
4. **33 rows instead of 8:** Should have 8 canonical baselines

---

## What Works

1. ✅ Epic 1 curvature data is complete
2. ✅ Epic 2 has original_r values (just not ablated)
3. ✅ Epic 3 has noise sensitivity curves
4. ✅ Epic 4 has flip rate data
5. ✅ Visualization scripts exist and run

---

## Fix Plan

### Step 1: Fix Epic 2 (Re-run with correct script)
```bash
# Use mechanism_ablation.py instead of run_epic2_mechanism_ablation.py
python -m goal_3_prediction.lsft.mechanism_ablation \
    --adata_path $ADATA \
    --split_config $SPLIT \
    --annotation_path $ANNOTATIONS \
    --dataset_name adamson \
    --baseline_type lpm_selftrained \
    --output_dir results/manifold_law_diagnostics/epic2_mechanism_ablation
```

### Step 2: Fix Epic 3 (Run Lipschitz aggregation)
```python
# Aggregate Lipschitz constants from noise injection files
from noise_injection import analyze_noise_injection_results
# Run on each file and combine results
```

### Step 3: Fix Report Generation
1. Remove dataset prefix from baseline names
2. Fix column name lookups in cross-epic summary
3. Filter out "results" row

### Step 4: Regenerate 5-Epic Winner Grid
Once data is fixed, regenerate with:
```bash
python publication_package/generate_publication_reports.py
```

---

## Immediate Action Required

**The 5-epic winner grid is incomplete because:**
1. Epic 2: Used placeholder script instead of real implementation
2. Epic 3: Lipschitz not aggregated
3. Reports: Aggregation code has bugs

**Recommended:** Fix Epic 2 by re-running with `mechanism_ablation.py`, then fix report aggregation.

---

## Files That Need Changes

1. `run_all_epics_all_baselines.sh` - Point to correct Epic 2 script
2. `publication_package/generate_publication_reports.py` - Fix aggregation bugs
3. `publication_package/generate_cross_epic_analysis.py` - Fix baseline naming

---

**Status:** Ready to implement fixes


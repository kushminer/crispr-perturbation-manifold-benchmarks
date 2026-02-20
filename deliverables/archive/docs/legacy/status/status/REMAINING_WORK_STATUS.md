# Remaining Work Status - Manifold Law Diagnostic Suite

**Last Updated:** 2025-11-24

---

## Executive Summary

**Current Baseline Count:** ✅ **Fixed** - Now running all 8 baselines (was only 5)
- Updated `run_all_epics_all_baselines.sh` to include all 8 baselines
- `mean_response` excluded from diagnostic epics (special case, different handling)

**Major Issues Identified:**
1. ⚠️ **GEARS baseline:** Works in some contexts but fails in Epic 1/2/3 (empty results)
2. ⚠️ **Cross-dataset baselines:** K562/RPE1 also failing (empty results)  
3. ⚠️ **Epic 3:** Noise injection not fully implemented (only baseline)
4. ⚠️ **Summary reports:** Very sparse (only executive summary)

---

## Issue 1: Baseline Count Fixed ✅

**Problem:** Only 5 baselines were running instead of 8
- Previously: `lpm_selftrained`, `lpm_randomGeneEmb`, `lpm_randomPertEmb`, `lpm_scgptGeneEmb`, `lpm_scFoundationGeneEmb`
- 3 were commented out: `lpm_gearsPertEmb`, `lpm_k562PertEmb`, `lpm_rpe1PertEmb`

**Status:** ✅ **FIXED**
- Updated `run_all_epics_all_baselines.sh` to include all 8 baselines
- Now running: All 8 baselines across all datasets and epics
- Total experiments: 120 (8 baselines × 3 datasets × 5 epics)

---

## Issue 2: Epic 3 - Noise Injection Not Fully Implemented ⚠️

**Problem:** Epic 3 only generates baseline (noise_level=0) but doesn't actually inject noise.

**Location:** `src/goal_3_prediction/lsft/run_epic3_noise_injection.py` (lines 78-83)

**Current State:**
```
⚠️  Full noise injection requires extending LSFT pipeline with noise hooks
   Creating placeholder results structure...
```

**Why `epic3_noise_injection/baseline/` folder exists:**
- Epic 3 runs baseline LSFT evaluation first (no noise) to get `noise_level=0` metrics
- This baseline is stored in `epic3_noise_injection/baseline/` 
- Then it should run noisy versions (0.01, 0.05, 0.1, 0.2), but that part isn't implemented yet
- Currently creates placeholder CSV with NaN values for noisy conditions

**What Needs Fixing:**
1. Add noise injection hooks to `retrain_lpm_on_filtered_data()` function in `lsft.py`
2. Inject noise to embeddings (B matrix) or expression (Y matrix) 
3. Re-run evaluation with noise at each level (0.01, 0.05, 0.1, 0.2)
4. Currently only creates placeholder CSV structure with NaN values

**Files to modify:**
- `src/goal_3_prediction/lsft/lsft.py` - Add noise injection to `retrain_lpm_on_filtered_data()`
- `src/goal_3_prediction/lsft/run_epic3_noise_injection.py` - Implement actual noise loops

---

## Issue 3: GEARS Baseline - Inconsistent Behavior ⚠️

**Status:** ⚠️ **PARTIAL FAILURE - Needs Investigation**

### Working Contexts ✅
- **LSFT Resampling:** GEARS works perfectly (37-694 lines per dataset)
  - Adamson: 37 lines
  - K562: 490 lines  
  - RPE1: 694 lines
- **Epic 4 (Direction-Flip Probe):** Works (13-232 lines per dataset)
- **Epic 5 (Tangent Alignment):** Works (13-232 lines per dataset)

### Failing Contexts ❌
- **Epic 1 (Curvature Sweep):** Empty files (only headers, 1 line)
- **Epic 2 (Mechanism Ablation):** Empty files (only headers, 1 line)  
- **Epic 3 (Noise Injection baseline):** Empty files (only headers, 1 line)

**Error Pattern:**
- All failing epics use `evaluate_lsft_with_k_list()` which calls LSFT evaluation
- Error message: `"No results generated - all experiments may have failed"`
- Results DataFrame is empty (no rows added)
- Exceptions are caught silently at line 297-298 in `lsft_k_sweep.py`

**Root Cause Hypothesis:**
1. **Perturbation overlap issue:** GEARS embeddings may not cover all test perturbations
   - When `retrain_lpm_on_filtered_data()` is called for GEARS, test perturbations without GEARS embeddings might fail
   - The exception handler silently continues, so no results are added
   
2. **Embedding alignment issue:** GEARS uses gene symbols, may have alignment problems
   - GEARS embeddings are loaded using `construct_pert_embeddings()` with `source="gears"`
   - Test perturbations might not align properly if they're not in GEARS vocabulary
   
3. **Retraining failure:** The `retrain_lpm_on_filtered_data()` function might fail for GEARS embeddings
   - All retraining attempts might fail silently, leaving empty results

**Why Epic 4 & 5 Work:**
- Epic 4 & 5 don't use `evaluate_lsft_with_k_list()` - they use different code paths
- They work with existing baseline embeddings directly
- They don't need to retrain models on filtered subsets

**Investigation Steps:**
1. Add detailed error logging to `evaluate_lsft_with_k_list()` to see what exceptions occur
2. Check if GEARS test perturbations have embeddings (verify `B_test_baseline` is not None/empty)
3. Compare working LSFT resampling code with failing diagnostic epic code
4. Test with a single test perturbation to see exact error message

---

## Issue 4: Cross-Dataset Baselines (K562/RPE1) - Empty Results ⚠️

**Status:** ⚠️ **FAILING - Similar to GEARS**

**Problem:**
- `lpm_k562PertEmb` and `lpm_rpe1PertEmb` also returning empty results in Epic 1/2/3
- Same error pattern: `"No results generated - all experiments may have failed"`
- Results DataFrame is empty

**Likely Causes:**
1. **Source dataset availability:** Cross-dataset baselines need source dataset to be available
   - K562 baseline needs: `/Users/samuelminer/Documents/classes/nih_research/data_replogle_k562_essential/perturb_processed.h5ad`
   - RPE1 baseline needs: `/Users/samuelminer/Documents/classes/nih_research/data_replogle_rpe1_essential/perturb_processed.h5ad`
   - Paths might not be accessible or correct
   
2. **Perturbation name alignment:** Cross-dataset baselines use PCA from source dataset
   - Need to align perturbation names between source and target datasets
   - May have issues if perturbation names don't match

3. **Retraining failure:** Similar to GEARS - `retrain_lpm_on_filtered_data()` might fail for cross-dataset embeddings

**Investigation Steps:**
1. Verify source dataset paths exist and are accessible
2. Check perturbation name overlap between source and target datasets
3. Test cross-dataset embedding loading in isolation

---

## Issue 5: Summary Reports Are Sparse ⚠️

**Current Status:**
- Only `executive_summary.md` exists in `results/manifold_law_diagnostics/summary_reports/`
- Script `generate_diagnostic_summary.py` exists but output is minimal
- Only 1 figure exists: `curvature_sweep_all_baselines_datasets.png`

**What's Missing:**
- Detailed findings per epic
- Cross-epic comparisons  
- Baseline performance summaries
- Statistical analysis summaries (mean, CI, p-values)
- Additional visualizations for each epic
- Comparison tables

**Available Tools:**
- `generate_diagnostic_summary.py` - Exists but needs expansion
- `create_diagnostic_visualizations.py` - May exist for some visualizations

**Action Items:**
1. Run `generate_diagnostic_summary.py` to see what it produces
2. Expand summary generation to include:
   - Per-epic detailed findings
   - Cross-baseline comparisons
   - Statistical summaries
   - Key insights and conclusions

---

## Action Plan

### Priority 1: Fix GEARS and Cross-Dataset Baseline Failures

**Steps:**
1. **Add detailed error logging:**
   - Modify `evaluate_lsft_with_k_list()` to log full exception messages
   - Log when test perturbations are skipped
   - Log embedding loading issues
   
2. **Test GEARS embedding loading:**
   - Verify GEARS embeddings load correctly for all test perturbations
   - Check if `B_test_baseline` is None or empty
   - Test with a single test perturbation first
   
3. **Compare working vs failing code paths:**
   - Compare `evaluate_lsft()` (working in LSFT resampling) with `evaluate_lsft_with_k_list()` (failing)
   - Identify differences in how embeddings are handled

### Priority 2: Complete Epic 3 Implementation

**Steps:**
1. **Implement noise injection:**
   - Add `noise_level` and `noise_type` parameters to `retrain_lpm_on_filtered_data()`
   - Implement Gaussian noise injection on B matrix (embeddings)
   - Implement dropout noise if needed
   
2. **Update Epic 3 runner:**
   - Replace placeholder code with actual noise injection loops
   - Run evaluation for each noise level (0.01, 0.05, 0.1, 0.2)
   - Save results properly

### Priority 3: Expand Summary Reports

**Steps:**
1. Run existing summary generator and review output
2. Expand `generate_diagnostic_summary.py` to include:
   - Detailed per-epic summaries
   - Cross-baseline comparison tables
   - Statistical summaries
3. Generate additional visualizations

---

## Files Reference

### Key Files
- **Baseline definitions:** `src/goal_2_baselines/baseline_types.py`
- **Epic 3 runner:** `src/goal_3_prediction/lsft/run_epic3_noise_injection.py`
- **LSFT k-sweep (failing):** `src/goal_3_prediction/lsft/lsft_k_sweep.py`
- **LSFT evaluation (working):** `src/goal_3_prediction/lsft/lsft.py`
- **Summary generator:** `generate_diagnostic_summary.py`
- **Master runner:** `run_all_epics_all_baselines.sh`

### Result Locations
- **Epic 1:** `results/manifold_law_diagnostics/epic1_curvature/`
- **Epic 2:** `results/manifold_law_diagnostics/epic2_mechanism_ablation/`
- **Epic 3:** `results/manifold_law_diagnostics/epic3_noise_injection/`
- **Epic 4:** `results/manifold_law_diagnostics/epic4_direction_flip/`
- **Epic 5:** `results/manifold_law_diagnostics/epic5_tangent_alignment/`
- **Summaries:** `results/manifold_law_diagnostics/summary_reports/`

---

## Next Steps

1. ✅ **DONE:** Fix baseline count (all 8 baselines now running)
2. 🔄 **IN PROGRESS:** Investigate GEARS and cross-dataset baseline failures
3. ⏳ **PENDING:** Complete Epic 3 noise injection implementation
4. ⏳ **PENDING:** Expand summary reports

---

## Questions for User

1. **Epic 3:** Should we prioritize fixing Epic 3 noise injection, or focus on fixing GEARS failures first?
2. **GEARS:** Are the empty GEARS results blocking your analysis, or can we proceed with the working baselines?
3. **Summary Reports:** What level of detail do you need in the summary reports? Should they be poster-ready or more detailed analysis?

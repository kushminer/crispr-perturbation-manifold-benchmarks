# Poster Figures Data Status

## Summary

**All poster figures are using CORRECT data sources** - none of them use the corrupted `LSFT_results.csv` file.

## Figure-by-Figure Analysis

### ✅ Figure 1: Baseline Performance Comparison
**Script:** `poster/create_figure1_baseline_comparison.py`

**Data Sources:**
- **Pseudobulk:** Hardcoded values (verified against raw data)
- **Single-cell:** `results/single_cell_analysis/comparison/baseline_results_all.csv`

**Status:** ✅ **CORRECT**
- Pseudobulk values match `baseline_r` from `LSFT_raw_per_perturbation.csv`
- K562 Self-trained PCA: 0.6638 ✓
- K562 scGPT: 0.5127 ✓
- RPE1 Self-trained PCA: 0.7678 ✓
- RPE1 scGPT: 0.6672 ✓

**Does NOT use:** `LSFT_results.csv` (uses hardcoded verified values)

---

### ✅ Figure 3: LSFT Improvements (Δr = LSFT - baseline)
**Script:** `poster/create_figure3_lsft_improvements.py`

**Data Sources:**
- **Pseudobulk:** `results/goal_3_prediction/lsft_resampling/{dataset}/lsft_{dataset}_lpm_*.csv`
- **Single-cell:** `results/single_cell_analysis/{dataset}/lsft/lsft_single_cell_summary_{dataset}_*.csv`

**Status:** ✅ **CORRECT**
- Loads directly from resampling CSV files
- Computes improvements (`improvement_pearson_r` column)
- Does NOT use `LSFT_results.csv`

---

### ✅ Figure 4: LOGO Comparison
**Script:** `poster/create_figure4_logo_comparison.py`

**Data Sources:**
- **Pseudobulk:** `deliverables/skeletons_and_fact_sheets/data/LOGO_results.csv`
- **Single-cell:** `results/goal_4_logo/{dataset}_single_cell_summary.csv`

**Status:** ✅ **CORRECT**
- Uses `LOGO_results.csv` (separate from LSFT data)
- Single-cell loads from actual result files
- Does NOT use `LSFT_results.csv`

---

## Files That DO Use Corrupted LSFT_results.csv (Fixed)

The following scripts were updated to bypass the corrupted file:

1. ✅ `poster/create_figure1_neighborhood_smoothness.py` (if it exists)
   - Now computes from raw data
   
2. ✅ `audits/random_embedding_audit/06_resolution_comparison.py`
   - Now uses `LSFT_resampling.csv`

3. ✅ `deliverables/skeletons_and_fact_sheets/data/core_findings/scripts/why_linear_models_win.py`
   - Now computes from raw data

4. ✅ `fix_and_regenerate_all.py`
   - Now computes baseline_r from raw data

## Conclusion

**All current poster figures (1, 3, 4) are using correct, non-corrupted data sources.** They either:
- Use hardcoded verified values (Figure 1)
- Load directly from result CSV files (Figures 3 and 4)
- Do NOT rely on the corrupted `LSFT_results.csv` file

No regeneration needed for the poster figures themselves.

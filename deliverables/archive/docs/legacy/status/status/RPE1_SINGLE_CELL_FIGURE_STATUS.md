# RPE1 Single-Cell Data in Poster Figures

## Status: ✅ Complete

RPE1 single-cell resolution data is **complete** and **already included** in all relevant poster figures.

---

## Figure-by-Figure Status

### ✅ Figure 1: Baseline Performance Comparison

**Status:** Already includes RPE1 single-cell baseline data

**Source:** Hardcoded values in `deliverables/poster/create_figure1_baseline_comparison.py`

**Values (from `results/single_cell_analysis/rpe1/single_cell_baseline_summary.csv`):**
- Self-trained PCA: 0.395
- scGPT: 0.316
- scFoundation: 0.233
- GEARS: 0.203
- Random Gene: 0.203
- Random Pert: 0.203

**Action Required:** None - values are already correct ✓

---

### ✅ Figure 4: LOGO Comparison

**Status:** Already loads RPE1 single-cell LOGO data dynamically

**Source:** `results/single_cell_analysis/rpe1/logo/logo_single_cell_summary_rpe1_Transcription.csv`

**Values:**
- Self-trained PCA: 0.4144
- scGPT: 0.3436
- scFoundation: 0.2704
- GEARS: 0.2540
- Random Gene: 0.2535
- Random Pert: 0.2531

**Code Location:** `deliverables/poster/create_figure4_logo_comparison.py` lines 94-104

**Action Required:** None - code already loads RPE1 data ✓

---

### ⚠️ Figure 3: LSFT Improvements

**Status:** RPE1 single-cell LSFT data is **not available**

**Reason:** No RPE1 single-cell LSFT results exist (no `results/single_cell_analysis/rpe1/lsft/` directory)

**Current Behavior:**
- Code in `deliverables/poster/create_figure3_lsft_improvements.py` attempts to load RPE1 LSFT data (lines 125-144)
- All RPE1 single-cell LSFT values are set to `None`
- Figure 3 shows RPE1 pseudobulk LSFT improvements, but not single-cell

**Note:** This is expected - RPE1 single-cell LSFT was not run. Only baseline and LOGO single-cell evaluations were completed for RPE1.

---

## Summary

| Figure | RPE1 Single-Cell Data | Status |
|--------|----------------------|--------|
| Figure 1 (Baseline) | Included | ✅ Complete |
| Figure 4 (LOGO) | Loaded dynamically | ✅ Complete |
| Figure 3 (LSFT) | Not available | ⚠️ Expected (data doesn't exist) |

**Conclusion:** All available RPE1 single-cell data is already included in the poster figures. No updates needed.


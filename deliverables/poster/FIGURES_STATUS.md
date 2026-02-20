# Poster Figures Status

## Created Figures

### Figure 1: Neighborhood Smoothness Curve
- **Files**: 
  - `figure1_neighborhood_smoothness.png` (277KB)
  - `figure1_neighborhood_smoothness.pdf` (29KB)
- **Script**: `create_figure1_neighborhood_smoothness.py`
- **Data Source**: `skeletons_and_fact_sheets/data/LSFT_raw_per_perturbation.csv` (raw per-perturbation data)
- **Description**: Mean Pearson r vs. neighborhood size (k) with 95% confidence intervals
- **Status**: ✅ Uses correct data (NOT corrupted LSFT_results.csv)

### Figure 2: Similarity → Error
- **Files**:
  - `figure2_similarity_error.png` (1.1MB)
  - `figure2_similarity_error.pdf` (65KB)
  - `figure2_similarity_error_combined.png` (1.2MB)
  - `figure2_similarity_error_combined.pdf` (61KB)
- **Script**: `create_figure2_similarity_error.py`
- **Data Source**: `skeletons_and_fact_sheets/data/LSFT_raw_per_perturbation.csv`
- **Description**: Scatter plot with best-fit lines showing higher neighbor similarity → lower prediction error
- **Status**: ✅ Uses correct data

### Other Existing Figures
- `figure1_baseline_comparison_*.png/pdf` - Baseline performance comparison
- `figure3_lsft_improvements_*.png/pdf` - LSFT improvements
- `figure4_logo_comparison_*.png/pdf` - LOGO comparison

---

## Data Sources

### ✅ CORRECT Data Sources (Currently Used)
- `skeletons_and_fact_sheets/data/LSFT_raw_per_perturbation.csv` - Raw per-perturbation LSFT results
- `skeletons_and_fact_sheets/data/LSFT_resampling.csv` - LSFT resampling results with CIs
- `results/goal_3_prediction/lsft_resampling/*/lsft_*.csv` - Individual baseline resampling files

### ❌ DELETED Corrupted File
- ~~`skeletons_and_fact_sheets/data/LSFT_results.csv`~~ - **DELETED** (had Adamson values duplicated to all datasets)

---

## Fixes Applied

1. **Deleted corrupted file**: `skeletons_and_fact_sheets/data/LSFT_results.csv`
2. **Updated Figure 1 script**: Now computes mean r values directly from `LSFT_raw_per_perturbation.csv` instead of using corrupted file
3. **Figure 2**: Already uses correct raw data source

All poster figures now use correct, verified data sources.


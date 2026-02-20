# Skeletons and Fact Sheets

This folder contains skeleton versions of key analysis reports and a dataset fact sheet.

## Contents

### 1. `RESAMPLING_FINDINGS_REPORT_SKELETON.md`
- **Purpose:** Skeleton version of the resampling findings report
- **Content:** Methods and observations only (no commentary or interpretation)
- **Key Features:**
  - **Prominent attribution** to Ahlmann-Eltze et al., Nature Methods 2025 at the top
  - Clear distinction between "What We Keep" (from the paper) vs "Our Methodological Contributions" (LSFT, LOGO, uncertainty quantification)
  - **Critical design choice section:** Explains that all baselines use the same ridge regression framework; we're testing embedding quality, not comparing architectures
- **Includes:**
  - Bootstrap and permutation test procedures
  - LSFT and LOGO evaluation methods
  - Baseline descriptions (with attribution to the original paper)
  - Performance tables with confidence intervals and attribution captions
  - Statistical comparisons (scGPT embeddings vs Random embeddings, etc.)

### 2. `lsft_analysis_skeleton.md`
- **Purpose:** Skeleton version of the LSFT (Local Similarity-Filtered Training) analysis
- **Content:** Methods and observations only (no commentary or interpretation)
- **Key Features:**
  - **Prominent attribution** to Ahlmann-Eltze et al., Nature Methods 2025 at the top
  - Clear "What We Keep" vs "Our Methodological Contribution" sections
  - **Critical design choice section:** Explains embedding evaluation framework
- **Includes:**
  - Detailed LSFT procedure explanation (one perturbation at a time evaluation)
  - Baseline descriptions (with attribution to the original paper)
  - Performance tables by dataset and top percentage (with attribution captions)
  - Fraction of perturbations improved statistics

### 3. `DATASET_FACT_SHEET.md`
- **Purpose:** Fact sheet about the raw datasets (Adamson and Replogle)
- **Content:** Dataset characteristics, publication information, and data access
- **Includes:**
  - Publication details and citations
  - Dataset dimensions and characteristics
  - Data processing information
  - Comparison tables

### 4. `data/` Folder

**Purpose:** Contains all machine-readable CSV files with evaluation results

**Contents:**
- **Raw Per-Perturbation Data:**
  - `LSFT_raw_per_perturbation.csv` (9,744 rows) - Individual perturbation results for LSFT
  - `LOGO_raw_per_perturbation.csv` (19,305 rows) - Individual perturbation results for LOGO

- **Summary Statistics:**
  - `LSFT_results.csv` (72 rows) - LSFT summary (point estimates, all top_k)
  - `LSFT_resampling.csv` (24 rows) - LSFT resampling (with CIs, top_k=0.05)
  - `LOGO_results.csv` (27 rows) - LOGO summary (point estimates with CIs)
  - `LOGO_resampling.csv` (27 rows) - LOGO resampling (with bootstrap CIs)

**See `data/README.md` for complete documentation of all CSV files.**

## Key Methodological Points

These skeleton files emphasize:

1. **Attribution:** All baselines come from Ahlmann-Eltze et al., Nature Methods 2025. We extend their work with LSFT and LOGO evaluation frameworks.

2. **Design Choice:** All baselines use the same ridge regression architecture (Y = A × K × B). What varies is the source of embeddings (A and B), not the prediction model. This isolates representation quality from architectural differences.

3. **Similarity Computation:**
   - **What is compared:** Feature matrices (B embeddings), NOT target matrices (Y expression changes)
   - **No lookahead bias:** Embedding space is defined by training data only; test data is projected into this pre-defined space
   - **Procedure:** For training-data-based embeddings, PCA is fit on training data (`fit_transform`), then test data is transformed (`transform`) into that space

4. **What's Novel:**
   - LSFT: One-at-a-time evaluation using similarity-filtered training data
   - LOGO: Functional class holdout for biological extrapolation testing
   - Uncertainty quantification: Bootstrap CIs and permutation tests

5. **What's from the Paper:**
   - All 8 linear baselines + mean_response
   - Ridge regression model architecture
   - Data splits (GEARS methodology)
   - Core evaluation metrics

## Usage

These skeleton files are designed to provide:
- **Methods documentation:** Clear explanation of evaluation procedures with proper attribution
- **Raw observations:** Uninterpreted performance metrics and statistics
- **Dataset reference:** Quick access to dataset characteristics
- **Attribution clarity:** Explicit distinction between replicated methods and novel contributions

They can be used for:
- Manuscript preparation
- Poster creation
- Documentation
- Reproducibility verification
- Grant proposals and presentations

---

**Last Updated:** 2025-11-21


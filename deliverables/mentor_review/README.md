# Mentor Review Package

**Linear Perturbation Prediction Evaluation Framework**  
**Resampling-Enabled Statistical Analysis**

---

## 📋 Package Contents

This package contains all materials needed for mentor review of the resampling-enabled evaluation framework.

### ✅ Required Components

1. **Minimum Data Tables** (`data_tables/`)
   - `A_LSFT_Summary_Table.csv` - LSFT performance summary
   - `B_LOGO_Summary_Table.csv` - LOGO performance summary
   - `C_Hardness_Performance_Regression_Table.csv` - Hardness-performance regression results

2. **Key Visuals** (`key_visuals/`)
   - 6 critical figures for manuscript/poster
   - High-resolution PNG files (300 DPI)

3. **Methods Appendix** (`METHODS_APPENDIX.md`)
   - Page 1: Resampling Methods (Bootstrap, Permutation, CI, Hardness, Metrics)
   - Page 2: Evaluation Splits (LSFT, LOGO, Rationale)

4. **Full Results** (`full_results.zip`)
   - Complete LSFT resampling results
   - Complete LOGO resampling results
   - Baseline comparisons
   - (Optional for deeper review)

---

## 🚀 Quick Start

### For Statistical Verification

1. **Open the data tables** (`data_tables/`)
   - Each CSV is <50 rows
   - Contains point estimates, CIs, and sample sizes
   - Verify bootstrap CIs are reasonable (typically ±0.01-0.05 for Pearson r)

2. **Review the methods** (`METHODS_APPENDIX.md`)
   - Bootstrap procedure: B=1,000 samples, percentile method
   - Permutation test: P=10,000 permutations, sign-flip test
   - Hardness metric: Mean cosine similarity to top-K training perturbations

3. **Examine key visuals** (`key_visuals/`)
   - Figures 1-2: LSFT performance and hardness relationship
   - Figures 3-4: LOGO performance and baseline comparison
   - Figures 5-6: Conceptual schematics

### For Deeper Review

1. **Extract `full_results.zip`**
2. **Navigate to specific results:**
   - `lsft_resampling/` - Per-perturbation LSFT results with CIs
   - `logo_resampling/` - LOGO results with CIs
   - `comparisons/` - Baseline comparison tables with p-values

---

## 📊 Data Table Descriptions

### A. LSFT Summary Table

**Columns:**
- `baseline`: Baseline type (selftrained, scgptGeneEmb, randomGeneEmb)
- `dataset`: Dataset name (adamson, k562, rpe1)
- `pearson_r`: Mean Pearson correlation (point estimate)
- `pearson_ci_lower`: 95% CI lower bound
- `pearson_ci_upper`: 95% CI upper bound
- `l2`: Mean L2 distance (point estimate)
- `n_test`: Number of test perturbations

**Interpretation:**
- Higher `pearson_r` is better (range: -1 to 1)
- Lower `l2` is better (range: 0 to ∞)
- CIs should not overlap zero for significant results

### B. LOGO Summary Table

**Same columns as LSFT Summary Table**

**Key Differences:**
- Evaluates functional class holdout (more challenging)
- Typically lower performance than LSFT (true generalization test)
- Sample sizes: n=5 (Adamson), n~397 (K562), n~313 (RPE1)

### C. Hardness-Performance Regression Table

**Columns:**
- `baseline`: Baseline type
- `dataset`: Dataset name
- `top_pct`: Similarity threshold (0.05 = top 5%)
- `slope`: Regression slope (hardness → performance)
- `slope_ci_lower`: 95% CI lower bound for slope
- `slope_ci_upper`: 95% CI upper bound for slope
- `r`: Correlation between hardness and performance
- `sample_size`: Number of test perturbations

**Interpretation:**
- Positive slope: Higher hardness → better performance (expected)
- Negative slope: Higher hardness → worse performance (unexpected, may indicate overfitting)
- CI should not overlap zero for significant relationships

---

## 🔬 Statistical Methods Summary

### Bootstrap Confidence Intervals

- **Method:** Percentile bootstrap
- **Samples:** B = 1,000
- **Coverage:** 95% (α = 0.05)
- **Application:** Mean Pearson r, mean L2, mean deltas, regression slopes

### Permutation Tests

- **Method:** Paired sign-flip permutation test
- **Permutations:** P = 10,000
- **Null hypothesis:** Mean delta = 0 (no difference between baselines)
- **Application:** Baseline comparisons (e.g., scGPT vs Random)

### Hardness Metric

- **Definition:** Mean cosine similarity to top-K most similar training perturbations
- **Range:** [0, 1], where 1 = very similar, 0 = very dissimilar
- **Interpretation:** High hardness → easier to predict (many similar training examples)

---

## 📁 Repository Structure

```
lpm-evaluation-framework-v2/
├── src/
│   ├── goal_3_prediction/
│   │   ├── lsft/              # LSFT evaluation
│   │   └── functional_class_holdout/  # LOGO evaluation
│   └── stats/
│       ├── bootstrapping.py   # Bootstrap CI utilities
│       └── permutation.py     # Permutation test utilities
├── results/
│   └── goal_3_prediction/
│       ├── lsft_resampling/   # LSFT results
│       └── functional_class_holdout_resampling/  # LOGO results
├── data/                      # Input data (see data/README.md)
└── deliverables/mentor_review/             # This package
```

---

## 🔧 How to Run Resampling

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run LSFT with Resampling

```bash
python -m goal_3_prediction.lsft.run_lsft_with_resampling \
    --adata data/adamson.h5ad \
    --split_config results/goal_2_baselines/splits/adamson_split.json \
    --baseline lpm_selftrained \
    --dataset adamson \
    --output_dir results/goal_3_prediction/lsft_resampling/adamson \
    --n_boot 1000
```

### Run LOGO with Resampling

```bash
python -m goal_3_prediction.functional_class_holdout.logo_resampling \
    --adata data/adamson.h5ad \
    --annotation data/annotations/adamson_go.tsv \
    --functional_class "Transcription" \
    --dataset adamson \
    --output_dir results/goal_3_prediction/functional_class_holdout_resampling/adamson \
    --n_boot 1000 \
    --n_perm 10000
```

---

## 📚 Dependencies

### Python Packages

- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning utilities (PCA, cosine similarity)
- `anndata` - Single-cell data structures
- `matplotlib` / `seaborn` - Visualization
- `scipy` - Statistical functions

### Data Dependencies

See `data/README.md` for complete data source documentation.

**Key data files:**
- Dataset files (`.h5ad`): Adamson, K562, RPE1
- Split configs (`.json`): Train/test/validation splits
- Annotations (`.tsv`): GO/Reactome functional class annotations
- Pretrained embeddings: scGPT, scFoundation (optional)

---

## 📝 Citation

If using this evaluation framework, please cite:

**Original Paper:**
```
[Citation for Nature Methods 2025 paper]
```

**This Framework:**
```
[Your citation when published]
```

---

## ❓ Questions?

For questions about:
- **Statistical methods:** See `METHODS_APPENDIX.md`
- **Data sources:** See `data/README.md`
- **Code structure:** See main `README.md` in repository root
- **Results interpretation:** See `RESAMPLING_FINDINGS_REPORT.md` in repository root

---

**Package Generated:** [Date]  
**Framework Version:** v2 (Resampling-Enabled)  
**Evaluation Scope:** LSFT + LOGO across 3 datasets (Adamson, K562, RPE1)


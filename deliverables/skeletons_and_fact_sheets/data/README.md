# CSV Data Files

This folder contains all machine-readable CSV files extracted from the evaluation framework results. Files are organized into two categories: **Raw Per-Perturbation Data** and **Summary Statistics**.

---

## Raw Per-Perturbation Data

These files contain individual perturbation-level results, allowing for detailed analysis and custom aggregations.

### 1. `LSFT_raw_per_perturbation.csv` (9,744 rows)

**Purpose:** Complete per-perturbation LSFT (Local Similarity-Filtered Training) results across all datasets, baselines, and top-K percentages.

**Content:**
- **8 baselines** × **3 datasets** × **3 top_k values** × **variable perturbations per dataset**
- Adamson: 12 test perturbations
- K562: 163 test perturbations  
- RPE1: 231 test perturbations

**Columns:**
- `dataset`: Dataset name (adamson, k562, rpe1)
- `baseline`: Baseline identifier (e.g., lpm_selftrained, lpm_scgptGeneEmb, lpm_randomGeneEmb)
- `test_perturbation`: Name of the test perturbation being evaluated
- `top_pct`: Top-K percentage as decimal (0.01, 0.05, 0.10)
- `local_train_size`: Number of training perturbations used after filtering
- `local_mean_similarity`: Mean cosine similarity to filtered training perturbations
- `local_max_similarity`: Maximum similarity to filtered training perturbations
- `local_min_similarity`: Minimum similarity to filtered training perturbations
- `performance_local_pearson_r`: Pearson correlation for LSFT prediction
- `performance_local_l2`: L2 distance for LSFT prediction
- `performance_baseline_pearson_r`: Baseline (full training set) Pearson correlation
- `performance_baseline_l2`: Baseline (full training set) L2 distance
- `improvement_pearson_r`: Improvement in Pearson r (local - baseline)
- `improvement_l2`: Improvement in L2 (baseline - local, since lower is better)
- `baseline_type`: Baseline identifier (duplicate of `baseline`)
- `pearson_r`: Standardized metric (same as `performance_local_pearson_r`)
- `l2`: Standardized metric (same as `performance_local_l2`)
- `hardness`: Hardness metric (mean similarity to training data)
- `embedding_similarity`: Embedding similarity metric
- `split_fraction`: Fraction of training data used

**Usage Example:**
```python
import pandas as pd
df = pd.read_csv('LSFT_raw_per_perturbation.csv')

# Filter for specific baseline and top_k
selftrained_5pct = df[(df['baseline'] == 'lpm_selftrained') & (df['top_pct'] == 0.05)]

# Calculate mean improvement per dataset
improvement_by_dataset = df.groupby(['dataset', 'baseline', 'top_pct'])['improvement_pearson_r'].mean()
```

---

### 2. `LOGO_raw_per_perturbation.csv` (19,305 rows)

**Purpose:** Complete per-perturbation LOGO (Functional Class Holdout) results across all datasets and baselines.

**Content:**
- **9 baselines** × **3 datasets** × **variable perturbations per dataset**
- Adamson: 5 test perturbations
- K562: 397 test perturbations
- RPE1: 313 test perturbations

**Columns:**
- `dataset`: Dataset name (adamson, k562, rpe1)
- `baseline`: Baseline identifier (includes mean_response)
- `perturbation`: Name of the perturbation being evaluated
- `class`: Functional class (typically "Transcription" for test set)
- `pearson_r`: Pearson correlation for this perturbation
- `l2`: L2 distance for this perturbation
- `split_type`: Type of split (e.g., "functional_class_holdout")
- `baseline_type`: Baseline identifier (duplicate of `baseline`)
- `class_name`: Functional class name (typically "Transcription")

**Usage Example:**
```python
import pandas as pd
df = pd.read_csv('LOGO_raw_per_perturbation.csv')

# Filter for specific baseline
scgpt_results = df[df['baseline'] == 'lpm_scgptGeneEmb']

# Calculate mean performance per dataset
mean_performance = df.groupby(['dataset', 'baseline'])[['pearson_r', 'l2']].mean()
```

---

## Summary Statistics

These files contain aggregated statistics (means, confidence intervals) suitable for high-level comparisons and visualizations.

### 3. `LSFT_results.csv` (72 rows)

**Purpose:** Summary LSFT results (point estimates) for all baselines across all datasets and top-K percentages.

**Content:**
- **8 baselines** × **3 datasets** × **3 top_k values** = 72 rows
- Contains mean performance metrics without confidence intervals

**Columns:**
- `dataset`: Dataset name (adamson, k562, rpe1)
- `baseline`: Baseline identifier
- `top_k`: Top-K percentage as decimal (0.01, 0.05, 0.10)
- `local_r`: Mean local LSFT Pearson correlation
- `baseline_r`: Mean baseline (full training set) Pearson correlation
- `local_l2`: Mean local LSFT L2 distance
- `baseline_l2`: Mean baseline (full training set) L2 distance
- `mean_similarity`: Mean cosine similarity across all perturbations

**Source:** Extracted from `lsft_analysis_skeleton.md`

---

### 4. `LSFT_resampling.csv` (24 rows)

**Purpose:** LSFT resampling results with bootstrap confidence intervals (top_pct=0.05 only).

**Content:**
- **8 baselines** × **3 datasets** = 24 rows
- Contains mean performance metrics with 95% bootstrap confidence intervals

**Columns:**
- `dataset`: Dataset name (adamson, k562, rpe1)
- `baseline`: Baseline identifier
- `top_k`: Top-K percentage (0.05, fixed for resampling results)
- `r_mean`: Mean Pearson correlation
- `r_ci_low`: Lower bound of 95% bootstrap CI for Pearson r
- `r_ci_high`: Upper bound of 95% bootstrap CI for Pearson r
- `l2_mean`: Mean L2 distance
- `l2_ci_low`: Lower bound of 95% bootstrap CI for L2
- `l2_ci_high`: Upper bound of 95% bootstrap CI for L2

**Source:** Extracted from `RESAMPLING_FINDINGS_REPORT_SKELETON.md` (LSFT Performance section with CIs)

---

### 5. `LOGO_results.csv` (27 rows)

**Purpose:** Summary LOGO results (point estimates with CIs from skeleton reports).

**Content:**
- **9 baselines** × **3 datasets** = 27 rows
- Contains mean performance metrics with confidence intervals

**Columns:**
- `dataset`: Dataset name (adamson, k562, rpe1)
- `baseline`: Baseline identifier (includes mean_response)
- `r_mean`: Mean Pearson correlation
- `r_ci_low`: Lower bound of 95% CI for Pearson r
- `r_ci_high`: Upper bound of 95% CI for Pearson r
- `l2_mean`: Mean L2 distance
- `l2_ci_low`: Lower bound of 95% CI for L2
- `l2_ci_high`: Upper bound of 95% CI for L2

**Source:** Extracted from `RESAMPLING_FINDINGS_REPORT_SKELETON.md` (LOGO Performance section)

---

### 6. `LOGO_resampling.csv` (27 rows)

**Purpose:** LOGO resampling results with bootstrap confidence intervals.

**Content:**
- **9 baselines** × **3 datasets** = 27 rows
- Contains mean performance metrics with 95% bootstrap confidence intervals

**Columns:**
- `dataset`: Dataset name (adamson, k562, rpe1)
- `baseline`: Baseline identifier (includes mean_response)
- `r_mean`: Mean Pearson correlation
- `r_ci_low`: Lower bound of 95% bootstrap CI for Pearson r
- `r_ci_high`: Upper bound of 95% bootstrap CI for Pearson r
- `l2_mean`: Mean L2 distance
- `l2_ci_low`: Lower bound of 95% bootstrap CI for L2
- `l2_ci_high`: Upper bound of 95% bootstrap CI for L2

**Source:** Extracted from `RESAMPLING_FINDINGS_REPORT_SKELETON.md` (LOGO Performance section with CIs)

---

## File Relationships

```
Raw Data (Per-Perturbation)
├── LSFT_raw_per_perturbation.csv  → Aggregated to → LSFT_results.csv
│                                         ↓
│                                    LSFT_resampling.csv (with CIs)
│
└── LOGO_raw_per_perturbation.csv  → Aggregated to → LOGO_results.csv
                                          ↓
                                     LOGO_resampling.csv (with CIs)
```

**Key Differences:**
- **Raw files**: Individual perturbation results (thousands of rows)
- **Summary files**: Aggregated means (tens of rows)
- **Resampling files**: Include bootstrap confidence intervals

---

## Data Generation

These CSV files are generated by scripts in the parent directory:

1. **Raw Data:**
   - `create_raw_data_csvs.py` - Concatenates standardized CSV files from results directories

2. **Summary Data:**
   - `create_csv_results.py` - Extracts summary tables from skeleton markdown files
   - `create_resampling_csv.py` - Extracts resampling results with CIs from skeleton markdown files

To regenerate these files, run:
```bash
cd skeletons_and_fact_sheets
python create_raw_data_csvs.py
python create_csv_results.py
python create_resampling_csv.py
```

---

## Baselines Reference

All baselines are from **Ahlmann-Eltze et al., Nature Methods 2025**:

1. **lpm_selftrained**: PCA on training data (genes and perturbations)
2. **lpm_scgptGeneEmb**: scGPT gene embeddings + PCA perturbation embeddings
3. **lpm_scFoundationGeneEmb**: scFoundation gene embeddings + PCA perturbation embeddings
4. **lpm_randomGeneEmb**: Random gene embeddings + PCA perturbation embeddings
5. **lpm_k562PertEmb**: PCA gene embeddings + K562 dataset perturbation embeddings
6. **lpm_rpe1PertEmb**: PCA gene embeddings + RPE1 dataset perturbation embeddings
7. **lpm_gearsPertEmb**: PCA gene embeddings + GEARS GO graph perturbation embeddings
8. **lpm_randomPertEmb**: PCA gene embeddings + random perturbation embeddings
9. **mean_response**: Simple baseline (mean expression change, LOGO only)

**Note:** All baselines use the same ridge regression framework (Y = A × K × B). The evaluation tests embedding quality, not architectural differences.

---

## Dataset Reference

- **adamson**: Adamson et al. dataset (n=12 test perturbations for LSFT, n=5 for LOGO)
- **k562**: Replogle K562 dataset (n=163 test perturbations for LSFT, n=397 for LOGO)
- **rpe1**: Replogle RPE1 dataset (n=231 test perturbations for LSFT, n=313 for LOGO)

---

## Citation

When using these data files, please cite:

1. **Original Paper (Baselines):**
   Ahlmann-Eltze, C., et al. (2025). Linear perturbation models for predicting cellular responses. *Nature Methods*. [DOI]

2. **This Evaluation Framework:**
   [Your citation for the LSFT and LOGO evaluation extensions]

---

**Last Updated:** 2025-11-21


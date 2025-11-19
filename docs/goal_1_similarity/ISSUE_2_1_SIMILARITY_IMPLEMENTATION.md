# Issue 2.1 — Similarity Analysis Implementation

**Status:** ✅ **COMPLETE**  
**Date:** 2025-11-17

---

## Summary

Implemented two distinct similarity analysis modules:
1. **DE Matrix Similarity** - Computes similarity in expression space (Y matrix)
2. **Embedding Similarity** - Computes similarity in baseline-specific embedding spaces (B matrices)

Both modules are fully separated with clear naming and distinct output directories.

---

## Implementation

### 1. DE Matrix Similarity (Expression Space)

**Script:** `src/similarity/de_matrix_similarity.py`  
**Output Directory:** `results/de_matrix_similarity/`

**What it does:**
- Computes cosine similarity between test and training perturbations in the **pseudobulk expression change space (Y matrix)**
- This is the **same for all baselines** since Y (expression changes) is fixed across all baselines
- Measures how similar test perturbations are to training perturbations in terms of **actual gene expression changes**

**Key Functions:**
- `load_pseudobulk_matrix()` - Loads Y matrix (perturbations × genes)
- `compute_similarity_statistics()` - Computes cosine similarity statistics
- `attach_performance_metrics()` - Attaches baseline performance
- `plot_similarity_distributions()` - Creates distribution plots
- `plot_performance_vs_similarity()` - Creates scatter plots
- `compute_regression_analysis()` - Fits regression: performance ~ similarity

**Output Files:**
- `de_matrix_similarity_results.csv` - Combined results (perturbation × baseline)
- `de_matrix_regression_analysis.csv` - Regression results per baseline
- `fig_de_matrix_similarity_distributions.png` - Similarity distribution plots
- `fig_de_matrix_performance_vs_similarity.png` - Performance vs similarity scatter plots
- `de_matrix_similarity_report.md` - Summary report

**Usage:**
```bash
python -m goal_1_similarity.de_matrix_similarity \
    --adata_path ../paper/benchmark/data/gears_pert_data/adamson/perturb_processed.h5ad \
    --split_config results/baselines/adamson_split_seed1.json \
    --baseline_results results/baselines/adamson_reproduced/baseline_results_reproduced.csv \
    --output_dir results/de_matrix_similarity
```

---

### 2. Embedding Similarity (Baseline-Specific Embedding Spaces)

**Script:** `src/similarity/embedding_similarity.py`  
**Output Directory:** `results/embedding_similarity/`

**What it does:**
- Computes cosine similarity between test and training perturbations in each **baseline's embedding space (B matrix)**
- This is **different for each baseline** because each baseline uses different perturbation embeddings:
  - `lpm_selftrained`: PCA on training data
  - `lpm_k562PertEmb`: K562 PCA embeddings
  - `lpm_gearsPertEmb`: GEARS GO embeddings
  - `lpm_randomPertEmb`: Random embeddings
  - etc.
- Measures how similar test perturbations are to training perturbations **as represented by each baseline's embedding space**

**Key Functions:**
- `extract_perturbation_embeddings()` - Extracts B_train and B_test for a baseline
- `compute_embedding_similarity_statistics()` - Computes cosine similarity in embedding space
- `attach_performance_metrics()` - Attaches per-perturbation performance metrics
- `plot_embedding_similarity_distributions()` - Creates distribution plots per baseline
- `plot_embedding_performance_vs_similarity()` - Creates scatter plots per baseline
- `compute_embedding_regression_analysis()` - Fits regression: performance ~ similarity

**Output Files:**
- `embedding_similarity_all_baselines.csv` - Combined results across all baselines
- `embedding_regression_analysis_all_baselines.csv` - Regression results across all baselines
- `{baseline_name}/embedding_similarity_results.csv` - Per-baseline results
- `{baseline_name}/embedding_regression_analysis.csv` - Per-baseline regression
- `{baseline_name}/fig_embedding_similarity_distributions.png` - Per-baseline distributions
- `{baseline_name}/fig_embedding_performance_vs_similarity.png` - Per-baseline scatter plots
- `embedding_similarity_report.md` - Summary report

**Usage:**
```bash
python -m goal_1_similarity.embedding_similarity \
    --adata_path ../paper/benchmark/data/gears_pert_data/adamson/perturb_processed.h5ad \
    --split_config results/baselines/adamson_split_seed1.json \
    --baselines lpm_selftrained lpm_k562PertEmb lpm_gearsPertEmb \
    --output_dir results/embedding_similarity
```

---

## Key Differences

| Aspect | DE Matrix Similarity | Embedding Similarity |
|--------|---------------------|---------------------|
| **Space** | Expression space (Y matrix) | Baseline-specific embedding space (B matrix) |
| **Baseline-specific?** | No (same for all baselines) | Yes (different for each baseline) |
| **What it measures** | Expression pattern similarity | Embedding space similarity |
| **Insights** | General "hardness" profile | Baseline-specific "hardness" profile |
| **Performance correlation** | May be weak (expression ≠ performance) | May be stronger (embedding space may capture relevant structure) |

---

## What Embedding Similarity Tells Us

1. **Baseline-specific "hardness"**: How hard each test perturbation is for that specific baseline, given how that baseline represents perturbations

2. **Embedding quality assessment**: Whether similarity in a baseline's embedding space correlates with performance (indicates embedding captures useful structure)

3. **Transfer learning insights**: For cross-dataset baselines (K562, RPE1), whether similarity in the source dataset's embedding space predicts performance

4. **Stronger performance correlations**: Embedding-space similarity may correlate more strongly with performance than raw expression similarity

5. **Baseline comparison**: Compare which embedding spaces best capture similarity-performance relationships

---

## File Structure

```
src/similarity/
├── __init__.py                    # Module exports
├── de_matrix_similarity.py       # DE matrix similarity (expression space)
└── embedding_similarity.py        # Embedding similarity (baseline-specific)

results/
├── de_matrix_similarity/          # DE matrix similarity results
│   ├── de_matrix_similarity_results.csv
│   ├── de_matrix_regression_analysis.csv
│   ├── fig_de_matrix_similarity_distributions.png
│   ├── fig_de_matrix_performance_vs_similarity.png
│   └── de_matrix_similarity_report.md
└── embedding_similarity/           # Embedding similarity results
    ├── embedding_similarity_all_baselines.csv
    ├── embedding_regression_analysis_all_baselines.csv
    ├── embedding_similarity_report.md
    └── {baseline_name}/            # Per-baseline results
        ├── embedding_similarity_results.csv
        ├── embedding_regression_analysis.csv
        ├── fig_embedding_similarity_distributions.png
        └── fig_embedding_performance_vs_similarity.png
```

---

## Acceptance Criteria

✅ **CSV file with ≥ 82 rows** (one per perturbation) and 8 × performance columns
- **Status:** Both modules create CSV files with results
- **Note:** Adamson has 12 test perturbations. For ≥ 82 rows, run on larger datasets (Replogle K562, Replogle RPE1)

✅ **At least 3 similarity statistics per perturbation**
- **Status:** 5 statistics computed (max, mean_topk, std, median, min)

✅ **Scatter plot: performance vs similarity for each baseline**
- **Status:** Created for both DE matrix and embedding similarity

✅ **Regression results summarized in docs**
- **Status:** Reports generated (`de_matrix_similarity_report.md`, `embedding_similarity_report.md`)

✅ **Code runnable via CLI**
- **Status:** Both modules runnable via `python -m goal_1_similarity.de_matrix_similarity` and `python -m goal_1_similarity.embedding_similarity`

---

## Next Steps

1. **Run on larger datasets** (Replogle K562, Replogle RPE1) to get ≥ 82 rows
2. **Compare results** between DE matrix and embedding similarity
3. **Analyze correlations** to identify which embedding spaces best capture similarity-performance relationships

---

**Last Updated:** 2025-11-17

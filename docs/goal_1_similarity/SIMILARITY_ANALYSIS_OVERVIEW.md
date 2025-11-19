# Similarity Analysis Overview

**Last Updated:** 2025-11-17

---

## Two Types of Similarity Analysis

This module provides two distinct similarity analyses, each answering different questions:

### 1. DE Matrix Similarity (Expression Space)

**Script:** `src/similarity/de_matrix_similarity.py`  
**Output Directory:** `results/de_matrix_similarity/`

**What it does:**
- Computes cosine similarity between test and training perturbations in the **pseudobulk expression change space (Y matrix)**
- This is the **same for all baselines** since Y (expression changes) is fixed across all baselines
- Measures how similar test perturbations are to training perturbations in terms of **actual gene expression changes**

**Key Question:** How similar are test perturbations to training perturbations in expression space?

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

**Key Question:** How similar are test perturbations to training perturbations in each baseline's embedding space, and does this correlate with performance?

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

## When to Use Each

### Use DE Matrix Similarity when:
- You want to understand the general "hardness" of test perturbations
- You want a baseline-independent measure of similarity
- You're interested in expression-level relationships

### Use Embedding Similarity when:
- You want to understand baseline-specific "hardness"
- You want to assess embedding quality (does similarity in embedding space correlate with performance?)
- You're interested in transfer learning insights (for cross-dataset baselines)
- You want to compare which embedding spaces best capture similarity-performance relationships

---

## Combined Analysis

Both analyses can be run and compared to answer:
1. **General question**: Are test perturbations similar to training in expression space?
2. **Baseline-specific question**: Are test perturbations similar to training in each baseline's embedding space?
3. **Correlation question**: Does similarity in embedding space correlate better with performance than similarity in expression space?

---

**Last Updated:** 2025-11-17


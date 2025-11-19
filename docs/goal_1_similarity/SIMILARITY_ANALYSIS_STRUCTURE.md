# Similarity Analysis Structure

**Last Updated:** 2025-11-17

---

## Module Structure

```
src/similarity/
├── __init__.py                    # Module exports (both analyses)
├── de_matrix_similarity.py       # DE matrix similarity (expression space)
└── embedding_similarity.py        # Embedding similarity (baseline-specific)
```

---

## Output Structure

### DE Matrix Similarity Results

```
results/de_matrix_similarity/
├── de_matrix_similarity_results.csv          # Combined results (perturbation × baseline)
├── de_matrix_regression_analysis.csv         # Regression results per baseline
├── fig_de_matrix_similarity_distributions.png # Similarity distribution plots
├── fig_de_matrix_performance_vs_similarity.png # Performance vs similarity plots
└── de_matrix_similarity_report.md            # Summary report
```

### Embedding Similarity Results

```
results/embedding_similarity/
├── embedding_similarity_all_baselines.csv     # Combined results across all baselines
├── embedding_regression_analysis_all_baselines.csv # Regression results across all baselines
├── embedding_similarity_report.md             # Summary report
└── {baseline_name}/                           # Per-baseline directory
    ├── embedding_similarity_results.csv       # Per-baseline results
    ├── embedding_regression_analysis.csv      # Per-baseline regression
    ├── fig_embedding_similarity_distributions.png # Per-baseline distributions
    └── fig_embedding_performance_vs_similarity.png # Per-baseline scatter plots
```

---

## File Naming Convention

### DE Matrix Similarity
- All files prefixed with `de_matrix_` or `fig_de_matrix_`
- Output directory: `results/de_matrix_similarity/`

### Embedding Similarity
- All files prefixed with `embedding_` or `fig_embedding_`
- Output directory: `results/embedding_similarity/`
- Per-baseline files in subdirectories: `{baseline_name}/`

---

## Usage Examples

### DE Matrix Similarity

```bash
python -m goal_1_similarity.de_matrix_similarity \
    --adata_path ../paper/benchmark/data/gears_pert_data/adamson/perturb_processed.h5ad \
    --split_config results/baselines/adamson_split_seed1.json \
    --baseline_results results/baselines/adamson_reproduced/baseline_results_reproduced.csv \
    --output_dir results/de_matrix_similarity \
    --k 5 \
    --seed 1
```

### Embedding Similarity

```bash
python -m goal_1_similarity.embedding_similarity \
    --adata_path ../paper/benchmark/data/gears_pert_data/adamson/perturb_processed.h5ad \
    --split_config results/baselines/adamson_split_seed1.json \
    --baselines lpm_selftrained lpm_k562PertEmb lpm_gearsPertEmb \
    --output_dir results/embedding_similarity \
    --k 5 \
    --pca_dim 10 \
    --ridge_penalty 0.1 \
    --seed 1
```

---

## Key Differences Summary

| Aspect | DE Matrix Similarity | Embedding Similarity |
|--------|---------------------|---------------------|
| **Script** | `de_matrix_similarity.py` | `embedding_similarity.py` |
| **Output Dir** | `results/de_matrix_similarity/` | `results/embedding_similarity/` |
| **File Prefix** | `de_matrix_*` | `embedding_*` |
| **Space** | Expression (Y matrix) | Baseline embedding (B matrix) |
| **Baseline-specific?** | No | Yes |
| **Per-baseline dirs?** | No | Yes |

---

**Last Updated:** 2025-11-17


# LSFT Resampling Evaluation - Methods and Observations

**Generated:** 2025-11-19  
**Last Updated:** 2025-11-21  

**Building on:** Ahlmann-Eltze et al., _Nature Methods_ 2025  
"Deep learning-based predictions of gene perturbation effects do not yet outperform simple linear baselines"  
doi: https://doi.org/10.1038/s41592-025-02772-6

**Our Extension:** Local similarity-filtered training (LSFT) and leave-one-GO-class-out (LOGO) evaluation of the published baselines, with bootstrap confidence intervals and permutation tests for statistical rigor.

**Bootstrap samples:** 1000 per evaluation  
**Permutations:** 10,000 for statistical tests

---

## Methods

### Foundation: Building on Nature Methods 2025

This evaluation framework extends the work of:

> **Deep learning-based predictions of gene perturbation effects do not yet outperform simple linear baselines.**  
> Constantin Ahlmann-Eltze, Wolfgang Huber, Simon Anders.  
> _Nature Methods_ 2025; doi: https://doi.org/10.1038/s41592-025-02772-6  
> Repository: https://github.com/const-ae/linear_perturbation_prediction-Paper

**What We Keep from Ahlmann-Eltze et al. (2025):**
- **All 8 linear baselines + mean_response:** We use the exact baseline definitions from the paper
- **Ridge regression model:** **Y = A × K × B**, where Y is expression changes, A is gene embeddings, K is the learned interaction matrix (via ridge regression), and B is perturbation embeddings
- **Data Splits:** Same train/test/validation splits, generated via GEARS `prepare_split(split='simulation')` methodology
- **Core evaluation metrics:** Pearson correlation (r) and L2 distance

**Our Methodological Contributions:**

We extend the Nature Methods framework in three ways:

1. **LSFT Evaluation (Local Similarity-Filtered Training):** Instead of training once on the full dataset, we evaluate each test perturbation individually using only similar training data. For each test perturbation, we calculate its similarity to training perturbations, filter the training set to top-K% most similar, retrain the model, and make a prediction for that single target perturbation. This tests local interpolation performance in perturbation manifolds. The metrics reported are **summarized statistics** (mean, CI) across all test perturbations.

2. **LOGO Evaluation (Functional Class Holdout):** We introduce functional class holdout (Transcription genes) to test biological extrapolation, going beyond the random splits used in the original paper. This evaluates whether models can predict Transcription gene responses using only non-Transcription training data.

3. **Uncertainty Quantification:** We add bootstrap confidence intervals (1000 samples) and permutation tests (10,000 permutations) to rigorously assess statistical significance of baseline comparisons.

### Critical Design Choice: Embedding Evaluation Framework

**All baselines use the identical prediction architecture: Ridge regression via Y = A × K × B.**

**What varies:** The source of embeddings A (gene space) and B (perturbation space).

**What does NOT vary:** The prediction model (ridge regression), the training procedure, or the evaluation metrics.

This design isolates the value of different representations while controlling for model complexity. **We are not comparing scGPT's transformer architecture against PCA; we are comparing whether scGPT's learned gene embeddings provide better features for linear prediction than PCA-derived features.**

**Example:**
- **lpm_scgptGeneEmb:** Extracts gene embeddings from scGPT's pretrained transformer model → uses these as matrix A → solves ridge regression (same as all other baselines)
- **lpm_selftrained:** Extracts gene embeddings via PCA on training data → uses these as matrix A → solves same ridge regression

**Rationale:** This approach enables fair comparison of representation quality independent of architectural differences. It tests whether the biological knowledge encoded in foundation model embeddings facilitates linear interpolation in locally dense perturbation manifolds, rather than testing whether transformer architectures outperform linear models.

**Implication for Findings:** When we report that "scGPT performs no better than random embeddings (p=0.056)", this means:
- **What we're showing:** scGPT's learned gene representations, when used as features in a ridge regression framework, don't outperform random features for local perturbation prediction
- **What we're NOT showing:** Whether scGPT's full architecture (transformer + attention + non-linear prediction head) would outperform linear models

This finding suggests that the biological knowledge encoded in scGPT's embeddings doesn't provide value for linear interpolation in locally dense manifolds, which is a more nuanced and interesting result than a direct architecture comparison.

### Bootstrap Procedure
- **Method:** Percentile bootstrap with 1000 samples
- **CI construction:** 95% confidence intervals (α = 0.05)
- **Statistic:** Mean Pearson r and mean L2 distance across test perturbations
- **Formula:** CI_α = [Q(α/2), Q(1 - α/2)] where Q(p) is the p-th quantile
- **Application:** Applied to per-perturbation performance metrics to estimate uncertainty in summary statistics

### Permutation Test Procedure
- **Method:** Paired sign-flip permutation test
- **Permutations:** 10,000 per comparison
- **Null hypothesis:** H₀: Mean(Δ) = 0
- **Test statistic:** Absolute mean delta
- **P-value:** Proportion of permuted datasets with |mean(Δ_perm)| ≥ |mean(Δ_obs)|
- **Application:** Used to test statistical significance of baseline comparisons (e.g., scGPT vs Random)

### Evaluation Metrics
- **Primary:** Pearson correlation (r) between predicted and observed expression changes
- **Secondary:** L2 distance (Euclidean norm) between predicted and observed expression vectors
- **Hardness metric:** Top-K cosine similarity to training perturbations (used in LSFT filtering)
- **Reporting:** All metrics are computed per-perturbation, then summarized as mean ± 95% CI across test set

### LSFT Evaluation (Local Similarity-Filtered Training)

**Procedure:**
1. **For each test perturbation individually:**
   - Calculate cosine similarity between the test perturbation's embedding and all training perturbation embeddings
   - Filter training set to top-K% most similar perturbations (K = 1%, 5%, or 10%)
   - Retrain the linear model (Y = A × K × B) on this filtered training set
   - Make prediction for the single test perturbation
   - Compute Pearson r and L2 for this single prediction

2. **Aggregation:**
   - Collect per-perturbation metrics across all test perturbations
   - Compute summary statistics: mean Pearson r, mean L2
   - Apply bootstrap to estimate 95% confidence intervals

**Key Difference from Standard Evaluation:**
- Standard evaluation: Train once on full training set, evaluate all test perturbations simultaneously
- LSFT: Train separately for each test perturbation using only similar training data, evaluate one perturbation at a time

**Baselines (from Ahlmann-Eltze et al. 2025):**

All 8 linear baselines are reproduced exactly as defined in the original Nature Methods paper. We evaluate these published baselines using our novel LSFT and LOGO frameworks.

1. **lpm_selftrained:** Gene embeddings (A) and perturbation embeddings (B) both from PCA on training data. [This was the best-performing baseline in the original paper]

2. **lpm_scgptGeneEmb:** Gene embeddings (A) extracted from scGPT's pretrained transformer model (512-dimensional embeddings), perturbation embeddings (B) from PCA on training data. **Note:** We extract embeddings from scGPT but use them as features in the same ridge regression framework as all other baselines. This tests whether foundation model gene representations provide better features for linear prediction, not whether the transformer architecture itself outperforms linear models.

3. **lpm_scFoundationGeneEmb:** Gene embeddings (A) extracted from scFoundation's pretrained transformer model, perturbation embeddings (B) from PCA on training data. **Note:** Same design as scGPT—embeddings are extracted and used as features in ridge regression, not the full transformer architecture.

4. **lpm_randomGeneEmb:** Gene embeddings (A) are random vectors, perturbation embeddings (B) from PCA on training data. Tests whether gene structure matters or if perturbation structure alone is sufficient.

5. **lpm_k562PertEmb:** Gene embeddings (A) from PCA on training data, perturbation embeddings (B) from PCA on Replogle K562 dataset (cross-dataset transfer). Tests whether perturbation embeddings transfer across datasets.

6. **lpm_rpe1PertEmb:** Gene embeddings (A) from PCA on training data, perturbation embeddings (B) from PCA on Replogle RPE1 dataset (cross-dataset transfer). Tests cross-dataset transfer for a different cell type.

7. **lpm_gearsPertEmb:** Gene embeddings (A) from PCA on training data, perturbation embeddings (B) from GEARS GO graph spectral embeddings. Tests whether graph-based perturbation representations help.

8. **lpm_randomPertEmb:** Gene embeddings (A) from PCA on training data, perturbation embeddings (B) are random vectors. Tests whether perturbation structure matters or if gene structure alone is sufficient.

**Datasets:** Adamson (n=12 test), K562 (n=163 test), RPE1 (n=231 test)

### LOGO Evaluation (Functional Class Holdout)

**Procedure:**
1. **Functional Class Annotation:** Genes are annotated into functional classes (e.g., Transcription, Translation, Transport) using Gene Ontology (GO) terms
2. **Holdout Strategy:** One functional class (Transcription) is designated as the test set; all other classes form the training set
3. **Evaluation:** Train on non-Transcription genes, test on Transcription genes. This tests biological extrapolation: can the model predict Transcription gene responses using only non-Transcription training data?
4. **Metrics:** Computed per-perturbation, then summarized as mean ± 95% CI

**Baselines Tested (9 models):**
- All 8 linear baselines listed above, plus:
- **mean_response:** Simple baseline predicting mean expression change across training data (no model training)

**Datasets:** Adamson (n=5 test), K562 (n=397 test), RPE1 (n=313 test)

---

## Observations

### LSFT Performance (top_pct=0.05)

**Table 1: LSFT Performance (top_pct=0.05)**  
*Baselines from Ahlmann-Eltze et al. (2025), evaluated using our similarity-filtered training procedure*

#### Adamson Dataset (n=12 test perturbations)

| Rank | Baseline | Pearson r (95% CI) | L2 (95% CI) |
|------|----------|-------------------|-------------|
| 1 | selftrained | 0.941 [0.900, 0.966] | 2.09 [1.66, 2.70] |
| 2 | scgptGeneEmb | 0.935 [0.892, 0.960] | 2.21 [1.78, 2.83] |
| 3 | scFoundationGeneEmb | 0.933 [0.887, 0.959] | 2.26 [1.82, 2.90] |
| 4 | randomGeneEmb | 0.932 [0.885, 0.959] | 2.30 [1.86, 2.95] |
| 5 | rpe1PertEmb | 0.932 [0.886, 0.960] | 2.26 [1.78, 2.92] |
| 6 | k562PertEmb | 0.932 [0.886, 0.961] | 2.23 [1.77, 2.88] |
| 7 | gearsPertEmb | 0.772 [0.562, 0.908] | 3.83 [2.70, 5.38] |
| 8 | randomPertEmb | 0.597 [0.353, 0.775] | 5.69 [4.28, 7.33] |

#### K562 Dataset (n=163 test perturbations)

| Rank | Baseline | Pearson r (95% CI) | L2 (95% CI) |
|------|----------|-------------------|-------------|
| 1 | selftrained | 0.705 [0.677, 0.734] | 3.54 [3.35, 3.75] |
| 2 | scgptGeneEmb | 0.666 [0.637, 0.696] | 3.84 [3.62, 4.07] |
| 3 | k562PertEmb | 0.657 [0.625, 0.689] | 3.85 [3.63, 4.07] |
| 4 | scFoundationGeneEmb | 0.656 [0.627, 0.687] | 3.96 [3.73, 4.19] |
| 5 | randomGeneEmb | 0.649 [0.620, 0.680] | 4.04 [3.80, 4.28] |
| 6 | rpe1PertEmb | 0.633 [0.599, 0.667] | 3.97 [3.74, 4.21] |
| 7 | gearsPertEmb | 0.461 [0.423, 0.497] | 5.14 [4.85, 5.43] |
| 8 | randomPertEmb | 0.307 [0.278, 0.339] | 5.82 [5.54, 6.09] |

#### RPE1 Dataset (n=231 test perturbations)

| Rank | Baseline | Pearson r (95% CI) | L2 (95% CI) |
|------|----------|-------------------|-------------|
| 1 | selftrained | 0.792 [0.773, 0.812] | 4.56 [4.40, 4.73] |
| 2 | scgptGeneEmb | 0.759 [0.738, 0.782] | 5.00 [4.81, 5.21] |
| 3 | rpe1PertEmb | 0.759 [0.736, 0.783] | 4.92 [4.73, 5.10] |
| 4 | k562PertEmb | 0.746 [0.722, 0.770] | 5.08 [4.89, 5.28] |
| 5 | scFoundationGeneEmb | 0.742 [0.718, 0.767] | 5.22 [5.01, 5.44] |
| 6 | randomGeneEmb | 0.738 [0.714, 0.764] | 5.34 [5.12, 5.57] |
| 7 | gearsPertEmb | 0.636 [0.599, 0.670] | 7.14 [6.88, 7.45] |
| 8 | randomPertEmb | 0.541 [0.507, 0.575] | 8.19 [7.90, 8.52] |

### LOGO Performance (Transcription class held out)

**Table 2: LOGO Performance (Transcription class held out)**  
*Baselines from Ahlmann-Eltze et al. (2025), evaluated using our functional class holdout procedure*

#### Adamson Dataset (n=5 test perturbations)

| Rank | Baseline | Pearson r (95% CI) | L2 (95% CI) |
|------|----------|-------------------|-------------|
| 1 | selftrained | 0.882 [0.842, 0.924] | 4.36 [2.88, 5.45] |
| 2 | rpe1PertEmb | 0.821 [0.772, 0.884] | 4.32 [2.65, 5.62] |
| 3 | k562PertEmb | 0.805 [0.759, 0.874] | 5.35 [3.22, 7.08] |
| 4 | scgptGeneEmb | 0.454 [0.204, 0.734] | 6.11 [3.90, 7.94] |
| 5 | gearsPertEmb | 0.417 [-0.003, 0.726] | 7.06 [4.28, 10.16] |
| 6 | scFoundationGeneEmb | 0.257 [-0.123, 0.660] | 7.26 [4.34, 9.80] |
| 7 | randomGeneEmb | 0.036 [-0.483, 0.573] | 8.07 [4.74, 11.02] |
| 8 | mean_response | 0.032 [-0.491, 0.571] | 8.08 [4.75, 11.04] |
| 9 | randomPertEmb | -0.004 [-0.542, 0.544] | 8.39 [4.96, 11.52] |

#### K562 Dataset (n=397 test perturbations)

| Rank | Baseline | Pearson r (95% CI) | L2 (95% CI) |
|------|----------|-------------------|-------------|
| 1 | selftrained | 0.632 [0.616, 0.650] | 4.57 [4.42, 4.72] |
| 2 | k562PertEmb | 0.610 [0.592, 0.629] | 4.72 [4.56, 4.89] |
| 3 | rpe1PertEmb | 0.600 [0.582, 0.618] | 3.68 [3.56, 3.81] |
| 4 | scgptGeneEmb | 0.486 [0.469, 0.502] | 5.26 [5.11, 5.44] |
| 5 | scFoundationGeneEmb | 0.384 [0.368, 0.399] | 5.83 [5.65, 6.02] |
| 6 | gearsPertEmb | 0.354 [0.334, 0.374] | 6.09 [5.89, 6.31] |
| 7 | randomGeneEmb | 0.341 [0.324, 0.356] | 6.07 [5.88, 6.27] |
| 8 | mean_response | 0.341 [0.324, 0.356] | 6.07 [5.88, 6.28] |
| 9 | randomPertEmb | 0.335 [0.318, 0.351] | 6.09 [5.91, 6.30] |

#### RPE1 Dataset (n=313 test perturbations)

| Rank | Baseline | Pearson r (95% CI) | L2 (95% CI) |
|------|----------|-------------------|-------------|
| 1 | selftrained | 0.804 [0.789, 0.821] | 5.62 [5.43, 5.81] |
| 2 | rpe1PertEmb | 0.795 [0.779, 0.812] | 5.78 [5.58, 5.98] |
| 3 | k562PertEmb | 0.787 [0.769, 0.805] | 4.84 [4.67, 5.00] |
| 4 | scgptGeneEmb | 0.725 [0.707, 0.745] | 6.68 [6.50, 6.87] |
| 5 | mean_response | 0.694 [0.674, 0.717] | 7.98 [7.76, 8.21] |
| 6 | randomGeneEmb | 0.694 [0.674, 0.716] | 7.98 [7.75, 8.21] |
| 7 | randomPertEmb | 0.694 [0.673, 0.717] | 8.00 [7.78, 8.24] |
| 8 | gearsPertEmb | 0.691 [0.670, 0.713] | 7.84 [7.62, 8.08] |
| 9 | scFoundationGeneEmb | 0.691 [0.671, 0.712] | 7.59 [7.38, 7.81] |

### Statistical Comparisons

#### scGPT vs Random Gene Embeddings

**LSFT (Adamson, top_pct=0.05):**
- scGPT: r = 0.935 [0.892, 0.960]
- Random: r = 0.932 [0.885, 0.959]
- Delta: r = +0.002
- Permutation p-value: p = 0.0557

**LOGO (Adamson):**
- scGPT embeddings: r = 0.454 [0.204, 0.734]
- Random embeddings: r = 0.036 [-0.483, 0.573]
- Delta: r = +0.418
- **Note:** Both use the same ridge regression framework; difference is in embedding source.

**LOGO (K562):**
- scGPT embeddings: r = 0.486 [0.469, 0.502]
- Random embeddings: r = 0.341 [0.324, 0.356]
- Delta: r = +0.145
- CIs do not overlap
- **Note:** Both use the same ridge regression framework; difference is in embedding source.

**LOGO (RPE1):**
- scGPT embeddings: r = 0.725 [0.707, 0.745]
- Random embeddings: r = 0.694 [0.674, 0.716]
- Delta: r = +0.031
- **Note:** Both use the same ridge regression framework; difference is in embedding source.

#### Self-trained vs scGPT Embeddings

- Adamson: Self-trained (PCA) r=0.941 vs scGPT embeddings r=0.935 (Δr=+0.006)
- K562: Self-trained (PCA) r=0.705 vs scGPT embeddings r=0.666 (Δr=+0.039)
- RPE1: Self-trained (PCA) r=0.792 vs scGPT embeddings r=0.759 (Δr=+0.033)
- **Note:** Both use the same ridge regression framework; comparison is between PCA-derived embeddings vs. scGPT-extracted embeddings as features.

### Statistical Precision

#### Bootstrap Confidence Intervals
- Average CI width for Pearson r: ~0.06-0.07
- Average CI width for L2: ~1.0-1.2
- Coverage: 95% confidence intervals based on 1000 bootstrap samples

#### Hardness-Performance Relationships
- top_pct=0.01: r ≈ 0.97
- top_pct=0.05: r ≈ 0.93
- top_pct=0.10: r ≈ 0.91

### Cross-Dataset Performance Patterns

**Performance Ranking (Best to Worst):**
1. Adamson: r=0.597-0.941, mean of top 6 ≈ 0.933
2. RPE1: r=0.541-0.792, mean of top 6 ≈ 0.746
3. K562: r=0.307-0.705, mean of top 6 ≈ 0.661


# Methods Appendix

**Linear Perturbation Prediction Evaluation Framework**  
**Resampling-Enabled Statistical Analysis**

---

## Page 1: Resampling Methods

### Bootstrap Procedure

**Purpose:** Estimate confidence intervals for mean performance metrics (Pearson correlation r and L2 distance) across test perturbations.

**Method:**
1. **Resampling:** For each evaluation (LSFT or LOGO), we generate B = 1,000 bootstrap samples by sampling test perturbations with replacement.
2. **Statistic computation:** For each bootstrap sample, compute the mean Pearson r and mean L2 distance.
3. **CI construction:** Use percentile bootstrap method to construct 95% confidence intervals:
   - Lower bound: 2.5th percentile of bootstrap distribution
   - Upper bound: 97.5th percentile of bootstrap distribution

**Formula:**
```
CI_α = [Q(α/2), Q(1 - α/2)]
```
Where Q(p) is the p-th quantile of the bootstrap distribution, and α = 0.05 for 95% confidence.

**Rationale:** Bootstrap resampling provides nonparametric confidence intervals that do not assume a specific distribution of performance metrics. This is particularly important when sample sizes are small (e.g., n=5 for Adamson LOGO) or when the distribution is non-normal.

**Implementation:** `stats/bootstrapping.py::bootstrap_mean_ci()`

---

### Permutation Test Procedure

**Purpose:** Test statistical significance of differences between baseline pairs using a nonparametric permutation test.

**Method:**
1. **Delta computation:** For each test perturbation, compute the performance delta: Δ[p] = Baseline_A[p] - Baseline_B[p]
2. **Null hypothesis:** H₀: Mean(Δ) = 0 (no difference between baselines)
3. **Permutation:** Generate P = 10,000 permuted datasets by randomly sign-flipping each delta (multiplying by ±1 with equal probability)
4. **Test statistic:** Compute the absolute mean delta for each permutation
5. **P-value:** Proportion of permuted datasets with absolute mean delta ≥ observed absolute mean delta

**Formula:**
```
p-value = (1/P) × Σ[I(|mean(Δ_perm)| ≥ |mean(Δ_obs)|)]
```
Where I(·) is the indicator function, and Δ_perm are permuted deltas.

**Alternative hypotheses:**
- `two-sided`: Test if mean(Δ) ≠ 0
- `greater`: Test if mean(Δ) > 0 (Baseline_A better)
- `less`: Test if mean(Δ) < 0 (Baseline_B better)

**Rationale:** Permutation tests are exact (when all permutations are enumerated) and make no distributional assumptions. They are robust to small sample sizes and non-normal distributions.

**Implementation:** `stats/permutation.py::paired_permutation_test()`

---

### CI Computation

**Bootstrap CIs for Summary Statistics:**
- **Mean Pearson r:** Bootstrap mean across test perturbations
- **Mean L2 distance:** Bootstrap mean across test perturbations
- **Mean delta (baseline comparisons):** Bootstrap mean of performance differences

**Bootstrap CIs for Regression Coefficients:**
- **Slope (hardness-performance regression):** Bootstrap slope estimates from resampled perturbations
- **Correlation coefficient:** Bootstrap correlation between hardness and performance

**Coverage:** All confidence intervals are 95% (α = 0.05).

**Bootstrap samples:** B = 1,000 for all analyses.

---

### Hardness Metric Definition

**Hardness** quantifies how similar a test perturbation is to the training data.

**Definition:**
For each test perturbation p and similarity threshold top_pct (e.g., 0.05 for top 5%):

1. **Compute similarities:** Cosine similarity between test perturbation embedding B_test[p] and all training perturbation embeddings B_train
2. **Select top-K:** Select the top K = ceil(n_train × top_pct) most similar training perturbations
3. **Hardness:** Mean cosine similarity to the selected top-K training perturbations

**Formula:**
```
hardness[p, top_pct] = (1/K) × Σ[cosine_similarity(B_test[p], B_train[k])]
```
Where k indexes the top-K most similar training perturbations.

**Interpretation:**
- **High hardness (≈1.0):** Test perturbation is very similar to training data → easier to predict
- **Low hardness (≈0.0):** Test perturbation is dissimilar to training data → harder to predict

**Rationale:** Hardness captures the "local density" of training data around each test perturbation in embedding space. This metric is central to understanding why similarity-based filtering (LSFT) works.

---

### Rationale for Metrics

**Primary Metric: Pearson Correlation (r)**

- **Why:** Captures directional agreement between predicted and actual expression changes, which is biologically meaningful (upregulation vs downregulation)
- **Range:** [-1, 1], where 1 = perfect positive correlation, 0 = no correlation, -1 = perfect negative correlation
- **Interpretation:** r > 0.7 is typically considered strong correlation in biological contexts
- **Advantage:** Scale-invariant, robust to outliers

**Secondary Metric: L2 Distance**

- **Why:** Captures magnitude of prediction error, important for understanding absolute prediction accuracy
- **Range:** [0, ∞), where 0 = perfect prediction, larger values = larger errors
- **Interpretation:** Lower is better; typically ranges from 2-8 for our evaluations
- **Advantage:** Directly interpretable in units of expression change

**Why both metrics:** Pearson r captures direction (biological relevance), while L2 captures magnitude (practical accuracy). Together, they provide a complete picture of model performance.

---

## Page 2: Evaluation Splits

### LSFT Description

**LSFT (Local Similarity-Filtered Training)** evaluates model performance when training data is filtered by similarity to each test perturbation.

**Procedure:**
1. **Split:** Use paper's recommended train/test/validation split (from split config JSON)
2. **For each test perturbation p:**
   - Compute cosine similarity between p and all training perturbations (in B embedding space)
   - Filter training perturbations to top K% most similar (K = 1%, 5%, 10%)
   - Retrain the full LPM model (Y = A × K × B) using only filtered training data
   - Evaluate on test perturbation p
   - Compare to baseline performance (trained on all data)

**Key Features:**
- **Per-perturbation filtering:** Each test perturbation gets its own filtered training set
- **Full model retraining:** A, K, and B are all recomputed on filtered data (for self-trained embeddings)
- **Local PCA space:** Test perturbations are transformed into the local PCA space of filtered training data
- **Multiple thresholds:** Evaluated at top 1%, 5%, and 10% similarity

**Output:** Per-perturbation performance metrics (Pearson r, L2) for each top_pct threshold, with bootstrap CIs for summary statistics.

**Rationale:** LSFT tests whether similarity-based filtering improves predictions. If perturbations with high similarity to training data perform better, this validates the embedding space and suggests that local training is beneficial.

---

### LOGO Description

**LOGO (Leave-One-Group-Out)** evaluates model performance on functional class holdout, testing true biological extrapolation.

**Procedure:**
1. **Functional class annotation:** Load GO/Reactome annotations mapping genes to functional classes
2. **Class selection:** Select a target functional class (e.g., "Transcription") to hold out
3. **Split creation:**
   - **Test set:** Perturbations targeting genes in the held-out class
   - **Training set:** All other perturbations (targeting genes in other classes)
4. **Model training:** Train LPM model on training set using all baselines
5. **Evaluation:** Evaluate on test set (held-out functional class)

**Key Features:**
- **Functional extrapolation:** Tests whether models can predict expression changes for perturbations in unseen functional classes
- **Biological relevance:** More challenging than random splits because functional classes represent coherent biological processes
- **All baselines:** Evaluates all 9 baselines to compare embedding strategies under extrapolation

**Output:** Per-perturbation performance metrics (Pearson r, L2) for each baseline, with bootstrap CIs for summary statistics.

**Rationale:** LOGO tests true biological generalization. If models perform well on held-out functional classes, this demonstrates that embeddings capture meaningful biological structure beyond dataset-specific patterns.

---

### Why These Splits Matter

**LSFT Split (Paper's Recommended Split):**
- **Purpose:** Standard evaluation split for fair comparison with baseline methods
- **Source:** Paper's canonical train/test/validation splits (from Nature Methods 2025)
- **Why it matters:** Ensures reproducibility and allows direct comparison with published results
- **Split logic:** Uses GEARS framework's `prepare_split(split='simulation')` for consistency

**LOGO Split (Functional Class-Based):**
- **Purpose:** Tests biological extrapolation beyond dataset-specific patterns
- **Source:** GO/Reactome functional class annotations
- **Why it matters:** More realistic evaluation scenario—in practice, we want to predict effects of perturbations targeting genes in functional classes not seen during training
- **Split logic:** Dynamic split based on functional class membership of perturbed genes

**Complementary Evaluation:**
- **LSFT:** Tests similarity-based filtering (can we improve predictions by using similar training examples?)
- **LOGO:** Tests functional extrapolation (can we predict effects for unseen biological processes?)

Together, LSFT and LOGO provide a comprehensive evaluation of model performance under different scenarios:
- **LSFT:** Optimistic scenario (similar training examples available)
- **LOGO:** Pessimistic scenario (no similar training examples, must extrapolate)

**Statistical Rigor:**
Both evaluation methods use:
- Bootstrap confidence intervals (1,000 samples) for uncertainty quantification
- Permutation tests (10,000 permutations) for statistical significance
- Standardized output formats for reproducibility

---

**End of Methods Appendix**


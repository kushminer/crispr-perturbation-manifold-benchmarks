# LSFT Analysis: Methods and Observations

**Building on:** Ahlmann-Eltze et al., _Nature Methods_ 2025  
"Deep learning-based predictions of gene perturbation effects do not yet outperform simple linear baselines"  
doi: https://doi.org/10.1038/s41592-025-02772-6

**Our Extension:** Local similarity-filtered training (LSFT) evaluation of the published baselines

---

## Methods

### Foundation: Building on Nature Methods 2025

This analysis extends the work of:

> **Deep learning-based predictions of gene perturbation effects do not yet outperform simple linear baselines.**  
> Constantin Ahlmann-Eltze, Wolfgang Huber, Simon Anders.  
> _Nature Methods_ 2025; doi: https://doi.org/10.1038/s41592-025-02772-6  
> Repository: https://github.com/const-ae/linear_perturbation_prediction-Paper

**What We Keep from Ahlmann-Eltze et al. (2025):**
- **All 8 linear baselines + mean_response:** We use the exact baseline definitions from the paper
- **Ridge regression model:** **Y = A × K × B** (same architecture)
- **Data Splits:** Same train/test/validation splits, generated via GEARS `prepare_split(split='simulation')` methodology
- **Core evaluation metrics:** Pearson correlation (r) and L2 distance

**Our Methodological Contribution:**

**LSFT Evaluation (Local Similarity-Filtered Training):** We introduce a novel evaluation framework that tests local interpolation performance. Instead of training once on the full dataset (as in the original paper), we evaluate each test perturbation individually using only similar training data. This tests whether models can leverage local manifold structure for improved predictions.

**LSFT Procedure:** Unlike the standard evaluation in the original paper (where all test perturbations are evaluated simultaneously), LSFT evaluates **one perturbation at a time**. For each test perturbation, we:
1. Calculate its similarity to all training perturbations
2. Filter the training set to the top-K% most similar perturbations
3. Retrain the model on this filtered training set
4. Make a prediction for that single target perturbation
5. Compute metrics (Pearson r, L2) for this single prediction

The metrics reported in this analysis are **summarized statistics** (mean, improvement, fraction improved) aggregated across all test perturbations.

### Critical Design Choice: Embedding Evaluation Framework

**All baselines use the identical prediction architecture: Ridge regression via Y = A × K × B.**

**What varies:** The source of embeddings A (gene space) and B (perturbation space).

**What does NOT vary:** The prediction model (ridge regression), the training procedure, or the evaluation metrics.

This design isolates the value of different representations while controlling for model complexity. **We are not comparing scGPT's transformer architecture against PCA; we are comparing whether scGPT's learned gene embeddings provide better features for linear prediction than PCA-derived features.**

**Example:**
- **lpm_scgptGeneEmb:** Extracts gene embeddings from scGPT's pretrained transformer model → uses these as matrix A → solves ridge regression (same as all other baselines)
- **lpm_selftrained:** Extracts gene embeddings via PCA on training data → uses these as matrix A → solves same ridge regression

**Rationale:** This approach enables fair comparison of representation quality independent of architectural differences. It tests whether the biological knowledge encoded in foundation model embeddings facilitates linear interpolation in locally dense perturbation manifolds, rather than testing whether transformer architectures outperform linear models.

**Implication for Findings:** When we report that foundation model embeddings perform similarly to random embeddings, this means:
- **What we're showing:** Foundation model gene representations, when used as features in a ridge regression framework, don't consistently outperform random features for local perturbation prediction
- **What we're NOT showing:** Whether foundation models' full architectures (transformer + attention + non-linear prediction head) would outperform linear models

This finding suggests that the biological knowledge encoded in foundation model embeddings doesn't provide value for linear interpolation in locally dense manifolds, which is a more nuanced and interesting result than a direct architecture comparison.

### LSFT Procedure (Local Similarity-Filtered Training)

**Step-by-Step Algorithm:**

For each test perturbation `t`:
1. **Similarity Calculation:** Compute cosine similarity between test perturbation embedding and all training perturbation embeddings in the embedding space defined by the baseline model
   - **What is compared:** Feature matrices (B embeddings), NOT target matrices (Y expression changes)
   - **Embedding space:** Defined by the baseline model (e.g., PCA space for `lpm_selftrained`, scGPT embedding space for `lpm_scgptGeneEmb`)
   - **No lookahead bias:** 
     - For training-data-based embeddings: PCA is fit on training data only, then test data is projected into that space
     - The embedding space is defined entirely by training data; test perturbations are simply projected into this pre-defined space
     - Test expression data (Y_test) is used only to identify which perturbation is being evaluated, not to define the embedding space
2. **Filtering:** Select top-K% most similar training perturbations (K = 1%, 5%, or 10%)
3. **Local Training:** Retrain the linear model (Y = A × K × B) using only the filtered training set
   - Note: For cross-dataset baselines (k562PertEmb, rpe1PertEmb), the perturbation embeddings come from the external dataset, but similarity is computed in the local embedding space
4. **Prediction:** Predict expression changes for the single test perturbation `t`
5. **Evaluation:** Compute Pearson correlation (r) and L2 distance between predicted and observed expression changes for perturbation `t`

**Aggregation:**
- Collect per-perturbation metrics (Pearson r, L2) across all test perturbations
- Compute summary statistics: mean Pearson r, mean L2, mean improvement
- Compute fraction of perturbations improved (local performance > baseline performance)

**Top Percentages Tested:** 1%, 5%, 10% (of training perturbations)

**Datasets:** 
- Adamson: 12 test perturbations
- K562: 163 test perturbations  
- RPE1: 231 test perturbations

### Baseline Descriptions

**Baselines (from Ahlmann-Eltze et al. 2025):**

All 8 linear baselines are reproduced exactly as defined in the original Nature Methods paper. We evaluate these published baselines using our novel LSFT framework.

1. **lpm_selftrained:** 
   - Gene embeddings (A): PCA on training data genes
   - Perturbation embeddings (B): PCA on training data perturbations
   - [This was the best-performing baseline in the original paper]

2. **lpm_scgptGeneEmb:**
   - Gene embeddings (A): Extracted from scGPT's pretrained transformer model (512-dimensional embeddings)
   - Perturbation embeddings (B): PCA on training data perturbations
   - **Note:** We extract embeddings from scGPT but use them as features in the same ridge regression framework as all other baselines. This tests whether foundation model gene representations provide better features for linear prediction, not whether the transformer architecture itself outperforms linear models.

3. **lpm_scFoundationGeneEmb:**
   - Gene embeddings (A): Extracted from scFoundation's pretrained transformer model
   - Perturbation embeddings (B): PCA on training data perturbations
   - **Note:** Same design as scGPT—embeddings are extracted and used as features in ridge regression, not the full transformer architecture. Tests an alternative foundation model approach.

4. **lpm_randomGeneEmb:**
   - Gene embeddings (A): Random vectors (no structure)
   - Perturbation embeddings (B): PCA on training data perturbations
   - Tests whether gene structure matters or if perturbation structure alone is sufficient

5. **lpm_k562PertEmb:**
   - Gene embeddings (A): PCA on training data genes
   - Perturbation embeddings (B): PCA on Replogle K562 dataset (cross-dataset transfer)
   - Tests whether perturbation embeddings transfer across datasets (same cell type)

6. **lpm_rpe1PertEmb:**
   - Gene embeddings (A): PCA on training data genes
   - Perturbation embeddings (B): PCA on Replogle RPE1 dataset (cross-dataset transfer)
   - Tests cross-dataset transfer for a different cell type

7. **lpm_gearsPertEmb:**
   - Gene embeddings (A): PCA on training data genes
   - Perturbation embeddings (B): GEARS GO graph spectral embeddings (graph-based)
   - Tests whether graph-based perturbation representations help

8. **lpm_randomPertEmb:**
   - Gene embeddings (A): PCA on training data genes
   - Perturbation embeddings (B): Random vectors (no structure)
   - Tests whether perturbation structure matters or if gene structure alone is sufficient

### Evaluation Metrics

**Primary Metric:**
- **Pearson correlation (r):** Correlation between predicted and observed expression changes across all genes for a perturbation

**Secondary Metric:**
- **L2 distance:** Euclidean norm of the difference between predicted and observed expression change vectors

**Improvement Metric:**
- **Improvement (r):** Local LSFT performance - Baseline performance (positive = improvement)
- **Improvement (L2):** Baseline L2 - Local LSFT L2 (positive = improvement, since lower L2 is better)

**Summary Statistics:**
- **Mean improvement:** Average improvement across all test perturbations
- **Fraction improved:** Proportion of test perturbations where local performance > baseline performance

---

## Observations

### Adamson Dataset (n=12 test perturbations)

**Table 1: LSFT Performance by Baseline and Top Percentage (Adamson)**  
*Baselines from Ahlmann-Eltze et al. (2025), evaluated using our similarity-filtered training procedure*

#### Performance by Baseline and Top Percentage

| Baseline | Top % | Local Pearson r | Baseline Pearson r | Improvement (r) | Local L2 | Baseline L2 | Improvement (L2) | Train Size | Mean Similarity |
|----------|-------|----------------|-------------------|-----------------|----------|-------------|------------------|------------|-----------------|
| lpm_gearsPertEmb | 1% | 0.7308 | 0.7485 | -0.0177 | 4.5969 | 4.3071 | -0.2898 | 1.0 | 0.9154 |
| lpm_gearsPertEmb | 5% | 0.7725 | 0.7485 | 0.0240 | 3.8256 | 4.3071 | 0.4816 | 4.0 | 0.8578 |
| lpm_gearsPertEmb | 10% | 0.7845 | 0.7485 | 0.0360 | 3.7153 | 4.3071 | 0.5919 | 7.0 | 0.8073 |
| lpm_k562PertEmb | 1% | 0.9191 | 0.9334 | -0.0142 | 2.4267 | 2.4399 | 0.0132 | 1.0 | 0.9993 |
| lpm_k562PertEmb | 5% | 0.9317 | 0.9334 | -0.0017 | 2.2334 | 2.4399 | 0.2065 | 4.0 | 0.9987 |
| lpm_k562PertEmb | 10% | 0.9330 | 0.9334 | -0.0004 | 2.3225 | 2.4399 | 0.1173 | 7.0 | 0.9976 |
| lpm_randomGeneEmb | 1% | 0.9247 | 0.7214 | 0.2033 | 2.3613 | 4.3444 | 1.9831 | 1.0 | 0.9539 |
| lpm_randomGeneEmb | 5% | 0.9323 | 0.7214 | 0.2110 | 2.3019 | 4.3444 | 2.0425 | 4.0 | 0.8996 |
| lpm_randomGeneEmb | 10% | 0.9117 | 0.7214 | 0.1904 | 2.5305 | 4.3444 | 1.8139 | 7.0 | 0.8508 |
| lpm_randomPertEmb | 1% | 0.5892 | 0.7075 | -0.1183 | 5.8694 | 4.5362 | -1.3332 | 1.0 | 0.6558 |
| lpm_randomPertEmb | 5% | 0.5974 | 0.7075 | -0.1100 | 5.6943 | 4.5362 | -1.1580 | 4.0 | 0.5523 |
| lpm_randomPertEmb | 10% | 0.5161 | 0.7075 | -0.1914 | 6.5551 | 4.5362 | -2.0188 | 7.0 | 0.4847 |
| lpm_rpe1PertEmb | 1% | 0.9187 | 0.9303 | -0.0116 | 2.4375 | 2.5187 | 0.0812 | 1.0 | 0.9997 |
| lpm_rpe1PertEmb | 5% | 0.9320 | 0.9303 | 0.0017 | 2.2629 | 2.5187 | 0.2557 | 4.0 | 0.9992 |
| lpm_rpe1PertEmb | 10% | 0.9308 | 0.9303 | 0.0005 | 2.3570 | 2.5187 | 0.1617 | 7.0 | 0.9984 |
| lpm_scFoundationGeneEmb | 1% | 0.9247 | 0.7767 | 0.1480 | 2.3613 | 3.9785 | 1.6172 | 1.0 | 0.9539 |
| lpm_scFoundationGeneEmb | 5% | 0.9330 | 0.7767 | 0.1562 | 2.2588 | 3.9785 | 1.7197 | 4.0 | 0.8996 |
| lpm_scFoundationGeneEmb | 10% | 0.9157 | 0.7767 | 0.1389 | 2.4505 | 3.9785 | 1.5280 | 7.0 | 0.8508 |
| lpm_scgptGeneEmb | 1% | 0.9247 | 0.8107 | 0.1140 | 2.3613 | 3.7328 | 1.3715 | 1.0 | 0.9539 |
| lpm_scgptGeneEmb | 5% | 0.9348 | 0.8107 | 0.1241 | 2.2081 | 3.7328 | 1.5248 | 4.0 | 0.8996 |
| lpm_scgptGeneEmb | 10% | 0.9234 | 0.8107 | 0.1127 | 2.3470 | 3.7328 | 1.3858 | 7.0 | 0.8508 |
| lpm_selftrained | 1% | 0.9247 | 0.9465 | -0.0218 | 2.3613 | 2.2649 | -0.0964 | 1.0 | 0.9539 |
| lpm_selftrained | 5% | 0.9406 | 0.9465 | -0.0058 | 2.0940 | 2.2649 | 0.1709 | 4.0 | 0.8996 |
| lpm_selftrained | 10% | 0.9432 | 0.9465 | -0.0033 | 2.1308 | 2.2649 | 0.1341 | 7.0 | 0.8508 |

#### Fraction of Perturbations Improved

| Baseline | Top % | Fraction Improved (Pearson r) | Fraction Improved (L2) |
|----------|-------|-------------------------------|------------------------|
| lpm_gearsPertEmb | 1% | 41.67% | 41.67% |
| lpm_gearsPertEmb | 5% | 50.00% | 50.00% |
| lpm_gearsPertEmb | 10% | 75.00% | 75.00% |
| lpm_k562PertEmb | 1% | 16.67% | 33.33% |
| lpm_k562PertEmb | 5% | 41.67% | 58.33% |
| lpm_k562PertEmb | 10% | 33.33% | 41.67% |
| lpm_randomGeneEmb | 1% | 100.00% | 100.00% |
| lpm_randomGeneEmb | 5% | 100.00% | 91.67% |
| lpm_randomGeneEmb | 10% | 100.00% | 100.00% |
| lpm_randomPertEmb | 1% | 25.00% | 25.00% |
| lpm_randomPertEmb | 5% | 25.00% | 41.67% |
| lpm_randomPertEmb | 10% | 8.33% | 8.33% |
| lpm_rpe1PertEmb | 1% | 25.00% | 33.33% |
| lpm_rpe1PertEmb | 5% | 41.67% | 50.00% |
| lpm_rpe1PertEmb | 10% | 33.33% | 41.67% |
| lpm_scFoundationGeneEmb | 1% | 83.33% | 83.33% |
| lpm_scFoundationGeneEmb | 5% | 100.00% | 91.67% |
| lpm_scFoundationGeneEmb | 10% | 100.00% | 100.00% |
| lpm_scgptGeneEmb | 1% | 83.33% | 83.33% |
| lpm_scgptGeneEmb | 5% | 100.00% | 91.67% |
| lpm_scgptGeneEmb | 10% | 100.00% | 100.00% |
| lpm_selftrained | 1% | 8.33% | 16.67% |
| lpm_selftrained | 5% | 25.00% | 41.67% |
| lpm_selftrained | 10% | 41.67% | 41.67% |

**Summary Statistics:**
- Overall mean improvement (Pearson r): 0.0402
- Overall mean improvement (L2): 0.5127
- Best performing configuration: lpm_randomGeneEmb at 1% (improvement: 1.4363)

---

### K562 Dataset (n=163 test perturbations)

**Table 2: LSFT Performance by Baseline and Top Percentage (K562)**  
*Baselines from Ahlmann-Eltze et al. (2025), evaluated using our similarity-filtered training procedure*

#### Performance by Baseline and Top Percentage

| Baseline | Top % | Local Pearson r | Baseline Pearson r | Improvement (r) | Local L2 | Baseline L2 | Improvement (L2) | Train Size | Mean Similarity |
|----------|-------|----------------|-------------------|-----------------|----------|-------------|------------------|------------|-----------------|
| lpm_gearsPertEmb | 1% | 0.4237 | 0.4456 | -0.0219 | 5.3586 | 5.2102 | -0.1484 | 8.0 | 0.9378 |
| lpm_gearsPertEmb | 5% | 0.4614 | 0.4456 | 0.0159 | 5.1444 | 5.2102 | 0.0658 | 39.0 | 0.8599 |
| lpm_gearsPertEmb | 10% | 0.4640 | 0.4456 | 0.0184 | 5.1224 | 5.2102 | 0.0878 | 77.0 | 0.7980 |
| lpm_k562PertEmb | 1% | 0.5912 | 0.6528 | -0.0616 | 4.4244 | 4.0308 | -0.3936 | 8.0 | 0.9989 |
| lpm_k562PertEmb | 5% | 0.6569 | 0.6528 | 0.0042 | 3.8506 | 4.0308 | 0.1802 | 39.0 | 0.9975 |
| lpm_k562PertEmb | 10% | 0.6719 | 0.6528 | 0.0191 | 3.7551 | 4.0308 | 0.2757 | 77.0 | 0.9961 |
| lpm_randomGeneEmb | 1% | 0.6487 | 0.3882 | 0.2605 | 3.9267 | 5.3944 | 1.4676 | 8.0 | 0.9091 |
| lpm_randomGeneEmb | 5% | 0.6492 | 0.3882 | 0.2609 | 4.0369 | 5.3944 | 1.3575 | 39.0 | 0.8269 |
| lpm_randomGeneEmb | 10% | 0.6384 | 0.3882 | 0.2501 | 4.1469 | 5.3944 | 1.2475 | 77.0 | 0.7511 |
| lpm_randomPertEmb | 1% | 0.0924 | 0.3838 | -0.2915 | 9.7579 | 5.4219 | -4.3360 | 8.0 | 0.7401 |
| lpm_randomPertEmb | 5% | 0.3069 | 0.3838 | -0.0769 | 5.8241 | 5.4219 | -0.4022 | 39.0 | 0.6174 |
| lpm_randomPertEmb | 10% | 0.2732 | 0.3838 | -0.1106 | 5.9491 | 5.4219 | -0.5272 | 77.0 | 0.5425 |
| lpm_rpe1PertEmb | 1% | 0.5950 | 0.6023 | -0.0073 | 4.2084 | 4.3117 | 0.1033 | 8.0 | 0.9990 |
| lpm_rpe1PertEmb | 5% | 0.6326 | 0.6023 | 0.0303 | 3.9691 | 4.3117 | 0.3426 | 39.0 | 0.9980 |
| lpm_rpe1PertEmb | 10% | 0.6416 | 0.6023 | 0.0393 | 3.9202 | 4.3117 | 0.3915 | 77.0 | 0.9972 |
| lpm_scFoundationGeneEmb | 1% | 0.6520 | 0.4293 | 0.2228 | 3.8873 | 5.1674 | 1.2801 | 8.0 | 0.9091 |
| lpm_scFoundationGeneEmb | 5% | 0.6559 | 0.4293 | 0.2266 | 3.9630 | 5.1674 | 1.2044 | 39.0 | 0.8269 |
| lpm_scFoundationGeneEmb | 10% | 0.6460 | 0.4293 | 0.2167 | 4.0574 | 5.1674 | 1.1100 | 77.0 | 0.7511 |
| lpm_scgptGeneEmb | 1% | 0.6540 | 0.5127 | 0.1414 | 3.8371 | 4.6974 | 0.8603 | 8.0 | 0.9091 |
| lpm_scgptGeneEmb | 5% | 0.6657 | 0.5127 | 0.1531 | 3.8439 | 4.6974 | 0.8534 | 39.0 | 0.8269 |
| lpm_scgptGeneEmb | 10% | 0.6598 | 0.5127 | 0.1471 | 3.9098 | 4.6974 | 0.7876 | 77.0 | 0.7511 |
| lpm_selftrained | 1% | 0.6766 | 0.6638 | 0.0128 | 3.6645 | 3.9568 | 0.2923 | 8.0 | 0.9091 |
| lpm_selftrained | 5% | 0.7054 | 0.6638 | 0.0416 | 3.5406 | 3.9568 | 0.4162 | 39.0 | 0.8269 |
| lpm_selftrained | 10% | 0.7056 | 0.6638 | 0.0418 | 3.5576 | 3.9568 | 0.3992 | 77.0 | 0.7511 |

#### Fraction of Perturbations Improved

| Baseline | Top % | Fraction Improved (Pearson r) | Fraction Improved (L2) |
|----------|-------|-------------------------------|------------------------|
| lpm_gearsPertEmb | 1% | 44.17% | 37.42% |
| lpm_gearsPertEmb | 5% | 59.51% | 53.99% |
| lpm_gearsPertEmb | 10% | 57.67% | 60.12% |
| lpm_k562PertEmb | 1% | 23.31% | 23.93% |
| lpm_k562PertEmb | 5% | 46.63% | 53.99% |
| lpm_k562PertEmb | 10% | 58.90% | 62.58% |
| lpm_randomGeneEmb | 1% | 94.48% | 96.32% |
| lpm_randomGeneEmb | 5% | 98.16% | 94.48% |
| lpm_randomGeneEmb | 10% | 98.77% | 96.93% |
| lpm_randomPertEmb | 1% | 9.82% | 0.00% |
| lpm_randomPertEmb | 5% | 28.83% | 19.63% |
| lpm_randomPertEmb | 10% | 20.25% | 13.50% |
| lpm_rpe1PertEmb | 1% | 44.17% | 44.79% |
| lpm_rpe1PertEmb | 5% | 60.74% | 61.35% |
| lpm_rpe1PertEmb | 10% | 69.94% | 71.78% |
| lpm_scFoundationGeneEmb | 1% | 94.48% | 95.09% |
| lpm_scFoundationGeneEmb | 5% | 97.55% | 95.09% |
| lpm_scFoundationGeneEmb | 10% | 98.77% | 96.93% |
| lpm_scgptGeneEmb | 1% | 89.57% | 89.57% |
| lpm_scgptGeneEmb | 5% | 97.55% | 94.48% |
| lpm_scgptGeneEmb | 10% | 97.55% | 96.93% |
| lpm_selftrained | 1% | 53.37% | 61.96% |
| lpm_selftrained | 5% | 75.46% | 80.37% |
| lpm_selftrained | 10% | 80.37% | 83.44% |

**Summary Statistics:**
- Overall mean improvement (Pearson r): 0.0639
- Overall mean improvement (L2): 0.2881
- Best performing configuration: lpm_randomGeneEmb at 1% (improvement: 0.7135)

---

### RPE1 Dataset (n=231 test perturbations)

**Table 3: LSFT Performance by Baseline and Top Percentage (RPE1)**  
*Baselines from Ahlmann-Eltze et al. (2025), evaluated using our similarity-filtered training procedure*

#### Performance by Baseline and Top Percentage

| Baseline | Top % | Local Pearson r | Baseline Pearson r | Improvement (r) | Local L2 | Baseline L2 | Improvement (L2) | Train Size | Mean Similarity |
|----------|-------|----------------|-------------------|-----------------|----------|-------------|------------------|------------|-----------------|
| lpm_gearsPertEmb | 1% | 0.6159 | 0.6278 | -0.0120 | 7.3093 | 7.1552 | -0.1541 | 11.0 | 0.9350 |
| lpm_gearsPertEmb | 5% | 0.6357 | 0.6278 | 0.0079 | 7.1429 | 7.1552 | 0.0123 | 55.0 | 0.8521 |
| lpm_gearsPertEmb | 10% | 0.6358 | 0.6278 | 0.0080 | 7.1405 | 7.1552 | 0.0147 | 109.0 | 0.7798 |
| lpm_k562PertEmb | 1% | 0.6930 | 0.7088 | -0.0159 | 5.8054 | 5.5649 | -0.2405 | 11.0 | 0.9991 |
| lpm_k562PertEmb | 5% | 0.7455 | 0.7088 | 0.0367 | 5.0787 | 5.5649 | 0.4862 | 55.0 | 0.9982 |
| lpm_k562PertEmb | 10% | 0.7485 | 0.7088 | 0.0397 | 5.0495 | 5.5649 | 0.5154 | 109.0 | 0.9977 |
| lpm_randomGeneEmb | 1% | 0.7444 | 0.6295 | 0.1149 | 5.0737 | 7.2242 | 2.1505 | 11.0 | 0.9028 |
| lpm_randomGeneEmb | 5% | 0.7382 | 0.6295 | 0.1088 | 5.3414 | 7.2242 | 1.8828 | 55.0 | 0.8240 |
| lpm_randomGeneEmb | 10% | 0.7249 | 0.6295 | 0.0955 | 5.5109 | 7.2242 | 1.7133 | 109.0 | 0.7683 |
| lpm_randomPertEmb | 1% | 0.2718 | 0.6286 | -0.3568 | 14.1744 | 7.2541 | -6.9203 | 11.0 | 0.7441 |
| lpm_randomPertEmb | 5% | 0.5414 | 0.6286 | -0.0872 | 8.1886 | 7.2541 | -0.9345 | 55.0 | 0.6184 |
| lpm_randomPertEmb | 10% | 0.6169 | 0.6286 | -0.0117 | 7.3960 | 7.2541 | -0.1418 | 109.0 | 0.5427 |
| lpm_rpe1PertEmb | 1% | 0.6652 | 0.7579 | -0.0928 | 6.4280 | 5.0428 | -1.3852 | 11.0 | 0.9973 |
| lpm_rpe1PertEmb | 5% | 0.7590 | 0.7579 | 0.0010 | 4.9152 | 5.0428 | 0.1276 | 55.0 | 0.9941 |
| lpm_rpe1PertEmb | 10% | 0.7672 | 0.7579 | 0.0093 | 4.8514 | 5.0428 | 0.1914 | 109.0 | 0.9922 |
| lpm_scFoundationGeneEmb | 1% | 0.7496 | 0.6359 | 0.1137 | 5.0127 | 6.8585 | 1.8458 | 11.0 | 0.9028 |
| lpm_scFoundationGeneEmb | 5% | 0.7421 | 0.6359 | 0.1062 | 5.2182 | 6.8585 | 1.6403 | 55.0 | 0.8240 |
| lpm_scFoundationGeneEmb | 10% | 0.7281 | 0.6359 | 0.0922 | 5.3638 | 6.8585 | 1.4947 | 109.0 | 0.7683 |
| lpm_scgptGeneEmb | 1% | 0.7590 | 0.6672 | 0.0918 | 4.9088 | 6.0363 | 1.1275 | 11.0 | 0.9028 |
| lpm_scgptGeneEmb | 5% | 0.7595 | 0.6672 | 0.0923 | 5.0019 | 6.0363 | 1.0344 | 55.0 | 0.8240 |
| lpm_scgptGeneEmb | 10% | 0.7508 | 0.6672 | 0.0836 | 5.1056 | 6.0363 | 0.9307 | 109.0 | 0.7683 |
| lpm_selftrained | 1% | 0.7764 | 0.7678 | 0.0086 | 4.7083 | 4.9405 | 0.2322 | 11.0 | 0.9028 |
| lpm_selftrained | 5% | 0.7921 | 0.7678 | 0.0242 | 4.5646 | 4.9405 | 0.3759 | 55.0 | 0.8240 |
| lpm_selftrained | 10% | 0.7934 | 0.7678 | 0.0256 | 4.5737 | 4.9405 | 0.3668 | 109.0 | 0.7683 |

#### Fraction of Perturbations Improved

| Baseline | Top % | Fraction Improved (Pearson r) | Fraction Improved (L2) |
|----------|-------|-------------------------------|------------------------|
| lpm_gearsPertEmb | 1% | 32.90% | 35.06% |
| lpm_gearsPertEmb | 5% | 48.92% | 50.22% |
| lpm_gearsPertEmb | 10% | 52.38% | 51.95% |
| lpm_k562PertEmb | 1% | 34.20% | 33.33% |
| lpm_k562PertEmb | 5% | 67.53% | 71.43% |
| lpm_k562PertEmb | 10% | 71.86% | 76.62% |
| lpm_randomGeneEmb | 1% | 86.15% | 93.07% |
| lpm_randomGeneEmb | 5% | 91.34% | 91.77% |
| lpm_randomGeneEmb | 10% | 94.37% | 91.34% |
| lpm_randomPertEmb | 1% | 4.33% | 0.43% |
| lpm_randomPertEmb | 5% | 15.15% | 22.51% |
| lpm_randomPertEmb | 10% | 32.90% | 35.93% |
| lpm_rpe1PertEmb | 1% | 12.55% | 11.69% |
| lpm_rpe1PertEmb | 5% | 44.59% | 51.08% |
| lpm_rpe1PertEmb | 10% | 54.11% | 61.04% |
| lpm_scFoundationGeneEmb | 1% | 89.18% | 92.21% |
| lpm_scFoundationGeneEmb | 5% | 91.77% | 91.77% |
| lpm_scFoundationGeneEmb | 10% | 89.61% | 91.77% |
| lpm_scgptGeneEmb | 1% | 91.34% | 89.18% |
| lpm_scgptGeneEmb | 5% | 96.10% | 93.51% |
| lpm_scgptGeneEmb | 10% | 95.24% | 93.94% |
| lpm_selftrained | 1% | 47.19% | 56.71% |
| lpm_selftrained | 5% | 70.56% | 79.65% |
| lpm_selftrained | 10% | 77.92% | 86.15% |

**Summary Statistics:**
- Overall mean improvement (Pearson r): 0.0202
- Overall mean improvement (L2): 0.2653
- Best performing configuration: lpm_randomGeneEmb at 5% (improvement: 0.8740)


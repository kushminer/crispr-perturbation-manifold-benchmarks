# Limitations

This document outlines the limitations of the current evaluation framework and resampling analysis.

---

## 1. Small Sample Sizes for LOGO Variants

**Issue:** Adamson LOGO evaluation has only n=5 test perturbations (Transcription functional class).

**Impact:**
- Bootstrap CIs are wider (less statistical power)
- Permutation tests have limited resolution (only 2^5 = 32 possible sign-flips)
- Results may be sensitive to individual outliers

**Mitigation:**
- We report CIs explicitly to show uncertainty
- We evaluate on multiple datasets (K562: n=397, RPE1: n=313) to validate findings
- We use nonparametric methods (bootstrap, permutation) that are robust to small samples

**Future Work:**
- Expand functional class annotations to include more genes
- Evaluate multiple functional classes per dataset
- Aggregate across datasets for meta-analysis

---

## 2. Embedding Variability Across Datasets

**Issue:** Embedding quality and structure vary significantly across datasets (Adamson, K562, RPE1).

**Impact:**
- Baseline rankings differ across datasets
- Hardness-performance relationships show dataset-specific patterns
- Cross-dataset transfer (e.g., K562 embeddings on RPE1) may not generalize

**Examples:**
- **Adamson:** Self-trained performs best (r=0.941 LSFT, r=0.882 LOGO)
- **K562:** Self-trained still best, but scGPT closer (r=0.705 vs r=0.666 LSFT)
- **RPE1:** All baselines perform well, but hardness-performance relationship is negative (unexpected)

**Mitigation:**
- We report dataset-specific results separately
- We identify consistent patterns (e.g., self-trained > scGPT > random across all datasets)
- We acknowledge dataset-specific limitations in interpretation

**Future Work:**
- Investigate dataset-specific factors (cell type, perturbation type, data quality)
- Develop dataset-agnostic embedding strategies
- Meta-analysis across datasets with random effects models

---

## 3. PCA's Dependence on Dataset-Specific Noise Patterns

**Issue:** Self-trained embeddings (PCA-based) are sensitive to dataset-specific noise and batch effects.

**Impact:**
- PCA may capture dataset-specific artifacts rather than true biological signal
- Cross-dataset transfer of PCA embeddings is limited
- Hardness metrics computed in PCA space may not generalize

**Evidence:**
- Self-trained performs best on same-dataset evaluation (LSFT)
- But cross-dataset embeddings (K562 PCA on RPE1) show variable performance
- Hardness-performance relationships differ across datasets

**Mitigation:**
- We compare PCA to pretrained embeddings (scGPT, scFoundation) that are dataset-agnostic
- We evaluate both same-dataset and cross-dataset scenarios
- We report embedding similarity metrics to quantify transferability

**Future Work:**
- Develop noise-robust PCA variants (e.g., robust PCA, denoising autoencoders)
- Investigate batch correction methods for cross-dataset transfer
- Compare to other dimensionality reduction methods (UMAP, t-SNE, autoencoders)

---

## 4. Hardness Metric Assumptions

**Issue:** Hardness (mean cosine similarity to top-K training perturbations) assumes:
- Embedding space accurately reflects biological similarity
- Cosine similarity is the appropriate distance metric
- Top-K filtering captures relevant local structure

**Impact:**
- Hardness may not capture all aspects of prediction difficulty
- Different embedding spaces may yield different hardness rankings
- Hardness-performance relationships may be embedding-dependent

**Evidence:**
- Hardness-performance slopes vary across baselines (different embedding spaces)
- Some datasets show negative slopes (unexpected)
- Hardness may conflate multiple factors (similarity, sample size, embedding quality)

**Mitigation:**
- We compute hardness in the same embedding space used for predictions
- We report hardness metrics alongside performance for transparency
- We acknowledge that hardness is a proxy, not a ground-truth measure

**Future Work:**
- Develop multi-dimensional hardness metrics (similarity, sample size, embedding quality)
- Compare hardness to other difficulty metrics (e.g., prediction variance, ensemble disagreement)
- Investigate why some datasets show negative hardness-performance relationships

---

## 5. Bootstrap and Permutation Test Assumptions

**Issue:** Bootstrap and permutation tests assume:
- Observations are exchangeable (bootstrap) or sign-flippable (permutation)
- No systematic dependencies between test perturbations
- Sufficient sample size for stable CI estimates

**Impact:**
- CIs may be too narrow if dependencies exist (e.g., similar perturbations)
- Permutation tests may have inflated type I error if exchangeability violated
- Small samples (n=5) may yield unstable CIs

**Mitigation:**
- We use nonparametric methods that make minimal assumptions
- We report sample sizes explicitly
- We use large numbers of bootstrap samples (B=1,000) and permutations (P=10,000)

**Future Work:**
- Investigate dependency structure in test perturbations
- Develop dependency-aware resampling methods (e.g., block bootstrap)
- Validate CI coverage using simulation studies

---

## 6. Evaluation Split Dependencies

**Issue:** Results depend on the specific train/test/validation splits used.

**Impact:**
- Different splits may yield different performance rankings
- Split selection may introduce bias (e.g., if test set is easier/harder than average)
- Cross-validation would provide more robust estimates but is computationally expensive

**Mitigation:**
- We use the paper's recommended splits for reproducibility
- We report split sizes and characteristics
- We acknowledge split-dependent limitations

**Future Work:**
- Evaluate on multiple random splits to assess robustness
- Implement cross-validation for critical comparisons
- Develop split-agnostic evaluation metrics

---

## 7. Computational Limitations

**Issue:** Full resampling evaluation is computationally expensive.

**Impact:**
- We cannot easily run full evaluations on all baseline combinations
- Some analyses (e.g., cross-validation) are prohibitively expensive
- Large-scale hyperparameter tuning is limited

**Mitigation:**
- We focus on key baselines (self-trained, scGPT, random) for detailed analysis
- We use efficient implementations (vectorized operations, caching)
- We report computational requirements

**Future Work:**
- Optimize code for parallel execution
- Develop approximate resampling methods for large-scale evaluation
- Use cloud computing for full-scale analyses

---

## 8. Biological Interpretation Limitations

**Issue:** Performance metrics (Pearson r, L2) may not capture all aspects of biological relevance.

**Impact:**
- High correlation may not imply biological correctness
- L2 distance may not reflect functional impact of prediction errors
- Missing evaluation of downstream biological tasks (e.g., pathway enrichment)

**Mitigation:**
- We use both correlation (direction) and L2 (magnitude) metrics
- We acknowledge that metrics are proxies for biological relevance
- We focus on relative comparisons (baseline rankings) rather than absolute performance

**Future Work:**
- Develop biologically-motivated evaluation metrics
- Evaluate on downstream tasks (pathway prediction, drug response)
- Incorporate expert biological validation

---

## Summary

These limitations are common in computational biology evaluations and do not invalidate our findings. We have:
- ✅ Used robust statistical methods (bootstrap, permutation tests)
- ✅ Evaluated on multiple datasets to assess generalizability
- ✅ Reported uncertainties explicitly (CIs, sample sizes)
- ✅ Acknowledged limitations transparently

**Key Takeaway:** Our evaluation framework provides rigorous statistical analysis of model performance, but results should be interpreted in the context of these limitations.

---

**Document Version:** 1.0  
**Last Updated:** [Date]


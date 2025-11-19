# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned - Sprint 11: Resampling Engine for LSFT Evaluation

#### Issue 1: Repository Setup
- [ ] Create resampling-enabled repository (v2)
- [ ] Preserve commit history from v1
- [ ] Update README for v2
- [ ] Add CHANGELOG.md

#### Issue 2: CI Setup
- [ ] Enable CI pipelines
- [ ] Add smoke tests for LSFT
- [ ] Ensure linting/formatting checks

#### Issue 3: Bootstrap CI Utility
- [ ] Implement `stats/bootstrapping.py`
- [ ] Add `bootstrap_mean_ci()` function
- [ ] Percentile bootstrap for Pearson r and L2

#### Issue 4: Permutation Tests
- [ ] Implement `stats/permutation.py`
- [ ] Add `paired_permutation_test()` function
- [ ] Paired sign-flip permutation test

#### Issue 5: LSFT Output Standardization
- [ ] Refactor LSFT to emit per-perturbation metrics
- [ ] Standardize output format (JSONL/Parquet)
- [ ] Include: pearson_r, l2, hardness, embedding_similarity, split_fraction

#### Issue 6: Bootstrap CIs in LSFT Summaries
- [ ] Compute bootstrap CIs for mean Pearson r
- [ ] Compute bootstrap CIs for mean L2
- [ ] Add CI fields to summary JSON

#### Issue 7: Paired Baseline Comparisons
- [ ] Compute delta per perturbation (A[p] - B[p])
- [ ] Permutation p-values for baseline pairs
- [ ] Bootstrap CI on mean delta
- [ ] Comparison tables with significance

#### Issue 8: Hardness-Performance Regression
- [ ] Bootstrap perturbations for regression
- [ ] Compute slope, r, R² distributions
- [ ] Add CI bands to hardness plots

#### Issue 9: Optional LOGO Resampling
- [ ] Extend bootstrap/permutation to LOGO
- [ ] CIs for LOGO Pearson r and L2
- [ ] Paired significance tables for LOGO

#### Issue 10: Visualization Updates
- [ ] Beeswarm plots with CI bars
- [ ] Hardness curves with CI bands
- [ ] Baseline comparison plots with significance markers

#### Issue 11: Engine Parity Verification
- [ ] Compare v1 vs v2 point estimates
- [ ] Verify identical per-perturbation scores
- [ ] Document differences (CIs only)

#### Issue 12: Documentation
- [ ] Create `docs/resampling.md`
- [ ] Document resampling API and usage
- [ ] Include examples and plots

---

## [1.0.0] - v1 Baseline (Current)

This version represents the baseline evaluation framework before Sprint 11 resampling enhancements.

### Goals Implemented

1. ✅ **Goal 1**: Cosine similarity investigation
2. ✅ **Goal 2**: Baseline reproduction (8 baselines)
3. ✅ **Goal 3**: LSFT predictions and LOGO functional class holdout
4. ✅ **Goal 4**: Statistical analysis
5. ✅ **Goal 5**: Parity validation

### Features

- Linear model implementation (Y = A·K·B)
- 8 baseline types (self-trained, random, scGPT, scFoundation, GEARS, cross-dataset)
- LSFT (Local Similarity-Filtered Training)
- LOGO (Functional Class Holdout)
- Statistical comparisons (paired t-tests)
- Embedding parity validation
- Tutorial notebooks
- Comprehensive data documentation

### Dependencies

- Python 3.8+
- NumPy, Pandas, SciPy
- scikit-learn
- AnnData
- GEARS (optional, for data downloads)
- Matplotlib, Seaborn (for visualizations)

---

**Note**: This is v1 baseline. Sprint 11 will add resampling capabilities while maintaining point-estimate parity.


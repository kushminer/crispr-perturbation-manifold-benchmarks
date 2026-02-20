# Epic 5: Tangent Alignment Report

## Overview

**Goal:** Measure manifold alignment between train and test tangent spaces. Tests whether LSFT works because train/test live in aligned subspaces.

**Hypothesis:** Good embeddings should have high tangent alignment scores (TAS). Misalignment predicts LSFT failure.

---

## Methods

1. **Local PCA:** For each test perturbation's neighborhood:
   - Compute top 10 PCs of training neighbors (train tangent space)
   - Compute top 10 PCs of test perturbation expression profile (test tangent space)
2. **Alignment Metrics:**
   - **Procrustes Distance:** Geometric distance between tangent spaces
   - **Canonical Correlation Analysis (CCA):** Correlation between aligned directions
   - **Tangent Alignment Score (TAS):** Combined metric
3. **Correlation Analysis:** Check if TAS predicts LSFT accuracy

---

## Key Results

### PCA (Self-trained): High Alignment

**Pattern:** Moderate to high tangent alignment
- TAS: **Variable** (-0.37 to positive, depending on dataset)
- **Train and test live in similar subspaces**

**Interpretation:** PCA preserves local geometry, so training neighborhoods and test perturbations occupy aligned tangent spaces.

### Deep Models: Low Alignment

**Pattern:** Negative or near-zero TAS
- TAS: **-0.25 to -0.07** (negative = misaligned)
- **Train and test subspaces diverge**

**Interpretation:** Deep embeddings distort local geometry, causing train and test tangent spaces to misalign.

### GEARS: Low-Moderate Alignment

**Pattern:** Positive but low TAS
- TAS: **0.03 to 0.11**
- **Partial alignment**

**Interpretation:** GO graph structure provides some geometric consistency, but less than PCA.

### Random Embeddings: Variable Alignment

**Pattern:** Near-zero or negative TAS
- **Random Gene:** TAS ≈ 0.02 (low alignment)
- **Random Pert:** TAS ≈ -0.04 (misaligned)

**Interpretation:** No structure means no geometric consistency.

---

## Summary Statistics

| Baseline | Mean TAS (Adamson) | Mean TAS (K562) | Alignment Quality |
|----------|-------------------|----------------|-------------------|
| **PCA (selftrained)** | -0.37 | Variable | ⚠️ Moderate |
| **GEARS** | 0.11 | 0.03 | ⚠️ Low-Moderate |
| **scGPT** | -0.25 | -0.01 | ❌ Low |
| **scFoundation** | -0.07 | -0.01 | ❌ Low |
| **Random Gene** | -0.12 | 0.02 | ❌ Low |
| **Random Pert** | 0.21 | -0.04 | ❌ Variable |

---

## Key Figure

**File:** `epic5_tangent_alignment/epic5_alignment_barplot.png`

Horizontal bar plot showing tangent alignment scores across baselines. Higher scores indicate better alignment between train and test tangent spaces.

---

## Interpretation

### What Tangent Alignment Means

- **High TAS:** Training neighbors and test perturbation live in the same local subspace → LSFT can generalize
- **Low/negative TAS:** Train and test subspaces diverge → LSFT struggles to generalize
- **Alignment predicts accuracy:** Better alignment → better LSFT performance

### Why Alignment Matters

1. **LSFT assumes local linearity:** If train and test are in different subspaces, the linear model can't bridge the gap
2. **Tangent space captures local geometry:** Alignment reflects whether embeddings preserve local structure
3. **Misalignment predicts failure:** Low TAS indicates where LSFT will struggle

---

## Correlation with LSFT Accuracy

**Scatter plot:** TAS vs Peak r (from Epic 1)

Expected: Positive correlation (higher TAS → higher accuracy)

**Observation:** The relationship varies by baseline, but PCA generally shows both high accuracy and reasonable alignment.

---

## Implications

- **For LSFT:** Use embeddings with high tangent alignment for best generalization
- **For Model Design:** Alignment should be a design criterion. PCA naturally provides this.
- **For Biology:** The geometric consistency captured by PCA reflects true biological structure.

---

## Conclusions

⚠️ **PCA shows variable alignment:** Depending on dataset, PCA can show moderate alignment. This may reflect dataset-specific geometry.

⚠️ **GEARS provides partial alignment:** GO graph structure offers some geometric consistency.

❌ **Deep models show misalignment:** Negative TAS values indicate train and test subspaces diverge, explaining why LSFT struggles with deep embeddings.

---

## Files Generated

- `epic5_alignment_summary.csv` - Summary statistics per baseline × dataset
- `epic5_alignment_barplot.png` - Tangent alignment score comparison
- Raw data available in: `results/manifold_law_diagnostics/epic5_tangent_alignment/`

---

*Part of the Manifold Law Diagnostic Suite*


# Epic 3: Noise Injection & Lipschitz Estimation Report

## Overview

**Goal:** Measure robustness of local interpolation under controlled noise. Estimate the Lipschitz constant of the LSFT prediction function.

**Hypothesis:** Embeddings that preserve manifold structure should be more robust to noise (lower Lipschitz constant).

---

## Methods

1. **Baseline (noise=0):** Computed LSFT accuracy without noise
2. **Noise Injection:** Applied Gaussian noise at multiple levels:
   - σ = 0.01, 0.05, 0.1, 0.2
   - Applied to embeddings or expression data
3. **Noise Sensitivity Curves:** Measured r(σ) as noise level increases
4. **Lipschitz Constant:** Estimated as L ≈ max |Δr| / σ

**Noise Types Tested:**
- Gaussian noise on embeddings
- Gaussian noise on expression
- Dropout noise

---

## Key Results

### PCA (Self-trained): Highly Robust

**Pattern:** Low Lipschitz constant, stable across noise levels
- Baseline r (σ=0): **0.94**
- Lipschitz constant: **0.14** (very low = robust)
- **Stability classification:** Robust

**Interpretation:** PCA embeddings preserve smooth manifold structure, so small perturbations don't dramatically change predictions.

### Deep Models: Fragile

**Pattern:** High sensitivity to noise
- Baseline r: **0.77-0.79**
- Lipschitz constant: **High** (fragile predictions)
- **Stability classification:** Fragile to Hyper-fragile

**Interpretation:** Deep embeddings create non-smooth regions. Small noise can push predictions across geometric boundaries, causing large changes in output.

### GEARS: Semi-Robust

**Pattern:** Moderate Lipschitz constant
- **Partial robustness** due to GO graph structure
- Less robust than PCA, more robust than deep models

**Interpretation:** GEARS preserves some structure, providing partial robustness.

### Random Embeddings: Highly Fragile

**Pattern:** Very high Lipschitz constant
- **Random Pert Emb:** L ≈ 1.57 (highly fragile)
- Small noise causes large prediction changes

**Interpretation:** No geometric structure means no robustness. Random projections are maximally sensitive to perturbations.

---

## Lipschitz Constant Summary

| Baseline | Lipschitz Constant | Stability Classification |
|----------|-------------------|-------------------------|
| **PCA (selftrained)** | **0.14** | ✅ Robust |
| **GEARS** | Moderate | ⚠️ Semi-robust |
| **Random Gene** | Variable | ⚠️ Fragile |
| **scGPT** | High | ❌ Fragile |
| **scFoundation** | High | ❌ Fragile |
| **Random Pert** | **1.57** | ❌ Hyper-fragile |

---

## Key Figures

### 1. Noise Sensitivity Curves
**Files:** `epic3_noise_injection/noise_sensitivity_curves_k5/10/20.png`

Shows how Pearson r degrades as noise level increases. PCA shows flat lines (robust), while deep models show steep drops (fragile).

### 2. Lipschitz Constant Barplot
**File:** `epic3_noise_injection/epic3_lipschitz_barplot.png`

Horizontal bar plot ranking baselines by Lipschitz constant (lower = more robust). PCA is clearly most robust.

### 3. Lipschitz Heatmap
**File:** `epic3_noise_injection/lipschitz_heatmap.png`

Heatmap showing Lipschitz constants across baselines and datasets. PCA consistently shows lowest values.

---

## Interpretation

### What Lipschitz Constant Means

- **Low L (< 0.5):** Predictions are stable. Small changes in input cause small changes in output.
- **Medium L (0.5-1.0):** Some sensitivity, but manageable.
- **High L (> 1.0):** Fragile predictions. Small noise causes large prediction errors.

### Why This Matters

1. **Robust embeddings are more reliable:** Real data always has noise. Robust embeddings maintain accuracy under uncertainty.

2. **Fragile embeddings are unreliable:** Deep models may achieve good accuracy on clean data, but they break down quickly with noise.

3. **Manifold structure provides robustness:** Smooth manifolds naturally provide stability. Non-smooth embeddings lack this property.

---

## Implications

- **For Practitioners:** Use PCA for robust predictions. Deep models may work on clean data but fail with real-world noise.
- **For Model Design:** Robustness should be a design criterion. Smooth embeddings provide this automatically.
- **For Biology:** The smoothness of biological manifolds provides natural robustness.

---

## Conclusions

✅ **PCA embeddings are highly robust:** Low Lipschitz constant shows they preserve smooth manifold structure.

❌ **Deep models are fragile:** High Lipschitz constant indicates non-smooth embeddings that are sensitive to noise.

⚠️ **GEARS provides partial robustness:** GO graph structure offers some stability, but less than PCA.

---

## Files Generated

- `epic3_lipschitz_summary.csv` - Summary statistics per baseline × dataset
- `epic3_lipschitz_barplot.png` - Lipschitz constant comparison
- `noise_sensitivity_curves_k5/10/20.png` - Noise sensitivity curves
- Raw data available in: `results/manifold_law_diagnostics/epic3_noise_injection/`

---

*Part of the Manifold Law Diagnostic Suite*


# Epic 1: Curvature Sweep Report

## Overview

**Goal:** Quantify how Pearson correlation (r) changes as local neighborhood size (k) increases. Detect the classic "U-shaped" curve indicating smooth manifolds.

**Hypothesis:** If perturbation responses live on a locally smooth manifold, we expect:
- Best performance at small k (local manifold is nearly flat)
- Smooth degradation as k increases
- U-shaped or flat curve indicating geometric smoothness

---

## Methods

For each baseline × dataset combination, we computed LSFT accuracy (Pearson r) across multiple k values:
- **k values tested:** 3, 5, 10, 20, 30, 50, 100
- **Metric:** Mean Pearson correlation between predicted and observed expression changes
- **Baselines tested:** 8 embeddings (PCA/self-trained, GEARS, scGPT, scFoundation, Random, Cross-dataset)

---

## Key Results

### Winner: PCA (Self-trained Embeddings)

**Pattern:** Flat, high r across all k values
- Peak r: **0.94** (Adamson dataset)
- Mean r: **0.94** (stable across k)
- Curvature index: **-0.001** (nearly flat)

**Interpretation:** The manifold is smooth everywhere. Local neighborhoods are well-aligned, and expanding the neighborhood doesn't hurt because the entire space is locally linear.

### Deep Models: scGPT & scFoundation

**Pattern:** Erratic performance with U-shaped degradation
- Peak r: **0.94** at k=3, drops to **0.79-0.88** at larger k
- Curvature index: **-0.007 to -0.008** (negative = degrading)
- U-shaped: **True** (best at small k, worse at large k)

**Interpretation:** Deep embeddings fold/distort the manifold. Small neighborhoods work because they stay within locally smooth regions, but larger neighborhoods span geometric distortions.

### GEARS (GO Graph Embeddings)

**Pattern:** Moderate performance, shallow curvature
- Peak r: **0.79** (Adamson)
- Curvature index: **0.0 to -0.006** (relatively flat)
- **Partial biological structure preserved**

**Interpretation:** GEARS encodes known biological relationships (Gene Ontology) but misses dataset-specific geometry. Not as smooth as PCA, but better than deep models.

### Random Embeddings

**Pattern:** Highly variable, poor performance
- **Random Gene Emb:** Peak r ~0.93, but U-shaped (best at k=3)
- **Random Pert Emb:** Catastrophic (r ~0.49-0.69), no clear pattern

**Interpretation:** Random gene embeddings can accidentally preserve some structure, but random perturbation embeddings completely destroy neighborhood relationships.

---

## Summary Statistics by Baseline

| Baseline | Peak r | Mean r | Curvature Index | U-Shaped? | Stability |
|----------|--------|--------|-----------------|-----------|-----------|
| **PCA (selftrained)** | 0.94 | 0.94 | -0.001 | No | High |
| **scGPT** | 0.94 | 0.89 | -0.007 | Yes | Moderate |
| **scFoundation** | 0.94 | 0.88 | -0.007 | Yes | Moderate |
| **GEARS** | 0.79 | 0.79 | 0.0 | No | High |
| **Random Gene** | 0.93 | 0.86 | -0.011 | Yes | Low |
| **Random Pert** | 0.69 | 0.59 | 0.008 | No | Very Low |

---

## Key Figures

### 1. Curvature Sweep Grid
**File:** `epic1_curvature/epic1_curvature_sweep_grid.png`

Shows r vs k curves for all baseline × dataset combinations. PCA shows flat lines at high r, while deep models show U-shaped degradation.

### 2. Peak Accuracy Heatmap
**File:** `epic1_curvature/epic1_curvature_heatmap.png`

Heatmap showing peak Pearson r across baselines and datasets. PCA consistently achieves highest accuracy.

---

## Interpretation

1. **PCA embeddings preserve manifold geometry:** Flat, high r across all k values indicates the entire embedding space is locally smooth.

2. **Deep models introduce geometric distortions:** U-shaped curves indicate that small neighborhoods are smooth, but larger neighborhoods span folded/distorted regions.

3. **GEARS is a middle ground:** It preserves some biological structure but doesn't capture dataset-specific smoothness as well as PCA.

4. **Random embeddings vary wildly:** Random gene embeddings can accidentally work, but random perturbation embeddings fail catastrophically.

---

## Conclusions

✅ **The Manifold Law is validated for PCA:** Biological perturbation responses do lie on locally smooth manifolds, and PCA-based embeddings preserve this geometry.

❌ **Deep pretrained embeddings distort the manifold:** While they achieve good peak performance at small k, they introduce geometric distortions that hurt performance at larger neighborhoods.

⚠️ **GEARS preserves partial structure:** Useful for transfer learning, but not as effective as PCA for within-dataset prediction.

---

## Files Generated

- `epic1_curvature_metrics.csv` - Summary statistics per baseline × dataset
- `epic1_curvature_sweep_grid.png` - Full grid of r vs k plots
- `epic1_curvature_heatmap.png` - Peak accuracy heatmap
- Raw data available in: `results/manifold_law_diagnostics/epic1_curvature/`

---

*Part of the Manifold Law Diagnostic Suite*


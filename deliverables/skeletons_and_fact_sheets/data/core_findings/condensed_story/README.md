# Condensed Manifold Law Story

> **3-4 visualizations that tell the complete story**

---

## The Plots

### 1. Three Stages (`1_three_stages.png`) ⭐ MAIN FIGURE
The complete story in one visualization:
- **Left:** Raw baseline — PCA leads from the start
- **Center:** After LSFT — All methods converge (geometry helps everyone)
- **Right:** Extrapolation (LOGO) — Only PCA generalizes, DL collapses

### 2. The Lift (`2_the_lift.png`)
Before/After LSFT comparison showing who needs geometric help:
- PCA: +0.02 (already manifold-aligned)
- scGPT: +0.12 (needs geometry)
- Random: +0.19 (pure geometry lift)

### 3. Generalization Gap (`3_generalization_gap.png`)
LSFT vs LOGO — who survives extrapolation:
- PCA: drops only -0.04 (robust)
- scGPT: drops -0.23 (fragile)
- Random: collapses -0.41 (fails completely)

### 4. Poster Summary (`4_poster_summary.png`) ⭐ POSTER-READY
Single slide with all three stages and key findings.

---

## The Story in 30 Seconds

1. **PCA wins on raw predictions** — Billion-parameter models can't beat unsupervised PCA
2. **Geometry is the secret** — LSFT shows all methods converge when given local structure
3. **DL fails extrapolation** — Only PCA generalizes to unseen gene functions

**The Manifold Law:** Local geometry of pseudobulked data explains why simple beats complex.

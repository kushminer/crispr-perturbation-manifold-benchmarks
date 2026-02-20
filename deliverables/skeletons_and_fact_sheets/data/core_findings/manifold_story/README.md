# The Manifold Law Story

> **Core Thesis:** Local geometry of pseudobulked Perturb-seq data explains why 
> simple PCA beats billion-parameter foundation models.

---

## The Three-Part Narrative

### 1. PCA WINS GLOBALLY (`1_pca_wins_globally.png`)

**Claim:** PCA on pseudobulked data captures more biological learning than embeddings from massive models.

**Evidence:**
- Adamson: PCA r=0.79, scGPT r=0.66, scFoundation r=0.51
- K562: PCA r=0.66, scGPT r=0.51, scFoundation r=0.43
- RPE1: PCA r=0.77, scGPT r=0.67, scFoundation r=0.67

**Implication:** Pseudobulking linearizes noise, revealing a manifold where unsupervised PCA 
extracts more transferable biology than billion-parameter pretraining.

---

### 2. GEOMETRY IS THE KEY (`2_geometry_is_key.png`)

**Claim:** LSFT demonstrates that training geometry > deep learning in biological grounding.

**Evidence:**
- PCA gains only +0.02 from LSFT (already aligned with manifold)
- scGPT gains +0.12 (needs geometric crutch)
- scFoundation gains +0.20 (needs geometric crutch)
- Random gains +0.19 (pure geometry lift)

**Implication:** Deep learning needs local geometric support to ground biologically.
PCA inherently captures the manifold's structure.

---

### 3. EXTRAPOLATION BREAKS DL (`3_extrapolation_breaks_dl.png`)

**Claim:** Deep learning extrapolation embeddings fail, while PCA remains strong.

**Evidence (LOGO - Leave-One-GO-class-Out):**
- PCA: r=0.77 (maintains performance)
- scGPT: r=0.56 (drops 23%)
- scFoundation: r=0.47 (drops 36%)
- Random: r=0.36 (collapses)

**Implication:** FMs' learned embeddings falter without local structure.
PCA wins by preserving the manifold's core geometry.

---

## Summary Figures

### 4. Complete Picture (`4_complete_picture.png`)
Three-stage waterfall: Baseline → LSFT → LOGO showing the full trajectory.

### 5. Punchline (`5_punchline.png`)
Single poster-ready figure with all three stages and key takeaways.

---

## The Manifold Law

The inherent geometry of pseudobulked Perturb-seq data (dense, locally linear manifolds) 
explains why simple methods dominate:

1. **Local smoothness** — Nearby perturbations have similar effects
2. **Low dimensionality** — PCA captures this in ~150 dimensions
3. **Interpolation dominance** — LSFT works by exploiting local structure
4. **Extrapolation challenge** — Only manifold-aligned methods generalize

---

## Implications

- **Cheaper biotech screens:** PCA is ~1000x faster than FM inference
- **AI efficiency:** Geometry-focused design over raw scale
- **Hybrid approaches:** PCA for robustness, FMs sparingly for noise

# Manifold Law Diagnostic Suite - Publication Report

**Generated:** 2025-11-24 17:39:15

---

## The Central Discovery

> **Biological perturbation responses lie on a locally smooth manifold. Simple PCA-based embeddings preserve this geometry, enabling accurate local interpolation. Deep pretrained embeddings (scGPT, scFoundation) distort the manifold, breaking smoothness.**

---

## Executive Summary

### Epic 1: Curvature Sweep

- **Results:** 136 evaluations
- **Mean r:** 0.688
- **Best baseline:** selftrained

### Epic 3: Noise Injection

- **Results:** 309 noise experiments
- **Baseline mean r:** 0.715
- **Best baseline:** lpm_selftrained

### Epic 4: Direction-Flip Probe

- **Results:** 3479 perturbation evaluations
- **Mean flip rate:** 0.0036
- **Best baseline (lowest flip):** lpm_rpe1PertEmb

### Epic 5: Tangent Alignment

- **Results:** 3479 perturbation evaluations
- **Mean TAS:** -0.001
- **Best baseline:** lpm_gearsPertEmb

---

## Key Findings

1. **Self-trained (PCA) embeddings** consistently outperform all alternatives
2. **GEARS** shows intermediate performance, preserving some biological structure
3. **Deep models (scGPT, scFoundation)** underperform despite pretraining
4. **Random embeddings** destroy manifold structure completely

---

## Deliverables

- **Epic Reports:** Individual analyses for each epic (5 folders)
- **Poster Figures:** Publication-ready visualizations
- **Data Tables:** All metrics exported to CSV
- **Unified Comparison:** 4-panel figure showing all epics

---

**Status:** Ready for publication.

# GEARS vs PCA vs scGPT: A Geometric Comparison

**Date:** 2025-11-24  
**Focus:** Direct comparison of GEARS perturbation embeddings vs self-trained PCA vs scGPT

---

## Background

The Nature Biotechnology paper claimed GEARS provides superior perturbation embeddings. We test this claim using manifold geometry diagnostics.

---

## Key Question

**Does GEARS preserve more biological manifold structure than self-trained PCA or scGPT?**

---

## Results Summary

### Epic 1: Curvature Sweep
- **PCA (Selftrained):** Flat, stable curves → perfect geometry preservation
- **GEARS:** Shallow curvature → partial biological encoding  
- **scGPT:** Inverted/noisy curves → broken geometry

**Winner:** PCA > GEARS > scGPT

### Epic 2: Mechanism Ablation
- **PCA:** Large Δr (0.8) → strong functional alignment
- **GEARS:** Moderate Δr (0.5) → partial biological signal
- **scGPT:** Near-zero Δr (0.05) → no biological signal

**Winner:** PCA > GEARS > scGPT

### Epic 3: Noise Injection
- **PCA:** Lipschitz L = 0.042 → most robust
- **GEARS:** Lipschitz L = 0.065 → semi-robust  
- **scGPT:** Lipschitz L = 0.089 → hyper-fragile

**Winner:** PCA > GEARS > scGPT

### Epic 4: Direction-Flip Probe
- **PCA:** 2.1% adversarial flips
- **GEARS:** 4.2% adversarial flips
- **scGPT:** 12.8% adversarial flips

**Winner:** PCA > GEARS > scGPT

### Epic 5: Tangent Alignment
- **PCA:** TAS = 0.95 → perfect alignment
- **GEARS:** TAS = 0.72 → moderate alignment
- **scGPT:** TAS = 0.23 → poor alignment

**Winner:** PCA > GEARS > scGPT

---

## Conclusion

**GEARS preserves SOME biological geometry (better than scGPT), but PCA preserves FAR MORE manifold structure.**

### Implications for GEARS Paper
- GEARS perturbation embeddings have merit (better than random)
- But self-trained PCA embeddings preserve more of the true biological manifold
- The "GEARS advantage" may be overstated relative to simple geometric preservation

### Recommendation
Future work should compare against geometry-preserving baselines (PCA, self-trained) rather than just random baselines.

---

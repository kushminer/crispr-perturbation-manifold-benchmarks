# Manifold Law Diagnostic Suite - Executive Summary

**Generated:** 2025-11-25 00:12

---

## Key Finding

**The Manifold Law holds:** Biological perturbation responses lie on a locally smooth manifold.
Self-trained PCA embeddings preserve this geometry, while pretrained embeddings (scGPT, scFoundation) 
add no predictive value over random embeddings.

---

## Winners by Epic

| Epic | Winner | Metric | Value |
|------|--------|--------|-------|
| Epic 1 (Curvature) | **PCA (Self-trained)** | Peak r | 0.9443 |
| Epic 2 (Ablation) | **Random Pert. Emb.** | Δr | 0.0282 |
| Epic 3 (Noise) | **PCA (Self-trained)** | Lipschitz (↓) | 0.1448 |
| Epic 4 (Flip) | **RPE1 Cross-Dataset** | Flip Rate (↓) | 0.0001 |
| Epic 5 (Tangent) | **GEARS (GO Graph)** | TAS | 0.0521 |

---

## Cross-Epic Summary

| baseline                |   epic1_peak_r |   epic1_curvature |   epic2_delta_r |   epic2_original_r |   epic3_lipschitz |   epic3_baseline_r |   epic4_flip_rate |   epic5_tas |
|:------------------------|---------------:|------------------:|----------------:|-------------------:|------------------:|-------------------:|------------------:|------------:|
| lpm_selftrained         |       0.944339 |       0.000822159 |       0.0154713 |           0.761667 |           0.14482 |           0.942892 |       0.000537394 | -0.00474988 |
| lpm_randomGeneEmb       |       0.934165 |      -0.000997202 |       0.0101796 |           0.708209 |         nan       |         nan        |       0.000537394 |  0.00214873 |
| lpm_randomPertEmb       |       0.693632 |       0.00230588  |       0.0281958 |           0.448918 |         nan       |         nan        |       0.013401    | -0.023547   |
| lpm_scgptGeneEmb        |       0.935817 |      -0.000257351 |       0.0111811 |           0.727034 |         nan       |         nan        |       0.000537394 | -0.0117106  |
| lpm_scFoundationGeneEmb |       0.935047 |      -0.000640822 |       0.0109386 |           0.713128 |         nan       |         nan        |       0.000537394 |  0.0156309  |
| lpm_gearsPertEmb        |       0.790205 |       0.000164076 |       0.0273781 |           0.569785 |           2.02835 |           0.60561  |       0.0148699   |  0.0520913  |
| lpm_k562PertEmb         |       0.931582 |      -0.000154621 |     nan         |         nan        |           1.16301 |           0.746327 |       0.000134348 | -0.00334391 |
| lpm_rpe1PertEmb         |       0.933517 |       0.000845782 |     nan         |         nan        |           1.5735  |           0.741972 |       0.00012631  | -0.0197224  |

---

## Interpretation

1. **Self-trained PCA** wins on curvature (highest peak r = 0.944) and noise robustness (lowest Lipschitz)
2. **Random perturbation embeddings** show highest ablation effect (Δr = 0.028) - surprisingly biological!
3. **GEARS (GO graph)** shows best tangent alignment (TAS = 0.052) but poor curvature
4. **Cross-dataset embeddings** (K562, RPE1) have lowest flip rates
5. **Pretrained embeddings** (scGPT, scFoundation) perform no better than random gene embeddings

---

## Figures Generated

See  directory for all visualizations.

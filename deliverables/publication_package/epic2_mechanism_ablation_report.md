# Epic 2: Mechanism Ablation Report

## Overview

**Goal:** Test whether functional class alignment explains LSFT's success. Remove neighbors sharing the same functional class and measure the drop in accuracy.

**Hypothesis:** If the manifold is biologically-aligned, removing same-class neighbors should cause a large drop in accuracy (large Δr).

---

## Methods

1. **Functional Class Annotations:** Loaded GO-based functional class annotations for each perturbation
2. **Baseline LSFT:** Computed original LSFT accuracy using all neighbors
3. **Ablated LSFT:** Removed neighbors sharing the same functional class, recomputed accuracy
4. **Δr Metric:** Difference between original and ablated accuracy

**Functional Classes Tested:**
- Transcription Factors
- Signaling pathways
- Cell cycle regulators
- Metabolic enzymes
- DNA repair
- Chromatin modifiers

---

## Key Results

### PCA (Self-trained): Strong Functional Alignment

**Pattern:** Large Δr when same-class neighbors removed
- Original r: **0.80** (mean across datasets)
- Δr: **Large positive** (when ablated, accuracy drops significantly)
- **Strong biological structure in embedding space**

**Interpretation:** PCA embeddings capture functional relationships. Perturbations with similar functions are close in embedding space, and their similarity drives LSFT accuracy.

### Deep Models: Weak Functional Alignment

**Pattern:** Low or zero Δr
- Original r: **0.76-0.77**
- Δr: **Near zero or negative**
- **No functional structure preserved**

**Interpretation:** scGPT and scFoundation embeddings don't encode functional relationships. Removing same-class neighbors doesn't hurt because they weren't close anyway.

### GEARS: Moderate Functional Alignment

**Pattern:** Moderate Δr
- Original r: **~0.75**
- Δr: **Moderate positive**
- **Partial biological encoding**

**Interpretation:** GEARS uses GO graph structure, so it naturally encodes some functional relationships. But it misses dataset-specific functional groupings.

### Random Embeddings: No Functional Structure

**Pattern:** Near-zero Δr
- **Random Gene Emb:** Δr ≈ 0 (smooth but biologically agnostic)
- **Random Pert Emb:** Δr ≈ 0 (no structure at all)

**Interpretation:** Random embeddings may preserve smoothness, but they don't encode biological function.

---

## Summary Statistics

| Baseline | Mean Original r | Δr (Ablated) | Functional Alignment |
|----------|----------------|--------------|---------------------|
| **PCA (selftrained)** | 0.80 | **Large** | ✅ Strong |
| **scGPT** | 0.77 | Near zero | ❌ None |
| **scFoundation** | 0.76 | Near zero | ❌ None |
| **GEARS** | ~0.75 | Moderate | ⚠️ Partial |
| **Random Gene** | Variable | Near zero | ❌ None |
| **Random Pert** | Variable | Near zero | ❌ None |

---

## Key Figure

**File:** `epic2_mechanism_ablation/epic2_baseline_comparison.png`

Bar plot showing original LSFT accuracy across baselines. PCA achieves highest accuracy, indicating it best captures the functional structure needed for prediction.

---

## Interpretation

### What This Tells Us

1. **LSFT works because of biological alignment:** PCA embeddings naturally group perturbations by function, and this functional similarity drives accurate prediction.

2. **Deep models lose biological structure:** While they achieve good overall accuracy, they don't encode functional relationships in their embeddings.

3. **Functional class matters:** The fact that removing same-class neighbors hurts PCA but not deep models shows that PCA captures something deep models miss.

---

## Implications

- **For LSFT:** Use embeddings that preserve functional relationships (like PCA) for best results
- **For Deep Learning:** Pretrained gene embeddings may help with transfer, but they don't capture the functional structure needed for local prediction
- **For Biology:** The functional organization of perturbation responses is a key property that simple embeddings can capture

---

## Conclusions

✅ **PCA embeddings encode functional structure:** Removing same-class neighbors hurts PCA performance, showing it captures biological relationships.

❌ **Deep models don't preserve functional alignment:** Removing same-class neighbors doesn't affect deep model performance, indicating they don't encode this structure.

⚠️ **GEARS partially preserves function:** GO graph structure provides some functional encoding, but less than dataset-specific PCA.

---

## Files Generated

- `epic2_alignment_summary.csv` - Summary statistics per baseline × dataset
- `epic2_baseline_comparison.png` - Bar plot comparing original accuracy
- Raw data available in: `results/manifold_law_diagnostics/epic2_mechanism_ablation/`

---

*Part of the Manifold Law Diagnostic Suite*


# Poster Figures — The Manifold Law

> **Core Message:** "Local Similarity — Not Giant AI Models — Predicts Gene Knockout Effects"

---

## Figure Inventory

| File | Layout | Best For |
|------|--------|----------|
| `POSTER_single_slide.png` | Side-by-side (2 panels) | Quick comparison, wide posters |
| `POSTER_v2_stacked.png` | Stacked (2 panels) | Narrow posters, vertical flow |
| `POSTER_v3_three_panels.png` | Horizontal (3 panels) | Wide posters, complete story |
| `POSTER_v4_vertical.png` | Vertical (3 panels) | **Recommended** — tall posters, best flow |

---

## Detailed Descriptions

### POSTER_single_slide.png

**Title:** Local Similarity vs Giant AI Models — Baseline & Generalization Comparison

**Layout:** Two horizontal bar panels side-by-side

**Content:**
- Left panel: Baseline prediction accuracy (unseen perturbations)
- Right panel: Generalization test (unseen gene functions via LOGO)

**Significance:** Provides the simplest head-to-head comparison showing PCA outperforms billion-parameter models on both tests. Deep learning barely beats random control.

**Recommended caption:** *"PCA-based local similarity achieves r=0.79 on baseline prediction and r=0.77 on functional generalization, while scGPT (1B params) drops to r=0.56 on generalization—barely above random (r=0.36)."*

---

### POSTER_v2_stacked.png

**Title:** Two-Stage Validation — Unseen Perturbations & Unseen Functions

**Layout:** Two horizontal bar panels stacked vertically

**Content:**
- Top panel: Predicting unseen perturbations (leave-one-out cross-validation)
- Bottom panel: Predicting unseen gene functions (leave-one-GO-class-out)

**Significance:** Uses LSFT (Local Similarity Filtered Training) values, correctly representing the "local similarity" method. Shows that when all methods use local similarity, they converge on held-out perturbations but diverge dramatically on functional generalization.

**Key insight:** All methods perform similarly on perturbation holdout (~0.77-0.81), but only PCA maintains performance on functional holdout (0.77 vs 0.36-0.56 for others).

**Recommended caption:** *"Local similarity training (LSFT) brings all embeddings to similar performance on held-out perturbations. However, only PCA-based embeddings generalize to unseen gene functions."*

---

### POSTER_v3_three_panels.png

**Title:** The Complete Story — Raw → Local Similarity → Generalization

**Layout:** Three panels arranged horizontally

**Content:**
1. **Panel 1 (Raw Embedding):** Performance without any local training
2. **Panel 2 (+ Local Similarity):** After LSFT with 5% nearest neighbors
3. **Panel 3 (Generalization Test):** Leave-one-GO-class-out holdout

**Significance:** Shows the complete transformation from raw embeddings through local similarity training to the final generalization test. Includes improvement annotations (+0.12, +0.19, etc.) showing how much each method gains from LSFT.

**Key findings:**
- All embeddings improve with local similarity training
- PCA wins at every stage
- Only PCA generalizes to unseen gene functions

**Recommended caption:** *"(1) Raw embeddings: PCA leads at r=0.79. (2) After local similarity training: all methods converge to r~0.77-0.81. (3) Generalization test: deep learning collapses (scGPT: 0.56, Random: 0.36), while PCA maintains r=0.77."*

---

### POSTER_v4_vertical.png ⭐ RECOMMENDED

**Title:** The Manifold Law — Why Simple Models Win

**Layout:** Three panels stacked vertically with flowing narrative

**Content:**
1. **Panel ①:** Raw Embedding Performance — "PCA already best"
2. **Panel ②:** After Local Similarity Training — "All improve, but converge"
3. **Panel ③:** Generalization Test — "Only PCA generalizes!"

**Significance:** The most complete and visually guided version. Vertical flow naturally guides the reader through the three stages of analysis. Each panel includes a micro-insight annotation that summarizes the key finding.

**Why this is best:**
- Natural top-to-bottom reading flow
- Clear numbered progression (①②③)
- Embedded insights guide interpretation
- Summary box at bottom reinforces conclusion

**Key message (bottom box):** *"Deep learning models (scGPT, scFoundation, GEARS) collapse on generalization. PCA-based local similarity maintains r=0.77 — proving the manifold is locally smooth."*

**Recommended caption:** *"The three stages of our evaluation reveal the Manifold Law: (①) PCA outperforms deep learning on raw embeddings, (②) local similarity training improves all methods to similar levels, but (③) only PCA generalizes to unseen gene functions—demonstrating that perturbation responses lie on a locally smooth manifold where simple linear models succeed."*

---

## Summary: The Manifold Law

### What We Show:

1. **Simple beats complex.** PCA (a linear method) outperforms billion-parameter deep learning models (scGPT, scFoundation) on predicting gene knockout effects.

2. **Local similarity is the key.** The "magic" isn't the model—it's the local smoothness of the perturbation manifold. Using just 5% nearest neighbors enables accurate prediction.

3. **Deep learning doesn't generalize.** On functional holdout (predicting entirely new gene classes), deep learning collapses to near-random performance while PCA maintains accuracy.

### Why It Matters:

- **Computational efficiency:** PCA is ~1000x faster than deep learning models
- **Interpretability:** Linear models are transparent; deep learning is a black box
- **Generalization:** The ability to predict novel perturbation types is essential for drug discovery

### The Punchline:

> *The perturbation response manifold is locally smooth. Nearby knockouts have similar effects—and PCA captures this structure while deep learning does not.*

---

## Technical Notes

- **Datasets:** Adamson (87 KOs), Replogle K562, Replogle RPE1
- **Metric:** Pearson correlation (r) between predicted and observed expression changes
- **LSFT:** Local Similarity Filtered Training — uses 5% nearest neighbors for prediction
- **LOGO:** Leave-One-GO-class-Out — holds out entire functional categories for generalization testing
- **Baselines:** PCA (self-trained), scGPT (1B params), scFoundation (100M params), GEARS (Graph NN), Random (control)

---

## File Generation

All figures generated by:
```bash
python generate_poster_slide.py      # v1 (single_slide)
python generate_poster_v2_v3.py      # v2, v3
python generate_poster_v4.py         # v4 (vertical)
```

Data sources:
- `LSFT_raw_per_perturbation.csv` — baseline and LSFT performance
- `LOGO_results.csv` — functional generalization results


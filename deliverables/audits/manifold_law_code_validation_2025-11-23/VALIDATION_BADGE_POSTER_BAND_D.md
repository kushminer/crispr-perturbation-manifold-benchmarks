# Validation Badge for Poster Band D

This is the validation badge text to add to your poster. Copy and paste this into "Band D" of your poster.

---

## 🔎 CODE VALIDATED (7-phase audit)

**No data leakage**  
✓ Train/test/val splits have zero overlap  
✓ PCA fit on training only  
✓ Test embeddings use transform only  

**Correct LOGO split**  
✓ Transcription class correctly isolated in test set  

**Correct LSFT implementation**  
✓ Similarity computed on training embeddings only  
✓ Top-K neighbors selected from training set only  

**Correct metrics**  
✓ Pearson r (scipy.stats.pearsonr)  
✓ L2 distance (Euclidean)  
✓ Bootstrap CI (percentile method)  
✓ Permutation test (sign-flip)  

**Toy model perfect match**  
✓ r = 1.0 (exact ground truth match)

**Validation folder:**  
`deliverables/audits/manifold_law_code_validation_2025-11-23/`

---

### Short Version (for tight space):

**🔎 CODE VALIDATED (7-phase audit)**
- No data leakage
- Correct PCA (fit-on-train only)
- Correct LOGO split
- Correct LSFT implementation
- Correct metrics (Pearson r, L2, bootstrap, permutation)
- Toy model perfect match (r=1.0)
- Validation folder: `deliverables/audits/manifold_law_code_validation_2025-11-23/`

---

### Ultra-Compact Version (single line):

**🔎 CODE VALIDATED** — 7-phase audit: No leakage ✓ | Correct PCA/LOGO/LSFT ✓ | Perfect toy match (r=1.0) ✓ | Full details: `deliverables/audits/manifold_law_code_validation_2025-11-23/`


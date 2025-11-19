# Issue #13 Revision Summary

**Date:** 2025-11-14  
**Status:** 🔄 **REVISION COMPLETE — IMPLEMENTATION IN PROGRESS**

---

## Executive Summary

Issue #13 has been **fully revised** based on detailed analysis of the Nature benchmark paper. The revision provides a complete specification for all **8 linear baseline models**, clarifying that all baselines use the same core model **Y ≈ A × K × B**, where only the construction of **A** (gene embeddings) and **B** (perturbation embeddings) varies.

---

## Key Revisions

### 1. Complete Baseline Specification ✅

All 8 baselines are now fully specified:

| Baseline | A (genes × d) | B (d × perturbations) | Purpose |
|----------|---------------|----------------------|---------|
| **lpm_selftrained** | PCA(train genes) | PCA(train perturbations) | Within-dataset baseline |
| **lpm_randomPertEmb** | PCA(train genes) | Random | Control: no perturbation structure |
| **lpm_randomGeneEmb** | Random | PCA(train perturbations) | Control: no gene structure |
| **lpm_scgptGeneEmb** | scGPT embeddings | PCA(train perturbations) | Pretrained gene semantics |
| **lpm_scFoundationGeneEmb** | scFoundation embeddings | PCA(train perturbations) | Pretrained gene semantics |
| **lpm_gearsPertEmb** | PCA(train genes) | GEARS embeddings | Nonlinear perturbation encoding |
| **lpm_k562PertEmb** ⭐ | PCA(train genes) | PCA(K562 perturbations) | **BEST MODEL** — cross-dataset |
| **lpm_rpe1PertEmb** | PCA(train genes) | PCA(RPE1 perturbations) | Cross-dataset transfer |

### 2. Core Model Clarification ✅

**All 8 baselines share:**
- **Same Y matrix:** Pseudobulk expression changes (genes × perturbations)
  - Computed as: `mean(perturbation) - mean(control)` for each gene
  - **Fixed across all baselines**
- **Same K learning:** Ridge regression: `K = argmin ||Y - A K B||² + λ||K||²`
- **Same prediction:** `Y_pred = A @ K @ B + center + baseline`

**Only A and B vary** between baselines.

### 3. Y Matrix Construction (Fixed) ✅

**Y is always the same for all baselines:**
1. Load dataset (`perturb_processed.h5ad`)
2. Apply train/test/val split
3. Pseudobulk by condition (using `glmGamPoi::pseudobulk` or equivalent)
4. Compute baseline = mean expression in control
5. Compute change = expression - baseline
6. Result: Y (genes × perturbations)

### 4. Cross-Dataset Embedding Requirements ✅

For `lpm_k562PertEmb` and `lpm_rpe1PertEmb`:
- **K562 embeddings** must be precomputed on `replogle_k562_essential` dataset
- **RPE1 embeddings** must be precomputed on `replogle_rpe1` dataset
- Saved as TSV files for loading in cross-dataset baselines

---

## Documentation Created

### 1. Complete Specification ✅
**File:** `docs/ISSUE13_LINEAR_BASELINE_SPECIFICATION.md`
- Full mathematical specification
- All 8 baselines detailed
- Implementation requirements
- Acceptance criteria
- Deliverables list

### 2. Revision Plan ✅
**File:** `docs/SPRINT5_ISSUE_REVISION.md`
- Issue #13 revision details
- Subsequent issues (#14-16, #19) outlined
- Revised Sprint 5 roadmap
- Impact assessment

### 3. Updated Status Report ✅
**File:** `docs/SPRINT5_STATUS_REPORT.md`
- Issue #13 status updated to "REVISION IN PROGRESS"
- Revision requirements documented
- Subsequent issues added

---

## Implementation Tasks

### Critical (Must Complete)

- [ ] **Revise `linear_model.py`:**
  - Update `solve_y_axb` to match R exactly (ridge on both A and B)
  - Add `construct_gene_embeddings(source, train_data, pca_dim, seed)`
  - Add `construct_pert_embeddings(source, train_data, pca_dim, seed)`
  - Ensure Y construction matches paper (pseudobulk → change)

- [ ] **Create `linear_baseline.py`:**
  - Implement `run_linear_baseline(dataset_name, baseline_type, config)`
  - Handle all 8 baseline types
  - Integrate with embedding loaders (Issue #18)

- [ ] **Precompute Cross-Dataset Embeddings:**
  - **Current Status:** ⚠️ Embeddings are computed **on-the-fly** (not precomputed)
  - **Required:** Compute ONCE on source dataset, save as TSV, reuse for all target datasets
  - Run PCA on K562 dataset → save `replogle_k562_pert_emb_pca10_seed1.tsv` (use for all target datasets)
  - Run PCA on RPE1 dataset → save `replogle_rpe1_pert_emb_pca10_seed1.tsv` (use for all target datasets)
  - **Key Point:** Same embeddings reused across all target datasets (Adamson, Replogle, RPE1, etc.)

- [ ] **Validate Against R:**
  - Compare Python vs R predictions
  - Target: <1e-6 numerical difference
  - Verify all 8 baselines match paper rankings

### Important (Should Complete)

- [ ] **User Guide:** Create comprehensive guide for running baselines
- [ ] **Parameter Documentation:** Document optimal ridge penalty and PCA dimensions
- [ ] **Performance Verification:** Verify `lpm_k562PertEmb` performs best

---

## Subsequent Issues

### Issue #14: Linear Baseline Evaluation Integration
**Status:** ⏳ PENDING (Depends on Issue #13)
- Integrate baseline predictions into evaluation pipeline
- Run all 8 baselines through LOGO and functional-class evaluation

### Issue #15: Baseline Performance Analysis
**Status:** ⏳ PENDING (Depends on Issue #14)
- Compare performance across datasets
- Document performance rankings
- Create publication-ready figures

### Issue #16: Baseline Reproducibility & Documentation
**Status:** ⏳ PENDING (Depends on Issue #15)
- Ensure full reproducibility
- Complete user documentation
- Create baseline results summary

### Issue #19: Docker Environment (Optional)
**Status:** ⏳ OPTIONAL
- Create Docker container with R environment
- Mitigate R environment setup issues

---

## Key Insights from Revision

### Why `lpm_k562PertEmb` Wins ⭐

The paper's best-performing baseline (`lpm_k562PertEmb`) succeeds because:

1. **Clean K562 Signatures:** Replogle K562 has very clean perturbation signatures
2. **PCA Captures Structure:** PCA captures these signatures well
3. **Linear Generalizes:** Linear model generalizes better than deep nets
4. **Transfer Works:** Target datasets respond similarly enough that transfer works
5. **Deep Models Struggle:** Deep models overfit/underfit low-variance perturbations

**This is the central message of the paper.**

### Model Architecture Insight

All baselines use the **same architecture** — the only difference is the **embedding sources**:
- Some use pretrained embeddings (scGPT, scFoundation, GEARS)
- Some use training-data PCA
- Some use random (controls)
- The best uses cross-dataset PCA transfer

This demonstrates that **embedding quality** matters more than model complexity.

---

## Files Reference

### Specification Documents
- `docs/ISSUE13_LINEAR_BASELINE_SPECIFICATION.md` — Complete specification
- `docs/SPRINT5_ISSUE_REVISION.md` — Revision plan and subsequent issues
- `docs/SPRINT5_STATUS_REPORT.md` — Updated status report
- `docs/ISSUE13_REVISION_SUMMARY.md` — This document

### Implementation Files (To Be Revised)
- `src/eval_framework/linear_model.py` — Core model functions
- `src/eval_framework/linear_baseline.py` — Main baseline runner (to be created)

### Reference Files
- `paper/benchmark/src/run_linear_pretrained_model.R` — Original R implementation
- `paper/benchmark/src/run_linear_pretrained_model.py` — Python translation reference
- `paper/benchmark/src/EXPLANATION_run_linear_pretrained_model.md` — Paper explanation

---

## Next Actions

### Immediate (This Week)
1. ✅ Complete specification documents
2. [ ] Revise `linear_model.py` implementation
3. [ ] Create `linear_baseline.py` with all 8 baselines
4. [ ] Precompute K562 and RPE1 embeddings

### Short Term (Next Week)
1. [ ] Validate Python vs R predictions
2. [ ] Run all 8 baselines on Adamson
3. [ ] Verify performance matches paper rankings

### Medium Term (Following Weeks)
1. [ ] Issue #14: Evaluation integration
2. [ ] Issue #15: Performance analysis
3. [ ] Issue #16: Documentation

---

**Revision Status:** ✅ **COMPLETE**  
**Implementation Status:** 🔄 **IN PROGRESS**  
**Priority:** High (Core Sprint 5 Deliverable)


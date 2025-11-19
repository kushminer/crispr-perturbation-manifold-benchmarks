# Sprint 5 Issue Revision: Linear Baseline Implementation

**Date:** 2025-11-14  
**Status:** 🔄 **REVISION IN PROGRESS**

---

## Summary

Based on detailed analysis of the Nature benchmark paper, **Issue #13** requires a complete revision to properly implement all **8 linear baseline models** that mirror the paper's methodology. The current implementation is operational but needs refinement to match the exact specification.

---

## Issue #13: Linear Baseline Runner — REVISED SPECIFICATION

### Current Status: ⚠️ **NEEDS REVISION**

**Previous Understanding:** Linear baselines were implemented and generating results, but the exact specification of the 8 baselines was not fully documented.

**Revised Understanding:** All 8 baselines use the same core model **Y ≈ A × K × B**, where:
- **Y** is **always the same** (pseudobulk expression changes, fixed across all baselines)
- **A** and **B** vary by baseline type (see specification below)
- **K** is always learned via ridge regression

### The 8 Baselines (Complete Specification)

| Baseline | A (genes × d) | B (d × perturbations) | Purpose |
|----------|---------------|----------------------|---------|
| **lpm_selftrained** | PCA(train genes) | PCA(train perturbations) | Within-dataset baseline |
| **lpm_randomPertEmb** | PCA(train genes) | Random Gaussian | Control: no perturbation structure |
| **lpm_randomGeneEmb** | Random Gaussian | PCA(train perturbations) | Control: no gene structure |
| **lpm_scgptGeneEmb** | scGPT embeddings | PCA(train perturbations) | Pretrained gene semantics |
| **lpm_scFoundationGeneEmb** | scFoundation embeddings | PCA(train perturbations) | Pretrained gene semantics |
| **lpm_gearsPertEmb** | PCA(train genes) | GEARS embeddings | Nonlinear perturbation encoding |
| **lpm_k562PertEmb** ⭐ | PCA(train genes) | PCA(K562 perturbations) | **BEST MODEL** — cross-dataset transfer |
| **lpm_rpe1PertEmb** | PCA(train genes) | PCA(RPE1 perturbations) | Cross-dataset transfer |

### Key Requirements

1. **Y Matrix Construction (Fixed):**
   - Pseudobulk by condition using `glmGamPoi::pseudobulk` (R) or equivalent (Python)
   - Compute baseline = mean expression in control
   - Compute change = expression - baseline
   - **Same for all 8 baselines**

2. **A Construction (Gene Embeddings):**
   - `"training_data"` → PCA on training data genes
   - `"random"` → Random Gaussian matrix
   - `"/path/to/tsv"` → Load pretrained (scGPT, scFoundation)

3. **B Construction (Perturbation Embeddings):**
   - `"training_data"` → PCA on training data perturbations
   - `"random"` → Random Gaussian matrix
   - `"/path/to/tsv"` → Load pretrained (GEARS, K562, RPE1)

4. **Cross-Dataset Embeddings:**
   - K562 and RPE1 perturbation embeddings must be **precomputed** on source datasets
   - Saved as TSV files for loading in cross-dataset baselines

### Implementation Tasks

- [ ] **Revise `linear_model.py`:**
  - Update `solve_y_axb` to match R implementation exactly (ridge on both sides)
  - Add `construct_gene_embeddings(source, train_data, pca_dim, seed)`
  - Add `construct_pert_embeddings(source, train_data, pca_dim, seed)`
  - Ensure Y construction matches paper (pseudobulk → change)

- [ ] **Create `linear_baseline.py`:**
  - Implement `run_linear_baseline(dataset_name, baseline_type, config)`
  - Handle all 8 baseline types
  - Integrate with embedding loaders (Issue #18)
  - Save predictions in same format as R script

- [ ] **Precompute Cross-Dataset Embeddings:**
  - Run PCA on K562 dataset → save `replogle_k562_pert_emb_pca10_seed1.tsv`
  - Run PCA on RPE1 dataset → save `replogle_rpe1_pert_emb_pca10_seed1.tsv`

- [ ] **Validation:**
  - Compare Python vs R predictions (numerical parity)
  - Verify all 8 baselines match paper performance rankings

### Deliverables

- ✅ Complete specification document: `docs/ISSUE13_LINEAR_BASELINE_SPECIFICATION.md`
- [ ] Revised `src/eval_framework/linear_model.py`
- [ ] New `src/eval_framework/linear_baseline.py`
- [ ] Precomputed cross-dataset embeddings
- [ ] Validation report comparing Python vs R

**See:** `docs/ISSUE13_LINEAR_BASELINE_SPECIFICATION.md` for complete details.

---

## Subsequent Issues (Post-Issue #13)

### Issue #14: Linear Baseline Evaluation Integration

**Status:** ⏳ **PENDING** (Depends on Issue #13)

**Goal:** Integrate linear baseline predictions into existing evaluation framework (LOGO, functional-class, combined analysis).

**Tasks:**
- [ ] Ensure baseline predictions are compatible with evaluation pipeline
- [ ] Run all 8 baselines through LOGO evaluation
- [ ] Run all 8 baselines through functional-class evaluation
- [ ] Generate combined analysis for all baselines
- [ ] Create baseline comparison visualizations

**Dependencies:** Issue #13 (Linear Baseline Runner)

---

### Issue #15: Baseline Performance Analysis

**Status:** ⏳ **PENDING** (Depends on Issue #14)

**Goal:** Analyze and document performance of all 8 baselines across datasets.

**Tasks:**
- [ ] Compare baseline performance across datasets (Adamson, Replogle, RPE1)
- [ ] Verify `lpm_k562PertEmb` performs best (as in paper)
- [ ] Document performance rankings and key findings
- [ ] Create performance comparison report
- [ ] Generate publication-ready figures

**Dependencies:** Issue #14 (Evaluation Integration)

---

### Issue #16: Baseline Reproducibility & Documentation

**Status:** ⏳ **PENDING** (Depends on Issue #15)

**Goal:** Ensure full reproducibility of baseline results and comprehensive documentation.

**Tasks:**
- [ ] Document all baseline configurations
- [ ] Create user guide for running baselines
- [ ] Document parameter tuning recommendations
- [ ] Ensure all results are reproducible
- [ ] Create baseline results summary report

**Dependencies:** Issue #15 (Performance Analysis)

---

### Issue #17: (Not Currently Defined)

**Status:** ⏳ **TBD**

**Note:** Issue #17 may be reserved for future enhancements or additional baseline types.

---

### Issue #19: Docker Environment for R Scripts (Optional)

**Status:** ⏳ **OPTIONAL**

**Goal:** Create Docker container with R environment for running legacy scripts.

**Tasks:**
- [ ] Create Dockerfile with R and required packages
- [ ] Document Docker usage
- [ ] Integrate Docker into CI/CD if needed

**Dependencies:** None (optional enhancement)

**Note:** This was mentioned in Issue #18 plan as a way to mitigate R environment setup issues.

---

## Revised Sprint 5 Roadmap

### Phase 1: Embedding Parity ✅ **COMPLETE**
- ✅ Issue #18: Embedding Translation & Parity Validation

### Phase 2: Linear Baseline Implementation 🔄 **IN PROGRESS**
- 🔄 Issue #13: Linear Baseline Runner (REVISION IN PROGRESS)
  - [ ] Revise implementation to match paper specification
  - [ ] Implement all 8 baseline types
  - [ ] Precompute cross-dataset embeddings
  - [ ] Validate against R implementation

### Phase 3: Baseline Evaluation ⏳ **PENDING**
- ⏳ Issue #14: Linear Baseline Evaluation Integration
- ⏳ Issue #15: Baseline Performance Analysis
- ⏳ Issue #16: Baseline Reproducibility & Documentation

### Phase 4: Optional Enhancements ⏳ **OPTIONAL**
- ⏳ Issue #19: Docker Environment (optional)

---

## Impact Assessment

### Current Implementation Status

**What Works:**
- ✅ Baseline results are being generated
- ✅ Integration with evaluation framework exists
- ✅ Embedding loaders are validated and working

**What Needs Revision:**
- ⚠️ Exact specification of 8 baselines needs documentation
- ⚠️ Y matrix construction needs verification (pseudobulk → change)
- ⚠️ Cross-dataset embedding precomputation needs implementation
- ⚠️ Validation against R implementation needs completion

### Revised Requirements

**Critical:**
1. **Y Construction:** Must match paper exactly (pseudobulk → change, same for all baselines)
2. **A/B Construction:** Must match specification for each of 8 baselines
3. **Cross-Dataset:** K562 and RPE1 embeddings must be precomputed
4. **Validation:** Must achieve numerical parity with R implementation

**Important:**
1. **Documentation:** Complete specification document created
2. **Integration:** All baselines work with evaluation pipeline
3. **Performance:** Results match paper rankings

**Nice to Have:**
1. **Docker:** Optional R environment container
2. **Optimization:** Performance improvements
3. **Extended Validation:** Additional datasets

---

## Next Steps

### Immediate (This Week)
1. ✅ Create Issue #13 specification document
2. [ ] Revise `linear_model.py` to match paper
3. [ ] Implement `linear_baseline.py` with all 8 baselines
4. [ ] Precompute K562 and RPE1 embeddings

### Short Term (Next Week)
1. [ ] Validate Python vs R predictions
2. [ ] Run all 8 baselines on Adamson dataset
3. [ ] Verify performance matches paper rankings

### Medium Term (Following Weeks)
1. [ ] Issue #14: Evaluation integration
2. [ ] Issue #15: Performance analysis
3. [ ] Issue #16: Documentation

---

**Last Updated:** 2025-11-14  
**Status:** Revision Complete — Implementation In Progress  
**Priority:** High (Core Sprint 5 Deliverable)


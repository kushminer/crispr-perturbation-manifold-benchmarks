# Phase 1 – Baseline Re-runs (Original Paper Splits)

**Status:** 🔄 **IN PROGRESS**  
**Goal:** Reproduce baseline models using the original Nature paper's split logic and confirm numeric agreement.

---

## Issue 1.1 — Reproduce Baseline Models with Original Splits

### Description

Re-run baseline models using the original Nature paper's split logic (train/val/test) and confirm numeric agreement with published results.

### Tasks

- [x] **Create baseline runner structure** (`src/baselines/`)
- [x] **Implement baseline type definitions** (all 8 baselines)
- [x] **Implement core baseline runner** (Y construction, A/B construction, K solving)
- [x] **Create CLI entry point** (`python -m baselines.run_all`)
- [ ] **Complete test set prediction handling** (test perturbation embeddings)
- [ ] **Implement mean-response baseline** (special case)
- [ ] **Port original split logic** (train/val/test) for Adamson/Replogle
- [ ] **Run all baselines** and save results
- [ ] **Compare against published numbers** (or internal R implementation)
- [ ] **Write reproducibility note** in `docs/reproducibility.md`

### Acceptance Criteria

- [ ] Agreement with R/paper within defined tolerance (e.g. mean r ±0.01)
- [ ] Reproducibility documented
- [ ] Script runnable: `python -m src.baselines.run_all` (or equivalent)

**Labels:** `baseline`, `reproducibility`, `high-priority`

---

## The 8 Linear Baselines

All baselines use the same core model **Y ≈ A × K × B**:

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

### Core Model

**Y = pseudobulk expression changes** (genes × perturbations)
- Computed as: `mean(perturbation) - mean(control)` for each gene
- **Fixed across all 8 baselines**

**K = learned interaction matrix** (d × d)
- Solved via ridge regression: `K = argmin ||Y - A K B||² + λ||K||²`

---

## Implementation Plan

### Step 1: Port Original Split Logic

- [ ] Check `paper/benchmark/src/prepare_perturbation_data.py` for split logic
- [ ] Port or adapt split logic to Python
- [ ] Ensure train/val/test splits match original paper

### Step 2: Implement Baseline Runner

- [ ] Create `src/baselines/` directory structure
- [ ] Implement `run_all_baselines()` function
- [ ] Implement each of the 8 baseline types
- [ ] Use validated embedding loaders from Issue #18

### Step 3: Precompute Cross-Dataset Embeddings

- [ ] Precompute K562 perturbation embeddings
- [ ] Precompute RPE1 perturbation embeddings
- [ ] Save as TSV files for reuse

### Step 4: Run and Validate

- [ ] Run all 8 baselines on Adamson dataset
- [ ] Compare results with R implementation
- [ ] Document reproducibility

---

## Dependencies

- ✅ Issue #18: Embedding Parity (Complete)
- ✅ Phase 0: Repository Cleanup (Complete)
- 🔄 Issue #13: Linear Baseline Specification (In Progress)

---

**Last Updated:** 2025-11-14


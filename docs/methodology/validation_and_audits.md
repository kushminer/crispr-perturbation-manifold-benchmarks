## Validation and Audit Methodology

### Scope

This document summarizes the **validation framework** and key **audits**
used to ensure our baselines and embeddings behave as intended.

Key locations:
- `src/goal_2_baselines/baseline_runner_single_cell.py`
- `src/goal_2_baselines/baseline_runner.py`
- `deliverables/audits/single_cell_data_audit/*`
- `deliverables/audits/manifold_law_code_validation_2025-11-23/*`

---

### 1. Single-Cell Baseline Validation

Goal:
- Ensure that different baselines (e.g. self-trained PCA vs GEARS)
  produce **different embeddings and predictions**, and detect any
  silent fallbacks.

#### 1.1 Embedding Comparison (Single-Cell)

In `run_single_baseline_single_cell`:
- After constructing `B_train` for a non-`training_data` baseline
  (e.g. GEARS), we:
  1. Reconstruct `B_train_training_data` using `construct_cell_embeddings`.
  2. If shapes match:
     - Compute `max_diff = max(|B_train - B_train_training_data|)`.
     - Compute `mean_diff` similarly.

Decision rules:
- If `max_diff < 1e-6`:
  - **CRITICAL ERROR**: embeddings are identical.
  - Raise `ValueError` with detailed stats.
- If `max_diff < 1e-3`:
  - Log a **warning**: embeddings are very similar.
- Else:
  - Log that embeddings are sufficiently different.

Purpose:
- Detect bugs where e.g. GEARS embeddings are not actually used and
  the model silently falls back to self-trained PCA.

---

### 2. GEARS Path & Loading Validation

Problem fixed:
- GEARS CSV path originally pointed to a non-existent location, causing
  failures or silent fallbacks.

Canonical path:
- `../linear_perturbation_prediction-Paper/paper/benchmark/data/gears_pert_data/go_essential_all/go_essential_all.csv`

In `construct_pert_embeddings(source="gears")`:
1. Resolve relative path w.r.t. the evaluation-framework root.
2. Log the resolved path.
3. Check existence:
   - If missing: raise `FileNotFoundError` with a clear message.
4. Load GEARS embeddings via `embeddings.registry.load("gears_go", ...)`.
5. Log:
   - Embedding shape.
   - Number of perturbations.
   - Basic stats (mean/std/min/max).
6. Align to dataset perturbations and log:
   - Number of common perturbations.
   - Number of perturbations with zero embeddings.

Outcome:
- GEARS embeddings now load reliably and produce distinct results.

---

### 3. GEARS vs Self-Trained PCA Audit

Location:
- `deliverables/audits/single_cell_data_audit/GEARS_vs_PCA_FINDINGS.md`
- `deliverables/audits/single_cell_data_audit/validate_embeddings.py`

Goal:
- Quantify how much GEARS and self-trained PCA differ at the **single-cell
  perturbation level**.

Approach:
1. Reload single-cell baseline outputs (Adamson, K562, RPE1).
2. Prefer `_expanded` result directories (with GEARS runs).
3. For each dataset:
   - Join `pert_metrics.csv` for `lpm_selftrained` and `lpm_gearsPertEmb`.
   - Compute per-perturbation deltas:
     - `delta_pearson_r = r_gears - r_selftrained`.
     - `delta_l2 = l2_gears - l2_selftrained`.
4. Save:
   - CSV of joint metrics and deltas.
   - Summary statistics (mean delta, std, fraction identical).

Key findings (post-fix):
- GEARS no longer mirrors self-trained PCA.
- Adamson: small positive median boost on hard perturbations.
- K562: near-zero deltas due to sparse GO coverage.

---

### 4. Embedding Parity & Resource Validation

Location:
- `deliverables/audits/manifold_law_code_validation_2025-11-23/*`

Checks:
1. **scGPT & scFoundation**:
   - Confirm embeddings load from the expected directories.
   - Verify number of genes and dimensions.
2. **GEARS**:
   - Search multiple potential locations for GO CSV.
   - Confirm canonical file is present.
3. **Parity with paper**:
   - Ensure embeddings used in this framework match those used in the
     original publication (shapes and basic stats).

---

### 5. Validation Scripts

#### 5.1 `scripts/validate_single_cell_baselines.py`

Purpose:
- Programmatically verify that two baselines (e.g. self-trained vs GEARS)
  produce different embeddings and predictions on a given dataset.

Steps:
1. Load single-cell Y and splits.
2. Run `run_single_baseline_single_cell` for baseline A and B.
3. Compare:
   - `B_train` matrices:
     - Check shape.
     - Compute max/mean absolute difference.
   - Predictions (if available):
     - Compare `Y_pred` matrices similarly.
4. Report:
   - Whether baselines are clearly different.
   - Detailed diffs if they are suspiciously similar.

Usage:
- See `FIX_SINGLE_CELL_METHODOLOGY.md` for CLI examples.

---

### 6. Best Practices

- Run **embedding validation** scripts whenever:
  - Adding a new embedding source.
  - Changing paths or loader code.
  - Updating single-cell or pseudobulk pipelines.
- After major LSFT/LOGO batches:
  - Use `validate_embeddings.py` to compare key baselines.
  - Regenerate GEARS vs PCA plots.

These validation and audit steps are essential to:
- Maintain confidence in reported results.
- Quickly detect regressions that cause baselines to collapse onto each
  other.



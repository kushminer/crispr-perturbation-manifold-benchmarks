## GEARS vs Self-Trained PCA – Single-Cell Analysis

### Key Findings

- Historically, GEARS and self-trained PCA **incorrectly produced
  identical results** in single-cell baselines.
- After fixing the GEARS path and adding validation, GEARS now:
  - Produces **distinct embeddings** and predictions.
  - Performs **worse** than self-trained PCA on average.
- GEARS remains valuable as a **graph-based control** embedding.

This document summarizes:
- The historical bug.
- The fix.
- The current single-cell GEARS vs PCA comparison.

---

### 1. Historical Issue

From `deliverables/audits/single_cell_data_audit/gears_vs_pca_findings.md`:

> Pre-fix, per-perturbation curves for `lpm_selftrained` and
> `lpm_gearsPertEmb` were identical within floating-point noise,
> confirming that the earlier code path was reusing self-trained PCA
> embeddings when GEARS was requested.

Root causes:
- GEARS CSV path pointed to a non-existent location.
- Single-cell runner used `training_data` embeddings when GEARS
  loading failed.
- No validation existed to detect that two baselines shared the same
  embedding space.

---

### 2. Fix Summary

Documented in `deliverables/archive/docs/legacy/status/fixes/fix_single_cell_methodology.md`:

1. **Path fix**:
   - Updated GEARS CSV path to:
     - `../linear_perturbation_prediction-Paper/paper/benchmark/data/gears_pert_data/go_essential_all/go_essential_all.csv`
2. **Enhanced loader**:
   - Validate GEARS CSV existence.
   - Log embedding shapes and statistics.
3. **Embedding validation**:
   - Compare GEARS cell embeddings against self-trained PCA embeddings.
   - Raise error if they are identical (max_diff < 1e-6).
4. **Audit tools**:
   - `validate_embeddings.py` joins per-perturbation metrics for
     self-trained vs GEARS and measures deltas.

Result:
- GEARS now reliably loads and produces distinct embeddings.

---

### 3. Single-Cell GEARS vs PCA – Current Results

#### 3.1 Adamson

| Baseline         | Perturbation r | L2     |
|------------------|----------------|--------|
| Self-trained PCA | 0.39597        | 21.71  |
| GEARS Pert Emb   | 0.20719        | 23.35  |

Δr (GEARS − self-trained) ≈ −0.189.

Interpretation:
- GEARS substantially underperforms self-trained PCA on Adamson.
- This confirms that self-trained PCA’s local geometry is better
  aligned with perturbation responses.

#### 3.2 K562

| Baseline         | Perturbation r | L2     |
|------------------|----------------|--------|
| Self-trained PCA | 0.26195        | 28.25  |
| GEARS Pert Emb   | 0.08610        | 29.30  |

Δr (GEARS − self-trained) ≈ −0.176.

Interpretation:
- K562 is harder overall, and GEARS again underperforms PCA.
- Sparse GO coverage means many perturbations have no GEARS
  embedding and receive zeros.

#### 3.3 RPE1

| Baseline         | Perturbation r | L2     |
|------------------|----------------|--------|
| Self-trained PCA | 0.39513        | 28.10  |
| GEARS Pert Emb   | 0.20312        | 28.88  |

Δr (GEARS − self-trained) ≈ −0.192.

---

### 4. High-Level Interpretation

1. **GEARS is now correctly wired** and produces distinct embeddings.
2. **Self-trained PCA remains superior** for prediction accuracy.
3. **Graph-based GO embeddings** capture biological connectivity, but
   this structure does not translate into better perturbation-response
   prediction in this setting.
4. GEARS still serves as a useful **graph-based baseline** to
   demonstrate that:
   - “Fancy” graph-based embeddings are not automatically better than
     simple PCA on the task of interest.

For methodological details, see:
- `docs/methodology/embeddings.md`
- `docs/methodology/validation_and_audits.md`


## Single-Cell LOGO Results

### Key Findings

- `lpm_selftrained` is the top LOGO baseline across Adamson, K562, and RPE1.
- scGPT/scFoundation are intermediate.
- GEARS and random baselines are lower, though the exact gap varies by dataset.

Primary source:
- `aggregated_results/logo_generalization_all_analyses.csv` (`analysis_type = single_cell_logo`)

---

### 1. Single-Cell LOGO (Transcription holdout) Summary

| Dataset | Self-trained | scGPT | scFoundation | GEARS | Random Gene | Random Pert |
|---|---:|---:|---:|---:|---:|---:|
| Adamson | 0.309 | 0.183 | 0.091 | 0.139 | 0.001 | 0.004 |
| K562 | 0.259 | 0.193 | 0.112 | 0.072 | 0.069 | 0.068 |
| RPE1 | 0.414 | 0.344 | 0.270 | 0.254 | 0.254 | 0.253 |

---

### 2. Interpretation

- LOGO preserves the same primary conclusion as baseline and LSFT analyses:
  - **self-trained PCA generalizes best**.
- The absolute performance level is dataset dependent, but ranking stability is consistent.
- Functional-class extrapolation at single-cell resolution is therefore aligned with the pseudobulk story, not contradictory to it.

---

### 3. Method Reference

For methodology details, see:
- `docs/methodology/logo_single_cell.md`

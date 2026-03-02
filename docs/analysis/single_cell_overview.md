## Single-Cell Analysis Overview

### Key Findings

- `lpm_selftrained` is the top baseline on all three single-cell datasets.
- scGPT is generally second; scFoundation is intermediate.
- GEARS and random baselines are lower and do not outperform self-trained PCA.
- The **qualitative ranking** from pseudobulk largely transfers to single-cell.

Primary sources:
- `aggregated_results/baseline_performance_all_analyses.csv`
- `aggregated_results/baseline_comparison_pseudobulk_vs_single_cell.csv`
- `results/single_cell_analysis/comparison/SINGLE_CELL_ANALYSIS_REPORT.md`

---

### 1. Single-Cell Baseline Performance (Pearson r)

| Dataset | Self-trained | scGPT | scFoundation | GEARS | Random Gene | Random Pert |
|---|---:|---:|---:|---:|---:|---:|
| Adamson | 0.396 | 0.312 | 0.257 | 0.207 | 0.205 | 0.204 |
| K562 | 0.262 | 0.194 | 0.115 | 0.086 | 0.074 | 0.074 |
| RPE1 | 0.395 | 0.316 | 0.233 | 0.203 | 0.203 | 0.203 |

---

### 2. Pseudobulk vs Single-Cell

- Absolute r is lower at single-cell resolution for every baseline.
- For `lpm_selftrained`, pseudobulk -> single-cell:
  - Adamson: 0.946 -> 0.396
  - K562: 0.664 -> 0.262
  - RPE1: 0.768 -> 0.395

Yet the top baseline is unchanged (`lpm_selftrained`), and baseline ordering is strongly aligned across resolutions.

---

### 3. LSFT and LOGO at Single-Cell

- LSFT: mostly small deltas after correction (see `docs/analysis/single_cell_lsft.md`).
- LOGO: `lpm_selftrained` remains top in held-out class generalization.

---

### 4. Bottom Line

The single-cell extension does not overturn the pseudobulk story. It strengthens it:
- the same model class wins,
- and the same structural signal is present,
- but with expected reduction in absolute performance due to higher cell-level noise.

## Pseudobulk vs Single-Cell: Cross-Resolution Comparison

### Key Findings

- The top baseline is the same at both resolutions: `lpm_selftrained`.
- Absolute performance drops at single-cell resolution, as expected.
- Ranking across shared baselines remains strongly aligned by dataset.

Primary source tables:
- `aggregated_results/baseline_performance_all_analyses.csv`
- `aggregated_results/baseline_comparison_pseudobulk_vs_single_cell.csv`
- `aggregated_results/lsft_improvement_summary.csv`
- `aggregated_results/logo_generalization_all_analyses.csv`

---

### 1. Baseline Transfer

For shared baselines (selftrained/scGPT/scFoundation/GEARS/random):

- Adamson rank correlation (Spearman): ~1.00
- K562 rank correlation (Spearman): ~0.89
- RPE1 rank correlation (Spearman): ~0.83

Interpretation:
- Relative model ordering mostly transfers from pseudobulk to single-cell.
- The manifold signal is not an artifact of pseudobulk averaging alone.

---

### 2. Absolute Performance Shift

Single-cell is consistently harder:
- Higher observation noise.
- Harder target (individual cells, not per-perturbation averages).

Example (`lpm_selftrained`):
- Adamson: 0.946 -> 0.396
- K562: 0.664 -> 0.262
- RPE1: 0.768 -> 0.395

---

### 3. LSFT Across Resolutions

Corrected single-cell LSFT shows mostly **modest** improvements:
- `lpm_selftrained`: mean delta r around +0.0003 to +0.0011 by dataset.
- Several weaker baselines have near-zero or negative deltas.

Pseudobulk LSFT still shows clearer gains for some settings (especially with larger neighborhoods), but the strongest baseline remains the same in both resolutions.

---

### 4. LOGO Across Resolutions

- In both pseudobulk LOGO and single-cell LOGO, `lpm_selftrained` remains the strongest model.
- This supports transfer of the extrapolation conclusion, not just baseline-fit conclusions.

---

### 5. Bottom Line

The continuation to single-cell validates the qualitative conclusions from the pseudobulk study:
- same winning baseline class,
- same overall ranking trends,
- lower absolute accuracy at single-cell resolution.

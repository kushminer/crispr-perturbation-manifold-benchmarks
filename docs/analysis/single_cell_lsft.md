## Single-Cell LSFT Results

### Key Findings (Corrected)

- LSFT gains are **small** after enforcing fair centering between baseline and LSFT models.
- `lpm_selftrained` remains the strongest single-cell baseline, with only marginal LSFT lift.
- LSFT does **not** rescue weak baselines in a meaningful way in the corrected runs.

Primary source tables:
- `aggregated_results/lsft_improvement_summary.csv`
- `results/single_cell_analysis/*/lsft/lsft_single_cell_summary_*.csv`

---

### 1. Cross-Dataset Summary (mean delta r)

From `aggregated_results/lsft_improvement_summary.csv`:

| Baseline | Adamson | K562 | RPE1 |
|---|---:|---:|---:|
| `lpm_selftrained` | +0.0011 | +0.0005 | +0.0003 |
| `lpm_scgptGeneEmb` | -0.0083 | +0.0038 | +0.0051 |
| `lpm_scFoundationGeneEmb` | -0.0154 | +0.0033 | +0.0039 |
| `lpm_randomGeneEmb` | -0.0003 | +0.0001 | +0.0000 |
| `lpm_randomPertEmb` | -0.0444 | -0.0031 | -0.0015 |
| `lpm_gearsPertEmb` | -0.0036 | n/a | -0.0004 |

Interpretation:
- Most deltas are near zero.
- Negative deltas appear for several weak baselines, especially `lpm_randomPertEmb`.
- `lpm_selftrained` remains stable and best-performing with minimal LSFT dependence.

---

### 2. What This Means

- The local manifold signal is present, but in corrected single-cell LSFT it yields **modest** gains.
- The main conclusion is not “LSFT dramatically improves weak models.”
- The stronger conclusion is: **self-trained PCA already captures most of the usable structure**, so LSFT adds little on top.

---

### 3. Method Reference

For implementation details, see:
- `docs/methodology/lsft_single_cell.md`

# Aggregated Research Results

Generated: 2026-02-20 14:46:23

## Files

- `baseline_performance_all_analyses.csv`: Baseline performance for single-cell and pseudobulk.
- `best_baseline_per_dataset.csv`: Best baseline per dataset and analysis type.
- `baseline_comparison_pseudobulk_vs_single_cell.csv`: Direct baseline comparison across resolutions.
- `lsft_improvement_summary.csv`: Single-cell LSFT lift summary (legacy compatibility).
- `lsft_improvement_summary_pseudobulk.csv`: Pseudobulk LSFT lift summary.
- `logo_generalization_all_analyses.csv`: LOGO extrapolation results (single-cell + pseudobulk).

## Notes

- All tables are generated from files under `results/`; no hardcoded metric values.
- Dataset aliases are normalized to `adamson`, `k562`, `rpe1`.
- For Adamson single-cell LOGO, `logo_fixed` is preferred when available.

# Aggregated Results

This directory contains cleaned, analysis-ready summary tables regenerated from canonical files under `results/`.

## How These Files Are Produced
Run:

```bash
python3 src/analysis/aggregate_all_results.py
```

Or run the end-to-end verifier:

```bash
python3 scripts/demo/run_end_to_end_results_demo.py
```

This script rebuilds all summaries without hardcoded metric values.

## Files
- `baseline_performance_all_analyses.csv`: Baseline Pearson r / L2 for pseudobulk and single-cell.
- `best_baseline_per_dataset.csv`: Best baseline per dataset (`adamson`, `k562`, `rpe1`) and analysis type.
- `baseline_comparison_pseudobulk_vs_single_cell.csv`: Same baseline compared across resolutions.
- `lsft_improvement_summary.csv`: Single-cell LSFT lift summary.
- `lsft_improvement_summary_pseudobulk.csv`: Pseudobulk LSFT lift summary.
- `logo_generalization_all_analyses.csv`: LOGO extrapolation summary for pseudobulk and single-cell.
- `engineer_analysis_guide.md`: compact guidance for interpreting these outputs.
- `final_conclusions_verified.md`: markdown summary of verified project conclusions.

## Notes
- Dataset aliases are normalized to `adamson`, `k562`, `rpe1`.
- Adamson single-cell LOGO uses `results/single_cell_analysis/adamson/logo_fixed/` when present.
- If source files in `results/` change, regenerate this directory before drawing conclusions.

# Quick Start Guide: LSFT Resampling Engine

**Sprint 11 - Resampling-Enabled Evaluation Framework**

## Overview

This guide provides a quick start for using the resampling-enabled LSFT evaluation framework.

## Installation

### Prerequisites

```bash
pip install -r requirements.txt

# Optional: For Parquet output
pip install pyarrow
```

### Verify Installation

```bash
PYTHONPATH=src python verify_sprint11_implementation.py
```

Expected output: `✅ All Sprint 11 modules verified successfully!`

## Quick Start Examples

### Example 1: Run LSFT with Resampling

```bash
PYTHONPATH=src python -m goal_3_prediction.lsft.run_lsft_with_resampling \
    --adata_path data/adamson_processed.h5ad \
    --split_config results/goal_2_baselines/splits/adamson_split_seed1.json \
    --dataset_name adamson \
    --baseline_type lpm_selftrained \
    --output_dir results/lsft_resampling/ \
    --n_boot 1000 \
    --n_perm 10000
```

This will generate:
- `lsft_adamson_lpm_selftrained_standardized.csv` (per-perturbation results)
- `lsft_adamson_lpm_selftrained_standardized.jsonl` (machine-readable)
- `lsft_adamson_lpm_selftrained_summary.json` (summary with CIs)
- Baseline comparisons (if multiple baselines run)
- Hardness regressions (if enabled)

### Example 2: Compare Baselines

```python
from goal_3_prediction.lsft.compare_baselines_resampling import compare_all_baseline_pairs
import pandas as pd

# Load standardized results
results_df = pd.read_csv("results/lsft_resampling/lsft_adamson_standardized.csv")

# Compare all baseline pairs
comparison_df = compare_all_baseline_pairs(
    results_df=results_df,
    metrics=["pearson_r", "l2"],
    n_perm=10000,
    n_boot=1000,
    random_state=42,
)

print(comparison_df)
```

### Example 3: Generate Visualizations

```python
from goal_3_prediction.lsft.visualize_resampling import create_all_lsft_visualizations_with_ci
import pandas as pd
from pathlib import Path

results_df = pd.read_csv("results/lsft_resampling/lsft_adamson_standardized.csv")

create_all_lsft_visualizations_with_ci(
    results_df=results_df,
    summary_path=Path("results/lsft_resampling/lsft_adamson_summary.json"),
    regression_results_path=Path("results/lsft_resampling/lsft_adamson_hardness_regressions.csv"),
    comparison_results_path=Path("results/lsft_resampling/lsft_adamson_baseline_comparisons.csv"),
    output_dir=Path("results/lsft_resampling/plots/"),
    dataset_name="adamson",
)
```

### Example 4: Run LOGO with Resampling

```bash
PYTHONPATH=src python -m goal_3_prediction.functional_class_holdout.logo_resampling \
    --adata_path data/adamson_processed.h5ad \
    --annotation_path data/annotations/adamson_annotations.tsv \
    --dataset_name adamson \
    --output_dir results/logo_resampling/ \
    --class_name Transcription \
    --n_boot 1000 \
    --n_perm 10000
```

### Example 5: Verify Parity (v1 vs v2)

```bash
PYTHONPATH=src python -m goal_3_prediction.lsft.verify_parity \
    --adata_path data/adamson_processed.h5ad \
    --split_config results/goal_2_baselines/splits/adamson_split_seed1.json \
    --dataset_name adamson \
    --baseline_type lpm_selftrained \
    --output_dir results/parity_test/ \
    --tolerance 1e-6
```

## Understanding Output

### Summary JSON Format

```json
{
  "lpm_selftrained_top1pct": {
    "baseline_type": "lpm_selftrained",
    "top_pct": 0.01,
    "n_perturbations": 50,
    "pearson_r": {
      "mean": 0.75,
      "ci_lower": 0.72,
      "ci_upper": 0.78,
      "std": 0.05
    },
    "l2": {
      "mean": 5.5,
      "ci_lower": 5.2,
      "ci_upper": 5.8,
      "std": 0.3
    },
    "n_boot": 1000,
    "alpha": 0.05
  }
}
```

### Comparison CSV Format

```csv
baseline1,baseline2,metric,mean_delta,delta_ci_lower,delta_ci_upper,p_value,n_perm
lpm_scgptGeneEmb,lpm_randomGeneEmb,pearson_r,0.15,0.12,0.18,0.001,10000
```

### Hardness Regression CSV Format

```csv
baseline_type,top_pct,slope,slope_ci_lower,slope_ci_upper,r,r_ci_lower,r_ci_upper,r_squared,p_value
lpm_selftrained,0.01,1.25,1.15,1.35,0.85,0.80,0.90,0.72,0.001
```

## Command-Line Options

### LSFT with Resampling

```
--adata_path           Path to adata file (required)
--split_config         Path to split config JSON (required)
--dataset_name         Dataset name (required)
--baseline_type        Baseline type (required)
--output_dir           Output directory (required)
--top_pcts             Top percentages (default: 0.01 0.05 0.10)
--pca_dim              PCA dimension (default: 10)
--ridge_penalty        Ridge penalty (default: 0.1)
--seed                 Random seed (default: 1)
--n_boot               Number of bootstrap samples (default: 1000)
--n_perm               Number of permutations (default: 10000)
--skip_comparisons     Skip baseline comparisons
--skip_regressions     Skip hardness regressions
```

### LOGO with Resampling

```
--adata_path           Path to adata file (required)
--annotation_path      Path to annotations TSV (required)
--dataset_name         Dataset name (required)
--output_dir           Output directory (required)
--class_name           Holdout class (default: Transcription)
--baselines            Specific baselines to run (optional)
--n_boot               Number of bootstrap samples (default: 1000)
--n_perm               Number of permutations (default: 10000)
--skip_comparisons     Skip baseline comparisons
```

## Best Practices

### 1. Bootstrap Sample Size

- **Testing**: Use `n_boot=100` for quick tests
- **Production**: Use `n_boot=1000` for stable CIs
- **Publication**: Use `n_boot=1000-5000` for high precision

### 2. Permutation Test Size

- **Testing**: Use `n_perm=1000` for quick tests
- **Production**: Use `n_perm=10000` for stable p-values
- **Publication**: Use `n_perm=10000-50000` for high precision

### 3. Reproducibility

Always set `--seed` (or `random_state`) for reproducible results:

```bash
--seed 42
```

### 4. Point Estimate Parity

Before using v2, verify parity with v1:

```bash
python -m goal_3_prediction.lsft.verify_parity \
    --adata_path <data> \
    --split_config <split> \
    --dataset_name <name> \
    --baseline_type <baseline> \
    --output_dir <output>
```

### 5. Multiple Baselines

For comparisons, run multiple baselines and use `compare_all_baseline_pairs`:

```bash
# Run each baseline
for baseline in lpm_selftrained lpm_scgptGeneEmb lpm_randomGeneEmb; do
    python -m goal_3_prediction.lsft.run_lsft_with_resampling \
        --baseline_type $baseline \
        ...
done

# Then compare
python -c "
from goal_3_prediction.lsft.compare_baselines_resampling import compare_all_baseline_pairs
import pandas as pd

results_df = pd.read_csv('results/lsft_resampling/combined.csv')
comparison_df = compare_all_baseline_pairs(results_df)
comparison_df.to_csv('results/lsft_resampling/comparisons.csv')
"
```

## Troubleshooting

### Issue: Import Errors

**Solution**: Ensure `PYTHONPATH=src` is set:
```bash
export PYTHONPATH=src
# Or prefix commands with PYTHONPATH=src
```

### Issue: Missing Dependencies

**Solution**: Install required packages:
```bash
pip install -r requirements.txt
pip install pyarrow  # For Parquet output
```

### Issue: Slow Bootstrap/Permutation

**Solution**: 
- Reduce `n_boot` and `n_perm` for testing
- Use full values only for final results
- Consider parallelization for production

### Issue: Parity Verification Fails

**Solution**:
- Check that same random seed used in v1 and v2
- Verify input data is identical
- Adjust tolerance if needed (default: 1e-6)

### Issue: No Baseline Comparisons

**Solution**: Ensure results from multiple baselines are available. Baseline comparisons require at least 2 baselines.

## Next Steps

1. **Run Verification**: `python verify_sprint11_implementation.py`
2. **Test on Small Dataset**: Use small subset for initial testing
3. **Verify Parity**: Ensure v1 and v2 match
4. **Run Full Evaluation**: Use full datasets for production
5. **Generate Visualizations**: Create plots with CIs

## Resources

- **Full Documentation**: `docs/resampling.md`
- **API Reference**: See docstrings in source files
- **Examples**: See `docs/resampling.md` for more examples
- **Status Report**: `docs/SPRINT_11_FINAL_STATUS.md`

---

**Ready to use!** Start with a small test dataset and scale up.


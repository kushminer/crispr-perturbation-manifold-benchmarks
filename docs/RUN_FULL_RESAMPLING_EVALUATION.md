# Running Full Resampling Evaluation

This guide explains how to run the complete LSFT and LOGO resampling evaluation on all datasets.

## Overview

The resampling evaluation includes:
1. **LSFT with Resampling** - Local Similarity-Filtered Training with bootstrap CIs and permutation tests
2. **LOGO with Resampling** - Functional Class Holdout with bootstrap CIs and baseline comparisons

## Prerequisites

### 1. Data Files Required

You need the following datasets:

- **Adamson**: `../paper/benchmark/data/gears_pert_data/adamson/perturb_processed.h5ad`
- **Replogle K562**: `/Users/samuelminer/Documents/classes/nih_research/data_replogle_k562_essential/perturb_processed.h5ad`
  - Or set `K562_DATA_PATH` environment variable
- **Replogle RPE1**: `/Users/samuelminer/Documents/classes/nih_research/data_replogle_rpe1_essential/perturb_processed.h5ad`
  - Or set `RPE1_DATA_PATH` environment variable

### 2. Split Configurations

Split files should be in:
- `results/goal_2_baselines/splits/adamson_split_seed1.json`
- `results/goal_2_baselines/splits/replogle_k562_essential_split_seed1.json`
- `results/goal_2_baselines/splits/replogle_rpe1_essential_split_seed1.json`

### 3. Annotation Files

Functional class annotations:
- `data/annotations/adamson_functional_classes_enriched.tsv`
- `data/annotations/replogle_k562_functional_classes_go.tsv`

## Running LSFT with Resampling

### Quick Start (All Datasets & Baselines)

```bash
cd crispr-perturbation-manifold-benchmarks
./run_lsft_resampling_all.sh
```

This will:
- Run LSFT with resampling for **8 baselines** × **3 datasets** = **24 evaluations**
- Generate bootstrap CIs (n_boot=1000)
- Perform permutation tests (n_perm=10000)
- Create baseline comparisons
- Generate hardness regressions with CIs

### Expected Runtime

- **Per baseline**: ~10-30 minutes (depending on dataset size)
- **Total**: ~4-12 hours for all baselines across all datasets

### Output Structure

```
results/goal_3_prediction/lsft_resampling/
├── adamson/
│   ├── lsft_adamson_lpm_selftrained_standardized.csv
│   ├── lsft_adamson_lpm_selftrained_standardized.jsonl
│   ├── lsft_adamson_lpm_selftrained_standardized.parquet
│   ├── lsft_adamson_lpm_selftrained_summary.json
│   ├── lsft_adamson_lpm_selftrained_hardness_regressions.csv
│   ├── lsft_adamson_baseline_comparisons.csv
│   └── lsft_adamson_*_run.log
├── k562/
│   └── ...
└── rpe1/
    └── ...
```

### Running Individual Baselines

For testing or focused runs:

```bash
PYTHONPATH=src python -m goal_3_prediction.lsft.run_lsft_with_resampling \
    --adata_path ../paper/benchmark/data/gears_pert_data/adamson/perturb_processed.h5ad \
    --split_config results/goal_2_baselines/splits/adamson_split_seed1.json \
    --dataset_name adamson \
    --baseline_type lpm_selftrained \
    --output_dir results/goal_3_prediction/lsft_resampling/adamson \
    --n_boot 1000 \
    --n_perm 10000
```

## Running LOGO with Resampling

### Quick Start (All Datasets)

```bash
cd crispr-perturbation-manifold-benchmarks
./run_logo_resampling_all.sh
```

This will:
- Run LOGO with resampling for **3 datasets**
- Use **Transcription** class as holdout
- Generate bootstrap CIs for all baselines
- Perform baseline comparisons with permutation tests

### Expected Runtime

- **Per dataset**: ~30-60 minutes (runs all 9 baselines)
- **Total**: ~1.5-3 hours for all datasets

### Output Structure

```
results/goal_3_prediction/functional_class_holdout_resampling/
├── adamson/
│   ├── logo_adamson_transcription_standardized.csv
│   ├── logo_adamson_transcription_standardized.jsonl
│   ├── logo_adamson_transcription_standardized.parquet
│   ├── logo_adamson_transcription_summary.json
│   ├── logo_adamson_transcription_baseline_comparisons.csv
│   └── logo_adamson_transcription_run.log
├── replogle_k562/
│   └── ...
└── replogle_rpe1/
    └── ...
```

### Running Individual Datasets

```bash
PYTHONPATH=src python -m goal_3_prediction.functional_class_holdout.logo_resampling \
    --adata_path ../paper/benchmark/data/gears_pert_data/adamson/perturb_processed.h5ad \
    --annotation_path data/annotations/adamson_functional_classes_enriched.tsv \
    --dataset_name adamson \
    --output_dir results/goal_3_prediction/functional_class_holdout_resampling/adamson \
    --class_name Transcription \
    --n_boot 1000 \
    --n_perm 10000
```

## Parameter Configuration

### Quick Testing (Faster)

Edit the scripts to use:
```bash
N_BOOT=100      # Reduced bootstrap samples
N_PERM=1000     # Reduced permutations
```

### Production (Accurate)

Use defaults:
```bash
N_BOOT=1000     # Stable CIs
N_PERM=10000    # Stable p-values
```

### Publication (High Precision)

Edit to use:
```bash
N_BOOT=5000     # High-precision CIs
N_PERM=50000    # High-precision p-values
```

## Monitoring Progress

### Check Running Jobs

```bash
# Check active Python processes
ps aux | grep "run_lsft_with_resampling\|logo_resampling"

# Monitor output directories
watch -n 5 'ls -lh results/goal_3_prediction/lsft_resampling/*/ | tail -20'
```

### Check Logs

Each run generates a log file:
```bash
# LSFT logs
tail -f results/goal_3_prediction/lsft_resampling/adamson/lsft_adamson_lpm_selftrained_run.log

# LOGO logs
tail -f results/goal_3_prediction/functional_class_holdout_resampling/adamson/logo_adamson_transcription_run.log
```

## Generating Visualizations

After running evaluations, generate visualizations:

```python
from goal_3_prediction.lsft.visualize_resampling import create_all_lsft_visualizations_with_ci
import pandas as pd
from pathlib import Path

# Load results
results_df = pd.read_csv("results/goal_3_prediction/lsft_resampling/adamson/lsft_adamson_standardized.csv")

# Create visualizations
create_all_lsft_visualizations_with_ci(
    results_df=results_df,
    summary_path=Path("results/goal_3_prediction/lsft_resampling/adamson/lsft_adamson_summary.json"),
    regression_results_path=Path("results/goal_3_prediction/lsft_resampling/adamson/lsft_adamson_hardness_regressions.csv"),
    comparison_results_path=Path("results/goal_3_prediction/lsft_resampling/adamson/lsft_adamson_baseline_comparisons.csv"),
    output_dir=Path("results/goal_3_prediction/lsft_resampling/adamson/plots/"),
    dataset_name="adamson",
)
```

## Expected Outputs

### LSFT Resampling Outputs

For each baseline:
1. **Standardized Results** (CSV, JSONL, Parquet)
   - Per-perturbation metrics with standardized fields
   - Fields: `pearson_r`, `l2`, `hardness`, `embedding_similarity`, `split_fraction`

2. **Summary with CIs** (JSON)
   - Mean Pearson r and L2 with bootstrap CIs
   - Grouped by baseline and top_pct

3. **Baseline Comparisons** (CSV, JSON)
   - Paired comparisons with permutation p-values
   - Bootstrap CIs on mean deltas

4. **Hardness Regressions** (CSV, JSON)
   - Regression statistics with bootstrapped CIs
   - Slope, r, R² with confidence intervals

### LOGO Resampling Outputs

For each dataset:
1. **Standardized Results** (CSV, JSONL, Parquet)
   - Per-perturbation metrics for all baselines

2. **Summary with CIs** (JSON)
   - Mean Pearson r and L2 with bootstrap CIs per baseline

3. **Baseline Comparisons** (CSV, JSON)
   - Paired comparisons with permutation p-values

## Troubleshooting

### Issue: Out of Memory

**Solution**: Run baselines sequentially or reduce dataset size:
```bash
# Process one baseline at a time
for baseline in lpm_selftrained lpm_scgptGeneEmb; do
    ./run_lsft_resampling_all.sh  # Edit script to run only one baseline
done
```

### Issue: Slow Bootstrap/Permutation

**Solution**: 
- Reduce `n_boot` and `n_perm` for testing
- Use full values only for final results
- Consider running on a cluster for parallelization

### Issue: Missing Data Files

**Solution**: 
- Check paths in scripts match your setup
- Set environment variables: `K562_DATA_PATH`, `RPE1_DATA_PATH`
- Download missing datasets using GEARS API

### Issue: Missing Split Files

**Solution**: Generate splits first:
```bash
# Generate splits (if not already done)
PYTHONPATH=src python -m goal_2_baselines.run_all_datasets \
    --datasets adamson k562 rpe1
```

## Next Steps After Running

1. **Review Results**: Check summary JSON files for CIs
2. **Generate Visualizations**: Use visualization scripts
3. **Analyze Comparisons**: Review baseline comparison CSV files
4. **Check Parity**: Verify v1 vs v2 point estimates match
5. **Prepare for Publication**: Use results in tables/figures

---

**Ready to run!** Start with a single baseline/dataset to test, then scale up.


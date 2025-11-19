# Sprint 11 Next Steps

**Status**: ✅ All code implementation complete

## Completed Work

All 12 issues have been implemented and verified:
- ✅ Issues 1-2: Repository setup and CI (prepared/complete)
- ✅ Issues 3-4: Statistics modules (bootstrap, permutation)
- ✅ Issues 5-8: LSFT resampling integration
- ✅ Issue 9: LOGO resampling
- ✅ Issue 10: Enhanced visualizations
- ✅ Issue 11: Parity verification
- ✅ Issue 12: Documentation

## Immediate Next Steps

### 1. Verify Implementation (✅ Done)

Run the verification script:
```bash
cd evaluation_framework
PYTHONPATH=src python verify_sprint11_implementation.py
```

This verifies all modules can be imported and basic functions work.

### 2. Create GitHub Repository (User Action Required)

Follow the instructions in `REPOSITORY_SETUP_INSTRUCTIONS.md`:

**Option A: New Repository (Recommended)**
```bash
# 1. Create repository on GitHub (name: perturbench-resampling)
# 2. Clone locally
git clone <your-new-repo-url>
cd perturbench-resampling

# 3. Copy files from evaluation_framework
cp -r ../linear_perturbation_prediction-Paper/evaluation_framework/* .
cp -r ../linear_perturbation_prediction-Paper/evaluation_framework/.* . 2>/dev/null || true

# 4. Update README
mv V2_RESAMPLING_README.md README.md

# 5. Initial commit
git add .
git commit -m "Initial commit: v1 baseline (Sprint 11 - before resampling enhancements)"
git push -u origin main
```

### 3. Run Integration Tests

Once the repository is created:

**A. Test on Small Dataset**
```bash
# Use a small subset of data for initial testing
PYTHONPATH=src python -m goal_3_prediction.lsft.run_lsft_with_resampling \
    --adata_path <small_test_data.h5ad> \
    --split_config <test_split.json> \
    --dataset_name test \
    --baseline_type lpm_selftrained \
    --output_dir results/test_lsft_resampling/ \
    --n_boot 100 \
    --n_perm 1000
```

**B. Verify Parity**
```bash
PYTHONPATH=src python -m goal_3_prediction.lsft.verify_parity \
    --adata_path <test_data.h5ad> \
    --split_config <test_split.json> \
    --dataset_name test \
    --baseline_type lpm_selftrained \
    --output_dir results/parity_test/
```

**C. Test LOGO Resampling**
```bash
PYTHONPATH=src python -m goal_3_prediction.functional_class_holdout.logo_resampling \
    --adata_path <test_data.h5ad> \
    --annotation_path <annotations.tsv> \
    --dataset_name test \
    --output_dir results/test_logo_resampling/ \
    --class_name Transcription \
    --n_boot 100 \
    --n_perm 1000
```

### 4. Run Full Evaluation

After verification:

**A. LSFT with Resampling (All Baselines)**
```bash
# Run for each baseline
for baseline in lpm_selftrained lpm_randomGeneEmb lpm_scgptGeneEmb; do
    PYTHONPATH=src python -m goal_3_prediction.lsft.run_lsft_with_resampling \
        --adata_path data/adamson_processed.h5ad \
        --split_config results/splits/adamson_split_seed1.json \
        --dataset_name adamson \
        --baseline_type $baseline \
        --output_dir results/lsft_resampling/ \
        --n_boot 1000 \
        --n_perm 10000
done
```

**B. Generate Visualizations**
```python
from goal_3_prediction.lsft.visualize_resampling import create_all_lsft_visualizations_with_ci
import pandas as pd
from pathlib import Path

results_df = pd.read_csv("results/lsft_resampling/lsft_adamson_standardized.csv")
summary_path = Path("results/lsft_resampling/lsft_adamson_summary.json")
regression_path = Path("results/lsft_resampling/lsft_adamson_hardness_regressions.csv")
comparison_path = Path("results/lsft_resampling/lsft_adamson_baseline_comparisons.csv")

create_all_lsft_visualizations_with_ci(
    results_df=results_df,
    summary_path=summary_path,
    regression_results_path=regression_path,
    comparison_results_path=comparison_path,
    output_dir=Path("results/lsft_resampling/plots/"),
    dataset_name="adamson",
)
```

### 5. Update Main README

Once everything works, update the main README to:
- Point to resampling features
- Link to `docs/resampling.md`
- Note v1 vs v2 distinction

### 6. Publication Preparation

For manuscript/poster:
1. Run resampling on all datasets
2. Generate all visualizations with CIs
3. Include bootstrap CIs in tables
4. Report permutation test p-values for comparisons

## Troubleshooting

### Issue: Import errors
**Solution**: Ensure `PYTHONPATH=src` is set when running scripts.

### Issue: Missing dependencies
**Solution**: Install optional dependencies:
```bash
pip install pyarrow  # For Parquet output
```

### Issue: Parity verification fails
**Solution**: 
- Ensure same random seed used in v1 and v2
- Check tolerance settings (default: 1e-6)
- Verify input data is identical

### Issue: Bootstrap/permutation takes too long
**Solution**: 
- Reduce `n_boot` and `n_perm` for testing (e.g., 100/1000)
- Use full values (1000/10000) for final results
- Consider parallelization for production

## Files to Review

Before production use, review:
1. `docs/resampling.md` - Full API documentation
2. `docs/SPRINT_11_COMPLETION_SUMMARY.md` - Complete feature list
3. `CHANGELOG.md` - Version history
4. `verify_sprint11_implementation.py` - Verification script

## Questions?

- Check `docs/resampling.md` for detailed API reference
- See `docs/SPRINT_11_ISSUES_5-8_SUMMARY.md` for implementation details
- Review example usage in `docs/resampling.md`

---

**Ready for production once repository is created and integration tests pass!**


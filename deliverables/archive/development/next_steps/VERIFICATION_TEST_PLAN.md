# Verification Test Plan

**Date:** 2025-11-24

---

## Purpose

Verify that all fixes are working correctly:
1. GEARS/Cross-dataset baseline handling
2. Epic 3 noise injection
3. Summary report generation

---

## Test 1: GEARS Baseline Fix Verification

### Goal
Verify that GEARS baseline now produces results instead of empty files.

### Test Command
```bash
cd lpm-evaluation-framework-v2
python -m goal_3_prediction.lsft.curvature_sweep \
  --adata_path ../paper/benchmark/data/gears_pert_data/adamson/perturb_processed.h5ad \
  --split_config results/goal_2_baselines/splits/adamson_split_seed1.json \
  --dataset_name adamson \
  --baseline_type lpm_gearsPertEmb \
  --output_dir results/manifold_law_diagnostics/epic1_curvature \
  --k_list 5 10 20 \
  --pca_dim 10 \
  --ridge_penalty 0.1 \
  --seed 1
```

### Expected Results
- Output CSV should have > 1 line (header + data rows)
- Should see warnings for perturbations not in GEARS vocabulary (expected)
- Should successfully process perturbations that ARE in GEARS vocabulary

### Success Criteria
- CSV file has multiple rows of results
- Log shows successful processing (may skip some perturbations)
- No critical errors about `B_test_local` being None

---

## Test 2: Epic 3 Noise Injection Verification

### Goal
Verify that Epic 3 actually injects noise and runs evaluations at all noise levels.

### Test Command
```bash
cd lpm-evaluation-framework-v2
python -m goal_3_prediction.lsft.run_epic3_noise_injection \
  --adata_path ../paper/benchmark/data/gears_pert_data/adamson/perturb_processed.h5ad \
  --split_config results/goal_2_baselines/splits/adamson_split_seed1.json \
  --dataset_name adamson \
  --baseline_type lpm_selftrained \
  --output_dir results/manifold_law_diagnostics/epic3_noise_injection \
  --k_list 5 10 \
  --noise_levels 0.01 0.05 0.1 \
  --noise_type gaussian \
  --noise_target embedding \
  --pca_dim 10 \
  --seed 1
```

### Expected Results
- Baseline (noise=0) results generated
- Noisy results at each noise level (0.01, 0.05, 0.1) with actual values
- Output CSV should have mean_r and mean_l2 filled in (not NaN)

### Success Criteria
- All noise levels have results (not NaN)
- Log shows "Running noise level: X" for each level
- Output CSV has entries for baseline + all noise levels

---

## Test 3: Summary Report Generation

### Goal
Verify that summary reports load actual CSV files and generate comprehensive summaries.

### Test Command
```bash
cd lpm-evaluation-framework-v2
python3 generate_diagnostic_summary.py
```

### Expected Results
- Executive summary generated with actual data
- Detailed epic summaries created
- Statistics show meaningful values

### Success Criteria
- No errors loading CSV files
- Summary shows data from actual results
- Detailed summaries are comprehensive

---

## Test 4: Cross-Dataset Baseline Verification

### Goal
Verify that K562/RPE1 cross-dataset baselines work similarly to GEARS fix.

### Test Command
```bash
cd lpm-evaluation-framework-v2
python -m goal_3_prediction.lsft.curvature_sweep \
  --adata_path ../paper/benchmark/data/gears_pert_data/adamson/perturb_processed.h5ad \
  --split_config results/goal_2_baselines/splits/adamson_split_seed1.json \
  --dataset_name adamson \
  --baseline_type lpm_k562PertEmb \
  --output_dir results/manifold_law_diagnostics/epic1_curvature \
  --k_list 5 10 \
  --pca_dim 10 \
  --ridge_penalty 0.1 \
  --seed 1
```

### Expected Results
- Should work the same way as GEARS
- May have fewer results if source dataset doesn't have all perturbations
- No critical errors

---

## Quick Verification Script

Create a simple script to check current state:

```bash
#!/bin/bash
# Quick verification of fixes

echo "=== Checking GEARS Results ==="
for file in results/manifold_law_diagnostics/epic1_curvature/*gearsPertEmb*.csv; do
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file")
        if [ "$lines" -gt 1 ]; then
            echo "✅ $file: $lines lines"
        else
            echo "❌ $file: EMPTY ($lines lines)"
        fi
    fi
done

echo ""
echo "=== Checking Epic 3 Noise Injection ==="
for file in results/manifold_law_diagnostics/epic3_noise_injection/noise_injection_*.csv; do
    if [ -f "$file" ]; then
        nan_count=$(grep -c ",," "$file" || echo "0")
        if [ "$nan_count" -eq 0 ]; then
            echo "✅ $file: All values filled"
        else
            echo "⚠️  $file: $nan_count NaN entries"
        fi
    fi
done
```

---

## Next Steps After Verification

1. If tests pass: Proceed with full diagnostic suite run
2. If tests fail: Review error logs and debug
3. Generate comprehensive summaries once all epics complete


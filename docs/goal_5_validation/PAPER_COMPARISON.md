# Paper Comparison Framework

**Date:** 2025-11-17  
**Purpose:** Compare reproduced baseline results with published Nature paper results

---

## Comparison Methodology

This document provides a framework for comparing our reproduced baseline results with the published Nature paper results.

### Metrics for Comparison

1. **Pearson r** (correlation coefficient)
   - Primary metric in paper
   - Expected agreement: ±0.01

2. **L2** (Euclidean distance)
   - Additional metric we compute
   - May not be in paper, but useful for validation

---

## Our Results Summary

### Adamson Dataset (seed=1, 12 test perturbations)

| Baseline | Mean Pearson r | Mean L2 |
|----------|---------------|---------|
| lpm_selftrained | 0.946 | 2.265 |
| lpm_rpe1PertEmb | 0.937 | 1.943 |
| lpm_k562PertEmb | 0.929 | 2.413 |
| lpm_scgptGeneEmb | 0.811 | 3.733 |
| lpm_scFoundationGeneEmb | 0.777 | 3.979 |
| lpm_gearsPertEmb | 0.748 | 4.307 |
| lpm_randomGeneEmb | 0.721 | 4.344 |
| mean_response | 0.720 | 4.350 |
| lpm_randomPertEmb | 0.707 | 4.536 |

### Replogle K562 Essential (seed=1, 163 test perturbations)

| Baseline | Mean Pearson r | Mean L2 |
|----------|---------------|---------|
| lpm_selftrained | 0.665 | 4.069 |
| lpm_k562PertEmb | 0.653 | 4.139 |
| lpm_rpe1PertEmb | 0.628 | 3.305 |
| lpm_scgptGeneEmb | 0.513 | 4.833 |
| lpm_scFoundationGeneEmb | 0.418 | 5.358 |
| lpm_gearsPertEmb | 0.431 | 5.429 |
| lpm_randomGeneEmb | 0.375 | 5.601 |
| mean_response | 0.375 | 5.603 |
| lpm_randomPertEmb | 0.371 | 5.620 |

### Replogle RPE1 Essential (seed=1, 231 test perturbations)

| Baseline | Mean Pearson r | Mean L2 |
|----------|---------------|---------|
| lpm_selftrained | 0.764 | 4.726 |
| lpm_rpe1PertEmb | 0.758 | 4.815 |
| lpm_k562PertEmb | 0.737 | 4.030 |
| lpm_scgptGeneEmb | 0.664 | 5.831 |
| lpm_scFoundationGeneEmb | 0.637 | 6.635 |
| lpm_gearsPertEmb | 0.631 | 6.876 |
| lpm_randomGeneEmb | 0.633 | 6.999 |
| mean_response | 0.633 | 7.003 |
| lpm_randomPertEmb | 0.632 | 7.021 |

---

## Expected Paper Results

**Note:** Replace this section with actual paper results when available.

### Adamson Dataset

| Baseline | Paper Pearson r | Our Pearson r | Difference | Status |
|----------|----------------|---------------|------------|--------|
| lpm_selftrained | TBD | 0.946 | TBD | ⏳ |
| lpm_k562PertEmb | TBD | 0.929 | TBD | ⏳ |
| lpm_rpe1PertEmb | TBD | 0.937 | TBD | ⏳ |
| ... | ... | ... | ... | ... |

### Replogle K562 Essential

| Baseline | Paper Pearson r | Our Pearson r | Difference | Status |
|----------|----------------|---------------|------------|--------|
| lpm_selftrained | TBD | 0.665 | TBD | ⏳ |
| lpm_k562PertEmb | TBD | 0.653 | TBD | ⏳ |
| ... | ... | ... | ... | ... |

### Replogle RPE1 Essential

| Baseline | Paper Pearson r | Our Pearson r | Difference | Status |
|----------|----------------|---------------|------------|--------|
| lpm_selftrained | TBD | 0.764 | TBD | ⏳ |
| lpm_rpe1PertEmb | TBD | 0.758 | TBD | ⏳ |
| ... | ... | ... | ... | ... |

---

## Comparison Criteria

### Agreement Thresholds

- **Excellent Agreement**: |difference| ≤ 0.005
- **Good Agreement**: |difference| ≤ 0.01
- **Acceptable Agreement**: |difference| ≤ 0.02
- **Needs Investigation**: |difference| > 0.02

### Factors Affecting Comparison

1. **Random Seed**: Paper may use different seed
2. **Split Logic**: Slight differences in train/test/val splits
3. **Data Preprocessing**: Pseudobulk computation differences
4. **Numerical Precision**: Floating-point differences
5. **Implementation Details**: Minor algorithmic differences

---

## Validation Checklist

- [ ] Load paper results (from supplementary materials or R output)
- [ ] Align baseline names (paper naming vs our naming)
- [ ] Compare on same datasets (Adamson, K562, RPE1)
- [ ] Check split consistency (same train/test/val)
- [ ] Verify metric computation (Pearson r formula)
- [ ] Document differences > 0.01
- [ ] Investigate large discrepancies

---

## Comparison Script

Use the validation script to compare:

```bash
python -m goal_2_baselines.validate_against_r \
    --python_results results/baselines/adamson_reproduced/baseline_results_reproduced.csv \
    --r_results path/to/paper/results.csv \
    --output_dir validation/paper_comparison \
    --tolerance 0.01
```

---

## Notes

- Our implementation uses seed=1 for all splits
- Paper may use different seeds or split logic
- Some baselines may not be in paper (e.g., mean_response)
- Cross-dataset baselines (k562PertEmb, rpe1PertEmb) are key for validation

---

**Status:** ⏳ **Awaiting Paper Results**  
**Last Updated:** 2025-11-17


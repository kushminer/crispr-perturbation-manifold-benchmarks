# Manifold Law Diagnostic Suite - Current Status

**Date:** 2025-11-24  
**Status:** ⚠️ **PARTIALLY COMPLETE - FIXES APPLIED**

---

## Current Status Summary

| Epic | Status | Data Available | Notes |
|------|--------|----------------|-------|
| **Epic 1: Curvature** | ✅ Complete | 8/8 baselines | Peak r and curvature computed |
| **Epic 2: Ablation** | ❌ Incomplete | 0/8 baselines | Placeholder script used - needs re-run |
| **Epic 3: Lipschitz** | ⚠️ Partial | 4/8 baselines | Only selftrained, gears, k562, rpe1 have data |
| **Epic 4: Flip Rate** | ✅ Complete | 8/8 baselines | Direction-flip probe complete |
| **Epic 5: Tangent** | ✅ Complete | 8/8 baselines | TAS computed for all |

---

## Cross-Epic Summary (Fixed)

| Baseline | Epic 1 Peak r | Epic 3 Lipschitz | Epic 4 Flip Rate | Epic 5 TAS |
|----------|---------------|------------------|------------------|------------|
| **lpm_selftrained** | **0.944** 🏆 | **0.145** 🏆 | 0.0005 | -0.005 |
| lpm_randomGeneEmb | 0.934 | N/A | 0.0005 | 0.002 |
| lpm_randomPertEmb | 0.694 | N/A | 0.0134 | -0.024 |
| lpm_scgptGeneEmb | 0.936 | N/A | 0.0005 | -0.012 |
| lpm_scFoundationGeneEmb | 0.935 | N/A | 0.0005 | 0.016 |
| lpm_gearsPertEmb | 0.790 | 2.028 | 0.0149 | **0.052** 🏆 |
| lpm_k562PertEmb | 0.932 | 1.163 | 0.0001 | -0.003 |
| lpm_rpe1PertEmb | 0.934 | 1.574 | **0.0001** 🏆 | -0.020 |

---

## Fixes Applied

### ✅ Fix 1: Report Aggregation (Complete)
- Fixed baseline naming (removed dataset prefix)
- Fixed column lookups in cross-epic summary
- Regenerated `baseline_summary_fixed.csv`
- Regenerated `5epic_winner_grid_fixed.png`

### ✅ Fix 2: Epic 3 Lipschitz Aggregation (Complete)
- Computed Lipschitz from noise injection files
- Used k-sweep baseline for files missing 0.0 noise level
- Created `lipschitz_summary_complete.csv`

### ⏳ Fix 3: Epic 2 Re-run (Pending)
- Created `fix_and_rerun_epics.sh` script
- Ready to run with correct `mechanism_ablation.py`
- **Requires manual execution** (takes ~30-60 minutes)

---

## Outstanding Issues

### Epic 2: Ablation Not Run ❌
**Root Cause:** Wrong script used (`run_epic2_mechanism_ablation.py` has placeholders)

**To Fix:** Run the correct script:
```bash
cd /Users/samuelminer/Documents/classes/nih_research/linear_perturbation_prediction-Paper/lpm-evaluation-framework-v2
./fix_and_rerun_epics.sh
```

**Time Estimate:** 30-60 minutes for all baselines × datasets

### Epic 3: Incomplete Data ⚠️
**Root Cause:** Some noise injection files have placeholder values (empty mean_r)

**To Fix:** Re-run noise injection for:
- lpm_randomGeneEmb
- lpm_randomPertEmb
- lpm_scgptGeneEmb
- lpm_scFoundationGeneEmb

---

## Key Findings (with available data)

### Winners by Epic:
1. **Epic 1 (Curvature):** `lpm_selftrained` - Highest peak r (0.944)
2. **Epic 3 (Lipschitz):** `lpm_selftrained` - Lowest Lipschitz (0.145) = most robust
3. **Epic 4 (Flip Rate):** `lpm_rpe1PertEmb` - Lowest flip rate (0.0001)
4. **Epic 5 (TAS):** `lpm_gearsPertEmb` - Highest TAS (0.052)

### Interpretation:
- **Self-trained PCA** dominates on curvature and noise robustness
- **Cross-dataset embeddings** (k562, rpe1) have lowest flip rates
- **GEARS** has highest tangent alignment but poor curvature

---

## Files Generated

- `publication_package/final_tables/baseline_summary_fixed.csv`
- `publication_package/poster_figures/5epic_winner_grid_fixed.png`
- `results/manifold_law_diagnostics/epic3_noise_injection/lipschitz_summary_complete.csv`
- `fix_and_rerun_epics.sh` (ready to run for Epic 2)

---

## Next Steps

1. **Optional:** Run `./fix_and_rerun_epics.sh` to complete Epic 2
2. **Optional:** Re-run noise injection for remaining baselines
3. After fixes, run `python publication_package/fix_aggregation_and_regenerate.py` again


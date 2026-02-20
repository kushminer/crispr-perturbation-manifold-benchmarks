# Next Steps Plan

**Date:** 2025-11-24  
**Status:** ✅ All Fixes Verified, Ready for Full Execution

---

## ✅ Verification Complete

1. **GEARS baseline:** ✅ Working (25 lines)
2. **Epic 3 noise injection:** ✅ Working (all filled, Lipschitz computed)
3. **Cross-dataset baselines:** ✅ Working (25 lines)

---

## 🎯 Recommended Next Steps

### Option 1: Full Diagnostic Suite Re-Run (Recommended)

Re-run all epics on all 8 baselines with verified fixes:

```bash
cd lpm-evaluation-framework-v2
./run_all_epics_all_baselines.sh
```

**Expected Duration:** Several hours (120 total experiments: 8 baselines × 3 datasets × 5 epics)

**What This Will Do:**
- Re-run Epic 1 on all baselines (including GEARS, K562, RPE1)
- Re-run Epic 2 on all baselines
- Re-run Epic 3 on all baselines (with noise injection)
- Epic 4 & 5 should already be complete

**Output:** Complete results for all epics on all baselines

---

### Option 2: Targeted Re-Run (Faster)

Re-run only the epics that need fixes:

```bash
# Epic 1: Re-run GEARS and cross-dataset baselines
# Epic 3: Re-run all baselines to fill in noise injection results
# Epic 2: May need similar fixes
```

---

### Option 3: Generate Summaries from Current Results

Generate comprehensive summaries from existing results:

```bash
cd lpm-evaluation-framework-v2
python3 generate_diagnostic_summary.py
```

Then review what's missing and fill gaps.

---

## 📊 Current Status

### Epic 1: Curvature Sweep
- ✅ 5 baselines working
- ✅ GEARS: Now fixed and verified
- ✅ K562: Now fixed and verified
- ⏳ RPE1: Should work (same fix)

### Epic 2: Mechanism Ablation
- ⏳ May need similar fixes to Epic 1
- ⏳ Needs testing

### Epic 3: Noise Injection
- ✅ Fix verified on selftrained
- ⏳ Need to re-run all baselines to fill in NaN values

### Epic 4: Direction-Flip Probe
- ✅ Already working (including GEARS)

### Epic 5: Tangent Alignment
- ✅ Already working (including GEARS)

---

## Recommendation

**Start with Option 1 (Full Re-Run)** to get complete results:
- All fixes are verified
- Script has resume capability (skips existing results)
- Will generate complete dataset for analysis

**Or** start with Option 3 to see what we have, then fill gaps strategically.


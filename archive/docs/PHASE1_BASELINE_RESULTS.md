# Phase 1 - Baseline Reproduction Results

**Date:** 2025-11-17  
**Status:** ✅ **ALL BASELINES COMPLETE**

---

## Summary

Successfully reproduced all 9 baseline models (8 linear + mean-response) on the Adamson dataset with seed=1.

---

## Baseline Results (Adamson, seed=1)

| Baseline | Mean Pearson r | Mean L2 | Test Perturbations | Status |
|----------|---------------|---------|-------------------|--------|
| **lpm_selftrained** | **0.946** | **2.265** | 12 | ✅ |
| **lpm_rpe1PertEmb** | **0.937** | **1.943** | 12 | ✅ |
| **lpm_k562PertEmb** | **0.929** | **2.413** | 12 | ✅ |
| **lpm_scgptGeneEmb** | **0.811** | **3.733** | 12 | ✅ |
| **lpm_scFoundationGeneEmb** | **0.777** | **3.979** | 12 | ✅ |
| **lpm_gearsPertEmb** | **0.748** | **4.307** | 12 | ✅ |
| **lpm_randomGeneEmb** | **0.721** | **4.344** | 12 | ✅ |
| **mean_response** | **0.720** | **4.350** | 12 | ✅ |
| **lpm_randomPertEmb** | **0.707** | **4.536** | 12 | ✅ |

**Note:** L2 (Euclidean distance) - lower is better. L2 = sqrt(sum((y_true - y_pred)²))

---

## Implementation Details

### ✅ Completed Features

1. **Split Logic**
   - Fixed condition matching (exact "ctrl" vs containing "ctrl")
   - Proper train/test/val split (61/12/14 for Adamson)

2. **Gene Alignment**
   - Ensembl ID → Gene Symbol mapping for scGPT/scFoundation
   - Common gene alignment for cross-dataset baselines
   - Handles missing genes with zero padding

3. **Cross-Dataset Embeddings**
   - K562: Direct h5ad path loading
   - RPE1: Direct h5ad path loading (from `/Users/samuelminer/Documents/classes/nih_research/data_replogle_rpe1_essential`)
   - Proper PCA projection with gene alignment

4. **GEARS Alignment**
   - Perturbation name cleaning ("+ctrl" removal)
   - Gene symbol matching
   - Zero padding for missing perturbations

5. **Path Resolution**
   - Relative path resolution for checkpoints
   - Proper handling of data directories

6. **Mean-Response Baseline**
   - Simple baseline predicting mean expression
   - Integrated into default baseline list

---

## Files Generated

- `results/baselines/adamson_reproduced/baseline_results_reproduced.csv` - Complete results

---

## Configuration

**Parameters:**
- PCA dimension: 10
- Ridge penalty: 0.1
- Random seed: 1
- Train/Test/Val split: 61/12/14 perturbations

**Data:**
- Dataset: Adamson
- Total conditions: 87
- Test perturbations: 12

---

## Next Steps

1. ✅ **All Baselines Working** - Complete
2. ⏳ **Validate Against R** - Compare results with original R implementation
3. ⏳ **Documentation** - Document baseline reproduction process
4. ⏳ **Reproducibility** - Create reproducibility report

---

## Key Achievements

✅ **9/9 baselines working**  
✅ **Cross-dataset embeddings functional** (K562, RPE1)  
✅ **Pretrained embeddings integrated** (scGPT, scFoundation, GEARS)  
✅ **Mean-response baseline added**  
✅ **Proper gene/perturbation alignment**  

---

**Status:** ✅ **Baseline Reproduction Complete**  
**Ready for:** Validation against R implementation


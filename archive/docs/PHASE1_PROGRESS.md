# Phase 1 – Baseline Re-runs Progress

**Date:** 2025-11-14  
**Status:** 🔄 **IN PROGRESS**

---

## Issue 1.1 — Reproduce Baseline Models with Original Splits

### ✅ Completed

1. **Baseline Runner Structure Created**
   - `src/baselines/__init__.py` - Module initialization
   - `src/baselines/baseline_types.py` - Baseline type definitions and configs
   - `src/baselines/baseline_runner.py` - Core runner implementation
   - `src/baselines/run_all.py` - CLI entry point

2. **Core Functionality Implemented**
   - ✅ Y matrix construction (pseudobulk expression changes)
   - ✅ Split configuration loading
   - ✅ Gene embedding construction (training_data, random, scGPT, scFoundation)
   - ✅ Perturbation embedding construction (training_data, random, GEARS, K562, RPE1)
   - ✅ K matrix solving via ridge regression
   - ✅ Baseline configuration system

3. **All 8 Baseline Types Defined**
   - ✅ lpm_selftrained
   - ✅ lpm_randomPertEmb
   - ✅ lpm_randomGeneEmb
   - ✅ lpm_scgptGeneEmb
   - ✅ lpm_scFoundationGeneEmb
   - ✅ lpm_gearsPertEmb
   - ✅ lpm_k562PertEmb
   - ✅ lpm_rpe1PertEmb

### ✅ Recently Completed

1. **Split Logic Porting** ✅
   - Created `src/baselines/split_logic.py`
   - Implemented simple random split (for Adamson/Replogle)
   - Implemented norman-specific split logic
   - Optional GEARS integration if available
   - Can load existing split files

2. **Test Set Prediction Handling** ✅
   - Updated `construct_pert_embeddings()` to return test embeddings
   - For training_data PCA: uses fitted PCA to transform test data
   - For random: generates random test embeddings
   - Properly computes Y_pred = A @ K @ B_test + center
   - Handles cases where test embeddings unavailable

3. **Mean-Response Baseline** ✅
   - Implemented `run_mean_response_baseline()`
   - Always predicts mean expression across training perturbations
   - Integrated into `run_all_baselines()`

### ✅ Recently Completed (Precomputation)

4. **Cross-Dataset Embedding Precomputation** ✅
   - ✅ Precomputed K562 perturbation embeddings
     - File: `results/replogle_k562_pert_emb_pca10_seed1.tsv`
     - Dimensions: 10 × 1,093 perturbations
     - ✅ **VALIDATED** - Perfect parity with original script (1.00000000 cosine, 0.00e+00 diff)
   - ✅ Precomputed RPE1 perturbation embeddings
     - File: `results/replogle_rpe1_pert_emb_pca10_seed1.tsv`
     - Dimensions: 10 × 1,544 perturbations
     - ⏳ Parity validation pending (same validated loader as K562)
   - ✅ Created precomputation and validation scripts
   - ✅ Integrated into baseline configuration

### 🔄 Remaining

5. **Validation and Comparison**
   - Run all baselines on Adamson dataset
   - Compare results with R implementation
   - Document reproducibility

---

## Implementation Details

### Current Architecture

```
src/baselines/
├── __init__.py              # Module exports
├── baseline_types.py        # BaselineType enum, BaselineConfig, get_baseline_config()
├── baseline_runner.py       # Core runner: run_all_baselines(), run_single_baseline()
└── run_all.py              # CLI entry point
```

### Key Functions

1. **`compute_pseudobulk_expression_changes()`**
   - Computes Y matrix (genes × perturbations)
   - Same for all 8 baselines
   - Formula: `Y_{i,j} = mean(gene i in pert j) - mean(gene i in ctrl)`

2. **`construct_gene_embeddings()`**
   - Handles: training_data (PCA), random, scGPT, scFoundation
   - Returns: A matrix (genes × d)

3. **`construct_pert_embeddings()`**
   - Handles: training_data (PCA), random, GEARS, K562, RPE1
   - Returns: B matrix (d × perturbations)

4. **`run_single_baseline()`**
   - Constructs A and B
   - Solves for K via ridge regression
   - Makes predictions (needs completion for test set)
   - Computes metrics

5. **`run_all_baselines()`**
   - Orchestrates running all baselines
   - Saves results to CSV

---

## Next Steps

### Immediate (High Priority)

1. **Complete Test Set Prediction**
   - Implement proper test perturbation embedding construction
   - Handle alignment between test perturbations and embedding labels
   - Update `run_single_baseline()` to make real predictions

2. **Port Split Logic**
   - Port `prepare_perturbation_data.py` split logic to Python
   - Ensure it matches R implementation exactly
   - Test on Adamson dataset

3. **Precompute Cross-Dataset Embeddings**
   - Run PCA on K562 dataset → save TSV
   - Run PCA on RPE1 dataset → save TSV

### Short Term

4. **Implement Mean-Response Baseline**
   - Simple baseline: always predict mean expression
   - Useful for comparison

5. **Run Full Pipeline**
   - Run all 8 baselines on Adamson
   - Save results
   - Compare with R implementation

6. **Documentation**
   - Write reproducibility note
   - Document any discrepancies

---

## Usage

Once complete, run baselines with:

```bash
python -m baselines.run_all \
    --adata_path data/gears_pert_data/adamson/perturb_processed.h5ad \
    --split_config results/split_config.json \
    --output_dir results/baselines \
    --pca_dim 10 \
    --ridge_penalty 0.1 \
    --seed 1
```

---

**Last Updated:** 2025-11-14


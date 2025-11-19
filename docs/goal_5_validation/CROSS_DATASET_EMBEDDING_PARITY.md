# Cross-Dataset Embedding Parity Validation

**Date:** 2025-11-14  
**Status:** ✅ **K562 VALIDATED**, ⏳ **RPE1 PRECOMPUTED**

---

## Summary

Cross-dataset perturbation embeddings (K562 and RPE1) have been precomputed for use in cross-dataset baseline models (`lpm_k562PertEmb`, `lpm_rpe1PertEmb`).

### K562 Embeddings ✅ **VALIDATED**

- **Precomputed:** `results/replogle_k562_pert_emb_pca10_seed1.tsv`
- **Source Data:** `/Users/samuelminer/Documents/classes/nih_research/data_replogle_k562_essential/perturb_processed.h5ad`
- **Dimensions:** 10 × 1,093 perturbations
- **Parity Status:** ✅ **PERFECT PARITY** with original script

**Parity Metrics:**
- Mean cosine similarity: **1.00000000**
- Min cosine similarity: **1.00000000**
- Mean Pearson r: **1.00000000**
- Max absolute difference: **0.00e+00**
- Mean absolute difference: **0.00e+00**

**Validation:**
- Compared against: `validation/legacy_runs/replogle_k562_pert_emb_pca10_seed1.tsv` (from `extract_pert_embedding_pca.py`)
- All thresholds met: ✓ mean_cosine >= 0.99, ✓ min_cosine >= 0.95, ✓ mean_pearson >= 0.99

### RPE1 Embeddings ✅ **PRECOMPUTED**

- **Precomputed:** `results/replogle_rpe1_pert_emb_pca10_seed1.tsv`
- **Source Data:** `/Users/samuelminer/Documents/classes/nih_research/data_replogle_rpe1_essential/perturb_processed.h5ad`
- **Dimensions:** 10 × 1,544 perturbations
- **Parity Status:** ⏳ **Precomputed** (parity validation pending legacy script output)

**Note:** RPE1 embeddings were generated using the same validated `pca_perturbation` loader that achieved perfect parity for K562. Parity validation can be completed when legacy script output is available.

---

## Precomputation Process

### Script Used

```bash
python -m goal_2_baselines.precompute_cross_dataset_embeddings \
    --dataset replogle_k562_essential \
    --adata_path /path/to/perturb_processed.h5ad \
    --output_path results/replogle_k562_pert_emb_pca10_seed1.tsv \
    --pca_dim 10 \
    --seed 1
```

### Implementation

- Uses validated `pca_perturbation` embedding loader from Issue #18
- Computes PCA on pseudobulk expression profiles
- Format: rows = dimensions (PC1-PC10), columns = perturbations
- TSV format matching original `extract_pert_embedding_pca.py` output

---

## Usage in Baselines

These precomputed embeddings are automatically loaded by:

- **`lpm_k562PertEmb`** → loads `results/replogle_k562_pert_emb_pca10_seed1.tsv`
- **`lpm_rpe1PertEmb`** → loads `results/replogle_rpe1_pert_emb_pca10_seed1.tsv`

Configured in `src/baselines/baseline_types.py`:

```python
BaselineType.K562_PERT_EMB:
    pert_embedding_args={
        "embedding_path": "results/replogle_k562_pert_emb_pca10_seed1.tsv",
    }

BaselineType.RPE1_PERT_EMB:
    pert_embedding_args={
        "embedding_path": "results/replogle_rpe1_pert_emb_pca10_seed1.tsv",
    }
```

---

## Validation Results

### K562 Parity Validation

**Command:**
```bash
python -m goal_2_baselines.validate_cross_dataset_embeddings \
    --dataset replogle_k562_essential \
    --new_embedding results/replogle_k562_pert_emb_pca10_seed1.tsv \
    --legacy_embedding validation/legacy_runs/replogle_k562_pert_emb_pca10_seed1.tsv
```

**Results:**
```
Mean cosine similarity: 1.00000000
Min cosine similarity:  1.00000000
Mean Pearson r:        1.00000000
Max absolute diff:     0.00e+00
Mean absolute diff:    0.00e+00
Perturbations compared: 1092
Dimensions: 10

✓ All parity checks passed!
```

---

## Files Generated

- ✅ `results/replogle_k562_pert_emb_pca10_seed1.tsv` - K562 embeddings (validated)
- ✅ `results/replogle_rpe1_pert_emb_pca10_seed1.tsv` - RPE1 embeddings (precomputed)
- ✅ `validation/legacy_runs/replogle_k562_pert_emb_pca10_seed1.tsv` - Legacy K562 output (for comparison)

---

## Next Steps

1. ✅ K562 embeddings precomputed and validated
2. ✅ RPE1 embeddings precomputed
3. ⏳ RPE1 parity validation (when legacy script output available)
4. ✅ Baseline configs updated to use precomputed embeddings

---

**Last Updated:** 2025-11-14


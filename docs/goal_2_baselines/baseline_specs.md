# Baseline Specifications

This document consolidates the baseline model specifications from ISSUE13 documentation.

## Overview

This repository reproduces **8 linear baseline models** plus **1 mean-response baseline** from the Nature (2024) paper. All baselines use the same core model:

**Y ≈ A × K × B**

Where:
- **Y** (genes × perturbations): Pseudobulk expression changes (target - control)
- **A** (genes × d): Gene embedding matrix
- **K** (d × d): Learned interaction matrix (via ridge regression)
- **B** (d × perturbations): Perturbation embedding matrix

**Ridge Regression:**
```
K = argmin ||Y - A K B||² + λ(||A||² + ||B||²)
```

Default hyperparameters:
- `pca_dim = 10`
- `ridge_penalty = 0.1`
- `seed = 1`

## The 8 Linear Baselines

### 1. `lpm_selftrained` (Baseline: Within-Dataset)

**Configuration:**
```python
gene_embedding = "training_data"
pert_embedding = "training_data"
```

**A (Gene Embeddings):**
- **Source:** PCA on training data gene expression
- **Method:** `PCA(n_components=10).fit_transform(X_train.T)`
- **Shape:** (genes × 10)
- **Note:** Uses only training perturbations to compute gene structure

**B (Perturbation Embeddings):**
- **Source:** PCA on training data perturbation profiles
- **Method:** `PCA(n_components=10).fit_transform(X_train)`
- **Shape:** (10 × perturbations)
- **Note:** Uses only training perturbations

**K:** Learned via ridge regression on training data

**Purpose:** Fully self-contained baseline, no external embeddings.

---

### 2. `lpm_randomPertEmb` (Control: No Perturbation Structure)

**Configuration:**
```python
gene_embedding = "training_data"
pert_embedding = "random"
```

**A (Gene Embeddings):**
- Same as `lpm_selftrained`: PCA(train genes)

**B (Perturbation Embeddings):**
- **Source:** Random Gaussian matrix
- **Method:** `np.random.default_rng(seed).normal(0, 1, size=(10, n_perturbations))`
- **Shape:** (10 × perturbations)
- **Note:** Each perturbation gets a random 10-dimensional vector

**K:** Learned via ridge regression

**Purpose:** Tests "Is perturbation structure necessary?" (Answer: Yes — this performs very poorly)

---

### 3. `lpm_randomGeneEmb` (Control: No Gene Structure)

**Configuration:**
```python
gene_embedding = "random"
pert_embedding = "training_data"
```

**A (Gene Embeddings):**
- **Source:** Random Gaussian matrix
- **Method:** `np.random.default_rng(seed).normal(0, 1, size=(n_genes, 10))`
- **Shape:** (genes × 10)
- **Note:** Each gene gets a random 10-dimensional vector

**B (Perturbation Embeddings):**
- Same as `lpm_selftrained`: PCA(train perturbations)

**K:** Learned via ridge regression

**Purpose:** Tests "Is gene structure necessary?" (Answer: Yes — this performs very poorly)

---

### 4. `lpm_scgptGeneEmb` (Pretrained Gene Embeddings: scGPT)

**Configuration:**
```python
gene_embedding = "/path/to/scgpt_embeddings.tsv"  # From extract_gene_embedding_scgpt.py
pert_embedding = "training_data"
```

**A (Gene Embeddings):**
- **Source:** scGPT encoder embeddings (pretrained)
- **Extraction:** `extract_gene_embedding_scgpt.py` → `encoder.embedding.weight` from checkpoint
- **Shape:** (genes × 512) → subset to (genes × 10) via PCA or direct use
- **Note:** Fixed, pretrained gene feature vectors from scGPT transformer

**B (Perturbation Embeddings):**
- Same as `lpm_selftrained`: PCA(train perturbations)

**K:** Learned via ridge regression

**Purpose:** Tests "Do scGPT's pretrained gene embeddings help a linear model?"

---

### 5. `lpm_scFoundationGeneEmb` (Pretrained Gene Embeddings: scFoundation)

**Configuration:**
```python
gene_embedding = "/path/to/scfoundation_embeddings.tsv"  # From extract_gene_embedding_scfoundation.py
pert_embedding = "training_data"
```

**A (Gene Embeddings):**
- **Source:** scFoundation positional embeddings (pretrained)
- **Extraction:** `extract_gene_embedding_scfoundation.py` → `gene.state_dict.model.pos_emb.weight` from checkpoint
- **Shape:** (genes × 768) → subset to (genes × 10) via PCA or direct use
- **Note:** Fixed, pretrained gene feature vectors from scFoundation transformer

**B (Perturbation Embeddings):**
- Same as `lpm_selftrained`: PCA(train perturbations)

**K:** Learned via ridge regression

**Purpose:** Tests "Do scFoundation's pretrained gene embeddings help a linear model?"

---

### 6. `lpm_gearsPertEmb` (Pretrained Perturbation Embeddings: GEARS)

**Configuration:**
```python
gene_embedding = "training_data"
pert_embedding = "/path/to/gears_pert_embeddings.tsv"  # From extract_pert_embedding_from_gears.R
```

**A (Gene Embeddings):**
- Same as `lpm_selftrained`: PCA(train genes)

**B (Perturbation Embeddings):**
- **Source:** GEARS GO graph spectral embeddings
- **Extraction:** `extract_pert_embedding_from_gears.R` → spectral embedding of GO similarity graph
- **Shape:** (2 × perturbations) → may need to pad/expand to (10 × perturbations) or use as-is
- **Note:** Nonlinear perturbation encoding from GEARS deep graph model

**K:** Learned via ridge regression

**Purpose:** Tests "How well does a linear model perform if perturbations are represented by GEARS' nonlinear encoding?"

---

### 7. `lpm_k562PertEmb` ⭐ **BEST MODEL** (Cross-Dataset Transfer: K562)

**Configuration:**
```python
gene_embedding = "training_data"
pert_embedding = "/path/to/replogle_k562_pert_embeddings.tsv"  # From extract_pert_embedding_pca.R on K562
```

**A (Gene Embeddings):**
- **Source:** PCA on **training dataset** genes (e.g., Adamson)
- **Method:** `PCA(n_components=10).fit_transform(X_train.T)` on **target dataset**
- **Shape:** (genes × 10)
- **Note:** Uses gene structure from the **target dataset** (where we're predicting)

**B (Perturbation Embeddings):**
- **Source:** PCA on **Replogle K562** perturbations (different dataset!)
- **Extraction:** `extract_pert_embedding_pca.R` on `replogle_k562_essential` dataset
- **Method:** Pseudobulk K562 → PCA → (10 × K562_perturbations)
- **Shape:** (10 × perturbations)
- **Note:** Uses perturbation structure from **source dataset** (K562), transferred to target

**K:** Learned via ridge regression on target dataset training data

**Purpose:** Tests cross-dataset transfer: "Can we use clean perturbation signatures from K562 to predict on Adamson/RPE1?"

---

### 8. `lpm_rpe1PertEmb` (Cross-Dataset Transfer: RPE1)

**Configuration:**
```python
gene_embedding = "training_data"
pert_embedding = "/path/to/replogle_rpe1_pert_embeddings.tsv"  # From extract_pert_embedding_pca.R on RPE1
```

**A (Gene Embeddings):**
- Same as `lpm_k562PertEmb`: PCA on target dataset genes

**B (Perturbation Embeddings):**
- **Source:** PCA on **Replogle RPE1** perturbations (different dataset!)
- **Extraction:** `extract_pert_embedding_pca.R` on `replogle_rpe1_essential` dataset
- **Method:** Pseudobulk RPE1 → PCA → (10 × RPE1_perturbations)
- **Shape:** (10 × perturbations)
- **Note:** Uses perturbation structure from **source dataset** (RPE1), transferred to target

**K:** Learned via ridge regression on target dataset training data

**Purpose:** Tests cross-dataset transfer: "Can we use clean perturbation signatures from RPE1 to predict on Adamson/K562?"

---

### 9. `mean_response` (Mean Expression Baseline)

**Configuration:**
- Special case: no embeddings, just mean expression

**Method:**
- For each gene, predict the mean expression change across all training perturbations
- `y_pred[gene] = mean(Y_train[gene, :])`

**Purpose:** Simple baseline to compare against structured models

---

## Implementation Details

### Data Processing

1. **Pseudobulk Expression Changes (Y)**
   - Compute mean expression per condition
   - Subtract control baseline: `Y = mean(perturbation) - mean(control)`
   - Same for all baselines

2. **Train/Test/Val Split**
   - Use original split logic from `prepare_perturbation_data.py`
   - Default: 70% train, 15% test, 15% val
   - Control always in training set
   - Seed: 1 (for reproducibility)

### Embedding Construction

See `docs/EMBEDDING_CHECKPOINT_SOURCES.md` for details on loading pretrained embeddings.

### Cross-Dataset Embeddings

For `lpm_k562PertEmb` and `lpm_rpe1PertEmb`:
1. Load source dataset (K562 or RPE1)
2. Compute PCA on source dataset perturbations
3. Transform target dataset perturbations into source PCA space
4. Gene alignment: common genes only

---

## Reproducibility

See `docs/shared/REPRODUCIBILITY.md` for detailed reproducibility documentation.

---

**References:**
- Original documentation: `docs/ISSUE13_LINEAR_BASELINE_SPECIFICATION.md`
- Revision notes: `docs/ISSUE13_REVISION_SUMMARY.md`


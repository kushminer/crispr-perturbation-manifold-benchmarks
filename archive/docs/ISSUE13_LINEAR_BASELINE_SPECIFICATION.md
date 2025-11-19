# Issue #13: Linear Baseline Runner — Complete Specification

**Status:** 🔄 **REVISION IN PROGRESS**  
**Priority:** High  
**Dependencies:** Issue #18 (Embedding Parity) ✅ Complete

---

## Executive Summary

Issue #13 implements the **8 linear baseline models** from the Nature benchmark paper. All baselines use the same core ridge-regularized bilinear model **Y ≈ A × K × B**, where the only difference is how **A** (gene embeddings) and **B** (perturbation embeddings) are constructed.

---

## Core Linear Model

All 8 baselines share the same mathematical foundation:

### Model Equation

\[
Y \approx A \cdot K \cdot B
\]

Where:
- **Y** = pseudobulk gene expression **changes** (genes × perturbations)
  - \(Y_{i,j} = \text{mean expression of gene } i \text{ in perturbation } j - \text{mean expression of gene } i \text{ in ctrl}\)
  - Shape: typically 8k–10k genes × 100–300 perturbations
- **A** = gene embedding matrix (genes × d)
- **B** = perturbation embedding matrix (d × perturbations)
- **K** = learned interaction matrix (d × d) via ridge regression
- **d** = PCA dimension (default: 10)

### Solution Method

All baselines use the same ridge regression solver:

```python
def solve_y_axb(Y, A, B, A_ridge=0.01, B_ridge=0.01):
    """
    Solve Y = A K B for K using ridge regression.
    
    K = argmin_K ||Y - A K B||² + λ_A ||K||² + λ_B ||K||²
    """
    # Center Y
    center = np.mean(Y, axis=1, keepdims=True)
    Y_centered = Y - center
    
    # Ridge solution
    AtA = A.T @ A + A_ridge * np.eye(A.shape[1])
    BBt = B @ B.T + B_ridge * np.eye(B.shape[0])
    
    K = np.linalg.solve(AtA, A.T @ Y_centered @ B.T)
    K = np.linalg.solve(BBt, K.T).T
    
    return {"K": K, "center": center}
```

### Y Matrix Construction (Fixed Across All Baselines)

**Y is always the same** for all 8 baselines:

1. **Load dataset** (`perturb_processed.h5ad`)
2. **Apply train/test/val split** (from `prepare_perturbation_data.py`)
3. **Pseudobulk** using `glmGamPoi::pseudobulk` (R) or equivalent (Python)
   - Groups cells by `condition` and `clean_condition`
   - Averages expression within each group
4. **Compute baseline** = mean expression in control conditions
5. **Compute change** = `assay(psce, "change") = assay(psce, "X") - baseline`

**Shape:** genes × perturbations (typically 8k–10k × 100–300)

---

## The 8 Linear Baselines

Each baseline is uniquely defined by how **A** and **B** are constructed:

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

**Purpose:** Tests "Is perturbation structure necessary?" (Answer: Yes — this performs poorly)

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

**Why It Wins:**
- Replogle K562 has **very clean perturbation signatures**
- PCA captures them well
- Linear model generalizes better than deep nets
- Target datasets respond similarly enough that transfer works
- Deep models overfit/underfit low-variance perturbations

**This is the central message of the paper.**

---

### 8. `lpm_rpe1PertEmb` (Cross-Dataset Transfer: RPE1)

**Configuration:**
```python
gene_embedding = "training_data"
pert_embedding = "/path/to/replogle_rpe1_pert_embeddings.tsv"  # From extract_pert_embedding_pca.R on RPE1
```

**A (Gene Embeddings):**
- Same as `lpm_k562PertEmb`: PCA(train genes) on target dataset

**B (Perturbation Embeddings):**
- **Source:** PCA on **Replogle RPE1** perturbations
- **Extraction:** `extract_pert_embedding_pca.R` on `replogle_rpe1` dataset
- **Method:** Pseudobulk RPE1 → PCA → (10 × RPE1_perturbations)
- **Shape:** (10 × perturbations)

**K:** Learned via ridge regression

**Purpose:** Tests cross-dataset transfer (weaker than K562, but same idea)

---

## Summary Table

| Baseline | A (genes × d) | B (d × perturbations) | Key Feature |
|----------|---------------|----------------------|-------------|
| **lpm_selftrained** | PCA(train genes) | PCA(train perturbations) | Within-dataset baseline |
| **lpm_randomPertEmb** | PCA(train genes) | Random Gaussian | No perturbation structure |
| **lpm_randomGeneEmb** | Random Gaussian | PCA(train perturbations) | No gene structure |
| **lpm_scgptGeneEmb** | scGPT embeddings | PCA(train perturbations) | Pretrained gene semantics |
| **lpm_scFoundationGeneEmb** | scFoundation embeddings | PCA(train perturbations) | Pretrained gene semantics |
| **lpm_gearsPertEmb** | PCA(train genes) | GEARS embeddings | Nonlinear perturbation encoding |
| **lpm_k562PertEmb** ⭐ | PCA(train genes) | PCA(K562 perturbations) | **BEST** — cross-dataset transfer |
| **lpm_rpe1PertEmb** | PCA(train genes) | PCA(RPE1 perturbations) | Cross-dataset transfer |

---

## Implementation Requirements

### Core Functions

#### 1. `solve_y_axb(Y, A, B, A_ridge, B_ridge)`
- **Input:** Y (genes × perts), A (genes × d), B (d × perts), ridge penalties
- **Output:** `{"K": K, "center": center}`
- **Algorithm:** Ridge regression solution (see above)

#### 2. `construct_gene_embeddings(source, train_data, pca_dim, seed)`
- **Handles:**
  - `"training_data"` → PCA on train genes
  - `"random"` → Random Gaussian
  - `"/path/to/tsv"` → Load pretrained embeddings (scGPT, scFoundation)
- **Returns:** A (genes × d)

#### 3. `construct_pert_embeddings(source, train_data, pca_dim, seed)`
- **Handles:**
  - `"training_data"` → PCA on train perturbations
  - `"random"` → Random Gaussian
  - `"/path/to/tsv"` → Load pretrained embeddings (GEARS, K562, RPE1)
- **Returns:** B (d × perturbations)

#### 4. `prepare_expression_changes(adata, train_test_config)`
- **Fixed across all baselines:**
  - Load `perturb_processed.h5ad`
  - Apply train/test/val split
  - Pseudobulk by condition
  - Compute baseline (mean ctrl)
  - Compute change = expression - baseline
- **Returns:** Y (genes × perturbations), labels

### Baseline Runner

#### Main Function: `run_linear_baseline(dataset_name, baseline_type, config)`

**Parameters:**
- `dataset_name`: "adamson", "replogle_k562_essential", "replogle_rpe1"
- `baseline_type`: One of the 8 baseline names
- `config`: YAML config with:
  - `pca_dim`: 10 (default)
  - `ridge_penalty`: 0.1 (default)
  - `seed`: 1 (default)
  - `train_test_config_id`: Split configuration ID
  - `gene_embedding`: Source for A
  - `pert_embedding`: Source for B

**Workflow:**
1. Load dataset and train/test split
2. Prepare Y (expression changes) — **same for all baselines**
3. Construct A based on `baseline_type`
4. Construct B based on `baseline_type`
5. Solve for K: `solve_y_axb(Y_train, A_train, B_train, ridge, ridge)`
6. Predict: `Y_pred = A @ K @ B + center + baseline`
7. Save predictions: `all_predictions.json`, `gene_names.json`
8. Run evaluation: LOGO, functional-class, combined

### Integration with Embedding Loaders (Issue #18)

The baseline runner must use validated embedding loaders:

- **scGPT:** `embeddings.registry.load("scgpt_gene", checkpoint_dir=..., subset_genes=...)`
- **scFoundation:** `embeddings.registry.load("scfoundation_gene", checkpoint_path=..., demo_h5ad=...)`
- **GEARS:** `embeddings.registry.load("gears_go", source_csv=...)`
- **PCA (K562/RPE1):** `embeddings.registry.load("pca_perturbation", adata_path=...)`

### Cross-Dataset Embedding Precomputation

For `lpm_k562PertEmb` and `lpm_rpe1PertEmb`, perturbation embeddings must be **precomputed ONCE** on the source datasets, then **reused for all target datasets**.

**Current Status:** ⚠️ Embeddings are currently computed **on-the-fly** using the `pca_perturbation` loader. According to the paper specification, they should be **precomputed once** and saved as TSV files.

**Why Precompute:**
- **Efficiency:** Avoid recomputing the same embeddings for each target dataset
- **Reproducibility:** Same embeddings used across all experiments
- **Paper Alignment:** Matches the R implementation which uses precomputed TSV files

**Precomputation Process:**

1. **K562 embeddings (compute ONCE):**
   ```python
   # Run ONCE on K562 dataset (source)
   from embeddings.registry import load
   k562_emb_result = load(
       "pca_perturbation",
       adata_path="/path/to/replogle_k562_essential/perturb_processed.h5ad",
       n_components=10,
       seed=1
   )
   # Save to: results/replogle_k562_pert_emb_pca10_seed1.tsv
   # Format: rows = dimensions (10), columns = perturbations
   k562_emb_result.values.T.to_csv("results/replogle_k562_pert_emb_pca10_seed1.tsv", sep="\t")
   ```

2. **RPE1 embeddings (compute ONCE):**
   ```python
   # Run ONCE on RPE1 dataset (source)
   rpe1_emb_result = load(
       "pca_perturbation",
       adata_path="/path/to/replogle_rpe1/perturb_processed.h5ad",
       n_components=10,
       seed=1
   )
   # Save to: results/replogle_rpe1_pert_emb_pca10_seed1.tsv
   rpe1_emb_result.values.T.to_csv("results/replogle_rpe1_pert_emb_pca10_seed1.tsv", sep="\t")
   ```

**Usage in Baselines:**
- When running `lpm_k562PertEmb` on **any target dataset** (Adamson, Replogle, RPE1, etc.), load the **same precomputed K562 TSV file**
- When running `lpm_rpe1PertEmb` on **any target dataset**, load the **same precomputed RPE1 TSV file**

**Key Point:** The embeddings are computed **once on the source dataset** (K562 or RPE1), then **reused for all target datasets**. They are NOT recomputed for each target dataset.

---

## Acceptance Criteria

### Functional Requirements

- [ ] **All 8 baselines implemented** with correct A/B construction
- [ ] **Y matrix construction** matches paper (pseudobulk → change)
- [ ] **Ridge regression solver** matches R implementation numerically
- [ ] **Integration with embedding loaders** (Issue #18) working
- [ ] **Cross-dataset embeddings** precomputed and loadable
- [ ] **Predictions saved** in same format as R script (`all_predictions.json`)

### Validation Requirements

- [ ] **Numerical parity** with R `run_linear_pretrained_model.R` outputs
  - Mean absolute difference < 1e-6 for predictions
  - Pearson correlation > 0.99 between Python and R predictions
- [ ] **All 8 baselines run** on Adamson dataset
- [ ] **Cross-dataset baselines** (K562, RPE1) run on Adamson
- [ ] **Results match paper** performance rankings:
  - `lpm_k562PertEmb` performs best
  - `lpm_selftrained` performs well
  - Random baselines perform poorly

### Integration Requirements

- [ ] **CLI task:** `--task run-baselines` or `--task linear-baseline`
- [ ] **Config-driven:** All 8 baselines configurable via YAML
- [ ] **Evaluation integration:** Baselines feed into LOGO/functional-class evaluation
- [ ] **Output format:** Compatible with existing evaluation pipeline

---

## Dependencies

### Required (✅ Complete)
- ✅ Issue #18: Embedding Parity — All embedding loaders validated
- ✅ Embedding loaders: scGPT, scFoundation, GEARS, PCA

### External Assets
- ✅ scGPT checkpoint: `data/models/scgpt/scGPT_human/`
- ✅ scFoundation checkpoint: `data/models/scfoundation/`
- ✅ GEARS GO graph: `paper/benchmark/data/gears_pert_data/go_essential_all/go_essential_all.csv`
- ✅ Datasets: Adamson, Replogle K562, Replogle RPE1

---

## Deliverables

1. **Core Implementation:**
   - `src/eval_framework/linear_baseline.py` — Main baseline runner
   - `src/eval_framework/linear_model.py` — Core model functions (revised)
   - `src/eval_framework/embedding_construction.py` — A/B construction utilities

2. **Configuration:**
   - `configs/baseline_*.yaml` — Configs for each baseline type
   - `configs/baseline_all.yaml` — Run all 8 baselines

3. **Documentation:**
   - `docs/ISSUE13_LINEAR_BASELINE_SPECIFICATION.md` — This document
   - `docs/LINEAR_BASELINE_USER_GUIDE.md` — User guide
   - `docs/LINEAR_BASELINE_RESULTS.md` — Results analysis

4. **Validation:**
   - `validation/linear_baseline_parity/` — Comparison with R outputs
   - `validation/linear_baseline_parity/parity_report.csv`

5. **Results:**
   - `results/baselines/*/` — Predictions and evaluations for all baselines

---

## Next Steps

1. **Revise `linear_model.py`:**
   - Update `solve_y_axb` to match R implementation exactly
   - Add `construct_gene_embeddings` and `construct_pert_embeddings` functions
   - Ensure Y construction matches paper (pseudobulk → change)

2. **Implement baseline runner:**
   - Create `linear_baseline.py` with `run_linear_baseline` function
   - Integrate with embedding loaders
   - Handle all 8 baseline types

3. **Precompute cross-dataset embeddings:**
   - Run PCA on K562 and RPE1 datasets
   - Save embeddings for cross-dataset baselines

4. **Validate against R:**
   - Run R script on subset
   - Compare Python vs R predictions
   - Ensure numerical parity

5. **Integration:**
   - Add CLI task
   - Update configs
   - Integrate with evaluation pipeline

---

**Last Updated:** 2025-11-14  
**Status:** Specification Complete — Implementation In Progress  
**Priority:** High (Blocking downstream analysis)


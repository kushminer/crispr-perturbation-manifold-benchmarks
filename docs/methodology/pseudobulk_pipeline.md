## Pseudobulk Baseline Pipeline

### Scope

This document describes the **pseudobulk baseline pipeline**:
how we go from raw single-cell data to perturbation-level predictions
for all linear baselines.

Primary entry points:
- `src/goal_2_baselines/baseline_runner.py`
- `compute_pseudobulk_expression_changes` in the same file

---

### 1. Data Preparation

**Inputs**
- AnnData `perturb_processed.h5ad` per dataset (e.g. Adamson, K562, RPE1)
- Split config JSON specifying train / test / val perturbations

**Steps**
1. Filter cells to those whose `condition` appears in any split.
2. Clean condition names by removing the `+ctrl` suffix.
3. Identify control cells (`condition == "ctrl"`).
4. Compute the **baseline** expression:
   - Mean expression across control cells for each gene.
5. For each non-control condition:
   - Average expression across all cells from that condition.
   - Subtract the baseline to get **perturbation-level change**.
6. Stack all perturbations into a Y matrix:
   - Shape: `genes × perturbations`.
7. Align split labels:
   - Map split names (train/test/val) to perturbation IDs.

Code reference:
- `compute_pseudobulk_expression_changes` in `baseline_runner.py`.

---

### 2. Embedding Construction

We factor the prediction model as:

> \\(Y \\approx A K B\\),

where:
- \\(Y\\): genes × perturbations (pseudobulk changes)
- \\(A\\): gene embeddings
- \\(B\\): perturbation embeddings
- \\(K\\): learned interaction matrix

#### 2.1 Gene Embeddings (A)

Constructed by `construct_gene_embeddings`:
- **Self-trained PCA** (`source="training_data"`):
  - PCA on rows of Y (genes as observations).
  - Output shape: `genes × pca_dim`.
- **Random** (`source="random"`):
  - Gaussian matrix with shape `genes × pca_dim`.
- **scGPT / scFoundation**:
  - Load pretrained embeddings via `embeddings.registry`.
  - Align to dataset gene names (with optional symbol mapping).
  - Fill missing genes with zeros.

Key parameters:
- `pca_dim` (default: 10)
- `seed`
- `embedding_args` (checkpoint paths, gene name mapping)

#### 2.2 Perturbation Embeddings (B)

Constructed by `construct_pert_embeddings`:
- **Self-trained PCA** (`source="training_data"`):
  - PCA on columns of Y (perturbations).
  - Output shape: `pca_dim × perturbations`.
- **Random** (`source="random"`):
  - Gaussian matrix with shape `pca_dim × perturbations`.
- **GEARS** (`source="gears"`):
  - Load GO-graph-based embeddings from a CSV.
  - Align to perturbation names (strip `+ctrl`).
  - Non-covered perturbations get zero vectors.
- **Cross-dataset PCA (K562_PCA, RPE1_PCA)**:
  - Fit PCA on a source dataset.
  - Align genes between source and target.
  - Transform target perturbations into the source PCA space.

Code reference:
- `construct_gene_embeddings` and `construct_pert_embeddings`
  in `src/goal_2_baselines/baseline_runner.py`.

---

### 3. Model Fitting

We learn the interaction matrix \\(K\\) via ridge regression:

1. Compute center:
   - `center = Y_train.mean(axis=1, keepdims=True)`.
2. Center training data:
   - `Y_centered = Y_train - center`.
3. Solve for K:
   - `solve_y_axb(Y=Y_centered, A=A, B=B_train, ridge_penalty=λ)`.
   - Returns `K` with shape `pca_dim × pca_dim`.

Implementation:
- `shared/linear_model.solve_y_axb`.

---

### 4. Prediction and Evaluation

**Prediction**
- Compute:
  - `Y_pred = A @ K @ B_test + center`.
  - Same shape as `Y_test`.

**Metrics**
- For each test perturbation:
  - Pearson correlation (r) between predicted and true gene changes.
  - L2 distance between predicted and true vector.
- Aggregate:
  - Mean r and mean L2 across all test perturbations.

Code reference:
- `run_single_baseline` and `run_all_baselines` in `baseline_runner.py`.

---

### 5. Design Choices & Assumptions

- Pseudobulk differences are **change from control**, not raw expression.
- PCA dimension is fixed at 10 for comparability across baselines.
- Ridge penalty default: 0.1 (tunable via config).
- Cross-dataset baselines rely on **gene name alignment**;
  missing genes are dropped via intersection.

This pipeline is the reference implementation for all **pseudobulk**
baseline results used in the analysis docs.



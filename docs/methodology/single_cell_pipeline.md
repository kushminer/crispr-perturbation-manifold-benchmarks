## Single-Cell Pipeline

### Scope

This document describes the **single-cell prediction pipeline**:
how we sample cells, compute expression changes, build embeddings, train
linear models, and evaluate at cell and perturbation levels.

Primary entry points:
- `src/goal_2_baselines/single_cell_loader.py`
- `src/goal_2_baselines/baseline_runner_single_cell.py`

This is the structured reference version of
`deliverables/archive/docs/legacy/single_cell_methodology_report.md`.

---

### 1. Data Preparation

**Inputs**
- AnnData `perturb_processed.h5ad` per dataset.
- Split config JSON with condition-level splits (train / test / val).

**Steps**
1. Filter cells to those whose `condition` is in any split.
2. Clean condition names:
   - Add `obs["clean_condition"]` by stripping `+ctrl`.
3. Identify control cells (`condition == "ctrl"`).
4. Compute baseline expression:
   - Mean expression across control cells per gene.
5. For each non-control condition:
   - Sample up to `n_cells_per_pert` cells (default: 50).
   - For each sampled cell:
     - Subtract baseline → **cell-level change vector**.
     - Assign a synthetic cell ID: `"{perturbation}_{i}"`.
     - Record mapping `cell_to_pert[cell_id] = perturbation`.
6. Build Y matrix:
   - Stack cell change vectors as columns:
   - Shape: `genes × cells`.
7. Derive cell-level splits:
   - Map each cell to its perturbation.
   - Assign to train/test/val based on perturbation’s split.

Code reference:
- `compute_single_cell_expression_changes` in `single_cell_loader.py`.

---

### 2. Embedding Construction

We reuse the same linear-factorization idea:

> \\(Y \\approx A K B\\)

but now with **cells** instead of perturbations.

#### 2.1 Gene Embeddings (A)

Identical to pseudobulk:
- Self-trained PCA on `Y_train` (genes as observations).
- Pretrained scGPT / scFoundation aligned to dataset genes.
- Random Gaussian matrix.

Implementation:
- `construct_gene_embeddings` from `goal_2_baselines.baseline_runner`.

#### 2.2 Cell Embeddings (B)

Constructed by `construct_cell_embeddings` in
`baseline_runner_single_cell.py`:

1. **Cell PCA** (`method="cell_pca"`, for `pert_embedding_source="training_data"`):
   - PCA on `Y_cells.T` (cells as observations).
   - Output: `B_train` with shape `pca_dim × n_train_cells`.
   - Test cells are projected via the same PCA.

2. **Perturbation-level embeddings** (`pert_embedding_source` ≠ `training_data`):
   - Compute **cell-level pseudobulk**:
     - Average cell-level Y across cells for each perturbation.
   - Construct **perturbation embeddings** using
     `construct_pert_embeddings` (e.g. GEARS, cross-dataset PCA).
   - Expand perturbation embeddings to cells via lookup:
     - Cells from the same perturbation share the same embedding.

3. **Random embeddings**:
   - Random Gaussian for train/test cells (used for `RANDOM_PERT_EMB`).

Key function:
- `_expand_per_pert_embeddings_to_cells` maps B_perts → B_cells.

---

### 3. Model Fitting (Single-Cell)

Given:
- `Y_train` (genes × train_cells)
- `A` (genes × pca_dim)
- `B_train` (pca_dim × train_cells)

We:
1. Compute center:
   - `center = Y_train_np.mean(axis=1, keepdims=True)`.
2. Center:
   - `Y_centered = Y_train_np - center`.
3. Solve for K:
   - `solve_y_axb(Y_centered, A, B_train, ridge_penalty)`.
4. Predict on test cells:
   - `Y_pred_test = A @ K @ B_test + center`.

Implementation:
- `run_single_baseline_single_cell` in
  `baseline_runner_single_cell.py`.

---

### 4. Metrics and Aggregation

#### 4.1 Cell-Level Metrics

For each test cell:
- Compute Pearson r between `y_true` and `y_pred`.
- Compute L2 distance.
- Store in `cell_metrics[cell_id]`.

#### 4.2 Perturbation-Level Aggregation

To compare with pseudobulk:
- Group cells by `cell_to_pert`.
- Average metrics across cells from the same perturbation.
- Store in `pert_metrics[perturbation]`.

Code reference:
- `aggregate_cell_metrics_by_perturbation` in `single_cell_loader.py`.

---

### 5. Baseline Runner (Single-Cell)

**Entry point**:
- `run_all_baselines_single_cell`:
  - Loads data and splits.
  - Calls `compute_single_cell_expression_changes`.
  - Runs a list of `BaselineType` values through
    `run_single_baseline_single_cell`.
  - Writes:
    - `single_cell_baseline_summary.csv`
    - Per-baseline `cell_metrics.csv` and `pert_metrics.csv`.

Default baselines (if none specified):
- Self-trained PCA
- Random Gene Emb
- Random Pert Emb

We extend this to additional baselines such as scGPT, scFoundation, GEARS.

---

### 6. Design Decisions & Assumptions

- **Sampling**: Fixed 50 cells per perturbation (if available) to
  control runtime and variance.
- **Baseline computation**: Always relative to control cells.
- **Embedding dimensionality**: PCA dimension fixed at 10.
- **Evaluation**: Both cell-level and perturbation-level metrics
  are used to understand performance.

For results and interpretations, see:
- `docs/analysis/single_cell_overview.md`
- `docs/analysis/single_cell_baselines.md`



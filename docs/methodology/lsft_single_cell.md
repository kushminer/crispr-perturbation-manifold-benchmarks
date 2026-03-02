## LSFT Single-Cell Methodology

### Scope

This document describes **Local Similarity-Filtered Training (LSFT)**
applied at the **single-cell** level.

Primary entry points:
- `src/goal_3_prediction/lsft/lsft_single_cell.py`

---

### 1. Intuition

LSFT assumes that the perturbation response manifold is **locally smooth**:
cells with similar responses live near each other in embedding space.

Instead of training on all cells, LSFT:
1. Computes similarities between a test cell and all training cells.
2. Selects the top-k% most similar training cells.
3. Retrains the linear model using only these neighbors.
4. Predicts the test cell using the locally trained model.

This tests whether **local geometry alone** is sufficient for accurate
prediction and how performance changes as we vary neighborhood size.

---

### 2. Data & Embeddings

LSFT operates on the same single-cell Y matrix as the baseline pipeline:
- `Y_train`: genes × train_cells
- `Y_test`: genes × test_cells

Embeddings:
- `A`: gene embeddings (as in baseline).
- `B_train`: cell embeddings for train cells.
- `B_test`: cell embeddings for test cells.

By default, LSFT uses the **same A and cell embedding method** that the
baseline uses (e.g. self-trained PCA, GEARS, etc.).

---

### 3. Similarity Computation

Function:
- `compute_cell_similarities(B_test, B_train)` in `lsft_single_cell.py`.

Steps:
1. Take `B_test` (pca_dim × n_test) and `B_train` (pca_dim × n_train).
2. Transpose to cell-major form (cells × dims).
3. Compute cosine similarity between each test cell and all train cells:
   - Result: `similarities` with shape `n_test × n_train`.

This matrix tells us, for each test cell, which training cells are most
similar in embedding space.

---

### 4. Neighbor Selection

Function:
- `filter_training_cells(similarities, train_cell_ids, top_pct)`.

For each test cell:
1. Take the similarity vector to all train cells.
2. Sort indices by similarity (descending).
3. Keep the top `top_pct` fraction of training cells.
4. Return:
   - `filtered_ids`: list of selected train cell IDs.
   - `filtered_indices`: corresponding indices.
   - `selected_similarities`: similarity values.

Typical values:
- `top_pcts = [0.01, 0.05, 0.10]` (1%, 5%, 10% of neighbors).

---

### 5. Local Model Fitting

For each test cell and each `top_pct`:
1. Extract filtered training data:
   - `Y_train_filtered = Y_train[:, filtered_indices]`.
   - `B_train_filtered = B_train[:, filtered_indices]`.
2. Center (fair-comparison rule):
   - Compute one global center from the full training set baseline model:
     - `center_baseline = mean(Y_train, axis=1)`.
   - Reuse that same center for LSFT retraining:
     - `Y_centered = Y_train_filtered - center_baseline`.
3. Solve for `K`:
   - `solve_y_axb(Y_centered, A, B_train_filtered, ridge_penalty)`.
4. Predict the test cell:
   - Take its embedding: `B_test[:, test_idx:test_idx+1]`.
   - `y_pred = (A @ K @ test_cell_embedding + center_baseline).flatten()`.
5. Compute metrics:
   - Pearson r and L2 vs `y_true` for that cell.

All results are recorded per cell and per `top_pct`.

---

### 6. Aggregation & Outputs

LSFT single-cell pipeline outputs:
- Per-cell metrics:
  - `lsft_pearson_r`, `lsft_l2`.
- Per-perturbation aggregates:
  - Mean and std of LSFT metrics over cells.
- Delta vs baseline:
  - `delta_r = lsft_pearson_r - baseline_r`.

The implementation also writes:
- CSV of cell-level results for each baseline and dataset.
- Aggregated LSFT comparison tables for analysis.

Code reference:
- `evaluate_lsft_single_cell` and
  `run_lsft_single_cell_all_baselines`
  in `lsft_single_cell.py`.

---

### 7. Design Choices

- Similarity metric: **cosine similarity** in cell embedding space.
- Neighborhood sizes: small percentages (1–10%) to test **locality**.
- Embeddings:
  - Can be PCA, GEARS, or other baseline-specific embeddings.
- Evaluation:
  - Focus on **change in r** vs the global baseline.

For interpretations of LSFT results, see:
- `docs/analysis/single_cell_lsft.md`.


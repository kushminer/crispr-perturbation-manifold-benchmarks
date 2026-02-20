## LOGO Single-Cell Methodology

### Scope

This document describes the **Leave-One-GO-Out (LOGO)** evaluation at
the **single-cell** level: how we define holdout functional classes,
sample cells, train models, and measure extrapolation performance.

Primary entry points:
- `src/goal_4_logo/logo_single_cell.py`

---

### 1. Problem Setting

LOGO tests **biological extrapolation**:
- Train on all perturbations **except** those belonging to a selected
  GO functional class (e.g. `Transcription`).
- Evaluate on held-out perturbations from that class.

Goal:
- Measure how well the model generalizes to **novel functional classes**
  not seen during training.

---

### 2. Data & Annotations

Inputs:
- AnnData `perturb_processed.h5ad` with perturbation-level metadata.
- Annotation TSV mapping perturbations to GO functional classes:
  - e.g. `data/annotations/adamson_functional_classes_enriched.tsv`.

The annotation file provides:
- `perturbation_id`
- `functional_class` (e.g. Transcription, Translation, etc.).

---

### 3. Single-Cell Expression Changes (LOGO)

Function:
- `compute_single_cell_expression_changes_logo` in `logo_single_cell.py`.

Steps:
1. Clean condition names: create `clean_condition` without `+ctrl`.
2. Compute control baseline:
   - Mean expression across control cells for each gene.
3. Identify target (holdout) perturbations:
   - Those whose functional class matches the selected `class_name`.
4. For each perturbation:
   - Sample up to `n_cells_per_pert` cells (default: 50).
   - Subtract baseline to get cell-level change vectors.
   - Assign synthetic cell IDs: `"{perturbation}_{i}"`.
5. Split into:
   - **Train cells**: perturbations **not** in holdout set.
   - **Test cells**: perturbations in holdout set.

Outputs:
- `Y_df`: genes × cells (all sampled cells).
- `split_labels`: `{"train": [cell_ids], "test": [cell_ids]}`.
- `cell_to_pert`: mapping from cell ID to perturbation.

---

### 4. Embeddings and Model

LOGO shares the same embedding and model structure as the single-cell
baseline pipeline:

1. Gene embeddings `A`:
   - Self-trained PCA or pretrained embeddings.
2. Cell embeddings `B`:
   - Typically cell-level PCA on Y_train cells.
3. Linear model:
   - Train K using train cells (non-holdout).
   - Predict on test cells (holdout functional class).

For a given baseline:
- Construct `A` and `B_train` as in the single-cell baseline runner.
- Solve for `K` via ridge regression.
- Compute `Y_pred` for test cells.

---

### 5. Metrics

Per-cell:
- Pearson r and L2 between true and predicted expression.

Per-perturbation:
- Aggregate metrics across cells from the same perturbation.

Per-functional-class:
- Aggregate perturbation-level metrics across all held-out
  perturbations in the target class.

These aggregations allow:
- Cell-level, perturbation-level, and class-level scores.

---

### 6. Implementation Summary

Function:
- `run_logo_single_cell` in `logo_single_cell.py`.

Key arguments:
- `adata_path`: path to `perturb_processed.h5ad`.
- `annotation_path`: functional class annotation TSV.
- `dataset_name`: identifier (e.g. `adamson`).
- `output_dir`: directory to write LOGO results.
- `class_name`: functional class to hold out (default: `Transcription`).
- `baseline_types`: list of `BaselineType`s to evaluate.

Output artifacts:
- Cell-level and perturbation-level metrics per baseline.
- CSV summaries per dataset and baseline.

---

### 7. Design Considerations

- Holdout design:
  - Entire functional class is held out, not just random perturbations.
- Sampling:
  - Same `n_cells_per_pert` as baseline/LSFT for comparability.
- Embeddings:
  - Use the same A/B construction as the corresponding baseline to
    isolate the effect of **functional extrapolation**.

For results and interpretations of LOGO experiments, see:
- `docs/analysis/single_cell_logo.md`.



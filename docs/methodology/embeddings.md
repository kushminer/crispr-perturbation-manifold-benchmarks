## Embedding Methodology

### Scope

This document summarizes how we construct **gene** and
**perturbation/cell** embeddings across all baselines.

Key modules:
- `src/goal_2_baselines/baseline_runner.py`
- `src/goal_2_baselines/baseline_runner_single_cell.py`
- `src/embeddings/*.py`

---

### 1. Gene Embeddings (A)

Function:
- `construct_gene_embeddings(source, train_data, gene_names, pca_dim, seed, embedding_args)`

Inputs:
- `train_data`: genes × train_perturbations (or cells for single-cell).
- `gene_names`: list of gene IDs (var_names).

#### 1.1 Self-Trained PCA (`source="training_data"`)

- Treat genes as observations and perturbations as features.
- Apply PCA with `pca_dim` components.
- Output:
  - `A`: genes × pca_dim
  - Labels = `gene_names`.

Pros:
- Data-specific and task-specific.
- Consistently best-performing baseline.

#### 1.2 Random (`source="random"`)

- Draw Gaussian matrix with shape `genes × pca_dim`.
- Fixed random seed per baseline for reproducibility.

Purpose:
- Control baseline to test whether geometry alone matters
  (especially with LSFT).

#### 1.3 scGPT (`source="scgpt"`)

- Use `embeddings.registry.load("scgpt_gene", ...)`.
- Align pretrained gene symbol embeddings to dataset genes:
  - Optionally use `gene_name_mapping` (e.g. Ensembl → symbol).
  - Create a full `A_full` with zeros for genes not in the embedding.

#### 1.4 scFoundation (`source="scfoundation"`)

- Similar to scGPT, but using scFoundation checkpoints:
  - `load("scfoundation_gene", checkpoint_path, demo_h5ad, ...)`.
- Align embedding rows to dataset genes using symbol mapping.

---

### 2. Perturbation / Cell Embeddings (B)

Function:
- `construct_pert_embeddings(source, train_data, pert_names, pca_dim, seed, embedding_args, test_data, test_pert_names)`
- `construct_cell_embeddings` (single-cell).

#### 2.1 Self-Trained Perturbation PCA (`source="training_data"`)

- For pseudobulk:
  - PCA on columns of Y (perturbations).
  - `B_train`: pca_dim × train_perturbations.
  - `B_test`: pca_dim × test_perturbations (via same PCA).

Used for:
- Self-trained PCA baseline’s B matrix.

#### 2.2 Random Perturbation Embeddings (`source="random"`)

- Gaussian random matrix for train/test perturbations.
- Used by `lpm_randomPertEmb`.

Purpose:
- Pure control for perturbation geometry.

#### 2.3 GEARS GO Graph Embeddings (`source="gears"`)

Loader:
- `embeddings/gears_go_perturbation.py` (`load_gears_go_embedding`).

Inputs:
- GO similarity CSV (`go_essential_all.csv`) with columns:
  - `source`, `target`, `importance`.

Steps:
1. Load CSV and build a gene graph.
2. Compute a spectral embedding:
   - Symmetrize adjacency.
   - Add small ridge on diagonal.
   - Top eigenvectors scaled by sqrt(eigenvalues).
3. Align to dataset perturbations:
   - Clean perturbation names (strip `+ctrl`).
   - Intersect with GEARS item labels.
   - Embed only overlapping perturbations.
4. Create full B matrix:
   - Non-covered perturbations are zeros.
5. Optionally build `B_test` for test perturbations via the same alignment.

Key path:
- `../linear_perturbation_prediction-Paper/paper/benchmark/data/gears_pert_data/go_essential_all/go_essential_all.csv`

#### 2.4 Cross-Dataset PCA (K562_PCA / RPE1_PCA)

- Fit PCA on a **source dataset** (K562/RPE1 pseudobulk).
- Align genes between source and target:
  - Intersect gene sets.
- Transform target perturbations using source PCA.

Purpose:
- Test transfer of embedding geometry between datasets.

---

### 3. Single-Cell Cell Embeddings

Function:
- `construct_cell_embeddings(Y_cells, cell_ids, cell_to_pert, pca_dim, seed, method)`

Methods:
1. `method="cell_pca"`:
   - PCA directly on cells.
   - Output: `B` (pca_dim × n_cells).
2. `method="pert_pca"`:
   - Aggregate to perturbation level first.
   - PCA on perturbations.
   - Map perturbation embeddings back down to cells.

Used in:
- Single-cell baselines when `pert_embedding_source="training_data"`.

---

### 4. Baseline Types and Embedding Configs

Defined in:
- `src/goal_2_baselines/baseline_types.py`.

Key baselines:
- `lpm_selftrained`:
  - A: training_data
  - B: training_data
- `lpm_randomGeneEmb`:
  - A: random
  - B: training_data
- `lpm_randomPertEmb`:
  - A: training_data
  - B: random
- `lpm_scgptGeneEmb`:
  - A: scgpt
  - B: training_data
- `lpm_scFoundationGeneEmb`:
  - A: scfoundation
  - B: training_data
- `lpm_gearsPertEmb`:
  - A: training_data
  - B: GEARS GO graph embeddings.

---

### 5. Validation & Debugging

To guard against embedding bugs:
- **Diagnostics**:
  - Log shapes and basic stats (mean/std/min/max).
  - Log first few labels.
- **Equality checks** (single-cell):
  - Compare B matrices against `training_data` PCA embeddings.
  - Raise if max difference < 1e-6 (identical).

See:
- `validation_and_audits.md` for higher-level audits.



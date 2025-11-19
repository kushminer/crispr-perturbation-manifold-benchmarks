## Sprint 5 – Issue #18 Plan: Embedding Translation & Parity Validation

### 1. Goal
- Reproduce every embedding extraction workflow from the Nature benchmark (`paper/benchmark/src`) inside the Python evaluation framework.
- Guarantee numerical parity between the original scripts (R or Python) and the new Python-native modules before any downstream baseline work (#13–#16).

### 2. Source Script Inventory
| Script | Lang | Purpose | Inputs (current) | Notes / Restrictions |
| --- | --- | --- | --- | --- |
| `extract_gene_embedding_scgpt.py` | Python | Pull encoder weights from scGPT transformer | Requires access to `scGPT_human` checkpoint (`args.json`, `best_model.pt`, `vocab.json`) | Paths hardcoded to `/home/...`; assets are **not** in repo. Need configurable path/env + lightweight subset of genes for parity. |
| `extract_gene_embedding_scfoundation.py` | Python | Extract positional embeddings from scFoundation | Needs `demo.h5ad` + `models.ckpt` checkpoint | Also points to `/home/...`; must introduce configurable path and identify storage location for checkpoints. |
| `extract_pert_embedding_from_gears.R` | R | Build perturbation embeddings from GO similarity graph (GEARS) | `data/gears_pert_data/go_essential_all/go_essential_all.csv` | Computes igraph spectral embedding; translation needed. |
| `extract_pert_embedding_pca.R` | R | PCA embeddings from processed perturbation datasets | `data/gears_pert_data/<dataset>/perturb_processed.h5ad` | Already have a partial Python translation, but parity needs to be proven + unified interface. |

### 3. Target Python Modules
Planned location: `evaluation_framework/src/embeddings/`

| Module | Responsibilities |
| --- | --- |
| `scgpt_gene.py` | Load scGPT checkpoint, expose `load_embeddings(config)` returning `(matrix, metadata)`; support subset filtering for parity. |
| `scfoundation_gene.py` | Same for scFoundation positional embeddings. |
| `gears_go_perturbation.py` | Port R logic (Graph build → spectral embedding) using `networkx` or `igraph` bindings. |
| `pca_perturbation.py` | Finalize translation of PCA script, add deterministic pseudobulk + change computations + dataset hooks. |
| `registry.py` | Map logical embedding names (e.g., `scgpt_gene`) to loader functions + metadata (inputs, default params). |

### 4. Parity Harness Design
- CLI entry: `--task validate-embeddings`
- Config additions:
  - `embedding_validation.subset_config_path`: YAML describing each embedding, subset selection, and source script invocation.
  - `embedding_validation.working_dir`: scratch directory for raw/original outputs.
- Workflow per embedding:
  1. **Materialize subset inputs** (e.g., sample of GO edges, subset of `perturb_processed.h5ad`, limited gene vocabulary). Store under `validation/embedding_subsets/<name>/`.
  2. **Run legacy script** via subprocess (Rscript / python) with the original arguments, targeting the subset data + scratch working dir.
  3. **Run new Python loader** to produce embeddings on the exact subset.
  4. **Align indices** (respecting gene / perturbation labels + ordering).
  5. **Compare metrics**: mean cosine, min cosine, mean Pearson r. Thresholds: ≥0.99 / ≥0.95 / ≥0.99, respectively.
  6. **Record hash** of subset source artifacts (SHA256 per file) for reproducibility (`validation/embedding_subsets/hashes_<name>.json`).
  7. **Persist report** to `validation/embedding_script_parity_report.csv` and save visual diffs (scatter + hist) to `validation/embedding_script_parity_plots/<name>_*.png`.

### 5. Subset Strategy (initial draft)
| Embedding | Subset Approach | Rationale |
| --- | --- | --- |
| scGPT gene | Limit to ≤200 genes drawn from shared vocabulary; subset `best_model.pt` weights by index list → reduces load while keeping identical ordering. Need utility to extract submatrix without loading entire 1M parameter set into GPU. |
| scFoundation gene | Similar subset of gene names; rely on `demo.h5ad` metadata to map names. |
| GEARS GO perturbation | Sample 150 genes / edges from `go_essential_all.csv`, maintain connectivity; store as CSV subset. |
| PCA perturbation (per dataset) | Take first 100 perturbations from `perturb_processed.h5ad`, drop extra genes (top 500 HVGs). Save trimmed `.h5ad` for both R and Python to consume. |

> **Update (2025-11-14):** `src/embeddings/build_embedding_subsets.py` now materializes the GO and Replogle subsets into `validation/embedding_subsets/` with hashes captured in `manifest.json`. scGPT checkpoint download remains the outstanding blocker.

### 6. Immediate Next Steps
1. ✅ Document source inventory (this file).
2. 🔄 Locate / request access paths for scGPT & scFoundation checkpoints; if unavailable, create TODO + placeholder config entries.
3. 🔄 Design subset extraction utilities (`validation/subset_utils.py`).
4. 🔄 Scaffold embedding module interface + registry.
5. 🔄 Implement parity harness + CLI plumbing.

### 7. Blockers / Dependencies
- External checkpoints for scGPT/scFoundation (must confirm storage location or provide download instructions).
- R runtime with required packages (`tidyverse`, `argparser`, `Matrix`, `igraph`, `irlba`, etc.) for running legacy scripts. Optional Issue #19 (Docker) would mitigate this.

### 8. Deliverables Recap
- Python modules under `src/embeddings/`
- `validation/embedding_script_parity_report.csv`
- `validation/embedding_script_parity_plots/`
- Subset hashes and raw inputs for reproducibility
- CLI task `validate-embeddings`



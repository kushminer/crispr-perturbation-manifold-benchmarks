# Data Sources & Environment Setup

## Local Paths

| Dataset | Default env var | Example path |
| --- | --- | --- |
| Adamson UPR (pseudobulk + single-cell) | `DATA_ADAMSON` | `/Users/<user>/Documents/classes/nih_research/data_adamson/perturb_processed.h5ad` |
| Replogle K562 Essential | `DATA_K562` | `/Users/<user>/Documents/classes/nih_research/data_replogle_k562_essential/perturb_processed.h5ad` |
| Replogle RPE1 Essential | `DATA_RPE1` | `/Users/<user>/Documents/classes/nih_research/data_replogle_rpe1_essential/perturb_processed.h5ad` |

Set these variables in `.env` (copy from `.env.example`). Runner scripts
read them directly; you can override via CLI flags if needed.

## Splits

Split configs live in `results/single_cell_analysis/*_split_config.json`.
Auto-generated if missing, but you can edit them to control train/test
perturbations. Keep dataset-specific splits committed for
reproducibility.

## External References

| Dataset | Source | Notes |
| --- | --- | --- |
| Adamson UPR | GEO: GSE149383 | Preprocessed to `.h5ad`; includes condition labels + GO annotations. |
| Replogle K562 Essential | Replogle et al. 2020 (Cell) | Provided via lab data portal; ensure gene names align with GO files. |
| Replogle RPE1 Essential | Replogle et al. 2020 (Cell) | Same preprocessing pipeline as K562. |

## Embedding Resources

| Resource | Path | Description |
| --- | --- | --- |
| scGPT gene embeddings | `data/models/scgpt/scGPT_human` | Required for `lpm_scgptGeneEmb`. |
| scFoundation checkpoints | `data/models/scfoundation/models.ckpt` + `demo.h5ad` | Used by `lpm_scFoundationGeneEmb`. |
| GEARS perturbation embeddings | `../paper/benchmark/data/gears_pert_data/` | CSV used by `lpm_gearsPertEmb`. |

Ensure these paths are accessible before running baselines/LSFT.

## Environment

1. Install dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate lpm
   ```
2. Verify Python version via `.python-version` (pyenv) if using editors
   like VS Code / Cursor.
3. Optional: run `pip install -r requirements.txt` if using plain venv.

## Data Integrity Checks

- `deliverables/audits/single_cell_data_audit/cell_counts.py` reports per-perturbation
  cell counts and sparsity.
- `deliverables/audits/single_cell_data_audit/validate_embeddings.py` ensures GEARS,
  scGPT, etc., produce unique metrics.

## Privacy / Licensing

All datasets listed above are publicly available (GEO / published
supplements). Follow the original licenses when redistributing data or
derivative results. Document any new datasets in this file and update
`.env.example` accordingly.



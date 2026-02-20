# LPM Evaluation Pipeline

## Overview

```
┌────────────┐   ┌────────────┐   ┌────────┐   ┌───────────┐
│ Baselines  │ → │ LSFT sweep │ → │ LOGO   │ → │ Publication│
└────────────┘   └────────────┘   └────────┘   └───────────┘
       │                │               │             │
       └── audits ──────┴─────── results ┴── figures ─┘
```

The pipeline operates on each dataset (Adamson, K562, RPE1) to validate
the Manifold Law diagnostics at single-cell resolution.

## Stages

1. **Baselines** (`run_single_cell_baselines.sh`)
   - Loads dataset splits and runs every baseline via
     `baseline_runner_single_cell.py`.
   - Outputs: `results/single_cell_analysis/<dataset>/<baseline>/`.

2. **LSFT** (`run_single_cell_lsft.sh`)
   - Executes `lsft_single_cell.py` for each baseline with configurable
     top-percent filters.
   - Outputs: `results/single_cell_analysis/<dataset>/lsft/`.

3. **LOGO** (`run_single_cell_logo.sh`)
   - Uses `logo_single_cell.py` to hold out GO classes (default:
     “Transcription”).
   - Outputs: `results/single_cell_analysis/<dataset>/logo/`.

4. **Analysis & Reporting**
   - `src/analysis/pseudobulk_vs_single_cell.py` generates unified tables
     and figures (`results/single_cell_analysis/comparison/`).
   - `deliverables/audits/single_cell_data_audit/*.py` runs the GEARS vs PCA audit.

5. **Publication Package**
   - `deliverables/publication_package/generate_all_reports.sh` regenerates epic
     plots, summary tables, and the 5-epic winner grid.

## Dependencies

| Stage | Inputs | Outputs | Consumers |
| --- | --- | --- | --- |
| Baselines | `.env` dataset paths, split configs | `results/.../<baseline>/pert_metrics.csv` | LSFT, comparison report |
| LSFT | Baseline metrics, cell embeddings | `lsft_single_cell_summary_*.csv` | Comparison report, story plots |
| LOGO | Baseline models | `logo_single_cell_summary_*.csv` | Story plots, publication |
| Audits | Baseline results | `deliverables/audits/.../output/*.csv/.png` | README, integrity notes |
| Publication | Results CSVs | `deliverables/publication_package/poster_figures/*.png` | Manuscript / poster |

## Command Reference

| Command | Description |
| --- | --- |
| `bash run_single_cell_baselines.sh` | Run baselines for all datasets |
| `bash run_single_cell_lsft.sh` | Run LSFT sweeps |
| `bash run_single_cell_logo.sh` | Run LOGO experiments |
| `python src/analysis/pseudobulk_vs_single_cell.py` | Regenerate summary tables/plots |
| `python deliverables/audits/single_cell_data_audit/validate_embeddings.py` | Compare baselines (e.g., GEARS vs PCA) |
| `python deliverables/audits/single_cell_data_audit/audit_visuals.py` | Render audit plots |
| `bash deliverables/publication_package/generate_all_reports.sh` | Regenerate publication figures |

## Tips

- Use `results/README.md` to understand where each output lands.
- Record reruns / major fixes in `docs/CHANGELOG.md`.
- For partial reruns, target specific datasets via the scripts’ CLI flags
  (see script headers for usage).



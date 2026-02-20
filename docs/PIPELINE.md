# Pipeline

## Scope

This repository evaluates CRISPR perturbation prediction baselines and their
local/generalization extensions:

- Baselines (goal 2)
- LSFT (goal 3, local training)
- LOGO (goal 3/4, functional-class holdout)
- Cross-resolution aggregation (pseudobulk vs single-cell)

## Preferred End-to-End Entry Point

Run the single demo to regenerate aggregate tables and verified conclusions:

```bash
python3 scripts/demo/run_end_to_end_results_demo.py
```

This writes refreshed outputs into `aggregated_results/`.

## Core Rebuild Steps

1. Aggregate all available outputs:

```bash
python3 src/analysis/aggregate_all_results.py
```

2. Build pseudobulk vs single-cell comparisons:

```bash
python3 src/analysis/pseudobulk_vs_single_cell.py
```

3. (Optional) Re-run execution batches (dataset-level outputs):

```bash
bash scripts/execution/run_single_cell_baselines.sh
bash scripts/execution/run_single_cell_lsft.sh
bash scripts/execution/run_single_cell_logo.sh
```

## Output Locations

- Active aggregate outputs: `aggregated_results/`
- Local experiment outputs: `results/` (gitignored in this repo)
- Archived historical outputs/docs: `deliverables/archive/`

## Notebook Walkthrough

- `tutorials/tutorial_end_to_end_results.ipynb`

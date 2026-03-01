# Pipeline

## Scope

This repository evaluates CRISPR perturbation prediction baselines and their local and extrapolation extensions:

- Baselines
- LSFT
- LOGO
- Cross-resolution aggregation from pseudobulk to single-cell

## Preferred End-to-End Entry Point

Run the single demo to regenerate aggregate tables and the verified conclusions:

```bash
python3 scripts/demo/run_end_to_end_results_demo.py
```

This refreshes outputs in `aggregated_results/` from local `results/` artifacts.

## Core Rebuild Steps

1. Aggregate all available outputs:

```bash
python3 src/analysis/aggregate_all_results.py
```

2. Build pseudobulk vs single-cell comparison summaries:

```bash
python3 src/analysis/pseudobulk_vs_single_cell.py
```

3. Optional dataset-level reruns:

```bash
bash scripts/execution/run_single_cell_baselines.sh
bash scripts/execution/run_single_cell_lsft.sh
bash scripts/execution/run_single_cell_logo.sh
```

## Output Locations

- `aggregated_results/`: committed summary tables and markdown conclusions
- `results/`: local experiment outputs used to build those summaries

## Notebook Walkthrough

- `tutorials/tutorial_end_to_end_results.ipynb`

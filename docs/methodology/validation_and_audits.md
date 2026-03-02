# Validation and Audit Methodology

## Scope

The maintained validation surface in this repository is small and explicit:

- unit tests in `tests/`
- parity artifacts in `validation/`
- runtime checks in the baseline runners
- targeted validation helpers under `scripts/`

## Main Validation Layers

### 1. Unit-Level Validation

The reusable code path is covered by pytest tests for:

- IO and config handling
- linear model solving
- metrics
- bootstrapping and validation helpers
- functional-class utilities

Run with:

```bash
PYTHONPATH=src python3 -m pytest tests -q
```

### 2. Embedding and Parity Validation

Committed parity artifacts live under `validation/`:

- `validation/embedding_parity/embedding_parity_report.md`
- `validation/embedding_parity/embedding_script_parity_report.csv`
- `validation/paper_python_vs_r/paper_python_vs_r_comparison.csv`
- `validation/r_parity/r_parity_validation.csv`

These files document that the Python implementation remained aligned with the reference implementation and saved embedding subsets.

### 3. Runtime Guards in Baseline Execution

The main baseline code contains checks to keep embedding sources and paths honest:

- `src/goal_2_baselines/baseline_runner.py`
- `src/goal_2_baselines/baseline_runner_single_cell.py`
- `src/goal_2_baselines/split_logic.py`

These are the places that matter if you are reviewing whether a baseline is truly using the requested embeddings and whether dataset paths resolve to the canonical repo layout.

### 4. Targeted Validation Helper

For single-cell baseline comparisons, use:

- `scripts/validate_single_cell_baselines.py`

This is the focused script for checking that two baselines produce distinct embeddings and predictions on the same dataset.

## Practical Review Standard

For this repo, a change is considered safe when all of the following hold:

1. `PYTHONPATH=src python3 -m pytest tests -q` passes.
2. `python3 -m compileall src scripts` passes.
3. `python3 scripts/demo/run_end_to_end_results_demo.py` still regenerates the committed aggregate summaries.

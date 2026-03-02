# Notebook Testing Guide

The maintained notebook surface is intentionally small.

## Notebook Under Test

- `tutorial_end_to_end_results.ipynb`

## What To Validate

1. The notebook opens cleanly.
2. The notebook executes top to bottom with a Python kernel that has the project requirements installed.
3. The demo script prints the verified conclusions.
4. The notebook reads the committed files in `aggregated_results/` without hardcoded local paths.

## Automated Check

Run:

```bash
cd tutorials
python3 test_notebooks.py
```

## Manual Check

From the repository root:

```bash
jupyter notebook tutorials/tutorial_end_to_end_results.ipynb
```

Run all cells and confirm the summary markdown and core tables render.

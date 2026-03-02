# Unit Test Suite

This directory contains the maintained unit tests for the reusable framework code under `src/`.

## Current Test Files

```text
tests/
├── conftest.py
├── test_bootstrapping.py
├── test_functional_class.py
├── test_hardness_api.py
├── test_io.py
├── test_linear_model.py
├── test_metrics.py
└── test_validation.py
```

## What The Suite Covers

- `test_bootstrapping.py`: bootstrap confidence intervals and correlation CI helpers.
- `test_functional_class.py`: functional-class holdout behavior and result formatting.
- `test_hardness_api.py`: hardness computations over target/train perturbation geometry.
- `test_io.py`: loading utilities for JSON, CSV/TSV, annotations, and gene metadata.
- `test_linear_model.py`: linear model fitting, prediction, PCA dimension handling, and reproducibility.
- `test_metrics.py`: Pearson/L2 metric behavior across normal and edge cases.
- `test_validation.py`: annotation, LOGO, hardness-bin, and class-holdout validation helpers.

## Run The Tests

```bash
PYTHONPATH=src pytest tests -q
```

Verbose run:

```bash
PYTHONPATH=src pytest tests -v
```

Collect-only check:

```bash
PYTHONPATH=src pytest tests --collect-only -q
```

## Current Suite Size

- `52` collected tests
- fast unit-level coverage of the shared framework utilities

## Notes

- The unit suite is intentionally scoped to reusable code paths, not full raw-data reruns.
- Full framework reruns depend on the external datasets and model checkpoints documented in `data/README.md` and `docs/data_sources.md`.

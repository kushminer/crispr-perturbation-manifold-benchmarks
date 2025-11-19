# Unit Test Suite

Comprehensive unit tests for the linear perturbation evaluation framework.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_metrics.py          # Metrics computation tests
├── test_io.py               # I/O function tests
├── test_linear_model.py     # Linear model fitting tests
├── test_logo_hardness.py    # LOGO + hardness evaluation tests
├── test_functional_class.py # Functional-class holdout tests
├── test_combined_analysis.py # Combined analysis tests
└── test_validation.py       # Validation function tests
```

## Running Tests

```bash
# Run all tests
PYTHONPATH=src pytest tests/

# Run with verbose output
PYTHONPATH=src pytest tests/ -v

# Run specific test file
PYTHONPATH=src pytest tests/test_metrics.py

# Run specific test
PYTHONPATH=src pytest tests/test_metrics.py::test_compute_metrics_perfect_correlation

# Run with coverage
PYTHONPATH=src pytest tests/ --cov=src/eval_framework --cov-report=html
```

## Test Coverage

- **45 tests** covering all major modules
- **Runtime**: <2 seconds
- **Coverage**: Tests all evaluation modules, I/O, metrics, and validation

## Test Categories

### Metrics Tests (`test_metrics.py`)
- Perfect correlation
- Negative correlation
- No correlation
- Constant values
- Edge cases (NaN, different lengths)

### I/O Tests (`test_io.py`)
- JSON expression loading
- CSV/TSV loading
- Annotation loading
- Gene names handling
- Error handling

### Linear Model Tests (`test_linear_model.py`)
- Basic model fitting
- PCA dimension capping
- Prediction functionality
- Reproducibility
- Shape validation

### LOGO Tests (`test_logo_hardness.py`)
- Similarity matrix computation
- Hardness bin assignment
- Full LOGO evaluation
- Cluster blocking
- Small dataset handling

### Functional-Class Tests (`test_functional_class.py`)
- Basic class holdout
- Min class size filtering
- Class representation
- Insufficient training data handling

### Combined Analysis Tests (`test_combined_analysis.py`)
- Result table loading
- Summary computation
- Heatmap generation
- Empty data handling

### Validation Tests (`test_validation.py`)
- Annotation quality checks
- LOGO integrity validation
- Hardness bin validation
- Class holdout validation
- Summary formatting

## Fixtures

Shared fixtures in `conftest.py`:
- `random_seed`: Fixed random seed for reproducibility
- `synthetic_expression_matrix`: 50×100 expression matrix
- `synthetic_annotations`: Balanced class annotations
- `temp_dir`: Temporary directory for test outputs
- `sample_predictions_json`: Sample prediction files
- `sample_logo_results`: Sample LOGO results DataFrame
- `sample_class_results`: Sample class results DataFrame

## Continuous Integration

Tests are designed to:
- Run quickly (<30 seconds)
- Be deterministic (fixed seeds)
- Cover edge cases
- Handle empty/missing data gracefully
- Validate error conditions


# Dependency Analysis (Issue 0.5)

**Date:** 2025-11-14  
**Status:** ✅ **COMPLETE**

---

## Dependency Audit Results

### Core Dependencies (Required)

| Package | Version | Used In | Purpose |
|---------|---------|---------|---------|
| `numpy` | >=1.24 | All modules | Numerical computations |
| `pandas` | >=2.0 | All modules | Data manipulation |
| `scipy` | >=1.11 | embeddings, eval_framework | Scientific computing (sparse matrices, eigs) |
| `scikit-learn` | >=1.3 | embeddings, eval_framework | PCA, machine learning utilities |
| `anndata` | >=0.10 | embeddings | Single-cell data handling |
| `pyyaml` | >=6.0 | eval_framework | Configuration file parsing |
| `matplotlib` | >=3.7 | visualization | Plotting |
| `seaborn` | >=0.12 | visualization, comparison | Statistical visualizations |
| `umap-learn` | >=0.5 | visualization | UMAP dimensionality reduction |
| `torch` | >=2.0 | embeddings (scGPT, scFoundation) | PyTorch for model checkpoint loading |

### Testing Dependencies

| Package | Version | Used In | Purpose |
|---------|---------|---------|---------|
| `pytest` | >=7.0 | tests/ | Test framework |
| `pytest-cov` | >=4.0 | tests/ | Test coverage |

### Optional Dependencies (Commented Out)

| Package | Version | Used In | Purpose | Notes |
|---------|---------|---------|---------|-------|
| `gseapy` | >=1.0 | `annotate_classes.py` | GO enrichment analysis | Only needed for annotation generation |
| `mygene` | >=3.2 | `annotate_classes.py` | Gene ID mapping | Only needed for annotation generation |
| `requests` | >=2.31 | (via mygene) | HTTP requests | Only needed if using mygene |

**Note:** Optional dependencies are commented out in `requirements.txt`. Uncomment if you need to use `annotate_classes.py` for GO enrichment.

---

## Changes Made

### ✅ Added Missing Dependency
- **`torch>=2.0`** - Required for scGPT and scFoundation embedding loaders (was missing!)

### ✅ Removed Unused Dependencies
- None removed (all were being used)

### ✅ Organized Dependencies
- Grouped into: Core, PyTorch, Testing, Optional
- Commented out optional dependencies with clear notes

---

## Verification

### Core Functionality
All core dependencies are required for:
- ✅ Data loading and preprocessing
- ✅ Embedding extraction (scGPT, scFoundation, GEARS, PCA)
- ✅ Linear baseline models
- ✅ LOGO evaluation
- ✅ Functional-class evaluation
- ✅ Combined analysis
- ✅ Visualization

### Optional Functionality
Optional dependencies are only needed for:
- Annotation generation via GO enrichment (`annotate_classes.py`)

---

## Fresh Environment Test

To verify a fresh environment works:

```bash
# Create fresh environment
python -m venv test_env
source test_env/bin/activate  # or `test_env\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Test core functionality
PYTHONPATH=src python -c "from eval_framework.io import load_expression_dataset; print('✓ Core imports work')"
PYTHONPATH=src python -c "from embeddings.registry import get; print('✓ Embeddings work')"
PYTHONPATH=src python -c "from eval_framework.linear_model import LinearModel; print('✓ Models work')"
```

---

**Last Updated:** 2025-11-14


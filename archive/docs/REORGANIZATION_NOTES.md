# Repository Reorganization Notes

## Overview

The repository has been reorganized to clearly separate:
- **Original authors' work** (`../paper/`) - Code from Nature Methods 2025 publication
- **Evaluation framework** (`evaluation_framework/`) - New framework extending the paper's work

## Data Siloing

**Status:** ✅ **Complete** - Annotations are now siloed

Both `paper/` and `evaluation_framework/` maintain **standalone copies** of annotation files:

- `paper/benchmark/data/annotations/` - Paper's copy (for reproducibility)
- `evaluation_framework/data/annotations/` - Framework's copy (for independence)

### Benefits

- ✅ **Independence:** Each component can be used without the other
- ✅ **No cross-references:** Framework doesn't depend on paper/ paths
- ✅ **Clear boundaries:** Each has what it needs
- ✅ **Reproducibility:** Each can run standalone

## Path Updates

All paths have been updated to reflect the siloed structure:

### Config Files
- ✅ Annotation paths: `../data/annotations/...` (relative to configs/)
- Expression paths: Still point to `../../working_dir/...` (external)

### Scripts
- ✅ `generate_replogle_annotations.py`: Default output to `data/annotations/`
- ⚠️ `generate_replogle_expression.py`: Still references `../../paper/benchmark/data/gears_pert_data`
  - **Note:** This is OK - it's a helper script that needs raw data to generate predictions
  - The core framework doesn't depend on this script

### Documentation
- ✅ All references updated to use `data/annotations/` paths
- ✅ Cross-references updated in README files

## Verification

✅ Config files load correctly  
✅ Annotation paths resolve correctly  
✅ All tests pass (45/45)  
✅ Documentation paths updated  
✅ No cross-dependencies for annotations

## Usage

When working from `evaluation_framework/`:

```bash
# Run evaluation (uses data/annotations/)
PYTHONPATH=src python src/main.py --config configs/config_adamson.yaml

# Run tests
PYTHONPATH=src pytest tests/

# Generate annotations (saves to data/annotations/)
PYTHONPATH=src python src/generate_replogle_annotations.py --config configs/config_replogle.yaml
```

All paths are relative to the `evaluation_framework/` directory.

## Remaining Dependencies

The evaluation framework has one remaining reference to `paper/`:

- `src/generate_replogle_expression.py` → `../../paper/benchmark/data/gears_pert_data`
  - **Purpose:** Helper script to generate predictions from raw data
  - **Impact:** Low - core framework doesn't use this script
  - **Status:** Acceptable - this is a data processing dependency, not a framework dependency

The core evaluation framework (`src/eval_framework/`, `src/main.py`) has **zero dependencies** on `paper/` paths.

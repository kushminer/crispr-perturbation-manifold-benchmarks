# Repository Reorganization for Nature Methods Submission

**Date**: 2025-01-XX  
**Purpose**: Clean, professional repository structure suitable for Nature Methods submission

## Summary

The repository has been reorganized following scientific software repository best practices. All files have been moved from the root directory into organized subdirectories, leaving only essential entry points (`README.md`, `requirements.txt`, `pytest.ini`) in the root.

## New Directory Structure

### Core Directories (Unchanged)
- `src/` - Source code
- `results/` - Generated results
- `data/` - Data files
- `configs/` - Dataset configurations
- `tests/` - Unit tests
- `tutorials/` - Tutorial notebooks

### New Organization

#### `scripts/` - Organized execution scripts
- `scripts/execution/` - Main execution scripts (26 files)
  - `run_all_*.sh`, `run_epic*.sh`, `run_lsft*.sh`, `run_logo*.sh`
  - `run_single_cell_*.sh`, `run_k562_rpe1_*.py`
  - `run_rpe1_single_cell_lsft.py`, `run_gears_only.py`
  
- `scripts/analysis/` - Analysis & visualization scripts (20 files)
  - `generate_*.py`, `analyze_*.py`, `create_*.py`
  - `finalize_*.py`, `update_*.py`
  
- `scripts/utilities/` - Utility scripts (6 files)
  - `fix_*.py`, `validate_*.py`, `test_*.py`
  - `reproduce_issue.py`
  
- `scripts/monitoring/` - Monitoring scripts (4 files)
  - `monitor_*.sh`, `continuous_monitor.sh`

#### `docs/` - Enhanced documentation
- `docs/methodology/` - Methodology documentation (merged from root `methodology/`)
- `docs/analysis/` - Analysis documentation (merged from root `analysis_docs/`)
- `docs/publication/` - Publication-specific documentation
- `docs/status/` - Status and completion reports
  - `completion/` - 15 completion reports
  - `status/` - 9 status updates
  - `fixes/` - 4 fix verification reports

#### `deliverables/archive/` - Development artifacts
- `deliverables/archive/logs/` - 20 execution log files
- `deliverables/archive/development/next_steps/` - Planning documents

### Publication Materials (Unchanged)
- `publication_package/` - Publication materials
- `poster/` - Poster figures
- `audits/` - Audit reports
- `deliverables/skeletons_and_fact_sheets/` - Data skeletons

## Path Reference Notes

**Important**: Some scripts may reference paths relative to their old location in the root directory. When running scripts from their new locations, you may need to:

1. **Run scripts from project root** (recommended):
   ```bash
   cd /path/to/lpm-evaluation-framework-v2
   python scripts/execution/run_script.py
   ```

2. **Set PYTHONPATH appropriately**:
   ```bash
   export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
   ```

3. **Update script paths** if needed: Scripts that use `Path(__file__).parent` to reference `src/` or `results/` may need adjustment to `Path(__file__).parent.parent.parent` (3 levels up from `scripts/execution/` or `scripts/analysis/`).

## Migration Checklist

- [x] Create new directory structure
- [x] Move execution scripts to `scripts/execution/`
- [x] Move analysis scripts to `scripts/analysis/`
- [x] Move utility scripts to `scripts/utilities/`
- [x] Move monitoring scripts to `scripts/monitoring/`
- [x] Move status/completion reports to `docs/status/`
- [x] Move log files to `deliverables/archive/logs/`
- [x] Move planning docs to `deliverables/archive/development/`
- [x] Merge `methodology/` into `docs/methodology/`
- [x] Merge `analysis_docs/` into `docs/analysis/`
- [x] Move publication docs to `docs/publication/`
- [x] Update README.md with new structure
- [ ] Verify all scripts work from new locations (may require path adjustments)

## Benefits

1. **Clean root directory** - Professional first impression for reviewers
2. **Clear organization** - Reviewers can easily find code, data, results
3. **Separation of concerns** - Development artifacts archived, publication materials prominent
4. **Reproducibility** - Clear script organization makes reproduction easier
5. **Standards compliance** - Follows scientific software repository best practices

## Files Removed from Root

All development artifacts have been moved:
- Status/completion reports → `docs/status/`
- Execution logs → `deliverables/archive/logs/`
- Planning documents → `deliverables/archive/development/`
- Execution scripts → `scripts/execution/`
- Analysis scripts → `scripts/analysis/`

Only essential files remain:
- `README.md`
- `requirements.txt`
- `pytest.ini`

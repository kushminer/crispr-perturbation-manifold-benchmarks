# Repository Cleanup Plan

**Date:** 2025-11-24  
**Status:** ✅ **COMPLETED**

**Before:** ~12.5 GB  
**After:** ~5.5 GB  
**Saved:** ~7 GB (56%)

---

## Current Structure (Before Cleanup)

| Directory | Size | Status | Action |
|-----------|------|--------|--------|
| `lpm-evaluation-framework-v2/` | 5.4 GB | ✅ KEEP | Main framework (active) |
| `paper_backup_20251119_114402/` | 3.6 GB | 🗑️ DELETE | Old backup |
| `evaluation_framework/` | 3.1 GB | 🗑️ DELETE | Older version, duplicates v2 |
| `paper/` | 441 MB | ✅ KEEP | Paper files |
| `illustrations/` | 1.1 MB | ✅ KEEP | Design files |
| `deliverables/archive/` | 96 KB | ⚠️ REVIEW | Deprecated scripts |
| `data/` | 20 KB | ✅ KEEP | Annotations |
| `deliverables/skeletons_and_fact_sheets/` | 8 KB | ⚠️ REVIEW | Planning docs |
| `reference_data/` | 4 KB | ✅ KEEP | Reference data |
| `validation/` | 0 B | 🗑️ DELETE | Empty directory |

---

## Cleanup Actions

### Phase 1: Remove Obvious Duplicates (Safe)

1. **Delete `paper_backup_20251119_114402/`** (3.6 GB)
   - This is a dated backup of the paper directory
   - The current `paper/` directory is the active version

2. **Delete empty `validation/`** (0 B)
   - Empty directory at root level

### Phase 2: Merge and Remove evaluation_framework (Careful)

3. **Merge unique files from `evaluation_framework/` → `lpm-evaluation-framework-v2/`**
   - Check for any unique annotations or results
   - Then delete the duplicate directory

### Phase 3: Clean Within lpm-evaluation-framework-v2

4. **Review and clean subdirectories:**
   - `audits/` (1.1 MB) - audit logs, likely can be trimmed
   - `deliverables/archive/` (376 KB) - old versions
   - `deliverables/mentor_review/` (1.2 MB) - review documents
   - `publication_figures/` vs `publication_package/` - possible overlap

5. **Clean results directories:**
   - Remove intermediate/debug outputs
   - Keep only final results

### Phase 4: Organize Top Level

6. **Create clean top-level structure:**
   ```
   linear_perturbation_prediction-Paper/
   ├── lpm-evaluation-framework-v2/  # Main framework
   ├── paper/                         # Paper files
   ├── illustrations/                 # Design files
   ├── data/                          # Shared data
   └── README.md                      # Repository overview
   ```

---

## Proposed Clean Structure

```
linear_perturbation_prediction-Paper/
├── README.md                    # Repository overview
├── lpm-evaluation-framework-v2/ # Main evaluation framework
│   ├── src/                     # Source code
│   ├── configs/                 # Configuration files
│   ├── data/                    # Framework-specific data
│   ├── results/                 # Experiment results
│   ├── publication_package/     # Publication-ready outputs
│   ├── tests/                   # Unit tests
│   ├── tutorials/               # Usage tutorials
│   └── docs/                    # Documentation
├── paper/                       # Paper source files
│   ├── benchmark/               # Benchmark code
│   ├── notebooks/               # Analysis notebooks
│   └── plots/                   # Generated plots
├── illustrations/               # Design assets
└── data/                        # Shared annotations
```

---

## Safety Checklist

Before deleting anything:

- [ ] Verify `lpm-evaluation-framework-v2/` has all needed files
- [ ] Check for unique files in `evaluation_framework/` not in v2
- [ ] Confirm `paper/` is more recent than `paper_backup_*/`
- [ ] Back up to external location if desired

---

## Commands to Execute

### Phase 1: Safe Deletions
```bash
# Delete empty validation directory
rm -rf validation/

# Delete paper backup (after confirming paper/ is current)
rm -rf paper_backup_20251119_114402/
```

### Phase 2: Merge and Remove
```bash
# First, check for unique files in evaluation_framework
diff -rq evaluation_framework/data lpm-evaluation-framework-v2/data

# Copy any unique files, then delete
rm -rf evaluation_framework/
```

---

## Estimated Result

| Before | After | Savings |
|--------|-------|---------|
| 12.5 GB | 5.8 GB | 6.7 GB (53%) |

---

## Cleanup Completed

### Actions Taken:

1. ✅ Deleted `paper_backup_20251119_114402/` (3.6 GB)
2. ✅ Deleted `evaluation_framework/` (3.1 GB) after preserving unique files
3. ✅ Deleted empty `validation/` directory
4. ✅ Removed `validation/legacy_runs/` (222 MB)
5. ✅ Removed `validation/embedding_subsets/` (215 MB)
6. ✅ Merged `publication_figures/` into `publication_package/poster_figures/`
7. ✅ Cleaned `__pycache__`, `.DS_Store`, `.pyc` files
8. ✅ Preserved unique annotation file before deletion

### Final Structure:

```
linear_perturbation_prediction-Paper/   (~5.5 GB)
├── lpm-evaluation-framework-v2/        (5.0 GB) - Main framework
├── paper/                              (441 MB) - Paper files
├── illustrations/                      (1.1 MB) - Design files
├── deliverables/archive/                            (96 KB)  - Deprecated scripts
├── data/                               (20 KB)  - Shared annotations
├── reference_data/                     (4 KB)   - Reference data
├── deliverables/skeletons_and_fact_sheets/          (8 KB)   - Planning docs
├── CLEANUP_PLAN.md                     - This file
└── README.md
```


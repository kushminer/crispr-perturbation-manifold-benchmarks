# Sprint 11 Progress Tracker

**Status**: 🚧 In Progress

## Issues Status

| Issue | Title | Status | Notes |
|-------|-------|--------|-------|
| 1 | Create resampling-enabled repository (v2) | ✅ PREPARED | Setup guides ready; user action required |
| 2 | Set up CI pipelines | ✅ COMPLETE | `.github/workflows/ci.yml` created |
| 3 | Implement bootstrap CI utility | ✅ COMPLETE | `src/stats/bootstrapping.py` + tests |
| 4 | Implement paired permutation test | ✅ COMPLETE | `src/stats/permutation.py` created |
| 5 | LSFT output standardization | 📋 Ready | Will standardize per-perturbation output |
| 6 | Add bootstrap CIs to LSFT summaries | 📋 Ready | Depends on Issue 3 |
| 7 | Add paired baseline comparisons | 📋 Ready | Depends on Issues 4 & 6 |
| 8 | Hardness-performance regression | 📋 Ready | Depends on Issue 3 |
| 9 | Optional LOGO resampling | 📋 Ready | Depends on Issues 3 & 4 |
| 10 | Update visualizations | 📋 Ready | Depends on Issues 6, 7, 8 |
| 11 | Engine parity verification | 📋 Ready | Compare v1 vs v2 |
| 12 | Documentation | 📋 Ready | Create `docs/resampling.md` |

## Completed Work

### Issue 1: Repository Setup ✅

**Files Created**:
- `CHANGELOG.md` - Tracks Sprint 11 changes
- `V2_RESAMPLING_README.md` - Template README for v2
- `REPOSITORY_SETUP_INSTRUCTIONS.md` - Step-by-step setup guide
- `SPRINT_11_SETUP.md` - Repository strategy overview
- `docs/SPRINT_11_ISSUE1_STATUS.md` - Issue 1 status

**Status**: Ready for user to create GitHub repository.

### Issue 2: CI Setup ✅

**Files Created**:
- `.github/workflows/ci.yml` - GitHub Actions workflow

**Features**:
- Lint and format checking (black, isort, flake8)
- Unit tests (pytest)
- Smoke test (imports, CLI, key modules)
- Runs on push/PR to main or sprint11-resampling branch

### Issue 3: Bootstrap CI Utility ✅

**Files Created**:
- `src/stats/__init__.py` - Stats module init
- `src/stats/bootstrapping.py` - Bootstrap functions
- `tests/test_bootstrapping.py` - Unit tests

**Functions**:
- `bootstrap_mean_ci()` - Percentile bootstrap for mean metrics
- `bootstrap_correlation_ci()` - Bootstrap for correlation coefficients

**Tested**: ✓ Works correctly with realistic Pearson r and L2 values

### Issue 4: Permutation Tests ✅

**Files Created**:
- `src/stats/permutation.py` - Permutation test functions

**Functions**:
- `paired_permutation_test()` - Sign-flip permutation test
- `paired_permutation_test_two_sample()` - Wrapper for two groups

**Tested**: ✓ Works correctly with paired deltas

## Next Steps

1. **User Action**: Create GitHub repository (Issue 1)
2. **After Repo Created**: CI will run automatically (Issue 2)
3. **Continue Implementation**: 
   - Issue 5: Standardize LSFT output format
   - Issue 6: Add bootstrap CIs to summaries
   - Issue 7: Add paired baseline comparisons
   - Issue 8: Hardness-performance regression

## Statistics Module Structure

```
src/stats/
├── __init__.py          # Module exports
├── bootstrapping.py     # Bootstrap CI functions
└── permutation.py       # Permutation test functions
```

## Notes

- All preparation work is in `evaluation_framework/`
- User needs to create GitHub repository before proceeding
- CI workflow is ready to activate once repo exists
- Core statistics modules (Issues 3-4) are complete and tested
- Ready to integrate with LSFT (Issues 5-8)

# Sprint 11 - Issue 1 Status

## Issue 1: Create Resampling-Enabled Repository (v2)

**Status**: ✅ **PREPARED** (Ready for user action)

### What's Been Prepared

1. **CHANGELOG.md**
   - Created with Sprint 11 entries
   - Documents all planned enhancements
   - Tracks v1 baseline and v2 changes

2. **V2_RESAMPLING_README.md**
   - Template README for the v2 repository
   - Explains v1 vs v2 differences
   - Documents new resampling features
   - Usage examples for LSFT with resampling

3. **REPOSITORY_SETUP_INSTRUCTIONS.md**
   - Step-by-step guide for creating the new repository
   - Three options: New repo, Fork, or Local copy
   - Post-setup verification checklist

4. **SPRINT_11_SETUP.md**
   - Overview of repository strategy
   - File checklist
   - Next steps after repository creation

5. **docs/SPRINT_11_RESAMPLING_ENGINE.md**
   - Epic documentation
   - All 12 issues overview
   - Design decisions

### What Needs User Action

**Creating the GitHub Repository**:

Choose one of these options:

1. **New Repository** (Recommended):
   - Create `perturbench-resampling` on GitHub
   - Follow `REPOSITORY_SETUP_INSTRUCTIONS.md` Option 1

2. **Fork Current Repository**:
   - Fork if current repo is on GitHub
   - Follow `REPOSITORY_SETUP_INSTRUCTIONS.md` Option 2

3. **Local Development Copy**:
   - Create local copy first to test
   - Follow `REPOSITORY_SETUP_INSTRUCTIONS.md` Option 3

### Files Ready to Copy

Once the repository is created, these files are ready:
- ✅ All source code (`src/`)
- ✅ Configuration files (`configs/`)
- ✅ Documentation (`docs/`)
- ✅ Tests (`tests/`)
- ✅ `requirements.txt` (scipy already included)
- ✅ `pytest.ini`
- ✅ `.gitignore` (from parent repo)
- ✅ `CHANGELOG.md` (new, ready)
- ✅ `V2_RESAMPLING_README.md` (rename to `README.md`)

### Verification Checklist

After creating the repository, verify:

- [ ] Repository exists on GitHub (or locally)
- [ ] All `evaluation_framework/` files copied
- [ ] `README.md` updated (from `V2_RESAMPLING_README.md`)
- [ ] `CHANGELOG.md` present
- [ ] `requirements.txt` present
- [ ] Can install: `pip install -r requirements.txt`
- [ ] Basic imports work: `PYTHONPATH=src python -c "from goal_1_similarity import *"`
- [ ] Git initialized and first commit made

### Next Steps After Repository Creation

1. ✅ Issue 1: Repository created
2. → Issue 2: Set up CI pipelines (prepare workflows)
3. → Issue 3: Implement bootstrap CI utility

### Notes

- **scipy already in requirements.txt**: No additional dependencies needed for basic resampling
- **Point estimate parity**: v2 must match v1 exactly (only adds CIs)
- **Documentation**: All Sprint 11 work is documented in CHANGELOG.md

---

**Ready to proceed to Issue 2 once repository is created.**


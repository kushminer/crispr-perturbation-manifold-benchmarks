# Repository Setup Instructions for Sprint 11 (v2)

## Overview

Issue 1 requires creating a new repository for the resampling-enabled v2 engine. This document provides step-by-step instructions.

## Prerequisites

- GitHub account with repository creation permissions
- Git installed locally
- Current `evaluation_framework/` directory accessible

## Step-by-Step Instructions

### Option 1: Create New GitHub Repository (Recommended)

1. **Create repository on GitHub**:
   - Go to https://github.com/new
   - Repository name: `perturbench-resampling` (or your preferred name)
   - Description: "Resampling-enabled evaluation engine (v2) for linear perturbation prediction"
   - Visibility: Public or Private (your choice)
   - **Do NOT** initialize with README, .gitignore, or license (we'll add these)

2. **Clone and prepare locally**:
   ```bash
   # Clone the empty repository
   git clone <your-new-repo-url>
   cd perturbench-resampling
   
   # Copy all files from current evaluation_framework
   cp -r ../linear_perturbation_prediction-Paper/evaluation_framework/* .
   cp -r ../linear_perturbation_prediction-Paper/evaluation_framework/.* . 2>/dev/null || true
   
   # Update README for v2
   mv V2_RESAMPLING_README.md README.md
   
   # Ensure CHANGELOG.md exists (already created)
   # CHANGELOG.md should already be in place
   ```

3. **Update requirements.txt** (if needed):
   ```bash
   # Check if scipy is already in requirements.txt (it should be)
   grep -i scipy requirements.txt
   # If missing, add it:
   # scipy>=1.11
   ```

4. **Initial commit**:
   ```bash
   git add .
   git commit -m "Initial commit: v1 baseline (Sprint 11 - before resampling enhancements)"
   git push -u origin main
   ```

5. **Verify**:
   - Check GitHub repository has all files
   - Clone in a fresh directory and verify it installs/runs

### Option 2: Fork Current Repository

If the current repository is on GitHub:

1. **Fork on GitHub**:
   - Navigate to the original repository
   - Click "Fork" button
   - Choose your account/organization

2. **Rename the fork** (optional but recommended):
   - Go to repository Settings → General
   - Scroll to "Repository name"
   - Rename to `perturbench-resampling`

3. **Clone locally**:
   ```bash
   git clone <your-fork-url>
   cd perturbench-resampling
   ```

4. **Create Sprint 11 branch**:
   ```bash
   git checkout -b sprint11-resampling
   ```

5. **Update README and CHANGELOG**:
   ```bash
   # Replace README with v2 version
   mv V2_RESAMPLING_README.md README.md
   # CHANGELOG.md should already exist
   ```

6. **Commit and push**:
   ```bash
   git add .
   git commit -m "Sprint 11: Initialize resampling-enabled v2"
   git push -u origin sprint11-resampling
   ```

### Option 3: Local Development Copy (Test First)

If you want to test locally before creating GitHub repo:

```bash
# Create local copy
cd /Users/samuelminer/Documents/classes/nih_research/
cp -r linear_perturbation_prediction-Paper/evaluation_framework \
     perturbench-resampling-dev

cd perturbench-resampling-dev

# Update README
mv V2_RESAMPLING_README.md README.md

# Initialize git (optional, for local tracking)
git init
git add .
git commit -m "Initial: v1 baseline (Sprint 11 prep)"

# When ready, add remote:
# git remote add origin <your-repo-url>
# git push -u origin main
```

## Post-Setup Verification

After setting up the repository, verify:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify imports work
PYTHONPATH=src python -c "from goal_1_similarity import *; print('✓ Imports work')"

# 3. Run basic smoke test (if tests exist)
PYTHONPATH=src pytest tests/ -v -k "test_linear_model or test_metrics" || echo "Some tests may fail, this is expected"

# 4. Check key files exist
ls -la README.md CHANGELOG.md requirements.txt pytest.ini src/main.py
```

## Files Created for v2

The following files have been prepared:

- ✅ `CHANGELOG.md` - Tracks Sprint 11 changes
- ✅ `V2_RESAMPLING_README.md` - README template for v2 (rename to README.md)
- ✅ `SPRINT_11_SETUP.md` - This setup guide
- ✅ `docs/SPRINT_11_RESAMPLING_ENGINE.md` - Epic documentation

## Next Steps

After repository is created:

1. ✅ Issue 1 complete: Repository exists and mirrors v1
2. → Issue 2: Set up CI pipelines
3. → Issue 3: Implement bootstrap CI utility

## Notes

- **Preserve history**: If forking, git history is automatically preserved
- **Point estimates**: v2 must produce identical point estimates to v1
- **Documentation**: All v2 enhancements are documented in CHANGELOG.md
- **Testing**: Run smoke tests to ensure v1 functionality still works before adding resampling features

## Troubleshooting

### Issue: Files not copying correctly
- Make sure to copy hidden files: `cp -a source/. dest/`
- Check for `.gitignore` and `.github/` directories

### Issue: Import errors after copying
- Verify `PYTHONPATH=src` is set
- Check that all `__init__.py` files are present
- Ensure `requirements.txt` dependencies are installed

### Issue: Git history not preserved
- If using Option 1 (new repo), history starts fresh (this is fine)
- If using Option 2 (fork), history is automatically preserved


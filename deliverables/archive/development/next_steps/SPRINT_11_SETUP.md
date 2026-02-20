# Sprint 11 Setup: Resampling-Enabled Repository (v2)

## Issue 1: Create Resampling-Enabled Repository

This document outlines the setup for creating a new resampling-enabled repository for Sprint 11.

## Repository Strategy

**Approach**: Create a new repository (`perturbench-resampling` or similar) that:
- Starts from exact state of current `evaluation_framework/` (v1 baseline)
- Adds resampling enhancements (bootstrap CIs, permutation tests)
- Maintains point-estimate parity with v1
- Allows A/B comparison between v1 and v2 outputs

## Repository Creation Options

### Option A: New GitHub Repository (Recommended)

1. **Create new GitHub repository**:
   - Name: `perturbench-resampling` (or your preferred name)
   - Description: "Resampling-enabled evaluation engine (v2) for linear perturbation prediction"
   - Initialize with: README (we'll replace it)

2. **Clone locally**:
   ```bash
   git clone <new-repo-url>
   cd perturbench-resampling
   ```

3. **Copy current evaluation_framework**:
   ```bash
   # From parent directory
   cp -r evaluation_framework/* perturbench-resampling/
   cd perturbench-resampling
   ```

4. **Initial commit**:
   ```bash
   git add .
   git commit -m "Initial commit: v1 baseline (before resampling enhancements)"
   git push origin main
   ```

### Option B: Fork Current Repository

If the current repository is on GitHub:
1. Fork the repository on GitHub
2. Rename the fork to indicate it's the resampling version
3. Create `sprint11-resampling` branch for all Sprint 11 work

### Option C: Local Copy (For Development)

Create a local copy for development before pushing to GitHub:
```bash
cd /path/to/parent
cp -r evaluation_framework evaluation_framework_v2_resampling
cd evaluation_framework_v2_resampling
# Make changes, then initialize git when ready
```

## Files to Update for v2

Once the new repository structure is created, update:

1. **CHANGELOG.md** (create):
   - Add Sprint 11 entry
   - Document resampling enhancements

2. **README.md**:
   - Update title to indicate "v2 - Resampling-Enabled"
   - Add note about resampling features
   - Document how it extends v1

3. **requirements.txt**:
   - Add any new dependencies for resampling (scipy, statsmodels if not already present)

## Next Steps After Repository Creation

1. ✅ Repository exists and mirrors v1 exactly
2. ✅ All tests pass (verify v1 functionality)
3. ✅ README updated for v2
4. ✅ CHANGELOG.md created
5. Proceed to Issue 2 (CI setup)

## Verification Checklist

After creating the new repository:

- [ ] Repository exists and contains all v1 code
- [ ] All files from `evaluation_framework/` are present
- [ ] `requirements.txt` is present
- [ ] Tests directory exists
- [ ] README.md exists (ready to update)
- [ ] Can install and run: `pip install -r requirements.txt`
- [ ] Basic smoke test works: `python -m pytest tests/` (if dependencies installed)
- [ ] Git history preserved (if forking) or initial commit created

## Notes

- **Preserve commit history**: If forking, history is automatically preserved
- **Keep v1 accessible**: Maintain original repo for comparison
- **Version separation**: Clear distinction between v1 (current) and v2 (resampling)


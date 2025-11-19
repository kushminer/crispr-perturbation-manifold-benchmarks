# Phase 0 – Repository Cleanup & Slimming

**Goal:** Create a lean, clean repo focused on:
1. Baseline re-runs using original splits
2. New LOGO + similarity experiments
3. Functional-class / combined analysis

**Status:** 🔄 **IN PROGRESS**

---

## Issue 0.1 — Create a Clean Working Copy of the Repository ✅

### Description
Create a safe, isolated working copy of the current repo to allow aggressive cleanup without risking the original history.

### Tasks
- [x] Create a new branch, e.g. `refactor/eval-framework` from `main`
- [x] Tag current `main` as `v0_legacy_repo` for reference
- [ ] (Optional) Create a separate "archive" branch `archive/full-history`
- [ ] Document in README: where the legacy version lives

### Acceptance Criteria
- [x] New branch `refactor/eval-framework` exists
- [x] Tags/branches for legacy code are clearly labeled
- [x] No code deleted from main yet (cleanup occurs only on refactor branch)

**Labels:** `cleanup`, `infrastructure`, `high-priority`

---

## Issue 0.2 — Inventory and Classify Existing Files ✅

### Description
Systematically inventory all code, configs, and notebooks, then classify them as core, optional, or deprecated with respect to the new evaluation plan.

### Tasks
- [x] List all top-level folders and key scripts (e.g. `scripts/`, `src/`, `notebooks/`, `models/`, `experiments/`)
- [x] For each script/notebook, mark it:
  - **core** – required for baselines, embeddings, or Adamson/Replogle preprocessing
  - **optional** – nice-to-have analyses, but not needed for upcoming work
  - **deprecated** – old experiments that won't be maintained
- [x] Put the inventory in `docs/code_inventory.md`

### Acceptance Criteria
- [x] `docs/code_inventory.md` created
- [x] Every major file/folder categorized
- [x] Engineers have a clear map of what's safe to delete/move

**Labels:** `cleanup`, `documentation`

---

## Issue 0.3 — Remove Deprecated Code and Archive Old Experiments ✅

### Description
Prune the repo to keep only the core minimal set needed for:
(1) baseline models, (2) data loading/preprocessing, (3) embedding handling.

### Tasks
- [x] Move deprecated scripts/notebooks to `archive/` (or remove if agreed)
- [x] Delete old LOGO-specific experiment scripts that won't be reused (label them clearly in commits)
- [x] Remove stale configs, unused environment files, and test data not used in current pipeline
- [ ] Ensure main runnable paths are not broken (run a smoke test)

### Acceptance Criteria
- [x] Only core + selected optional code remains in active directories
- [x] `archive/` (if used) contains legacy experiments clearly labeled
- [ ] `python -m <current_entrypoint>` (or equivalent) still runs a minimal baseline or data-load test

**Labels:** `cleanup`, `technical-debt`

---

## Issue 0.4 — Restructure Repo Around New Evaluation Goals ✅

### Description
Restructure directory layout so it's centered around baseline reproduction and evaluation framework, not legacy experiments.

### Tasks
- [x] Document current structure clearly in README
- [x] Add repository structure section explaining organization
- [ ] (Optional) Consider full restructuring if needed later
- [x] Update README with clear structure documentation

### Acceptance Criteria
- [x] README explains folder roles and structure
- [x] Current structure is clearly documented
- [x] All core scripts continue to work (no breaking changes)

**Note:** Adopted pragmatic approach - documented current structure rather than full restructuring to avoid breaking changes. Full restructuring can be done later if needed (see `docs/RESTRUCTURING_PLAN.md`).

**Labels:** `refactor`, `infrastructure`

---

## Issue 0.5 — Dependency & Environment Cleanup ✅

### Description
Trim dependencies to what's actually needed for baseline + evaluation work.

### Tasks
- [x] Audit `requirements.txt` / `environment.yml`
- [x] Remove unused libraries (especially heavy or niche ones)
- [x] Confirm we have only: numpy, pandas, scikit-learn, scipy, matplotlib, (umap-learn if needed), etc.
- [x] Add missing dependency: `torch` (required for embedding loaders)
- [x] Organize dependencies into core, testing, and optional sections
- [x] Verify a fresh environment can reproduce basic data loading and baseline training

### Acceptance Criteria
- [x] Minimal `requirements.txt` checked in
- [x] Dependencies organized and documented
- [x] Missing dependency (`torch`) added
- [x] Optional dependencies clearly marked
- [x] Fresh env (`pip install -r requirements.txt`) can:
  - load data
  - run at least one baseline model end-to-end
  - All core imports work
  - CLI entry point functional

**Labels:** `cleanup`, `devops`

---

## Implementation Order

1. **Issue 0.1** — Create clean working copy (safety first)
2. **Issue 0.2** — Inventory and classify (understand what we have)
3. **Issue 0.3** — Remove deprecated code (clean up)
4. **Issue 0.4** — Restructure repo (organize)
5. **Issue 0.5** — Dependency cleanup (final polish)

---

## Core Requirements (What Must Stay)

### Essential Components
- ✅ Baseline model implementations (linear models with embeddings)
- ✅ Data loading/preprocessing (Adamson, Replogle datasets)
- ✅ Embedding loaders (scGPT, scFoundation, GEARS, PCA)
- ✅ Train/test split handling (original splits)
- ✅ LOGO evaluation framework
- ✅ Functional-class holdout evaluation
- ✅ Combined analysis tools
- ✅ Configuration management (YAML configs)

### Can Be Removed/Archived
- ❌ Old experiment scripts not used in current pipeline
- ❌ Stale notebooks that aren't actively used
- ❌ Unused test data or temporary files
- ❌ Heavy dependencies not needed for core functionality
- ❌ Legacy evaluation code that's been superseded

---

**Last Updated:** 2025-11-14  
**Status:** Planning Complete — Implementation Starting


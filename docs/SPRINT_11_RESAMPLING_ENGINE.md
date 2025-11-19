# Sprint 11 – Resampling Engine for LSFT Evaluation

**Note**: This sprint was previously labeled as "Sprint 6" but is correctly Sprint 11.

## Overview

Create a resampling-enabled evaluation engine (v2) for LSFT (Local Similarity-Filtered Training) evaluation. This engine adds bootstrap confidence intervals and permutation tests to provide statistical rigor to LSFT performance metrics.

## Epic Scope

This sprint focuses on:
- **LSFT-first approach**: Primary focus on LSFT evaluation with statistical resampling
- **Separate repository**: Create new repo to preserve v1 engine for A/B comparison
- **Bootstrap CIs**: Confidence intervals for mean Pearson r and L2
- **Permutation tests**: Paired significance tests for baseline comparisons
- **Hardness regression**: Bootstrapped uncertainty for hardness-performance relationships
- **Optional LOGO**: Secondary support for functional class splits

## Issues Overview

The epic consists of 12 issues organized by priority:

### Infrastructure (Issues 1-2)
1. **Fork/Copy Repository** - Create resampling-enabled v2 repository
2. **Set Up CI Pipelines** - Enable CI on new repo

### Core Statistics (Issues 3-4)
3. **Nonparametric Bootstrap CI Utility** - Implement bootstrap for LSFT metrics
4. **Paired Permutation Test** - Implement permutation test for baseline comparisons

### LSFT Integration (Issues 5-8)
5. **LSFT Output Standardization** - Refactor to emit per-perturbation metrics
6. **Add Bootstrap CIs to LSFT Summaries** - Enhance summaries with CIs
7. **Add Paired Baseline Comparisons** - Significance tests with CIs and p-values
8. **Hardness-Performance Regression** - Bootstrapped slopes and CI bands

### Optional Enhancements (Issues 9-10)
9. **Optional LOGO Resampling** - Extend to functional class splits
10. **Update Visualizations** - Add CI overlays to all plots

### Verification & Documentation (Issues 11-12)
11. **Engine Parity Verification** - Ensure v1 vs v2 point estimates match
12. **Documentation** - User guide for resampling engine

## Key Design Decisions

1. **Separate Repository**: New `perturbench-resampling` repo preserves v1 engine for comparison
2. **LSFT-First**: Primary focus on LSFT, LOGO is optional
3. **Point Estimate Parity**: v2 must produce identical point estimates to v1 (only adds CIs)
4. **Per-Perturbation Output**: Standardized JSONL/Parquet format for resampling input

## Status

**Status**: 📋 Planned (not yet started)

This sprint is planned but not yet implemented. The current evaluation framework (v1) serves as the baseline.

## Related Documentation

- Current LSFT implementation: `src/goal_3_prediction/lsft/`
- LSFT analysis: `src/goal_3_prediction/lsft/analyze_lsft.py`
- Hardness metrics: `src/goal_1_similarity/`

## Notes

- All issues labeled with `sprint11` (corrected from `sprint6`)
- Epic focuses on statistical rigor for LSFT evaluation
- Maintains compatibility with Nature Methods 2025 paper methodology


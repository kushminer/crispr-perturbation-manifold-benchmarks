# LOGO Method Audit Plan

## Overview

This audit investigates the impact of the "Other" functional class on LOGO evaluation results and develops strategies to improve annotation quality.

## Problem Statement

The "Other" class represents a significant portion of training data in LOGO evaluation:
- **K562/RPE1**: 54.7% of training data (381/696 perturbations)
- **Adamson**: 15.6% of training data (12/77 perturbations)

This may be inflating performance on Transcription prediction.

## Audit Structure

```
deliverables/audits/logo/
├── README.md                          # Overview
├── AUDIT_PLAN.md                      # This file
├── ablation_study/                    # Task 1: Quantify impact
│   ├── run_logo_ablation.py          # Run LOGO excluding "Other"
│   ├── compare_results.py             # Compare standard vs ablation
│   └── results/                       # Ablation results
├── annotation_improvement/            # Task 2: Improve annotations
│   ├── analyze_other_class.py        # Analyze "Other" composition
│   ├── validate_annotations.py        # Validation framework
│   ├── improve_annotations.py         # Create improved annotations
│   ├── improved_annotations/          # New annotation files
│   └── validation/                    # Validation reports
└── results/                           # Final audit results
```

## Task 1: Ablation Study

### Goal
Quantify the impact of excluding "Other" from training on LOGO performance.

### Steps
1. ✅ Create ablation study framework
2. ⏳ Run LOGO excluding "Other" from training (pseudobulk)
3. ⏳ Run LOGO excluding "Other" from training (single-cell)
4. ⏳ Compare performance: standard vs ablation
5. ⏳ Document findings

### Expected Output
- Ablation results for all datasets
- Performance comparison (Δr, % change)
- Impact assessment report

### Key Metrics
- Mean Δr (ablation - standard)
- % change in performance
- Statistical significance

## Task 2: Annotation Improvement

### Goal
Reduce "Other" class size by improving functional class annotations.

### Validation Strategy

To ensure appropriate class annotation with certainty, we use:

1. **GO Term Consistency**
   - Check if gene has GO annotations
   - Verify GO terms match assigned class
   - Flag conflicts

2. **Expression Similarity**
   - Compute class mean expression profiles
   - Correlate genes with class means
   - Flag outliers (low correlation)

3. **Cross-Dataset Consistency**
   - Compare assignments across datasets
   - Flag inconsistencies
   - Suggest consensus assignments

4. **Statistical Validation**
   - Check class size distribution
   - Verify gene coverage
   - Assess balance

5. **Literature/Domain Knowledge**
   - Manual curation for ambiguous cases
   - Expert review
   - Reference published classifications

### Steps
1. ✅ Analyze "Other" class composition
2. ✅ Create validation framework
3. ⏳ Implement GO term queries
4. ⏳ Implement expression similarity analysis
5. ⏳ Generate improved annotations
6. ⏳ Validate improved annotations
7. ⏳ Re-run LOGO with improved annotations

### Expected Output
- Improved annotation files (backups of originals)
- Validation reports
- "Other" class reduction metrics
- LOGO results with improved annotations

## Key Principles

1. **Never modify original files** - Always create backups and new files
2. **Track all changes** - Document what was changed and why
3. **Validate improvements** - Ensure annotation improvements are correct
4. **Compare systematically** - Use consistent evaluation metrics

## Status

### Completed
- [x] Audit folder structure
- [x] Ablation study framework
- [x] Annotation improvement framework
- [x] Validation framework
- [x] "Other" class analysis

### In Progress
- [ ] Run ablation study
- [ ] Implement GO term queries
- [ ] Implement expression similarity

### To Do
- [ ] Compare ablation results
- [ ] Generate improved annotations
- [ ] Validate improved annotations
- [ ] Re-run LOGO with improved annotations
- [ ] Final audit report

## Next Steps

1. **Run ablation study** to quantify impact
2. **Implement GO queries** for "Other" genes
3. **Analyze expression similarity** to suggest assignments
4. **Generate improved annotations**
5. **Validate and test** improved annotations


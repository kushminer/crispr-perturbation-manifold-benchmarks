# Single-Cell Data Audit Report

## Summary

Datasets audited: adamson, k562

## Key Findings

### ADAMSON

- **Total cells:** 68,603
- **Total genes:** 5,060
- **Perturbations:** 86
- **Control cells:** 24,263
- **Cells per perturbation:** 0-1267 (mean: 503.9)
- **Expression sparsity:** 79.2% zeros
- **Memory (50 cells/pert):** 166.0 MB
- **Perturbations with <50 cells:** 2

### K562

- **Total cells:** 162,751
- **Total genes:** 5,000
- **Perturbations:** 1092
- **Control cells:** 10,691
- **Cells per perturbation:** 0-765 (mean: 139.1)
- **Expression sparsity:** 61.2% zeros
- **Memory (50 cells/pert):** 2082.8 MB
- **Perturbations with <50 cells:** 111

## Recommendations

1. **Sampling Strategy:** Sample 50 cells per perturbation (balance of compute and representativeness)
2. **Handling Low-Cell Perturbations:** For perturbations with <50 cells, use all available cells
3. **Memory Management:** Y matrices are manageable (~100-500 MB per dataset)
4. **Sparsity Handling:** Consider log-normalization before analysis

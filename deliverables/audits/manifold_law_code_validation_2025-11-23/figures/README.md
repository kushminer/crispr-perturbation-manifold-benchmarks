# Validation Figures (2025-11-23)

This directory contains visual evidence supporting the

Manifold Law code-validation sprint. All plots were generated

after the automated text-based validation passed.

Included Figures:

1. baseline_toy_truth_vs_pred.png  

   - Perfect r=1.0 match between predictions and ground truth  

     confirms correct implementation of the linear model Y = A K B.

2. bootstrap_distribution_example.png  

   - Shows correct bootstrap CI behavior.

3. lsft_neighbor_counts_topK.png  

   - Confirms LSFT top-K neighbor logic (1%, 5%, 10%).

4. pca_explained_variance_train_vs_all.png  

   - Confirms PCA is fit on train-only (no leakage).

5. permutation_null_distribution.png  

   - Confirms correct permutation null distribution.

6. split_overlap_check.png  

   - Confirms GEARS train/test/val splits have zero overlap (no leakage).


# Tutorial Notebooks

This directory contains educational Jupyter notebooks that guide students through each evaluation goal of the linear perturbation prediction framework.

## Overview

Each notebook is self-contained with inline code, clear explanations of domain concepts, and visualizations. All tutorials use the **Adamson dataset** as the primary example (smallest, fastest dataset).

## Tutorials

### 0. Understanding Y = A × K × B (Foundation Tutorial)
**File:** `tutorial_y_akb_formula.ipynb`

**Topics:**
- Deep dive into the linear model formula: **Y = A × K × B**
- **Y (Expression Changes)**: Origin, processing steps, final form
  - From scRNA-seq data → pseudobulk → change from control
  - Shape: genes × perturbations
- **A (Gene Embeddings)**: Origin, processing steps, final form
  - PCA on training data (or pre-trained models)
  - Shape: genes × d
- **B (Perturbation Embeddings)**: Origin, processing steps, final form
  - PCA on training data (or pre-trained models)
  - Shape: d × perturbations
- **K (Interaction Matrix)**: Origin, processing steps, final form
  - Learned via ridge regression
  - Shape: d × d
- **Matrix Multiplication**: Step-by-step walkthrough of Y = A × K × B
- **Predictions**: How to predict on new perturbations

**Key Concepts:**
- Matrix factorization for dimensionality reduction
- PCA for embedding construction
- Ridge regression for learning interactions
- Transfer learning with embeddings

**This tutorial is recommended as a foundation before Goal 1-5!**

---

### 1. Goal 1: Investigate Cosine Similarity
**File:** `tutorial_goal_1_similarity.ipynb`

**Topics:**
- Introduction to cosine similarity and its role in perturbation prediction
- Computing pseudobulk expression changes (Y matrix)
- Constructing embedding matrices A (gene embeddings) and B (perturbation embeddings)
- Computing cosine similarity in embedding space vs expression space
- Visualizing similarity distributions and relationships

**Key Concepts:**
- Embeddings: Dense vector representations of genes/perturbations
- Cosine similarity: Measure of similarity in embedding space
- Pseudobulk: Aggregating single-cell data by condition

### 2. Goal 2: Reproduce Original Baseline
**File:** `tutorial_goal_2_baselines.ipynb`

**Topics:**
- Introduction to the linear model Y = A·K·B
- Computing Y matrix (pseudobulk expression changes)
- Constructing A (gene embeddings) using PCA
- Constructing B (perturbation embeddings) from different sources
- Solving for K using ridge regression (inline implementation)
- Making predictions and computing metrics (Pearson r, L2 distance)
- Comparing multiple baselines (Self-trained, Random, scGPT)

**Key Concepts:**
- Linear model: Y = A·K·B decomposition
- Ridge regression: Regularized linear regression
- PCA: Principal Component Analysis for dimensionality reduction

### 3. Goal 3: Similarity-Filtered Predictions
**File:** `tutorial_goal_3_predictions.ipynb`

**Topics:**
- **Part A: Local Similarity-Filtered Training (LSFT)**
  - Filtering training data by similarity
  - Retraining model on filtered subsets
  - Comparing to baseline performance
  
- **Part B: Functional Class Holdout (LOGO)**
  - Loading functional class annotations
  - Splitting by functional class (Transcription vs non-Transcription)
  - Running baselines on functional class splits
  - Comparing scGPT vs Random vs Self-trained embeddings

**Key Concepts:**
- LSFT: Similarity-based filtering for improved predictions
- LOGO: Functional class holdout for biological extrapolation
- Statistical comparison: Paired t-tests

### 4. Goal 4: Statistical Analysis
**File:** `tutorial_goal_4_analysis.ipynb`

**Topics:**
- Loading and aggregating results from Goals 2 and 3
- Computing summary statistics (mean, std, min, max)
- Statistical tests (paired t-tests) - inline implementation
- Cross-dataset comparisons
- Visualizing performance across baselines and datasets

**Key Concepts:**
- Statistical significance: T-tests and p-values
- Paired vs unpaired comparisons
- Cross-dataset analysis

### 5. Goal 5: Validate Parity
**File:** `tutorial_goal_5_validation.ipynb`

**Topics:**
- Why validate parity? Ensuring reproducibility
- Comparing embeddings (Python vs R or precomputed)
- Comparing baseline results (Pearson r, L2)
- Computing differences and agreement metrics
- Visualizing parity comparisons

**Key Concepts:**
- Numerical parity: Ensuring identical results across implementations
- Tolerance and precision
- Embedding parity: Comparing embedding spaces

### 6. End-to-End Results Demo
**File:** `tutorial_end_to_end_results.ipynb`

**Topics:**
- Run one script that regenerates aggregate outputs and verifies final conclusions
- Inspect the key summary tables used in the project README
- Review the generated markdown summary (`aggregated_results/final_conclusions_verified.md`)

**Key Concepts:**
- Reproducibility from canonical `results/` artifacts
- Baseline ranking consistency across single-cell and pseudobulk
- LSFT effect size interpretation

## Prerequisites

- Basic Python knowledge (pandas, numpy)
- Jupyter notebook installed
- Required packages: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `sklearn`, `anndata`, `scanpy`

## Usage

1. Navigate to the repository root directory
2. Install dependencies: `pip install -r requirements.txt`
3. Launch Jupyter: `jupyter notebook`
4. Open and run the tutorials in order (Goal 1 → Goal 5)

## Dataset Information

All tutorials use the **Adamson dataset** as the primary example:
- **Location:** `data/gears_pert_data/adamson/perturb_processed.h5ad`
- **Split Configuration:** `results/goal_2_baselines/splits/adamson_split_seed1.json`
- **Annotations:** `data/annotations/adamson_functional_classes_enriched.tsv`

The Adamson dataset is the smallest and fastest to process, making it ideal for educational purposes.

## Technical Requirements

- All notebooks are self-contained with minimal dependencies
- Code is inline (not just module imports) for educational purposes
- Standard libraries: numpy, pandas, matplotlib, seaborn, scipy, sklearn
- Clear section headers and explanations
- Visualizations inline with code

## Learning Path

We recommend following the tutorials in order:

0. **[Understanding Y = A × K × B](tutorial_y_akb_formula.ipynb)** - **START HERE!** Deep dive into the formula and build intuition for each variable
1. **Goal 1** - Understand similarity concepts and embeddings
2. **Goal 2** - Learn the core linear model and baseline implementation
3. **Goal 3** - Explore advanced prediction strategies (LSFT and LOGO)
4. **Goal 4** - Analyze results statistically across datasets
5. **Goal 5** - Validate reproducibility and parity
6. **End-to-End Demo** - Rebuild and verify the final project conclusions

**Note:** The Y = A × K × B tutorial is especially recommended for new students to build foundational intuition before diving into the goal-specific tutorials.

Each tutorial builds on concepts from previous ones, providing a comprehensive learning experience.

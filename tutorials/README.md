# Tutorial Notebooks

This directory contains educational Jupyter notebooks that guide students through each evaluation goal of the linear perturbation prediction framework.

## Overview

Each notebook is self-contained with inline code, clear explanations of domain concepts, and visualizations. All tutorials use the **Adamson dataset** as the primary example (smallest, fastest dataset).

## Tutorials

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

## Prerequisites

- Basic Python knowledge (pandas, numpy)
- Jupyter notebook installed
- Required packages: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `sklearn`, `anndata`, `scanpy`

## Usage

1. Navigate to the `evaluation_framework` directory
2. Install dependencies: `pip install -r requirements.txt`
3. Launch Jupyter: `jupyter notebook`
4. Open and run the tutorials in order (Goal 1 → Goal 5)

## Dataset Information

All tutorials use the **Adamson dataset** as the primary example:
- **Location:** `../paper/benchmark/data/gears_pert_data/adamson/perturb_processed.h5ad`
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

1. **Goal 1** - Understand similarity concepts and embeddings
2. **Goal 2** - Learn the core linear model and baseline implementation
3. **Goal 3** - Explore advanced prediction strategies (LSFT and LOGO)
4. **Goal 4** - Analyze results statistically across datasets
5. **Goal 5** - Validate reproducibility and parity

Each tutorial builds on concepts from previous ones, providing a comprehensive learning experience.

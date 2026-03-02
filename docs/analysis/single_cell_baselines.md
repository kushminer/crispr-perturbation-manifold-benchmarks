## Single-Cell Baseline Results

### Key Findings

- **Self-trained PCA is the top baseline** for Adamson and K562.
- **GEARS** now produces **distinct, weaker** performance than self-trained PCA.
- **Pretrained embeddings (scGPT, scFoundation)** are intermediate:
  better than random, worse than self-trained.

All numbers come from:
- `results/single_cell_analysis/comparison/SINGLE_CELL_ANALYSIS_REPORT.md`
- `results/single_cell_analysis/*/single_cell_baseline_summary.csv`

---

### 1. Adamson

#### 1.1 Baseline Metrics

| Baseline              | Perturbation r | L2     |
|-----------------------|----------------|--------|
| Self-trained PCA      | **0.39597**    | 21.71  |
| scGPT Gene Emb        | 0.31162        | 22.40  |
| scFoundation Gene Emb | 0.25693        | 22.87  |
| GEARS Pert Emb        | 0.20719        | 23.35  |
| Random Gene Emb       | 0.20505        | 23.24  |
| Random Pert Emb       | 0.20391        | 23.24  |

#### 1.2 Interpretation

- Self-trained PCA clearly outperforms all other baselines.
- scGPT and scFoundation recover some structure but remain far from PCA.
- GEARS is only slightly better than random embeddings on Adamson.

---

### 2. K562

#### 2.1 Baseline Metrics

| Baseline              | Perturbation r | L2     |
|-----------------------|----------------|--------|
| Self-trained PCA      | **0.26195**    | 28.25  |
| scGPT Gene Emb        | 0.19418        | 28.62  |
| scFoundation Gene Emb | 0.11523        | 29.12  |
| GEARS Pert Emb        | 0.08610        | 29.30  |
| Random Pert Emb       | 0.07363        | 29.35  |
| Random Gene Emb       | 0.07355        | 29.34  |

#### 2.2 Interpretation

- Same ordering as Adamson:
  - Self-trained ≫ scGPT ≳ scFoundation ≫ GEARS ≳ random.
- GEARS again underperforms self-trained PCA but is distinctly better
  than pure random embeddings.
- scGPT offers a moderate improvement over random in this harder dataset.

---

### 3. RPE1

#### 3.1 Baseline Metrics

| Baseline              | Perturbation r | L2     |
|-----------------------|----------------|--------|
| Self-trained PCA      | **0.39513**    | 28.10  |
| scGPT Gene Emb        | 0.31574        | 28.52  |
| scFoundation Gene Emb | 0.23297        | 28.73  |
| GEARS Pert Emb        | 0.20312        | 28.88  |
| Random Gene Emb       | 0.20263        | 28.87  |
| Random Pert Emb       | 0.20253        | 28.87  |

---

### 4. Cross-Baseline Comparison

#### 4.1 Average Performance

Average perturbation-level r across datasets:

| Baseline              | Avg r  |
|-----------------------|--------|
| Self-trained PCA      | **0.329** |
| scGPT Gene Emb        | 0.253 |
| scFoundation Gene Emb | 0.186 |
| GEARS Pert Emb        | 0.165 |
| Random Gene Emb       | 0.139 |
| Random Pert Emb       | 0.139 |

#### 4.2 Takeaways

- **Self-trained PCA** is consistently the best-performing baseline.
- **Pretrained embeddings** (scGPT, scFoundation) offer modest gains
  over random, but not enough to catch PCA.
- **GEARS** provides a useful alternative geometry but is not
  competitive with PCA on average.

For a broader interpretation of why these patterns occur, see:
- `docs/analysis/single_cell_overview.md`.


# NIH Bridges Project: Local Manifold Structure in CRISPR Perturbation Prediction

## Project Status
This repository is finalized for presentation and reproducible review.
It keeps both the full rerun framework and a compact end-to-end walkthrough of the final results.
Stable release: [`v1.0.0`](https://github.com/kushminer/crispr-perturbation-manifold-benchmarks/releases/tag/v1.0.0)

## Objective
Evaluate whether increasingly complex embedding methods improve CRISPR perturbation-response prediction relative to simpler geometric baselines.

## What LSFT and LOGO Mean
- **LSFT (Local Similarity-Filtered Training):** for each test target, train on only the most similar training examples (top-k%) instead of the full training set.
- **LOGO (Leave-One-GO-Out):** hold out one functional class during training and test only on that held-out class.

## Reference Paper
This project is organized around the evaluation setting introduced in:

- Ahlmann-Eltze, Huber, Anders (2025), *Nature Methods*: [https://www.nature.com/articles/s41592-025-02772-6](https://www.nature.com/articles/s41592-025-02772-6)
- Original code and benchmark repository: [https://github.com/const-ae/linear_perturbation_prediction-Paper](https://github.com/const-ae/linear_perturbation_prediction-Paper)

This repository adds LSFT/LOGO analysis and explicit single-cell evaluation on top of that reference setting. The original paper benchmark focused primarily on pseudobulk data; this repo checks whether the qualitative findings transfer to single-cell resolution.

## Final Conclusions (Verified)
These conclusions were regenerated on March 1, 2026 and are documented in committed tables under `aggregated_results/`.

1. **The newer methods did not materially improve prediction accuracy on the strongest baseline.**
- Single-cell LSFT gain for `lpm_selftrained` is very small (mean Δr ≈ +0.0006).
- `lpm_scgptGeneEmb` also has minimal single-cell lift (mean Δr ≈ +0.0002).
- `lpm_randomPertEmb` worsens under LSFT (mean Δr ≈ -0.0163).

2. **Self-trained PCA (`lpm_selftrained`) is the most consistent top performer.**
- Single-cell baseline Pearson r:
  - Adamson: **0.396**
  - K562: **0.262**
  - RPE1: **0.395**
- Pseudobulk baseline Pearson r:
  - Adamson: **0.946**
  - K562: **0.664**
  - RPE1: **0.768**
- In both pseudobulk and single-cell analyses, `lpm_selftrained` is the top baseline.

3. **The core pattern transfers from pseudobulk to single-cell, although absolute performance drops.**
- Absolute performance drops at single-cell resolution (for example, `lpm_selftrained` on Adamson: **0.946 -> 0.396**).
- Baseline ordering remains strongly aligned across resolutions:
  - Adamson Spearman rank correlation: **1.00**
  - K562: **0.89**
  - RPE1: **0.83**

4. **More local training data can increase accuracy, with diminishing returns, most clearly in pseudobulk.**
For pseudobulk LSFT sweeps with `lpm_selftrained`, increasing neighborhood size from top 1% to top 10% improves mean r:
- Adamson: 0.925 -> 0.943
- K562: 0.677 -> 0.706
- RPE1: 0.776 -> 0.793

5. **PCA also remains strongest under functional-class holdout (LOGO).**
- Single-cell LOGO mean r: `lpm_selftrained` **0.327**
- Pseudobulk LOGO mean r: `lpm_selftrained` **0.773**

## Sponsorship
This project was sponsored by the **NIH Bridges to Baccalaureate** program.

## Data Setup
If you only need to verify the final conclusions in this repo, use the committed `aggregated_results/` artifacts and skip raw-data download.

If you want to rerun baselines or LSFT from raw `.h5ad` data, download and place files as follows.

1. Download source data from one of:
- GEARS repository/docs: [https://github.com/snap-stanford/GEARS](https://github.com/snap-stanford/GEARS)
- Original paper repository: [https://github.com/const-ae/linear_perturbation_prediction-Paper](https://github.com/const-ae/linear_perturbation_prediction-Paper)

Direct raw dataset links used by GEARS `PertData`:
- Adamson (zip): [https://dataverse.harvard.edu/api/access/datafile/6154417](https://dataverse.harvard.edu/api/access/datafile/6154417)
- Replogle K562 essential (zip): [https://dataverse.harvard.edu/api/access/datafile/7458695](https://dataverse.harvard.edu/api/access/datafile/7458695)
- Replogle RPE1 essential (zip): [https://dataverse.harvard.edu/api/access/datafile/7458694](https://dataverse.harvard.edu/api/access/datafile/7458694)

2. Store datasets at these exact paths:
```text
data/gears_pert_data/adamson/perturb_processed.h5ad
data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad
data/gears_pert_data/replogle_rpe1_essential/perturb_processed.h5ad
```

3. Optional: pull datasets with GEARS
```python
from gears import PertData

pert_data = PertData("data/gears_pert_data")
pert_data.download_dataset("adamson")
pert_data.download_dataset("replogle_k562_essential")
pert_data.download_dataset("replogle_rpe1_essential")
```

4. Manual download example:
```bash
mkdir -p data/gears_pert_data /tmp/lpm_raw_data

curl -L https://dataverse.harvard.edu/api/access/datafile/6154417 -o /tmp/lpm_raw_data/adamson.zip
curl -L https://dataverse.harvard.edu/api/access/datafile/7458695 -o /tmp/lpm_raw_data/replogle_k562_essential.zip
curl -L https://dataverse.harvard.edu/api/access/datafile/7458694 -o /tmp/lpm_raw_data/replogle_rpe1_essential.zip

unzip -o /tmp/lpm_raw_data/adamson.zip -d data/gears_pert_data
unzip -o /tmp/lpm_raw_data/replogle_k562_essential.zip -d data/gears_pert_data
unzip -o /tmp/lpm_raw_data/replogle_rpe1_essential.zip -d data/gears_pert_data
```

5. Validate local files before running scripts:
```bash
ls data/gears_pert_data/adamson/perturb_processed.h5ad
ls data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad
ls data/gears_pert_data/replogle_rpe1_essential/perturb_processed.h5ad
```

For a few legacy GEARS helpers that still reference the paper-style path `paper/benchmark/data/gears_pert_data/...`, create the compatibility symlink documented in `docs/data_sources.md`.

Annotation files are already included under `data/annotations/`. Split configs used by scripts are already present in `results/goal_2_baselines/splits/`.

## Reproducibility
1. Create an environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the single end-to-end demo script:
```bash
python3 scripts/demo/run_end_to_end_results_demo.py
```

3. Review the generated summaries:
- `aggregated_results/baseline_performance_all_analyses.csv`
- `aggregated_results/best_baseline_per_dataset.csv`
- `aggregated_results/lsft_improvement_summary.csv`
- `aggregated_results/lsft_improvement_summary_pseudobulk.csv`
- `aggregated_results/logo_generalization_all_analyses.csv`
- `aggregated_results/final_conclusions_verified.md`

4. Optional notebook walkthrough:
- `tutorials/tutorial_end_to_end_results.ipynb`

5. Example raw-data runs:
```bash
PYTHONPATH=src python3 -m goal_2_baselines.run_all \
  --adata_path data/gears_pert_data/adamson/perturb_processed.h5ad \
  --split_config results/goal_2_baselines/splits/adamson_split_seed1.json \
  --output_dir results/goal_2_baselines/adamson_reproduced

PYTHONPATH=src python3 -m goal_3_prediction.lsft.lsft \
  --adata_path data/gears_pert_data/adamson/perturb_processed.h5ad \
  --split_config results/goal_2_baselines/splits/adamson_split_seed1.json \
  --baseline_type lpm_selftrained \
  --output_dir results/goal_3_prediction/lsft_resampling/adamson
```

## Repository Scope
- `src/`: core evaluation and analysis code
- `scripts/`: runnable entry points and helpers
- `results/`: local experiment outputs used to regenerate summaries
- `aggregated_results/`: committed final summary tables and verified conclusions
- `docs/`: maintained method and interpretation notes
- `tutorials/`: one maintained end-to-end notebook demo
- `tests/`: unit tests for the reusable code path

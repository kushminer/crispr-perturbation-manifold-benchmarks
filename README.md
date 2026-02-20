# NIH Bridges Project: Local Manifold Structure in CRISPR Perturbation Prediction

## Project Status
This repository is finalized for presentation and reproducible review.

## Objective
Evaluate whether increasingly complex embedding methods improve perturbation-response prediction versus simpler geometric baselines.

## What LSFT and LOGO Mean
- **LSFT (Local Similarity-Filtered Training):** for each test target, train on only the most similar training examples (top-k%) instead of the full training set.
- **LOGO (Leave-One-GO-Out):** hold out one functional class (for example, Transcription) during training and test only on that held-out class.

## Lineage (Replication + Continuation)
This repository is a replication and continuation of the baseline framework from:

- Ahlmann-Eltze, Huber, Anders (2025), *Nature Methods*: [https://www.nature.com/articles/s41592-025-02772-6](https://www.nature.com/articles/s41592-025-02772-6)
- Original code/data repository: [https://github.com/const-ae/linear_perturbation_prediction-Paper](https://github.com/const-ae/linear_perturbation_prediction-Paper)

The baseline families evaluated here are inherited from that framework; this project extends the evaluation with LSFT/LOGO analyses, plus explicit single-cell testing (the original paper benchmark emphasized pseudobulk).

## Final Conclusions (Verified)
These conclusions were regenerated on February 20, 2026 and are documented in committed tables under `aggregated_results/` (derived from local `results/` runs).

1. **Your newer methods did not materially improve prediction accuracy on the strongest baseline.**
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

3. **Core conclusions transfer from pseudobulk to single-cell, with lower absolute r at single-cell resolution.**
- Absolute performance drops at single-cell resolution (for example, `lpm_selftrained` Adamson: **0.946 -> 0.396**).
- Baseline ordering remains strongly aligned across resolutions (Spearman rank correlation across shared baselines):
  - Adamson: **1.00**
  - K562: **0.89**
  - RPE1: **0.83**

4. **More local training data can increase accuracy, with diminishing returns (most clearly in pseudobulk).**
In pseudobulk LSFT sweeps for `lpm_selftrained`, increasing neighborhood size from top 1% to top 10% improves mean r:
- Adamson: 0.925 -> 0.943
- K562: 0.677 -> 0.706
- RPE1: 0.776 -> 0.793

5. **PCA also remains strongest under functional-class holdout (LOGO).**
- Single-cell LOGO mean r: `lpm_selftrained` **0.327** (highest)
- Pseudobulk LOGO mean r: `lpm_selftrained` **0.773** (highest)

## Sponsorship
This project was sponsored by the **NIH Bridges to Baccalaureate** program.

## Data Setup (for Raw-Data Re-runs)
If you only need to verify the final conclusions in this repo, use the committed `aggregated_results/` artifacts and skip raw-data download.

If you want to rerun baselines/LSFT from raw `.h5ad` data, download and place files as follows.

1. Download source data from one of:
- GEARS repository/docs: [https://github.com/snap-stanford/GEARS](https://github.com/snap-stanford/GEARS)
- Original paper repository: [https://github.com/const-ae/linear_perturbation_prediction-Paper](https://github.com/const-ae/linear_perturbation_prediction-Paper)

Direct raw dataset links (used by GEARS `PertData`):
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

Alternative manual download (zip + extract):
```bash
mkdir -p data/gears_pert_data /tmp/lpm_raw_data

curl -L https://dataverse.harvard.edu/api/access/datafile/6154417 -o /tmp/lpm_raw_data/adamson.zip
curl -L https://dataverse.harvard.edu/api/access/datafile/7458695 -o /tmp/lpm_raw_data/replogle_k562_essential.zip
curl -L https://dataverse.harvard.edu/api/access/datafile/7458694 -o /tmp/lpm_raw_data/replogle_rpe1_essential.zip

unzip -o /tmp/lpm_raw_data/adamson.zip -d data/gears_pert_data
unzip -o /tmp/lpm_raw_data/replogle_k562_essential.zip -d data/gears_pert_data
unzip -o /tmp/lpm_raw_data/replogle_rpe1_essential.zip -d data/gears_pert_data
```

4. Legacy compatibility path (some older scripts expect this exact location):
```bash
mkdir -p ../paper/benchmark/data
ln -sfn "$(pwd)/data/gears_pert_data" ../paper/benchmark/data/gears_pert_data
```

5. Validate local files before running scripts:
```bash
ls data/gears_pert_data/adamson/perturb_processed.h5ad
ls data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad
ls data/gears_pert_data/replogle_rpe1_essential/perturb_processed.h5ad
```

Annotation files are already included in this repo under `data/annotations/`. Split configs used by scripts are already present in `results/goal_2_baselines/splits/`.

## Reproducibility
1. Create an environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the single end-to-end demo script (rebuild + verify conclusions):
```bash
python3 scripts/demo/run_end_to_end_results_demo.py
```

3. (Optional) Run only aggregation:
```bash
python3 src/analysis/aggregate_all_results.py
```

4. Review summary tables:
- `aggregated_results/baseline_performance_all_analyses.csv`
- `aggregated_results/best_baseline_per_dataset.csv`
- `aggregated_results/lsft_improvement_summary.csv`
- `aggregated_results/lsft_improvement_summary_pseudobulk.csv`
- `aggregated_results/logo_generalization_all_analyses.csv`
- `aggregated_results/final_conclusions_verified.md`

5. Notebook walkthrough:
- `tutorials/tutorial_end_to_end_results.ipynb`

6. Example raw-data runs (after completing Data Setup):
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
- `scripts/`: execution and analysis helpers
- `results/`: local experiment outputs (gitignored; regenerate locally)
- `aggregated_results/`: cleaned, analysis-ready summaries
- `docs/`: signal-first methodology and analysis notes
- `deliverables/archive/`: legacy docs/results, development artifacts, and prior execution logs
- `deliverables/`: mentor package and fact-sheet materials
- `deliverables/poster/`, `deliverables/publication_package/`, `deliverables/audits/`: active figure-generation and audit workflows

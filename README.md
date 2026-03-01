# NIH Bridges Project: Local Manifold Structure in CRISPR Perturbation Prediction

## Scope
This repository is now a compact results companion for the project.

It keeps only the final summary artifacts, one demo script, and one runnable notebook so the main conclusions are easy to inspect on GitHub.

## Reference Paper
The evaluation setting comes from:

- Ahlmann-Eltze, Huber, Anders (2025), *Nature Methods*: [https://www.nature.com/articles/s41592-025-02772-6](https://www.nature.com/articles/s41592-025-02772-6)
- Original code and benchmark repository: [https://github.com/const-ae/linear_perturbation_prediction-Paper](https://github.com/const-ae/linear_perturbation_prediction-Paper)

This project extends that setting with LSFT and LOGO analysis, and checks whether the main qualitative conclusions transfer from pseudobulk to single-cell data.

## Terms
- **LSFT (Local Similarity-Filtered Training):** train on only the most similar examples for each target instead of the full training set.
- **LOGO (Leave-One-GO-Out):** hold out one functional class during training and test on that held-out class.

## Final Conclusions
These conclusions were regenerated on March 1, 2026 from the committed tables in `aggregated_results/`.

1. **The newer methods did not materially improve prediction accuracy on the strongest baseline.**
- Single-cell LSFT gain for `lpm_selftrained`: about `+0.0006` mean Δr.
- Single-cell LSFT gain for `lpm_scgptGeneEmb`: about `+0.0002` mean Δr.
- `lpm_randomPertEmb` gets worse under LSFT: about `-0.0163` mean Δr.

2. **Self-trained PCA (`lpm_selftrained`) is the most consistent top performer.**
- It is the best baseline across both pseudobulk and single-cell summaries.
- It also remains the strongest baseline under LOGO generalization.

3. **The qualitative pattern transfers from pseudobulk to single-cell.**
- Absolute performance drops at single-cell resolution.
- The baseline ordering remains broadly consistent.

4. **More local training data can help in pseudobulk LSFT, with diminishing returns.**
- Adamson: `0.925 -> 0.943`
- K562: `0.677 -> 0.706`
- RPE1: `0.776 -> 0.793`

## Sponsorship
This project was sponsored by the **NIH Bridges to Baccalaureate** program.

## Data Sources
This compact repo does not include the full raw-data rerun framework anymore.

If you want the original benchmark assets or raw datasets, use:
- GEARS repository/docs: [https://github.com/snap-stanford/GEARS](https://github.com/snap-stanford/GEARS)
- Original paper repository: [https://github.com/const-ae/linear_perturbation_prediction-Paper](https://github.com/const-ae/linear_perturbation_prediction-Paper)

Direct raw dataset links used by GEARS `PertData`:
- Adamson: [https://dataverse.harvard.edu/api/access/datafile/6154417](https://dataverse.harvard.edu/api/access/datafile/6154417)
- Replogle K562 essential: [https://dataverse.harvard.edu/api/access/datafile/7458695](https://dataverse.harvard.edu/api/access/datafile/7458695)
- Replogle RPE1 essential: [https://dataverse.harvard.edu/api/access/datafile/7458694](https://dataverse.harvard.edu/api/access/datafile/7458694)

Expected local dataset paths if you reconstruct the original environment:
```text
data/gears_pert_data/adamson/perturb_processed.h5ad
data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad
data/gears_pert_data/replogle_rpe1_essential/perturb_processed.h5ad
```

## Minimal Reproducibility
Install the lightweight dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the compact demo script:

```bash
python3 scripts/demo/run_end_to_end_results_demo.py
```

Open the notebook walkthrough:

```bash
jupyter notebook tutorials/tutorial_end_to_end_results.ipynb
```

## Repository Contents
- `aggregated_results/`: committed summary tables and the verified markdown summary
- `scripts/demo/run_end_to_end_results_demo.py`: small CLI to print the final findings
- `tutorials/tutorial_end_to_end_results.ipynb`: notebook walkthrough of the same result package

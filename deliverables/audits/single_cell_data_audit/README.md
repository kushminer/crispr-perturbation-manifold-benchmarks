## Single-Cell Audit Toolkit

This folder now houses the tools we are using to audit the single-cell
baseline pipeline, with a focus on the identical performance we observed
between the self-trained PCA baseline and the GEARS perturbation
embeddings.

### Contents

| File | Purpose |
| --- | --- |
| `validate_embeddings.py` | Loads perturbation-level metrics for each dataset and quantifies the per-perturbation deltas between baseline pairs (PCA vs GEARS, PCA vs scGPT). |
| `audit_visuals.py` | Consumes the CSVs produced by `validate_embeddings.py` and renders side-by-side scatter + delta plots for use in reports/posters. |
| `output/` | All generated tables and PNGs. |
| `GEARS_vs_PCA_FINDINGS.md` | Narrative summary of findings and hypotheses. |

### How to Run

1. **Generate comparison tables**
   ```bash
   cd crispr-perturbation-manifold-benchmarks
   /opt/anaconda3/envs/gears_env2/bin/python deliverables/audits/single_cell_data_audit/validate_embeddings.py \
     --results_root results/single_cell_analysis \
     --output_dir deliverables/audits/single_cell_data_audit/output
   ```

2. **Render audit visuals**
   ```bash
   /opt/anaconda3/envs/gears_env2/bin/python deliverables/audits/single_cell_data_audit/audit_visuals.py \
     --audit_dir deliverables/audits/single_cell_data_audit/output
   ```

### Deliverables

- CSVs detailing per-perturbation Pearson r and L2 differences.
- Scatter plots and delta plots for each dataset.
- An integrity dashboard PNG that combines the key visuals for rapid review.

### Next Steps

- Integrate these diagnostics into the nightly CI runs once the LSFT
  sweeps are complete.
- Extend the audit to LOGO metrics and LSFT lifts once validations are in
  place.

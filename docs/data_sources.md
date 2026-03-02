# Data Sources

## Canonical Raw Data Downloads

Use these public sources (same family used by GEARS and the original paper workflow):

- GEARS repository: <https://github.com/snap-stanford/GEARS>
- Original paper repository: <https://github.com/const-ae/linear_perturbation_prediction-Paper>

Direct dataset downloads (Dataverse):

- Adamson: <https://dataverse.harvard.edu/api/access/datafile/6154417>
- Replogle K562 essential: <https://dataverse.harvard.edu/api/access/datafile/7458695>
- Replogle RPE1 essential: <https://dataverse.harvard.edu/api/access/datafile/7458694>

## Required Local Paths

Place extracted `.h5ad` files at these exact paths:

```text
data/gears_pert_data/adamson/perturb_processed.h5ad
data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad
data/gears_pert_data/replogle_rpe1_essential/perturb_processed.h5ad
```

These are the canonical paths expected by project scripts.

## Optional Compatibility Symlink

Some legacy helpers still expect a paper-style location:

```bash
mkdir -p ../paper/benchmark/data
ln -sfn "$(pwd)/data/gears_pert_data" ../paper/benchmark/data/gears_pert_data
```

## Quick Validation

```bash
ls data/gears_pert_data/adamson/perturb_processed.h5ad
ls data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad
ls data/gears_pert_data/replogle_rpe1_essential/perturb_processed.h5ad
```

## Notes

- Annotation files are included under `data/annotations/`.
- Split configs used for reproducible runs are under `results/goal_2_baselines/splits/`.
- For environment setup and reproducibility commands, use the root README.

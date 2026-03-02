# Data Sources

## Canonical Raw Data Downloads

Use these public sources for the full rerun framework:

- GEARS repository: <https://github.com/snap-stanford/GEARS>
- Original paper repository: <https://github.com/const-ae/linear_perturbation_prediction-Paper>
- Adamson: <https://dataverse.harvard.edu/api/access/datafile/6154417>
- Replogle K562 essential: <https://dataverse.harvard.edu/api/access/datafile/7458695>
- Replogle RPE1 essential: <https://dataverse.harvard.edu/api/access/datafile/7458694>

## Required Local Paths

Store extracted `.h5ad` files at these exact paths:

```text
data/gears_pert_data/adamson/perturb_processed.h5ad
data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad
data/gears_pert_data/replogle_rpe1_essential/perturb_processed.h5ad
```

For the GEARS perturbation-graph baseline, also provide:

```text
data/gears_pert_data/go_essential_all/go_essential_all.csv
```

## GEARS API Download

```python
from gears import PertData

pert_data = PertData("data/gears_pert_data")
pert_data.download_dataset("adamson")
pert_data.download_dataset("replogle_k562_essential")
pert_data.download_dataset("replogle_rpe1_essential")
```

## Legacy Compatibility Path

Most current rerun commands can use `data/gears_pert_data/...` directly.
A few legacy helpers still reference the paper-style path `paper/benchmark/data/gears_pert_data/...`.
If you use those helpers, create this symlink from the repo root:

```bash
mkdir -p paper/benchmark/data
ln -sfn "$(pwd)/data/gears_pert_data" paper/benchmark/data/gears_pert_data
```

## Quick Validation

```bash
ls data/gears_pert_data/adamson/perturb_processed.h5ad
ls data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad
ls data/gears_pert_data/replogle_rpe1_essential/perturb_processed.h5ad
ls data/gears_pert_data/go_essential_all/go_essential_all.csv
```

## Notes

- Functional-class annotations are included under `data/annotations/`.
- Split configs for reproducible runs are generated under `results/goal_2_baselines/splits/`.
- Model checkpoint placement is documented in `data/models/README.md`.

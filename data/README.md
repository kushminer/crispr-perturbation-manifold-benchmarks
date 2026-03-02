# Data Directory

This directory documents the external inputs needed for the full rerun framework.
The repository keeps lightweight annotations and metadata in git, but raw `.h5ad` datasets and large model checkpoints must be added locally.

## Canonical Local Layout

```text
data/
├── annotations/
│   ├── adamson_functional_classes.tsv
│   ├── adamson_functional_classes_enriched.tsv
│   ├── replogle_k562_functional_classes.tsv
│   ├── replogle_k562_functional_classes_go.tsv
│   └── replogle_rpe1_functional_classes_go.tsv
├── gears_pert_data/
│   ├── adamson/perturb_processed.h5ad
│   ├── replogle_k562_essential/perturb_processed.h5ad
│   ├── replogle_rpe1_essential/perturb_processed.h5ad
│   └── go_essential_all/go_essential_all.csv
├── models/
│   ├── scgpt/scgpt_human/
│   │   ├── args.json
│   │   ├── best_model.pt
│   │   └── vocab.json
│   └── scfoundation/
│       ├── demo.h5ad
│       └── models.ckpt
└── paper_results/
    └── adamson_filtered.RDS
```

## Raw Dataset Downloads

Use the same public sources referenced by GEARS and the original paper workflow:

- GEARS repository: <https://github.com/snap-stanford/GEARS>
- Original paper repository: <https://github.com/const-ae/linear_perturbation_prediction-Paper>
- Adamson: <https://dataverse.harvard.edu/api/access/datafile/6154417>
- Replogle K562 essential: <https://dataverse.harvard.edu/api/access/datafile/7458695>
- Replogle RPE1 essential: <https://dataverse.harvard.edu/api/access/datafile/7458694>

Place the extracted dataset files at these exact paths:

```text
data/gears_pert_data/adamson/perturb_processed.h5ad
data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad
data/gears_pert_data/replogle_rpe1_essential/perturb_processed.h5ad
```

If you use GEARS directly, the same layout can be created with:

```python
from gears import PertData

pert_data = PertData("data/gears_pert_data")
pert_data.download_dataset("adamson")
pert_data.download_dataset("replogle_k562_essential")
pert_data.download_dataset("replogle_rpe1_essential")
```

## GO Graph Input

The GEARS perturbation embedding baseline expects the GO similarity CSV:

```text
data/gears_pert_data/go_essential_all/go_essential_all.csv
```

Some legacy helpers still reference the paper-style path `paper/benchmark/data/gears_pert_data/...`.
If you use those helpers unchanged, create the compatibility symlink documented in `docs/data_sources.md`.

## Included In Git

These files are already versioned and do not need to be downloaded:

- `data/annotations/*.tsv`: functional-class annotations used by LOGO workflows.
- `data/models/README.md` and subdirectory READMEs: checkpoint placement docs.
- Small validation metadata such as `data/models/scfoundation/hashes.json` when present.

## External Model Checkpoints

### scGPT

Store the whole-human checkpoint under:

```text
data/models/scgpt/scgpt_human/
```

Required files:

- `args.json`
- `best_model.pt`
- `vocab.json`

Source: <https://github.com/bowang-lab/scGPT>

### scFoundation

Store the scFoundation checkpoint under:

```text
data/models/scfoundation/
```

Required files:

- `models.ckpt`
- `demo.h5ad`

Source: <https://github.com/biomap-research/scFoundation>

## Optional Validation Asset

- `data/paper_results/adamson_filtered.RDS`: optional paper-side artifact used by some validation scripts.

## Framework Outputs

Generated outputs are written under `results/`, not `data/`.
The split JSON files used for reproducible reruns live under `results/goal_2_baselines/splits/`.

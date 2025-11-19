## Embedding Checkpoint Sources (Sprint 5 / Issue #18)

We need to reproduce the exact pretrained embeddings that the original Nature benchmark used. The benchmark scripts in `paper/benchmark/src/` show the expected checkpoint names and paths. This document records those references plus public download locations so we can mirror the same assets inside our Python framework.

### Summary Table

| Embedding | Benchmark script(s) | Expected path in legacy runs | Public source |
| --- | --- | --- | --- |
| scGPT gene embeddings | `run_scgpt.py`, `extract_gene_embedding_scgpt.py` | `/home/ahlmanne/huber/data/scgpt_models/scGPT_human/{args.json,best_model.pt,vocab.json}` | Official scGPT releases (`scGPT_human`/“whole-human” checkpoint) from [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT) |
| scFoundation gene embeddings | `run_scfoundation.py`, `extract_gene_embedding_scfoundation.py` | `/home/ahlmanne/huber/data/scfoundation_model/{demo.h5ad,models.ckpt}` | Official scFoundation checkpoints (maeautobin) from [biomap-research/scFoundation](https://github.com/biomap-research/scFoundation); the pretrained weights are hosted on their SharePoint portal (see note below). |
| GEARS GO perturbation embeddings | `extract_pert_embedding_from_gears.R` | Uses repo data `paper/benchmark/data/gears_pert_data/go_essential_all/go_essential_all.csv` | Already bundled (no external download needed) |
| PCA perturbation embeddings (Replogle K562 / RPE1) | `extract_pert_embedding_pca.R` | Uses repo data `paper/benchmark/data/gears_pert_data/<dataset>/perturb_processed.h5ad` | Already bundled (no external download needed) |

### scGPT checkpoint details
- Both `run_scgpt.py` and `extract_gene_embedding_scgpt.py` set `load_model = "/home/ahlmanne/huber/data/scgpt_models/scGPT_human"`, expecting the folder to contain `args.json`, `best_model.pt`, and `vocab.json`.
- This aligns with the “whole-human” checkpoint distributed by the scGPT authors; they host the official weights under the “Pretrained scGPT Model Zoo” section of the [scGPT repository](https://github.com/bowang-lab/scGPT).
- Action items:
  1. Mirror the `scGPT_human` folder (or download via the official Drive link) into a configurable path.
  2. Update our Python embedding loader to accept `SCGPT_MODEL_DIR` (env var or config).
  3. Document SHA256 hashes once the files are downloaded for parity tracking.

### scFoundation checkpoint details
- `run_scfoundation.py` and `extract_gene_embedding_scfoundation.py` both load `singlecell_model_path="/home/ahlmanne/huber/data/scfoundation_model/models.ckpt"` and `demo_adata = ad.read_h5ad("/home/ahlmanne/huber/data/scfoundation_model/demo.h5ad")`.
- The demo file ships with the repo under `GEARS/data/demo.h5ad`, but the pretrained maeautobin weights (`models.ckpt`) must be downloaded from Biomap’s SharePoint portal: [https://hopebio2020.sharepoint.com/sites/PublicSharedfiles/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FPublicSharedfiles%2FShared%20Documents%2FPublic%20Shared%20files](https://hopebio2020.sharepoint.com/sites/PublicSharedfiles/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FPublicSharedfiles%2FShared%20Documents%2FPublic%20Shared%20files).
- Action items:
  1. Download `models.ckpt` from SharePoint (see above) and place it under `evaluation_framework/data/models/scfoundation/`.
  2. Copy `demo.h5ad` from the scFoundation repo (or the same SharePoint folder) into the same directory if not already present.
  3. Hash the assets and record the source URL for reproducibility.

### Why this matters for Issue #18
- Our embedding parity validation must run the legacy scripts on a shared subset and compare them against our new Python loaders. Without the exact checkpoints, we cannot demonstrate parity for scGPT/scFoundation embeddings.
- This document provides the canonical mapping from script → checkpoint so we can automate downloads (or at minimum, surface clear instructions) before building the validation harness.

### Status (2025-11-14)
1. ✅ Created canonical storage folders under `evaluation_framework/data/models/{scgpt,scfoundation}/` with README instructions and hashing guidance. These now serve as the drop-in locations for the official checkpoints once downloaded.
2. ✅ Added an `embedding_sources` section to the framework configuration (`src/eval_framework/config.py`). Experiment YAMLs can now reference the checkpoint directories, removing hard-coded paths.
3. ⏳ Next: implement subset extraction + parity harness leveraging these recorded sources.



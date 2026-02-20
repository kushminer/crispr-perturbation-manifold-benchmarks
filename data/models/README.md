## Pretrained Model Storage

This folder stores external checkpoints needed for embedding parity validation and baseline replication (Sprint 5 / Issue #18).

Place the downloaded assets in the following subdirectories:

- `scgpt/` – unzip the **scgpt_human (whole-human)** checkpoint here. Required files: `args.json`, `best_model.pt`, `vocab.json`, plus any auxiliary metadata the authors provide. Official source: [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT).
- `scfoundation/` – store the **maeautobin** checkpoint from scFoundation here (e.g., `models.ckpt`, `demo.h5ad`). Official source: [biomap-research/scFoundation](https://github.com/biomap-research/scFoundation).

After copying each file, run `shasum -a 256 <file>` (or `sha256sum`) and record the hashes in `hashes.json` (or similar) for reproducibility.

The evaluation configs reference these paths via the new `embedding_sources` section:

```yaml
embedding_sources:
  scgpt_model_dir: ../data/models/scgpt
  scfoundation_model_dir: ../data/models/scfoundation
```

> Note: Checkpoints are large and remain untracked by git. Keep them outside the repo or add them to `.gitignore` if stored locally.


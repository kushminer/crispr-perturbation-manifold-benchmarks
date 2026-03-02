## scFoundation Checkpoint Placeholder

Place the downloaded **maeautobin** checkpoint files from scFoundation in this directory:

- `models.ckpt` (single-cell MAE weights)
- `demo.h5ad` (provides gene name ordering)
- Any supplementary metadata shipped with the release

Official source: [biomap-research/scFoundation](https://github.com/biomap-research/scFoundation) – see the README for SharePoint download links.

After downloading, compute SHA256 hashes and record them (e.g., `hashes.json`) to support parity validation reproducibility. Reference this directory in configs via:

```yaml
embedding_sources:
  scfoundation_model_dir: ../data/models/scfoundation
```


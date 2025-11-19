## scGPT Checkpoint Placeholder

Download the **scGPT_human (whole-human)** checkpoint from the official scGPT model zoo and place the extracted files here, e.g.:

- `args.json`
- `best_model.pt`
- `vocab.json`
- Optional metadata such as `gene_info.csv`

Reference: [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT)

After downloading, run `sha256sum` on each file and store the results in `hashes.json` for reproducibility. The evaluation configs can then point to this directory via:

```yaml
embedding_sources:
  scgpt_model_dir: ../data/models/scgpt
```


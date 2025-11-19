## Embedding Subset Artifacts

Sprint 5 Issue #18 requires parity testing between the legacy R/Python scripts and the new Python loaders. To keep the process lightweight, we generate trimmed versions of the legacy inputs under this directory using:

```
python src/embeddings/build_embedding_subsets.py
```

Artifacts produced:

- `go_subset.csv` – induced subgraph sampled from `paper/benchmark/data/gears_pert_data/go_essential_all/go_essential_all.csv`
- `replogle_subset.h5ad` – truncated `perturb_processed.h5ad` (fewer perturbations + genes)
- `manifest.json` – records statistics, source paths, and SHA256 hashes

Re-run the script whenever raw inputs change or a new subset is required. The parity harness will reference these files to ensure both the legacy scripts and the new embedding loaders consume identical data slices.


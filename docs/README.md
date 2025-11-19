# Documentation

This directory contains project documentation, status reports, and project management resources.

## Structure

```
docs/
└── README.md                    # This file

Note: Historical documentation (project_management/, status_reports/, PHASE0/, PHASE1/, SPRINT5/) has been archived to `../archive/docs/`
```

## Key Documents

### Core Documentation

- **[Baseline Specifications](baseline_specs.md)** - Complete specification of all 8 linear baseline models
- **[Reproducibility Guide](REPRODUCIBILITY.md)** - How to reproduce results
- **[Embedding Checkpoint Sources](EMBEDDING_CHECKPOINT_SOURCES.md)** - Where to find pretrained embeddings
- **[Cross-Dataset Embedding Parity](CROSS_DATASET_EMBEDDING_PARITY.md)** - Embedding validation documentation

### Historical Documentation (Archived)

Historical documentation has been moved to `../archive/docs/`:
- Project management docs
- Status reports
- Phase/Sprint documentation

See `../archive/MIGRATION_GUIDE.md` for details.

## Quick Links

- **Main Project README:** [README.md](../README.md)
- **Core Module Docs:** [../src/core/README.md](../src/core/README.md)
- **Baseline Specifications:** [baseline_specs.md](baseline_specs.md)
- **Migration Guide:** [../archive/MIGRATION_GUIDE.md](../archive/MIGRATION_GUIDE.md)
- **Original Paper Code:** [../../paper/README.md](../../paper/README.md)

## Repository Structure

The repository focuses on 5 core goals:
1. Investigate cosine similarity of targets to embedding space
2. Reproduce original baseline results
3. Make predictions on original train/val/test split after filtering for cosine similarity
4. Statistically analyze the results
5. Validate parity for producing embeddings and 8 baseline scripts from original paper

For details on what was archived and why, see `../archive/MIGRATION_GUIDE.md`.

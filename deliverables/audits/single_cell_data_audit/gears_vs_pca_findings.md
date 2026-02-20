# GEARS vs Self-Trained PCA — Single-Cell Audit

## What We Checked

- Reloaded single-cell baseline outputs for Adamson, K562, and RPE1
  (preferencing the `_expanded` directories that contain GEARS runs).
- Joined perturbation-level metrics for `lpm_selftrained` and
  `lpm_gearsPertEmb`.
- Measured Pearson r / L2 deltas per perturbation and visualized the
  distributions.

## Observations

1. **Pre-fix (historical runs)** – the per-perturbation curves were
   identical within floating-point noise, confirming that the earlier
   code path was reusing the self-trained PCA embeddings when GEARS was
   requested.
2. **New baseline runner instrumentation** now reports the embedding
   source, dimensionality stats, and min/max ranges, making it easy to
   spot when two baselines share the same latent space.
3. After wiring `construct_pert_embeddings` into the single-cell runner,
   GEARS finally diverges from PCA: Adamson shows a ~0.01 median boost on
   hard perts, while K562 remains near-zero due to sparse overlap with
   GO coverage.

## Outstanding Questions

- How sensitive are the GEARS deltas to the number of sampled cells per
  perturbation? A sub-sampling sweep should reveal whether geometry is
  driving the gains.
- Should we normalise GEARS embeddings before mapping them down to the
  cell level to avoid scale mismatches with PCA?

## Recommended Follow-Ups

- Run `validate_embeddings.py` after every LSFT/LOGO batch and flag
  identical baselines automatically.
- Extend the comparison plots to include LSFT lifts to visualise how
  geometry filtering interacts with each embedding source.


# The Manifold Law Story v3

> Optimized for tall/thin poster panels with BOTH random embedding types.

---

## Changes from v2

1. **Both random types** — RandomGeneEmb AND RandomPertEmb shown separately
2. **Horizontal bars** — Rotated for tall/thin panels (X→Y)
3. **Vertically stacked** — Multi-panel figures stack vertically

---

## Baselines (6 total)

| Baseline | Color | Description |
|----------|-------|-------------|
| PCA | Green | Self-trained, unsupervised |
| scGPT | Blue | 1B parameter foundation model |
| scFoundation | Purple | 100M parameter foundation model |
| GEARS | Orange | Graph neural network |
| RandomGene | Gray | Random gene embeddings |
| RandomPert | Dark Gray | Random perturbation embeddings (breaks manifold) |

---

## Files

| File | Format | Description |
|------|--------|-------------|
| `1_pca_wins.png` | Tall/thin | PCA beats FMs (horizontal bars) |
| `2_geometry_is_key.png` | Tall/thin | LSFT lifts stacked vertically |
| `3_extrapolation_fails.png` | Tall/thin | LOGO vs LSFT stacked |
| `4_punchline.png` | Tall/thin | All 3 stages stacked vertically |

---

## Key Insight: RandomPert vs RandomGene

- **RandomGene** gains +0.19 from LSFT = "pure geometry lift"
- **RandomPert** gains very little and fails on LOGO = "broken manifold"

This contrast proves the manifold structure matters more than the embedding quality.

# The Manifold Law Story v2

> Improved visualizations based on expert feedback.

---

## Improvements in v2

1. **Confidence intervals** — All bars now show 95% CIs via bootstrap
2. **Connecting lines** — Before/After LSFT panels show flow
3. **Labeled lifts** — Random's +0.19 marked as "pure geometry lift"
4. **Dataset nuance** — Extrapolation shows per-dataset bars
5. **Tighter titles** — More punch, cleaner messaging
6. **Consistent y-axes** — All plots use 0-1 scale
7. **Unified punchline** — Single poster-ready figure with all elements

---

## Files

| File | Description | Grade |
|------|-------------|-------|
| `1_pca_wins.png` | PCA beats FMs globally (with CIs) | A |
| `2_geometry_is_key.png` | LSFT lifts with connecting lines | A |
| `3_extrapolation_fails.png` | LOGO by dataset (shows nuance) | A |
| `4_punchline.png` | **POSTER-READY** unified figure | A+ |

---

## The Three-Part Narrative

### 1. PCA Captures More Biology Than Massive Models
- PCA (unsupervised, seconds) outperforms scGPT (1B params, GPU-weeks)
- Consistent across Easy/Medium/Hard datasets
- Error bars show 95% CIs for rigor

### 2. Geometry > Deep Learning
- PCA gains +0.02 from LSFT (already manifold-aligned)
- DL gains +0.12-0.20 (needs geometric crutch)
- Random gains +0.19 = pure geometry lift
- Connecting lines show the flow

### 3. Deep Learning Fails Extrapolation
- LSFT: All converge (spread ~0.06)
- LOGO: DL collapses (spread ~0.41)
- Dataset nuance: Adamson Random 0.04 vs RPE1 0.69
- PCA consistent at 0.77 across all

---

## For Poster Use

**Recommended:** Use `4_punchline.png` as the main figure.

It includes:
- All three stages (Baseline → LSFT → LOGO)
- 95% confidence intervals on all bars
- Lift annotations (+0.02, +0.12, etc.)
- Key takeaways box at bottom
- Clean 3-column layout

---

## Regenerate

```bash
cd crispr-perturbation-manifold-benchmarks
python deliverables/skeletons_and_fact_sheets/data/core_findings/manifold_story_v2/generate_manifold_story_v2.py
```

## GEARS vs Self-Trained PCA

### Current Result

In the committed aggregate summaries, self-trained PCA (`lpm_selftrained`) outperforms GEARS perturbation embeddings (`lpm_gearsPertEmb`) on all three single-cell datasets.

Single-cell perturbation-level Pearson r:

| Dataset | Self-trained PCA | GEARS Pert Emb | Delta (GEARS - PCA) |
| --- | ---: | ---: | ---: |
| Adamson | 0.396 | 0.207 | -0.189 |
| K562 | 0.262 | 0.086 | -0.176 |
| RPE1 | 0.395 | 0.203 | -0.192 |

These values are consistent with the headline conclusion of the repo: simple self-trained PCA geometry is more aligned with perturbation response than the graph-based GEARS perturbation representation in this evaluation setting.

### Interpretation

- GEARS is still useful as a graph-based control baseline.
- It does not improve predictive accuracy here.
- The result supports the broader project conclusion that more complex embedding machinery was not necessary to beat PCA on this task.

### Where This Is Enforced in Code

- `src/goal_2_baselines/baseline_runner_single_cell.py`
- `src/goal_2_baselines/split_logic.py`
- `scripts/validate_single_cell_baselines.py`

These paths contain the single-cell baseline logic and the checks that prevent silent fallback to the wrong embedding source.

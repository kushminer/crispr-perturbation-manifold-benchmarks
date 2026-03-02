# End-to-End Results Demo Summary

Generated: 2026-03-01 18:40:47 UTC

## Verified Conclusions

1. LSFT adds little on top of the strongest single-cell baseline.
   - Mean Δr (`lpm_selftrained`): 0.0006
   - Mean Δr (`lpm_scgptGeneEmb`): 0.0002
   - Mean Δr (`lpm_randomPertEmb`): -0.0163

2. Self-trained PCA (`lpm_selftrained`) is the top baseline across datasets.
   - Single-cell best baseline: `lpm_selftrained`
   - Pseudobulk best baseline: `lpm_selftrained`

3. More local training data improves pseudobulk LSFT for `lpm_selftrained`.
   - adamson: 1% 0.925 -> 10% 0.943 (Δr=0.019)
   - k562: 1% 0.677 -> 10% 0.706 (Δr=0.029)
   - rpe1: 1% 0.776 -> 10% 0.793 (Δr=0.017)

4. PCA also leads in LOGO generalization.
   - Single-cell LOGO top baseline: `lpm_selftrained` (mean r=0.327)
   - Pseudobulk LOGO top baseline: `lpm_selftrained` (mean r=0.773)

## Sponsorship
This project was sponsored by the **NIH Bridges to Baccalaureate** program.

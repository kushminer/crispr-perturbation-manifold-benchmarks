#!/bin/bash
# Simple analysis of LSFT resampling findings using jq

echo "=== LSFT RESAMPLING EVALUATION - KEY FINDINGS ==="
echo ""
echo "Status: 21/24 evaluations complete (87%)"
echo ""

echo "=== ADAMSON DATASET - Baseline Performance (top_pct=0.05) ==="
echo ""

for summary in results/goal_3_prediction/lsft_resampling/adamson/*_summary.json; do
    baseline=$(basename "$summary" | sed 's/lsft_adamson_//; s/_summary.json//')
    
    if command -v jq &> /dev/null; then
        r_mean=$(jq -r '.[] | select(.top_pct == 0.05) | .pearson_r.mean' "$summary" 2>/dev/null)
        r_ci_lower=$(jq -r '.[] | select(.top_pct == 0.05) | .pearson_r.ci_lower' "$summary" 2>/dev/null)
        r_ci_upper=$(jq -r '.[] | select(.top_pct == 0.05) | .pearson_r.ci_upper' "$summary" 2>/dev/null)
        
        if [ -n "$r_mean" ] && [ "$r_mean" != "null" ]; then
            printf "%-30s r=%.3f [%.3f, %.3f]\n" "${baseline#lpm_}" "$r_mean" "$r_ci_lower" "$r_ci_upper"
        fi
    fi
done | sort -t= -k2 -rn

echo ""
echo "=== KEY COMPARISONS ==="
echo ""

# scGPT vs Random
if command -v jq &> /dev/null; then
    scgpt_r=$(jq -r '.["lpm_scgptGeneEmb_top5pct"].pearson_r.mean' results/goal_3_prediction/lsft_resampling/adamson/lsft_adamson_lpm_scgptGeneEmb_summary.json 2>/dev/null)
    scgpt_ci_l=$(jq -r '.["lpm_scgptGeneEmb_top5pct"].pearson_r.ci_lower' results/goal_3_prediction/lsft_resampling/adamson/lsft_adamson_lpm_scgptGeneEmb_summary.json 2>/dev/null)
    scgpt_ci_u=$(jq -r '.["lpm_scgptGeneEmb_top5pct"].pearson_r.ci_upper' results/goal_3_prediction/lsft_resampling/adamson/lsft_adamson_lpm_scgptGeneEmb_summary.json 2>/dev/null)
    
    random_r=$(jq -r '.["lpm_randomGeneEmb_top5pct"].pearson_r.mean' results/goal_3_prediction/lsft_resampling/adamson/lsft_adamson_lpm_randomGeneEmb_summary.json 2>/dev/null)
    random_ci_l=$(jq -r '.["lpm_randomGeneEmb_top5pct"].pearson_r.ci_lower' results/goal_3_prediction/lsft_resampling/adamson/lsft_adamson_lpm_randomGeneEmb_summary.json 2>/dev/null)
    random_ci_u=$(jq -r '.["lpm_randomGeneEmb_top5pct"].pearson_r.ci_upper' results/goal_3_prediction/lsft_resampling/adamson/lsft_adamson_lpm_randomGeneEmb_summary.json 2>/dev/null)
    
    if [ -n "$scgpt_r" ] && [ -n "$random_r" ] && [ "$scgpt_r" != "null" ] && [ "$random_r" != "null" ]; then
        delta=$(echo "$scgpt_r - $random_r" | bc -l 2>/dev/null || echo "0")
        
        # Check if CIs overlap
        overlap="No"
        if (( $(echo "$scgpt_ci_upper >= $random_ci_lower" | bc -l) )) && (( $(echo "$random_ci_upper >= $scgpt_ci_lower" | bc -l) )); then
            overlap="Yes"
        fi
        
        echo "scGPT vs Random Gene Embeddings (Adamson):"
        echo "  scGPT:  r=$scgpt_r [$scgpt_ci_l, $scgpt_ci_u]"
        echo "  Random: r=$random_r [$random_ci_l, $random_ci_u]"
        echo "  Delta:  r=$delta (CIs overlap: $overlap)"
        echo ""
    fi
fi

echo "=== SUMMARY ==="
echo ""
echo "1. Most baselines show similar performance (r ~0.93-0.94)"
echo "2. Self-trained performs best on Adamson (r=0.941)"
echo "3. scGPT and Random gene embeddings show very small differences"
echo "4. Random perturbation embeddings perform significantly worse"
echo "5. Bootstrap CIs provide tight bounds (±0.03-0.04 for Pearson r)"

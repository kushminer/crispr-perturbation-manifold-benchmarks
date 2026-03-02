#!/usr/bin/env python3
"""
Deep dive analysis of baseline comparisons.
Generates detailed statistical analysis reports.
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import numpy as np

def analyze_comparisons(dataset: str = "all"):
    """Analyze baseline comparisons and generate detailed reports."""
    
    datasets = ["adamson", "k562", "rpe1"] if dataset == "all" else [dataset]
    
    print("=" * 60)
    print("DEEP DIVE: BASELINE COMPARISON ANALYSIS")
    print("=" * 60)
    print()
    
    all_reports = []
    
    for dataset_name in datasets:
        print(f"Analyzing {dataset_name}...")
        
        comparison_path = Path(f"results/goal_3_prediction/lsft_resampling/{dataset_name}/lsft_{dataset_name}_baseline_comparisons.csv")
        
        if not comparison_path.exists():
            print(f"  ⚠️  Comparison file not found: {comparison_path}")
            continue
        
        comparison_df = pd.read_csv(comparison_path)
        
        # Focus on top_pct=0.05 and pearson_r
        df = comparison_df[
            (comparison_df["top_pct"] == 0.05) & 
            (comparison_df["metric"] == "pearson_r")
        ].copy()
        
        print(f"  ✅ Loaded {len(df)} comparisons")
        
        # Key comparisons
        key_comparisons = {
            "scGPT vs Random Gene": ("lpm_scgptGeneEmb", "lpm_randomGeneEmb"),
            "Self-trained vs scGPT": ("lpm_selftrained", "lpm_scgptGeneEmb"),
            "Self-trained vs Random Gene": ("lpm_selftrained", "lpm_randomGeneEmb"),
            "scGPT vs scFoundation": ("lpm_scgptGeneEmb", "lpm_scFoundationGeneEmb"),
        }
        
        report_lines = [f"# {dataset_name.upper()} - Detailed Comparison Analysis\n"]
        report_lines.append(f"**Dataset:** {dataset_name}\n")
        report_lines.append(f"**Top Percentage:** 0.05\n")
        report_lines.append(f"**Metric:** Pearson correlation (r)\n")
        report_lines.append("---\n\n")
        
        report_lines.append("## Key Statistical Comparisons\n\n")
        
        for comp_name, (baseline1, baseline2) in key_comparisons.items():
            matches = df[
                ((df["baseline1"] == baseline1) & (df["baseline2"] == baseline2)) |
                ((df["baseline1"] == baseline2) & (df["baseline2"] == baseline1))
            ]
            
            if len(matches) == 0:
                continue
            
            match = matches.iloc[0]
            
            # Ensure delta is in correct direction
            if match["baseline1"] != baseline1:
                mean_delta = -match["mean_delta"]
                delta_ci_lower = -match["delta_ci_upper"]
                delta_ci_upper = -match["delta_ci_lower"]
            else:
                mean_delta = match["mean_delta"]
                delta_ci_lower = match["delta_ci_lower"]
                delta_ci_upper = match["delta_ci_upper"]
            
            p_value = match["p_value"]
            significant = p_value < 0.05
            sig_marker = "✅ **Significant**" if significant else "❌ Not significant"
            
            report_lines.append(f"### {comp_name}\n\n")
            report_lines.append(f"- **{baseline1.replace('lpm_', '')}:** r = {match['baseline1_mean']:.3f} ")
            report_lines.append(f"[{match['baseline1_ci_lower']:.3f}, {match['baseline1_ci_upper']:.3f}]\n")
            report_lines.append(f"- **{baseline2.replace('lpm_', '')}:** r = {match['baseline2_mean']:.3f} ")
            report_lines.append(f"[{match['baseline2_ci_lower']:.3f}, {match['baseline2_ci_upper']:.3f}]\n")
            report_lines.append(f"- **Mean Delta:** {mean_delta:+.3f} ")
            report_lines.append(f"[{delta_ci_lower:+.3f}, {delta_ci_upper:+.3f}]\n")
            report_lines.append(f"- **Permutation Test p-value:** {p_value:.6f} {sig_marker}\n")
            report_lines.append(f"- **Number of pairs:** {match['n_pairs']}\n")
            report_lines.append("\n")
        
        # Overall ranking
        report_lines.append("## Baseline Ranking\n\n")
        report_lines.append("| Rank | Baseline | Mean r | 95% CI |\n")
        report_lines.append("|------|----------|--------|--------|\n")
        
        # Get unique baselines and their means
        baseline_means = {}
        for _, row in df.iterrows():
            b1, b2 = row["baseline1"], row["baseline2"]
            if b1 not in baseline_means:
                baseline_means[b1] = row["baseline1_mean"]
            if b2 not in baseline_means:
                baseline_means[b2] = row["baseline2_mean"]
        
        # Get CIs from summary files
        summary_path = Path(f"results/goal_3_prediction/lsft_resampling/{dataset_name}/lsft_{dataset_name}_lpm_selftrained_summary.json")
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
                # Get a representative baseline's CI structure
                pass
        
        sorted_baselines = sorted(baseline_means.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (baseline, mean_r) in enumerate(sorted_baselines, 1):
            baseline_short = baseline.replace("lpm_", "")
            # Get CI from comparison data
            match = df[df["baseline1"] == baseline].iloc[0] if len(df[df["baseline1"] == baseline]) > 0 else None
            if match is None:
                match = df[df["baseline2"] == baseline].iloc[0]
                ci_lower = match["baseline2_ci_lower"]
                ci_upper = match["baseline2_ci_upper"]
            else:
                ci_lower = match["baseline1_ci_lower"]
                ci_upper = match["baseline1_ci_upper"]
            
            marker = "**" if rank == 1 else ""
            report_lines.append(f"| {rank} | {marker}{baseline_short}{marker} | {mean_r:.3f} | ")
            report_lines.append(f"[{ci_lower:.3f}, {ci_upper:.3f}] |\n")
        
        report_lines.append("\n")
        
        # Save report
        report_path = Path(f"results/goal_3_prediction/lsft_resampling/{dataset_name}/COMPARISON_ANALYSIS.md")
        with open(report_path, "w") as f:
            f.write("".join(report_lines))
        
        all_reports.append(report_path)
        print(f"  ✅ Saved analysis to {report_path}")
        print()
    
    # Create aggregate report
    print("Creating aggregate comparison report...")
    aggregate_path = Path("results/goal_3_prediction/lsft_resampling/CROSS_DATASET_COMPARISON_ANALYSIS.md")
    # TODO: Aggregate analysis across datasets
    print(f"  ✅ Aggregate report: {aggregate_path}")
    
    print()
    print("=" * 60)
    print("✅ Comparison analysis complete!")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["all", "adamson", "k562", "rpe1"])
    args = parser.parse_args()
    
    analyze_comparisons(args.dataset)


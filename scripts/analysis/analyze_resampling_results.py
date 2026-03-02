#!/usr/bin/env python3
"""
Comprehensive analysis of LSFT resampling results.
Generates a detailed evaluation report with statistical comparisons.
"""

import json
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

def load_all_summaries(results_dir: Path):
    """Load all summary JSON files across datasets."""
    summaries = defaultdict(dict)
    
    for summary_path in sorted(glob.glob(str(results_dir / "**" / "*_summary.json"), recursive=True)):
        dataset = summary_path.parent.name
        baseline = Path(summary_path).stem.replace(f"lsft_{dataset}_", "").replace("_summary", "")
        
        with open(summary_path) as f:
            data = json.load(f)
        
        summaries[dataset][baseline] = data
    
    return summaries

def analyze_baseline_performance(summaries: dict, top_pct: float = 0.05):
    """Analyze baseline performance across datasets."""
    results = []
    
    for dataset, baselines in summaries.items():
        for baseline, data in baselines.items():
            # Find the key for the specified top_pct
            key = None
            for k in data.keys():
                if f"top{int(top_pct*100)}pct" in k or f"top{top_pct}" in k:
                    key = k
                    break
            
            if key is None:
                # Use middle one if exact match not found
                keys = sorted(data.keys())
                key = keys[1] if len(keys) > 1 else keys[0]
            
            r_data = data[key]["pearson_r"]
            l2_data = data[key]["l2"]
            hardness_data = data[key].get("hardness", {})
            
            results.append({
                "dataset": dataset,
                "baseline": baseline,
                "top_pct": data[key]["top_pct"],
                "pearson_r_mean": r_data["mean"],
                "pearson_r_ci_lower": r_data["ci_lower"],
                "pearson_r_ci_upper": r_data["ci_upper"],
                "pearson_r_std": r_data["std"],
                "l2_mean": l2_data["mean"],
                "l2_ci_lower": l2_data["ci_lower"],
                "l2_ci_upper": l2_data["ci_upper"],
                "l2_std": l2_data["std"],
                "hardness_mean": hardness_data.get("mean", np.nan),
                "n_perturbations": data[key]["n_perturbations"],
                "n_boot": data[key].get("n_boot", 1000),
            })
    
    return pd.DataFrame(results)

def compare_key_baselines(df: pd.DataFrame):
    """Compare key baseline pairs (scGPT vs Random, etc.)."""
    comparisons = []
    
    # Key comparisons
    key_pairs = [
        ("lpm_scgptGeneEmb", "lpm_randomGeneEmb", "scGPT vs Random Gene"),
        ("lpm_scgptGeneEmb", "lpm_selftrained", "scGPT vs Self-trained"),
        ("lpm_selftrained", "lpm_randomGeneEmb", "Self-trained vs Random Gene"),
        ("lpm_scgptGeneEmb", "lpm_scFoundationGeneEmb", "scGPT vs scFoundation"),
    ]
    
    for baseline1, baseline2, label in key_pairs:
        for dataset in df["dataset"].unique():
            df_dataset = df[df["dataset"] == dataset]
            
            b1_data = df_dataset[df_dataset["baseline"] == baseline1]
            b2_data = df_dataset[df_dataset["baseline"] == baseline2]
            
            if len(b1_data) == 0 or len(b2_data) == 0:
                continue
            
            b1 = b1_data.iloc[0]
            b2 = b2_data.iloc[0]
            
            # Check if CIs overlap (indicating non-significant difference)
            # For Pearson r (higher is better)
            r_overlap = not (b1["pearson_r_ci_upper"] < b2["pearson_r_ci_lower"] or 
                            b2["pearson_r_ci_upper"] < b1["pearson_r_ci_lower"])
            
            # For L2 (lower is better)
            l2_overlap = not (b1["l2_ci_upper"] < b2["l2_ci_lower"] or 
                             b2["l2_ci_upper"] < b1["l2_ci_lower"])
            
            comparisons.append({
                "comparison": label,
                "dataset": dataset,
                "baseline1": baseline1,
                "baseline2": baseline2,
                "delta_pearson_r": b1["pearson_r_mean"] - b2["pearson_r_mean"],
                "r_ci_overlap": r_overlap,
                "r_significant": not r_overlap,
                "delta_l2": b2["l2_mean"] - b1["l2_mean"],  # b2 - b1 because lower is better
                "l2_ci_overlap": l2_overlap,
                "l2_significant": not l2_overlap,
            })
    
    return pd.DataFrame(comparisons)

def generate_report(results_dir: Path, output_path: Path):
    """Generate comprehensive analysis report."""
    
    print("Loading summaries...")
    summaries = load_all_summaries(results_dir)
    
    print("Analyzing baseline performance...")
    df = analyze_baseline_performance(summaries, top_pct=0.05)
    
    print("Comparing key baselines...")
    comparisons_df = compare_key_baselines(df)
    
    # Generate report
    report_lines = []
    report_lines.append("# LSFT Resampling Evaluation - Comprehensive Analysis")
    report_lines.append("")
    report_lines.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Overall statistics
    report_lines.append("## Overall Statistics")
    report_lines.append("")
    report_lines.append(f"- **Datasets evaluated:** {len(summaries)}")
    report_lines.append(f"- **Total baseline-dataset pairs:** {len(df)}")
    report_lines.append(f"- **Bootstrap samples per evaluation:** {df['n_boot'].iloc[0] if len(df) > 0 else 'N/A'}")
    report_lines.append("")
    
    # Performance by dataset
    report_lines.append("## Performance by Dataset")
    report_lines.append("")
    
    for dataset in sorted(df["dataset"].unique()):
        df_dataset = df[df["dataset"] == dataset].sort_values("pearson_r_mean", ascending=False)
        report_lines.append(f"### {dataset.upper()}")
        report_lines.append("")
        report_lines.append("| Baseline | Pearson r (95% CI) | L2 (95% CI) | Rank |")
        report_lines.append("|----------|-------------------|-------------|------|")
        
        for rank, (_, row) in enumerate(df_dataset.iterrows(), 1):
            r_str = f"{row['pearson_r_mean']:.3f} [{row['pearson_r_ci_lower']:.3f}, {row['pearson_r_ci_upper']:.3f}]"
            l2_str = f"{row['l2_mean']:.2f} [{row['l2_ci_lower']:.2f}, {row['l2_ci_upper']:.2f}]"
            baseline_short = row['baseline'].replace('lpm_', '')
            report_lines.append(f"| {baseline_short} | {r_str} | {l2_str} | {rank} |")
        
        report_lines.append("")
    
    # Key comparisons
    report_lines.append("## Key Baseline Comparisons")
    report_lines.append("")
    report_lines.append("### scGPT vs Random Gene Embeddings")
    report_lines.append("")
    
    scgpt_vs_random = comparisons_df[comparisons_df["comparison"] == "scGPT vs Random Gene"]
    if len(scgpt_vs_random) > 0:
        report_lines.append("| Dataset | Δ Pearson r | Significant? | Δ L2 | Significant? |")
        report_lines.append("|---------|-------------|--------------|------|--------------|")
        
        for _, row in scgpt_vs_random.iterrows():
            sig_r = "✅ Yes" if row["r_significant"] else "❌ No (CIs overlap)"
            sig_l2 = "✅ Yes" if row["l2_significant"] else "❌ No (CIs overlap)"
            report_lines.append(f"| {row['dataset']} | {row['delta_pearson_r']:+.3f} | {sig_r} | {row['delta_l2']:+.2f} | {sig_l2} |")
    else:
        report_lines.append("*No scGPT vs Random comparisons found.*")
    
    report_lines.append("")
    
    # Key insights
    report_lines.append("## Key Findings")
    report_lines.append("")
    
    # Find best performing baseline per dataset
    best_per_dataset = df.loc[df.groupby("dataset")["pearson_r_mean"].idxmax()]
    
    report_lines.append("### Top Performers")
    report_lines.append("")
    for _, row in best_per_dataset.iterrows():
        report_lines.append(f"- **{row['dataset']}**: {row['baseline'].replace('lpm_', '')} "
                          f"(Pearson r = {row['pearson_r_mean']:.3f} "
                          f"[{row['pearson_r_ci_lower']:.3f}, {row['pearson_r_ci_upper']:.3f}])")
    
    report_lines.append("")
    
    # Analyze confidence intervals
    report_lines.append("### Statistical Precision")
    report_lines.append("")
    
    # Calculate average CI width
    df["r_ci_width"] = df["pearson_r_ci_upper"] - df["pearson_r_ci_lower"]
    df["l2_ci_width"] = df["l2_ci_upper"] - df["l2_ci_lower"]
    
    avg_r_ci_width = df["r_ci_width"].mean()
    avg_l2_ci_width = df["l2_ci_width"].mean()
    
    report_lines.append(f"- **Average Pearson r CI width:** {avg_r_ci_width:.3f}")
    report_lines.append(f"- **Average L2 CI width:** {avg_l2_ci_width:.2f}")
    report_lines.append("")
    report_lines.append(f"Bootstrap CIs provide tight bounds on performance estimates, "
                       f"with average uncertainty of ±{avg_r_ci_width/2:.3f} for Pearson r.")
    report_lines.append("")
    
    # Hardness analysis
    report_lines.append("### Hardness Distribution")
    report_lines.append("")
    hardness_df = df.dropna(subset=["hardness_mean"])
    if len(hardness_df) > 0:
        avg_hardness = hardness_df["hardness_mean"].mean()
        report_lines.append(f"- **Average hardness (cosine similarity):** {avg_hardness:.3f}")
        report_lines.append(f"- **Hardness range:** [{hardness_df['hardness_mean'].min():.3f}, "
                          f"{hardness_df['hardness_mean'].max():.3f}]")
        report_lines.append("")
    
    # Save report
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))
    
    # Also save comparison tables as CSV
    if len(comparisons_df) > 0:
        comparisons_path = output_path.parent / "baseline_comparisons_analysis.csv"
        comparisons_df.to_csv(comparisons_path, index=False)
        print(f"✅ Saved comparison table to: {comparisons_path}")
    
    print(f"✅ Report generated: {output_path}")
    return df, comparisons_df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = Path("results/goal_3_prediction/lsft_resampling")
    
    output_path = results_dir.parent / "RESAMPLING_ANALYSIS_REPORT.md"
    
    df, comparisons_df = generate_report(results_dir, output_path)
    
    print("\n=== Summary ===")
    print(f"Analyzed {len(df)} baseline-dataset pairs")
    print(f"Found {len(comparisons_df)} key comparisons")
    print(f"Report saved to: {output_path}")


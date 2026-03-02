#!/usr/bin/env python3
"""
Analyze LOGO (functional class holdout) resampling results.
Generates detailed reports on functional class extrapolation performance.
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd

def analyze_logo_results():
    """Analyze LOGO resampling results and generate reports."""
    
    print("=" * 60)
    print("LOGO RESAMPLING - DETAILED ANALYSIS")
    print("=" * 60)
    print()
    
    datasets = ["adamson", "replogle_k562", "replogle_rpe1"]
    
    all_reports = []
    
    for dataset in datasets:
        print(f"Analyzing {dataset}...")
        
        # Try multiple naming conventions
        possible_names = [
            f"logo_{dataset}_Transcription_summary.json",
            f"logo_{dataset}_transcription_summary.json",
            f"logo_{dataset}_essential_Transcription_summary.json",
            f"logo_{dataset}_essential_transcription_summary.json",
        ]
        
        summary_path = None
        for name in possible_names:
            path = Path(f"results/goal_3_prediction/functional_class_holdout_resampling/{dataset}/{name}")
            if path.exists():
                summary_path = path
                break
        
        if not summary_path.exists():
            print(f"  ⚠️  Summary file not found for {dataset}")
            continue
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        # Try multiple naming conventions for standardized file
        possible_std_names = [
            f"logo_{dataset}_Transcription_standardized.csv",
            f"logo_{dataset}_transcription_standardized.csv",
            f"logo_{dataset}_essential_Transcription_standardized.csv",
            f"logo_{dataset}_essential_transcription_standardized.csv",
        ]
        
        standardized_path = None
        for name in possible_std_names:
            path = Path(f"results/goal_3_prediction/functional_class_holdout_resampling/{dataset}/{name}")
            if path.exists():
                standardized_path = path
                break
        
        if standardized_path.exists():
            df = pd.read_csv(standardized_path)
        else:
            df = None
        
        # Generate report
        report_lines = [f"# LOGO Resampling Analysis - {dataset.upper()}\n"]
        report_lines.append(f"**Functional Class:** Transcription (held out)\n")
        report_lines.append(f"**Evaluation Type:** Functional Class Holdout (LOGO)\n")
        report_lines.append("---\n\n")
        
        report_lines.append("## Baseline Performance (Mean ± 95% CI)\n\n")
        report_lines.append("| Baseline | Pearson r (95% CI) | L2 (95% CI) | n perturbations |\n")
        report_lines.append("|----------|-------------------|-------------|-----------------|\n")
        
        # Sort baselines by performance
        baseline_perf = []
        for baseline, data in summary.items():
            if isinstance(data, dict) and "pearson_r" in data:
                r_mean = data["pearson_r"]["mean"]
                baseline_perf.append((baseline, data))
        
        baseline_perf.sort(key=lambda x: x[1]["pearson_r"]["mean"], reverse=True)
        
        for baseline, data in baseline_perf:
            baseline_short = baseline.replace("lpm_", "")
            r_data = data["pearson_r"]
            l2_data = data["l2"]
            n_pert = data.get("n_perturbations", "N/A")
            
            r_str = f"{r_data['mean']:.3f} [{r_data['ci_lower']:.3f}, {r_data['ci_upper']:.3f}]"
            l2_str = f"{l2_data['mean']:.2f} [{l2_data['ci_lower']:.2f}, {l2_data['ci_upper']:.2f}]"
            
            marker = "**" if baseline_perf.index((baseline, data)) == 0 else ""
            report_lines.append(f"| {marker}{baseline_short}{marker} | {r_str} | {l2_str} | {n_pert} |\n")
        
        report_lines.append("\n")
        
        # Key insights
        if len(baseline_perf) > 0:
            best_baseline = baseline_perf[0][0].replace("lpm_", "")
            best_r = baseline_perf[0][1]["pearson_r"]["mean"]
            
            # Find scGPT and Random if available
            scgpt_data = summary.get("lpm_scgptGeneEmb")
            random_data = summary.get("lpm_randomGeneEmb")
            
            report_lines.append("## Key Findings\n\n")
            report_lines.append(f"- **Best performing baseline:** {best_baseline} (r={best_r:.3f})\n")
            
            if scgpt_data and random_data:
                scgpt_r = scgpt_data["pearson_r"]["mean"]
                random_r = random_data["pearson_r"]["mean"]
                delta = scgpt_r - random_r
                
                # Check if CIs overlap
                scgpt_ci_u = scgpt_data["pearson_r"]["ci_upper"]
                scgpt_ci_l = scgpt_data["pearson_r"]["ci_lower"]
                random_ci_u = random_data["pearson_r"]["ci_upper"]
                random_ci_l = random_data["pearson_r"]["ci_lower"]
                
                overlap = not (scgpt_ci_u < random_ci_l or random_ci_u < scgpt_ci_l)
                
                report_lines.append(f"- **scGPT vs Random Gene:** Δr = {delta:+.3f} ")
                report_lines.append(f"(scGPT: r={scgpt_r:.3f}, Random: r={random_r:.3f})\n")
                report_lines.append(f"  - CIs {'overlap' if overlap else 'do not overlap'} ")
                report_lines.append(f"(scGPT: [{scgpt_ci_l:.3f}, {scgpt_ci_u:.3f}], ")
                report_lines.append(f"Random: [{random_ci_l:.3f}, {random_ci_u:.3f}])\n")
            
            report_lines.append("\n")
            
            # Interpretation
            report_lines.append("## Interpretation\n\n")
            report_lines.append("LOGO evaluation tests **functional extrapolation**: ")
            report_lines.append("Can the model predict expression changes for perturbations ")
            report_lines.append("targeting genes in a functional class (Transcription) that ")
            report_lines.append("was held out during training?\n\n")
            
            report_lines.append("**Lower performance compared to LSFT** is expected, as this ")
            report_lines.append("tests true biological extrapolation rather than similarity-based filtering.\n\n")
        
        # Save report
        report_path = Path(f"results/goal_3_prediction/functional_class_holdout_resampling/{dataset}/LOGO_ANALYSIS.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, "w") as f:
            f.write("".join(report_lines))
        
        all_reports.append(report_path)
        print(f"  ✅ Saved analysis to {report_path}")
        print()
    
    print("=" * 60)
    print("✅ LOGO analysis complete!")
    print("=" * 60)
    print()
    print("Analysis reports saved to:")
    for report_path in all_reports:
        print(f"  - {report_path}")

if __name__ == "__main__":
    analyze_logo_results()


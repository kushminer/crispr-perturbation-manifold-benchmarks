#!/usr/bin/env python3
"""
Generate comprehensive summary report from all Manifold Law Diagnostic Suite results.
Enhanced version with detailed per-epic analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir / "src"))

RESULTS_DIR = base_dir / "results" / "manifold_law_diagnostics"
OUTPUT_DIR = RESULTS_DIR / "summary_reports"

def load_all_epic1_results():
    """Load all Epic 1 results and aggregate."""
    epic1_dir = RESULTS_DIR / "epic1_curvature"
    result_files = list(epic1_dir.glob("lsft_k_sweep_*.csv"))
    
    all_results = []
    for file in result_files:
        try:
            df = pd.read_csv(file)
            if len(df) == 0:
                continue
            # Parse dataset and baseline from filename
            parts = file.stem.replace("lsft_k_sweep_", "").split("_", 1)
            if len(parts) == 2:
                dataset, baseline = parts[0], "_".join(parts[1:])
                if "dataset" not in df.columns:
                    df["dataset"] = dataset
                if "baseline_type" not in df.columns and "baseline" not in df.columns:
                    df["baseline_type"] = baseline
                all_results.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()

def load_all_epic3_results():
    """Load all Epic 3 results."""
    epic3_dir = RESULTS_DIR / "epic3_noise_injection"
    result_files = list(epic3_dir.glob("noise_injection_*.csv"))
    
    all_results = []
    for file in result_files:
        try:
            df = pd.read_csv(file)
            if len(df) == 0:
                continue
            all_results.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()

def load_all_epic4_results():
    """Load all Epic 4 results."""
    epic4_dir = RESULTS_DIR / "epic4_direction_flip"
    result_files = list(epic4_dir.glob("direction_flip_probe_*.csv"))
    result_files = [f for f in result_files if not f.name.endswith("_results.csv")]
    
    all_results = []
    for file in result_files:
        try:
            df = pd.read_csv(file)
            if len(df) == 0:
                continue
            all_results.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()

def load_all_epic5_results():
    """Load all Epic 5 results."""
    epic5_dir = RESULTS_DIR / "epic5_tangent_alignment"
    result_files = list(epic5_dir.glob("tangent_alignment_*.csv"))
    
    all_results = []
    for file in result_files:
        try:
            df = pd.read_csv(file)
            if len(df) == 0:
                continue
            all_results.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()

def generate_comprehensive_summary():
    """Generate comprehensive summary with detailed analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading results from all epics...")
    epic1_df = load_all_epic1_results()
    epic3_df = load_all_epic3_results()
    epic4_df = load_all_epic4_results()
    epic5_df = load_all_epic5_results()
    
    print(f"Epic 1: {len(epic1_df)} results")
    print(f"Epic 3: {len(epic3_df)} results")
    print(f"Epic 4: {len(epic4_df)} results")
    print(f"Epic 5: {len(epic5_df)} results")
    print("")
    
    # Create comprehensive summary
    summary_lines = [
        "# Manifold Law Diagnostic Suite - Comprehensive Summary",
        "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        "This report summarizes findings from all 5 diagnostic epics across 8 baselines and 3 datasets.",
        "",
        "## Epic 1: Curvature Sweep",
        "",
    ]
    
    if len(epic1_df) > 0:
        summary_lines.append(f"**Total Results:** {len(epic1_df)} per-perturbation measurements")
        
        # By dataset
        for dataset in sorted(epic1_df['dataset'].unique()):
            dataset_df = epic1_df[epic1_df['dataset'] == dataset]
            baseline_col = 'baseline_type' if 'baseline_type' in dataset_df.columns else 'baseline'
            
            summary_lines.append(f"### {dataset.upper()} Dataset")
            summary_lines.append("")
            summary_lines.append("| Baseline | Mean r (k=5) | Mean r (k=10) | Mean r (k=20) |")
            summary_lines.append("|----------|--------------|---------------|---------------|")
            
            for baseline in sorted(dataset_df[baseline_col].unique()):
                baseline_df = dataset_df[dataset_df[baseline_col] == baseline]
                r5 = baseline_df[baseline_df['k'] == 5]['performance_local_pearson_r'].mean() if 'k' in baseline_df.columns else np.nan
                r10 = baseline_df[baseline_df['k'] == 10]['performance_local_pearson_r'].mean() if 'k' in baseline_df.columns else np.nan
                r20 = baseline_df[baseline_df['k'] == 20]['performance_local_pearson_r'].mean() if 'k' in baseline_df.columns else np.nan
                
                r5_str = f"{r5:.3f}" if not np.isnan(r5) else "N/A"
                r10_str = f"{r10:.3f}" if not np.isnan(r10) else "N/A"
                r20_str = f"{r20:.3f}" if not np.isnan(r20) else "N/A"
                
                summary_lines.append(f"| {baseline} | {r5_str} | {r10_str} | {r20_str} |")
            summary_lines.append("")
    
    # Epic 3 summary
    summary_lines.append("## Epic 3: Noise Injection & Robustness")
    summary_lines.append("")
    
    if len(epic3_df) > 0:
        baseline_count = len(epic3_df[epic3_df['noise_level'] == 0.0])
        noisy_count = len(epic3_df[epic3_df['noise_level'] > 0])
        nan_count = epic3_df['mean_r'].isna().sum()
        
        summary_lines.append(f"**Baseline Entries:** {baseline_count}")
        summary_lines.append(f"**Noisy Entries:** {noisy_count}")
        summary_lines.append(f"**NaN Entries:** {nan_count}")
        summary_lines.append("")
        
        if nan_count == 0:
            summary_lines.append("✅ All noise injection experiments complete!")
            summary_lines.append("")
            
            # Lipschitz constants
            if 'lipschitz_constant' in epic3_df.columns:
                valid_lipschitz = epic3_df['lipschitz_constant'].dropna()
                if len(valid_lipschitz) > 0:
                    summary_lines.append(f"**Mean Lipschitz Constant:** {valid_lipschitz.mean():.4f}")
                    summary_lines.append(f"**Max Lipschitz Constant:** {valid_lipschitz.max():.4f}")
                    summary_lines.append("")
        else:
            summary_lines.append(f"⚠️  {nan_count} entries still need to be filled with noise injection results")
            summary_lines.append("")
    
    # Epic 4 summary
    summary_lines.append("## Epic 4: Direction-Flip Probe")
    summary_lines.append("")
    
    if len(epic4_df) > 0:
        total_adversarial = epic4_df['n_adversarial'].sum() if 'n_adversarial' in epic4_df.columns else 0
        valid_rates = epic4_df['adversarial_rate'].dropna() if 'adversarial_rate' in epic4_df.columns else pd.Series()
        
        summary_lines.append(f"**Total Test Perturbations:** {len(epic4_df)}")
        summary_lines.append(f"**Total Adversarial Pairs:** {int(total_adversarial)}")
        if len(valid_rates) > 0:
            summary_lines.append(f"**Mean Adversarial Rate:** {valid_rates.mean():.4f}")
        summary_lines.append("")
    
    # Epic 5 summary
    summary_lines.append("## Epic 5: Tangent Alignment")
    summary_lines.append("")
    
    if len(epic5_df) > 0:
        if 'tangent_alignment_score' in epic5_df.columns:
            valid_tas = epic5_df['tangent_alignment_score'].dropna()
            if len(valid_tas) > 0:
                summary_lines.append(f"**Mean TAS:** {valid_tas.mean():.3f}")
                summary_lines.append(f"**TAS Range:** {valid_tas.min():.3f} - {valid_tas.max():.3f}")
        summary_lines.append("")
    
    summary_lines.append("---")
    summary_lines.append("")
    summary_lines.append("## Next Steps")
    summary_lines.append("")
    summary_lines.append("1. Review detailed findings in per-epic summaries")
    summary_lines.append("2. Generate visualizations for key findings")
    summary_lines.append("3. Perform cross-epic comparative analysis")
    summary_lines.append("4. Prepare final reports and figures")
    
    summary_text = "\n".join(summary_lines)
    
    output_path = OUTPUT_DIR / "comprehensive_summary.md"
    with open(output_path, "w") as f:
        f.write(summary_text)
    
    print(f"✅ Comprehensive summary saved to {output_path}")
    return summary_text

if __name__ == "__main__":
    print("Generating comprehensive diagnostic suite summary...")
    generate_comprehensive_summary()
    print("✅ Summary generation complete!")


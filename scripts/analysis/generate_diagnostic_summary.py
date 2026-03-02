#!/usr/bin/env python3
"""
Generate comprehensive summary report from all Manifold Law Diagnostic Suite results.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir / "src"))

RESULTS_DIR = base_dir / "results" / "manifold_law_diagnostics"
OUTPUT_DIR = RESULTS_DIR / "summary_reports"

def load_epic1_results():
    """Load all Epic 1 (Curvature Sweep) results."""
    epic1_dir = RESULTS_DIR / "epic1_curvature"
    result_files = list(epic1_dir.glob("lsft_k_sweep_*.csv"))
    
    all_results = []
    for file in result_files:
        try:
            df = pd.read_csv(file)
            if len(df) == 0:
                continue
            # Parse dataset and baseline from filename: lsft_k_sweep_{dataset}_{baseline}.csv
            parts = file.stem.replace("lsft_k_sweep_", "").split("_", 1)
            if len(parts) == 2:
                dataset, baseline = parts[0], "_".join(parts[1:])
                df["dataset"] = dataset
                df["baseline"] = baseline
                all_results.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()

def load_epic4_results():
    """Load all Epic 4 (Direction-Flip Probe) results."""
    epic4_dir = RESULTS_DIR / "epic4_direction_flip"
    result_files = list(epic4_dir.glob("direction_flip_probe_*.csv"))
    # Exclude aggregated results file
    result_files = [f for f in result_files if not f.name.endswith("_results.csv")]
    
    all_results = []
    for file in result_files:
        try:
            df = pd.read_csv(file)
            if len(df) == 0:
                continue
            # Parse dataset and baseline: direction_flip_probe_{dataset}_{baseline}.csv
            parts = file.stem.replace("direction_flip_probe_", "").split("_", 1)
            if len(parts) == 2:
                dataset, baseline = parts[0], "_".join(parts[1:])
                if "dataset" not in df.columns:
                    df["dataset"] = dataset
                if "baseline" not in df.columns:
                    df["baseline"] = baseline
                all_results.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()

def load_epic5_results():
    """Load all Epic 5 (Tangent Alignment) results."""
    epic5_dir = RESULTS_DIR / "epic5_tangent_alignment"
    result_files = list(epic5_dir.glob("tangent_alignment_*.csv"))
    
    all_results = []
    for file in result_files:
        try:
            df = pd.read_csv(file)
            if len(df) == 0:
                continue
            # Parse dataset and baseline: tangent_alignment_{dataset}_{baseline}.csv
            parts = file.stem.replace("tangent_alignment_", "").split("_", 1)
            if len(parts) == 2:
                dataset, baseline = parts[0], "_".join(parts[1:])
                if "dataset" not in df.columns:
                    df["dataset"] = dataset
                if "baseline" not in df.columns:
                    df["baseline"] = baseline
                all_results.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()

def load_epic2_results():
    """Load all Epic 2 (Mechanism Ablation) results."""
    epic2_dir = RESULTS_DIR / "epic2_mechanism_ablation"
    result_files = list(epic2_dir.glob("mechanism_ablation_*.csv"))
    # Exclude files in subdirectories
    result_files = [f for f in result_files if f.parent == epic2_dir]
    
    all_results = []
    for file in result_files:
        try:
            df = pd.read_csv(file)
            if len(df) == 0:
                continue
            # Parse dataset and baseline: mechanism_ablation_{dataset}_{baseline}.csv
            parts = file.stem.replace("mechanism_ablation_", "").split("_", 1)
            if len(parts) == 2:
                dataset, baseline = parts[0], "_".join(parts[1:])
                if "dataset" not in df.columns:
                    df["dataset"] = dataset
                if "baseline" not in df.columns:
                    df["baseline"] = baseline
                all_results.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()

def load_epic3_results():
    """Load all Epic 3 (Noise Injection) results."""
    epic3_dir = RESULTS_DIR / "epic3_noise_injection"
    result_files = list(epic3_dir.glob("noise_injection_*.csv"))
    
    all_results = []
    for file in result_files:
        df = pd.read_csv(file)
        all_results.append(df)
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()

def generate_executive_summary():
    """Generate executive summary report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    epic1_df = load_epic1_results()
    epic2_df = load_epic2_results()
    epic3_df = load_epic3_results()
    epic4_df = load_epic4_results()
    epic5_df = load_epic5_results()
    
    summary_lines = [
        "# Manifold Law Diagnostic Suite - Executive Summary",
        "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Status",
        f"- Epic 1 (Curvature Sweep): {len(epic1_df)} results loaded" if len(epic1_df) > 0 else "- Epic 1: No results yet",
        f"- Epic 2 (Mechanism Ablation): {len(epic2_df)} results loaded" if len(epic2_df) > 0 else "- Epic 2: No results yet",
        f"- Epic 3 (Noise Injection): {len(epic3_df)} results loaded" if len(epic3_df) > 0 else "- Epic 3: No results yet",
        f"- Epic 4 (Direction-Flip Probe): {len(epic4_df)} results loaded" if len(epic4_df) > 0 else "- Epic 4: No results yet",
        f"- Epic 5 (Tangent Alignment): {len(epic5_df)} results loaded" if len(epic5_df) > 0 else "- Epic 5: No results yet",
        "",
        "## Key Findings",
        "",
    ]
    
    # Epic 1 findings
    if len(epic1_df) > 0:
        summary_lines.append("### Epic 1: Curvature Sweep")
        summary_lines.append(f"- Total per-perturbation results: {len(epic1_df)}")
        summary_lines.append(f"- Datasets: {epic1_df['dataset'].nunique() if 'dataset' in epic1_df.columns else 'N/A'}")
        summary_lines.append(f"- Baselines: {epic1_df['baseline_type'].nunique() if 'baseline_type' in epic1_df.columns else epic1_df['baseline'].nunique() if 'baseline' in epic1_df.columns else 'N/A'}")
        if 'performance_local_pearson_r' in epic1_df.columns:
            valid_r = epic1_df['performance_local_pearson_r'].dropna()
            if len(valid_r) > 0:
                summary_lines.append(f"- Pearson r range: {valid_r.min():.3f} - {valid_r.max():.3f} (mean: {valid_r.mean():.3f})")
        if 'k' in epic1_df.columns:
            k_values = sorted(epic1_df['k'].unique())
            summary_lines.append(f"- K values tested: {k_values}")
        summary_lines.append("")
    
    # Epic 4 findings
    if len(epic4_df) > 0:
        summary_lines.append("### Epic 4: Direction-Flip Probe")
        summary_lines.append(f"- Total test perturbations analyzed: {len(epic4_df)}")
        if 'adversarial_rate' in epic4_df.columns:
            total_adversarial = epic4_df['n_adversarial'].sum() if 'n_adversarial' in epic4_df.columns else 0
            summary_lines.append(f"- Total adversarial pairs found: {int(total_adversarial)}")
            valid_rates = epic4_df['adversarial_rate'].dropna()
            if len(valid_rates) > 0:
                summary_lines.append(f"- Mean adversarial rate: {valid_rates.mean():.4f}")
                summary_lines.append(f"- Max adversarial rate: {valid_rates.max():.4f}")
        summary_lines.append("")
    
    # Epic 2 findings
    if len(epic2_df) > 0:
        summary_lines.append("### Epic 2: Mechanism Ablation")
        summary_lines.append(f"- Total perturbations analyzed: {len(epic2_df)}")
        if 'delta_r' in epic2_df.columns and epic2_df['delta_r'].notna().any():
            mean_delta_r = epic2_df['delta_r'].mean()
            summary_lines.append(f"- Mean Δr (drop from same-class removal): {mean_delta_r:.3f}")
        if 'functional_class' in epic2_df.columns:
            summary_lines.append(f"- Functional classes analyzed: {epic2_df['functional_class'].nunique()}")
        summary_lines.append("")
    
    # Epic 3 findings
    if len(epic3_df) > 0:
        summary_lines.append("### Epic 3: Noise Injection")
        summary_lines.append(f"- Total experiments: {len(epic3_df)}")
        if 'noise_level' in epic3_df.columns:
            noise_levels = sorted([n for n in epic3_df['noise_level'].unique() if pd.notna(n)])
            summary_lines.append(f"- Noise levels tested: {noise_levels}")
            baseline_count = len(epic3_df[epic3_df['noise_level'] == 0.0])
            noisy_count = len(epic3_df[epic3_df['noise_level'] > 0])
            summary_lines.append(f"- Baseline entries (noise=0): {baseline_count}")
            summary_lines.append(f"- Noisy entries: {noisy_count}")
            if 'mean_r' in epic3_df.columns:
                baseline_r = epic3_df[epic3_df['noise_level'] == 0.0]['mean_r'].dropna()
                if len(baseline_r) > 0:
                    summary_lines.append(f"- Baseline mean r: {baseline_r.mean():.3f}")
        summary_lines.append("")
    
    # Epic 5 findings
    if len(epic5_df) > 0:
        summary_lines.append("### Epic 5: Tangent Alignment")
        summary_lines.append(f"- Total test perturbations analyzed: {len(epic5_df)}")
        if 'tangent_alignment_score' in epic5_df.columns:
            valid_tas = epic5_df['tangent_alignment_score'].dropna()
            if len(valid_tas) > 0:
                summary_lines.append(f"- Mean TAS: {valid_tas.mean():.3f}")
                summary_lines.append(f"- TAS range: {valid_tas.min():.3f} - {valid_tas.max():.3f}")
        if 'cca_correlation' in epic5_df.columns:
            valid_cca = epic5_df['cca_correlation'].dropna()
            if len(valid_cca) > 0:
                summary_lines.append(f"- Mean CCA correlation: {valid_cca.mean():.3f}")
        summary_lines.append("")
    
    summary_lines.append("---")
    summary_lines.append("")
    summary_lines.append("## Summary Statistics")
    summary_lines.append("")
    
    # Overall stats
    if len(epic1_df) > 0:
        summary_lines.append(f"- **Epic 1 datasets:** {epic1_df['dataset'].nunique() if 'dataset' in epic1_df.columns else 'N/A'}")
        summary_lines.append(f"- **Epic 1 baselines:** {epic1_df['baseline'].nunique() if 'baseline' in epic1_df.columns else 'N/A'}")
    
    summary_text = "\n".join(summary_lines)
    
    output_path = OUTPUT_DIR / "executive_summary.md"
    with open(output_path, "w") as f:
        f.write(summary_text)
    
    print(f"✅ Executive summary saved to {output_path}")
    return summary_text

def generate_detailed_epic_summaries():
    """Generate detailed summaries for each epic."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    epic1_df = load_epic1_results()
    epic2_df = load_epic2_results()
    epic3_df = load_epic3_results()
    epic4_df = load_epic4_results()
    epic5_df = load_epic5_results()
    
    # Epic 1 detailed summary
    if len(epic1_df) > 0:
        epic1_summary = []
        epic1_summary.append("# Epic 1: Curvature Sweep - Detailed Summary\n")
        epic1_summary.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # By dataset
        for dataset in epic1_df['dataset'].unique():
            dataset_df = epic1_df[epic1_df['dataset'] == dataset]
            epic1_summary.append(f"## {dataset.upper()}\n")
            
            # By baseline
            for baseline in dataset_df['baseline_type'].unique() if 'baseline_type' in dataset_df.columns else dataset_df['baseline'].unique():
                baseline_df = dataset_df[dataset_df.get('baseline_type', dataset_df.get('baseline')) == baseline]
                epic1_summary.append(f"### {baseline}\n")
                
                if 'k' in baseline_df.columns:
                    for k in sorted(baseline_df['k'].unique()):
                        k_df = baseline_df[baseline_df['k'] == k]
                        if 'performance_local_pearson_r' in k_df.columns:
                            r_vals = k_df['performance_local_pearson_r'].dropna()
                            if len(r_vals) > 0:
                                epic1_summary.append(f"- k={k}: Mean r={r_vals.mean():.3f} (n={len(r_vals)})\n")
                epic1_summary.append("")
        
        with open(OUTPUT_DIR / "epic1_detailed_summary.md", "w") as f:
            f.write("".join(epic1_summary))
        print(f"✅ Epic 1 detailed summary saved")
    
    # Similar for other epics...
    print("✅ Detailed summaries generated")

if __name__ == "__main__":
    print("Generating diagnostic suite summary...")
    generate_executive_summary()
    print("\nGenerating detailed epic summaries...")
    generate_detailed_epic_summaries()
    print("✅ Summary generation complete!")


#!/usr/bin/env python3
"""
Create comprehensive visualizations from Manifold Law Diagnostic Suite results.
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
FIGURES_DIR = RESULTS_DIR / "summary_reports" / "figures"

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def load_epic1_results():
    """Load Epic 1 results."""
    epic1_dir = RESULTS_DIR / "epic1_curvature"
    summary_files = list(epic1_dir.glob("curvature_sweep_summary_*.csv"))
    
    all_results = []
    for file in summary_files:
        try:
            df = pd.read_csv(file)
            parts = file.stem.replace("curvature_sweep_summary_", "").split("_", 1)
            if len(parts) == 2:
                dataset, baseline = parts
                df["dataset"] = dataset
                df["baseline"] = baseline
                all_results.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file.name}: {e}")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()

def plot_curvature_sweep_all():
    """Plot curvature sweep results for all baselines and datasets."""
    df = load_epic1_results()
    
    if len(df) == 0:
        print("No Epic 1 results to plot")
        return
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get unique baselines and datasets
    if 'baseline' not in df.columns or 'dataset' not in df.columns:
        print("Missing baseline or dataset columns")
        return
    
    baselines = sorted(df['baseline'].unique())
    datasets = sorted(df['dataset'].unique())
    
    if len(baselines) == 0 or len(datasets) == 0:
        print("No baselines or datasets found to plot")
        return
    
    # Create subplot grid
    n_rows = len(baselines)
    n_cols = len(datasets)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Handle single subplot case
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, baseline in enumerate(baselines):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            
            subset = df[(df['baseline'] == baseline) & (df['dataset'] == dataset)]
            
            if len(subset) > 0 and 'k' in subset.columns and 'mean_r' in subset.columns:
                subset = subset.sort_values('k')
                k = subset['k'].values
                mean_r = subset['mean_r'].values
                std_r = subset['std_r'].values if 'std_r' in subset.columns else None
                
                if std_r is not None and not np.isnan(std_r).all():
                    ax.errorbar(k, mean_r, yerr=std_r, fmt='o-', linewidth=2, markersize=6, capsize=3)
                else:
                    ax.plot(k, mean_r, 'o-', linewidth=2, markersize=6)
                
                ax.set_xlabel('Neighborhood Size (k)', fontsize=10)
                ax.set_ylabel('Pearson r', fontsize=10)
                ax.set_title(f'{baseline}\n{dataset}', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log')
                ax.set_ylim([0, 1.1])
            
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{baseline}\n{dataset}', fontsize=11)
                ax.set_xlabel('Neighborhood Size (k)', fontsize=10)
                ax.set_ylabel('Pearson r', fontsize=10)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "curvature_sweep_all_baselines_datasets.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved curvature sweep plot to {output_path}")


def load_epic3_results():
    """Load Epic 3 (Noise Injection) results."""
    epic3_dir = RESULTS_DIR / "epic3_noise_injection"
    result_files = list(epic3_dir.glob("noise_injection_*.csv"))
    
    all_results = []
    for file in result_files:
        try:
            df = pd.read_csv(file)
            # Parse dataset and baseline from filename
            parts = file.stem.replace("noise_injection_", "").split("_", 1)
            if len(parts) == 2:
                dataset, baseline = parts
                df["dataset"] = dataset
                df["baseline"] = baseline
                all_results.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file.name}: {e}")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


def plot_noise_sensitivity_curves():
    """Plot noise sensitivity curves (r vs noise_level) for Epic 3."""
    df = load_epic3_results()
    
    if len(df) == 0:
        print("No Epic 3 results to plot")
        return
    
    # Filter to files with baseline data (noise_level=0)
    df_valid = df[df['mean_r'].notna()].copy()
    
    if len(df_valid) == 0:
        print("No valid Epic 3 results (all NaN)")
        return
    
    # Get unique combinations
    datasets = sorted(df_valid['dataset'].unique())
    baselines = sorted(df_valid['baseline'].unique())
    k_values = sorted(df_valid['k'].unique())
    
    # Create one plot per k value (focus on k=5, 10, 20 if available)
    for k in k_values:
        fig, axes = plt.subplots(len(baselines), len(datasets), 
                                 figsize=(5*len(datasets), 4*len(baselines)))
        
        if len(baselines) == 1 and len(datasets) == 1:
            axes = np.array([[axes]])
        elif len(baselines) == 1:
            axes = axes.reshape(1, -1)
        elif len(datasets) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, baseline in enumerate(baselines):
            for j, dataset in enumerate(datasets):
                ax = axes[i, j]
                
                subset = df_valid[(df_valid['baseline'] == baseline) & 
                           (df_valid['dataset'] == dataset) & 
                           (df_valid['k'] == k)].sort_values('noise_level')
                
                if len(subset) > 0:
                    noise_levels = subset['noise_level'].values
                    mean_r = subset['mean_r'].values
                    
                    # Plot baseline point differently
                    baseline_mask = noise_levels == 0.0
                    noisy_mask = (noise_levels > 0) & ~np.isnan(mean_r)
                    
                    if np.any(baseline_mask):
                        ax.plot(noise_levels[baseline_mask], mean_r[baseline_mask], 
                               'o', color='green', markersize=8, label='Baseline', zorder=3)
                    
                    if np.any(noisy_mask):
                        ax.plot(noise_levels[noisy_mask], mean_r[noisy_mask], 
                               'o-', linewidth=2, markersize=6, alpha=0.7, zorder=2)
                    
                    ax.set_xlabel('Noise Level (σ)', fontsize=10)
                    ax.set_ylabel('Pearson r', fontsize=10)
                    ax.set_title(f'{baseline}\n{dataset} (k={k})', fontsize=10, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim([-0.02, 0.22])
                    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{baseline}\n{dataset} (k={k})', fontsize=10)
                
        plt.tight_layout()
        output_path = FIGURES_DIR / f"noise_sensitivity_curves_k{k}.png"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved noise sensitivity plot (k={k}) to {output_path}")


def plot_lipschitz_heatmap():
    """Plot Lipschitz constant heatmap from noise sensitivity analysis."""
    analysis_file = RESULTS_DIR / "epic3_noise_injection" / "noise_sensitivity_analysis.csv"
    
    if not analysis_file.exists():
        print("No noise sensitivity analysis file found - skipping Lipschitz heatmap")
        return
    
    try:
        df = pd.read_csv(analysis_file)
        
        if len(df) == 0 or 'lipschitz_constant' not in df.columns:
            print("No Lipschitz constants in analysis file")
            return
        
        # If we have k and dataset columns, create a pivot
        if 'k' in df.columns:
            if 'dataset' in df.columns:
                pivot = df.pivot(index='k', columns='dataset', values='lipschitz_constant')
            else:
                pivot = pd.DataFrame({'lipschitz_constant': df['lipschitz_constant']}, index=df['k'])
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', 
                       cbar_kws={'label': 'Lipschitz Constant'})
            plt.title('Lipschitz Constants by k and Dataset', fontsize=14, fontweight='bold')
            plt.xlabel('Dataset', fontsize=12)
            plt.ylabel('k', fontsize=12)
            
            output_path = FIGURES_DIR / "lipschitz_heatmap.png"
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved Lipschitz heatmap to {output_path}")
        else:
            print("No k column found in analysis file - cannot create heatmap")
    except Exception as e:
        print(f"Warning: Could not create Lipschitz heatmap: {e}")


if __name__ == "__main__":
    print("Creating diagnostic suite visualizations...")
    print("=" * 70)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n1. Plotting curvature sweep (Epic 1)...")
    plot_curvature_sweep_all()
    
    print("\n2. Plotting noise sensitivity curves (Epic 3)...")
    plot_noise_sensitivity_curves()
    
    print("\n3. Plotting Lipschitz heatmap (Epic 3)...")
    plot_lipschitz_heatmap()
    
    print("\n" + "=" * 70)
    print("✅ Visualization generation complete!")
    print(f"Figures saved to: {FIGURES_DIR}")

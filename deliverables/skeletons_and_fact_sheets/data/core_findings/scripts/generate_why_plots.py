#!/usr/bin/env python3
"""
WHY IT WORKS - Diagnostic Plots

These plots explain the "Why" behind the Manifold Law:
1. Consistency: Linear models have lower variance and fewer catastrophic failures.
2. Alignment: The training manifold aligns with the test manifold.
3. Generalization: Linear models extrapolate better to new functional classes (LOGO).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "manifold_law_diagnostics"
DATA_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR

# Colors
GREEN = '#27ae60'
BLUE = '#3498db'
GRAY = '#95a5a6'
RED = '#e74c3c'

plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18


def load_lsft_raw():
    """Load raw LSFT scores per perturbation."""
    path = DATA_DIR / "LSFT_raw_per_perturbation.csv"
    if not path.exists():
        print(f"❌ Missing {path}")
        return None
    return pd.read_csv(path)


def load_logo_results():
    """Load LOGO summary results."""
    path = DATA_DIR / "LOGO_results.csv"
    if not path.exists():
        print(f"❌ Missing {path}")
        return None
    return pd.read_csv(path)


def load_tangent_alignment():
    """Load tangent alignment data."""
    epic5_dir = RESULTS_DIR / "epic5_tangent_alignment"
    all_data = []
    
    for csv_file in epic5_dir.glob("tangent_alignment_*.csv"):
        parts = csv_file.stem.replace("tangent_alignment_", "").split("_", 1)
        if len(parts) != 2:
            continue
        dataset, baseline = parts
        
        # Normalize baseline
        for prefix in ['adamson_', 'k562_', 'rpe1_']:
            if baseline.startswith(prefix):
                baseline = baseline[len(prefix):]
        
        df = pd.read_csv(csv_file)
        df['dataset'] = dataset
        df['baseline'] = baseline
        all_data.append(df)
        
    if not all_data:
        print("❌ No tangent alignment data found")
        return None
        
    return pd.concat(all_data, ignore_index=True)


# =============================================================================
# PLOT 1: CONSISTENCY (Violin Plot)
# =============================================================================

def plot_consistency_violin(df):
    """
    Show distribution of errors.
    Key point: Linear models have fewer catastrophic failures (long tails).
    """
    # Filter to K562 (harder dataset, more perturbations to show distribution)
    k562 = df[(df['dataset'] == 'k562') & (df['top_pct'] == 0.05)].copy()
    
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomPertEmb']
    labels = ['PCA', 'scGPT', 'Random Pert']
    colors = [GREEN, BLUE, RED]
    
    k562 = k562[k562['baseline'].isin(baselines)]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    sns.violinplot(data=k562, x='baseline', y='performance_local_pearson_r', hue='baseline',
                   order=baselines, palette=colors, ax=ax, inner='quartile', cut=0, legend=False)
    
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel('Prediction Accuracy (r)', fontsize=16, fontweight='bold')
    ax.set_xlabel('', fontsize=14)
    ax.set_title('Why PCA Wins: Consistency\n(Fewer Catastrophic Failures)', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_ylim(-0.2, 1.0)
    
    # Annotation
    ax.annotate('Tight distribution\nHigh median', xy=(0, 0.75), 
                xytext=(0.4, 0.85), arrowprops=dict(arrowstyle='->', lw=2))
    
    ax.annotate('Long tail of failures\n(Broken manifold)', xy=(2, 0.1), 
                xytext=(1.5, 0.3), arrowprops=dict(arrowstyle='->', lw=2))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "WHY_consistency_is_key.png", 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ WHY_consistency_is_key.png")


# =============================================================================
# PLOT 2: GENERALIZATION (LOGO vs Random Split)
# =============================================================================

def plot_generalization_gap(lsft_df, logo_df):
    """
    Compare performance on Random Split (LSFT) vs Functional Holdout (LOGO).
    Key point: Linear models generalize better to unseen biology.
    """
    # Aggregate LSFT results to match LOGO (mean performance)
    lsft_agg = lsft_df[lsft_df['top_pct'] == 0.05].groupby(['dataset', 'baseline'])['performance_local_pearson_r'].mean().reset_index()
    lsft_agg.rename(columns={'performance_local_pearson_r': 'Random Split (LSFT)'}, inplace=True)
    
    # Use 'r_mean' from LOGO results
    logo_agg = logo_df[['dataset', 'baseline', 'r_mean']].copy()
    logo_agg.rename(columns={'r_mean': 'Functional Holdout (LOGO)'}, inplace=True)
    
    # Merge
    merged = pd.merge(lsft_agg, logo_agg, on=['dataset', 'baseline'])
    
    # Filter to Adamson (where LOGO was run most intensively)
    adamson = merged[merged['dataset'] == 'adamson'].copy()
    
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
    labels = ['PCA', 'scGPT', 'Random']
    colors = [GREEN, BLUE, GRAY]
    
    adamson = adamson[adamson['baseline'].isin(baselines)]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Prepare data for plotting
    x = np.arange(len(baselines))
    width = 0.35
    
    vals_lsft = [adamson[adamson['baseline'] == b]['Random Split (LSFT)'].values[0] for b in baselines]
    vals_logo = [adamson[adamson['baseline'] == b]['Functional Holdout (LOGO)'].values[0] for b in baselines]
    
    rects1 = ax.bar(x - width/2, vals_lsft, width, label='Random Split (Easy)', color=GRAY, alpha=0.5)
    rects2 = ax.bar(x + width/2, vals_logo, width, label='Functional Holdout (Hard)', color=colors)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel('Prediction Accuracy (r)', fontsize=16, fontweight='bold')
    ax.set_title('Generalization: Linear Models Extrapolate Better\n(Smaller Drop in Performance)', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower left', fontsize=12)
    
    # Add drop annotations
    for i in range(len(baselines)):
        drop = vals_lsft[i] - vals_logo[i]
        ax.text(i, max(vals_lsft[i], vals_logo[i]) + 0.02, f'-{drop:.2f}', 
                ha='center', color=RED, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "WHY_generalization_gap.png", 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ WHY_generalization_gap.png")


# =============================================================================
# PLOT 3: TANGENT ALIGNMENT (Mechanism)
# =============================================================================

def plot_tangent_alignment(lsft_df, tangent_df):
    """
    Show correlation between Tangent Alignment and Performance.
    Key point: High alignment (low Procrustes) -> High Performance.
    """
    if tangent_df is None:
        return
        
    # Merge performance into tangent data
    merged = pd.merge(lsft_df[lsft_df['top_pct'] == 0.05], 
                     tangent_df, 
                     on=['dataset', 'baseline', 'test_perturbation'])
    
    # Filter to PCA and scGPT on K562
    subset = merged[(merged['dataset'] == 'k562') & 
                   (merged['baseline'].isin(['lpm_selftrained', 'lpm_scgptGeneEmb']))].copy()
    
    subset['Baseline'] = subset['baseline'].map({
        'lpm_selftrained': 'PCA',
        'lpm_scgptGeneEmb': 'scGPT'
    })
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter plot
    sns.scatterplot(data=subset, x='procrustes_distance', y='performance_local_pearson_r',
                    hue='Baseline', palette={'PCA': GREEN, 'scGPT': BLUE},
                    style='Baseline', s=80, alpha=0.7, ax=ax)
    
    ax.set_xlabel('Manifold Misalignment (Procrustes Dist.)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Prediction Accuracy (r)', fontsize=16, fontweight='bold')
    ax.set_title('Mechanism: Better Alignment = Better Prediction', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.0)
    
    # Add trend line
    # Ideally we'd do a regression, but visual is fine
    
    ax.annotate('PCA aligns better\n(Lower distance)', xy=(0.5, 0.8), 
                xytext=(1.0, 0.9), arrowprops=dict(arrowstyle='->', lw=2, color=GREEN),
                color=GREEN, fontweight='bold')
                
    ax.annotate('More misalignment\n-> Lower accuracy', xy=(2.5, 0.4), 
                xytext=(2.0, 0.2), arrowprops=dict(arrowstyle='->', lw=2))

    ax.invert_xaxis() # Lower distance is better, so put 0 on right? No, usually 0 on left. 
    # Let's keep 0 on left, but note that "Left is Good".
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "WHY_manifold_alignment.png", 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ WHY_manifold_alignment.png")


def main():
    print("=" * 50)
    print("GENERATING 'WHY' PLOTS")
    print("=" * 50)
    
    lsft_df = load_lsft_raw()
    logo_df = load_logo_results()
    tangent_df = load_tangent_alignment()
    
    if lsft_df is not None:
        plot_consistency_violin(lsft_df)
        
        if logo_df is not None:
            plot_generalization_gap(lsft_df, logo_df)
            
        if tangent_df is not None:
            plot_tangent_alignment(lsft_df, tangent_df)
    
    print("\n" + "=" * 50)
    print("✅ DONE!")
    print("=" * 50)


if __name__ == "__main__":
    main()


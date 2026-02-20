#!/usr/bin/env python3
"""
ULTRA-SIMPLE Core Finding Plots

Design principles:
- ONE message per plot
- 3-4 elements MAX
- Large text, readable from 6 feet
- Strong color contrast
- Annotations that tell the story
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "manifold_law_diagnostics"
OUTPUT_DIR = SCRIPT_DIR

# Simplified palette - only 3 colors needed
GREEN = '#27ae60'   # Winner
GRAY = '#95a5a6'    # Others
RED = '#e74c3c'     # Worst

plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14


def load_data():
    """Load k-sweep data."""
    epic1_dir = RESULTS_DIR / "epic1_curvature"
    all_data = []
    
    for csv_file in epic1_dir.glob("lsft_k_sweep_*.csv"):
        parts = csv_file.stem.replace("lsft_k_sweep_", "").split("_", 1)
        if len(parts) != 2:
            continue
        dataset, baseline = parts
        
        # Normalize baseline name
        for prefix in ['adamson_', 'k562_', 'rpe1_']:
            if baseline.startswith(prefix):
                baseline = baseline[len(prefix):]
        
        df = pd.read_csv(csv_file)
        df['dataset'] = dataset
        df['baseline'] = baseline
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


# =============================================================================
# PLOT 1: CURVATURE - Just 3 lines, one dataset
# =============================================================================

def plot_simple_curvature(df):
    """Ultra-simple curvature plot - 3 lines only."""
    
    # Adamson only, 3 key baselines
    adamson = df[df['dataset'] == 'adamson']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    baselines = [
        ('lpm_selftrained', 'PCA', GREEN, 'o'),
        ('lpm_scgptGeneEmb', 'scGPT', GRAY, 's'),
        ('lpm_randomGeneEmb', 'Random', RED, '^'),
    ]
    
    for bl_id, label, color, marker in baselines:
        bl_data = adamson[adamson['baseline'] == bl_id]
        if len(bl_data) == 0:
            continue
        
        k_means = bl_data.groupby('k')['performance_local_pearson_r'].mean()
        ax.plot(k_means.index, k_means.values, 
                marker=marker, markersize=12, linewidth=3,
                color=color, label=label)
    
    # Highlight optimal zone
    ax.axvspan(3, 10, alpha=0.15, color=GREEN)
    ax.annotate('Optimal\n(k=5-10)', xy=(6, 0.97), fontsize=14, 
                ha='center', fontweight='bold', color=GREEN)
    
    ax.set_xlabel('Number of Neighbors (k)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Prediction Accuracy (r)', fontsize=16, fontweight='bold')
    ax.set_title('Small Neighborhoods → Best Predictions', 
                fontsize=20, fontweight='bold', pad=20)
    
    ax.set_ylim(0.5, 1.02)
    ax.set_xlim(2, 55)
    ax.set_xscale('log')
    ax.legend(loc='lower left', fontsize=14, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "SIMPLE_1_curvature.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ {output_path.name}")


# =============================================================================
# PLOT 2: BASELINE - Horizontal bars, 4 methods only
# =============================================================================

def plot_simple_baseline(df):
    """Ultra-simple baseline comparison - 4 horizontal bars."""
    
    # k=5, all datasets combined
    k5 = df[df['k'] == 5]
    
    # 4 key baselines only
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 
                 'lpm_scFoundationGeneEmb', 'lpm_randomGeneEmb']
    labels = ['PCA\n(Self-trained)', 'scGPT', 'scFoundation', 'Random']
    
    values = []
    for bl in baselines:
        bl_data = k5[k5['baseline'] == bl]
        values.append(bl_data['performance_local_pearson_r'].mean() if len(bl_data) > 0 else 0)
    
    # Colors: winner green, others gray
    colors = [GREEN if i == 0 else GRAY for i in range(len(values))]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, height=0.6, edgecolor='white', linewidth=2)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=14)
    ax.set_xlabel('Prediction Accuracy (r)', fontsize=16, fontweight='bold')
    ax.set_title('PCA Beats Deep Learning', fontsize=20, fontweight='bold', pad=20)
    ax.set_xlim(0, 1.0)
    
    # Add value labels
    for i, (bar, v) in enumerate(zip(bars, values)):
        color = 'white' if v > 0.5 else 'black'
        ax.text(v - 0.05, i, f'{v:.2f}', va='center', ha='right',
               fontsize=16, fontweight='bold', color=color)
    
    # Add annotation
    ax.annotate('Deep learning models\nperform like random!', 
                xy=(0.75, 1.5), fontsize=12, style='italic',
                ha='center', color=GRAY,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "SIMPLE_2_baseline.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ {output_path.name}")


# =============================================================================
# PLOT 3: SIMILARITY - Just 3 bars (Low/Med/High)
# =============================================================================

def plot_simple_similarity(df):
    """Ultra-simple similarity plot - 3 bars only."""
    
    # k=5, PCA only
    pca_k5 = df[(df['k'] == 5) & (df['baseline'] == 'lpm_selftrained')]
    
    if 'local_mean_similarity' not in pca_k5.columns:
        print("⚠️ No similarity data")
        return
    
    # Create 3 bins
    pca_k5 = pca_k5.copy()
    pca_k5['sim_bin'] = pd.qcut(pca_k5['local_mean_similarity'], 3, 
                                 labels=['Low\nSimilarity', 'Medium\nSimilarity', 'High\nSimilarity'])
    
    bin_means = pca_k5.groupby('sim_bin')['performance_local_pearson_r'].mean()
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    colors = [RED, GRAY, GREEN]
    x = range(3)
    bars = ax.bar(x, bin_means.values, color=colors, width=0.6, 
                  edgecolor='white', linewidth=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(bin_means.index, fontsize=14)
    ax.set_ylabel('Prediction Accuracy (r)', fontsize=16, fontweight='bold')
    ax.set_title('Similar Neighbors → Better Predictions', 
                fontsize=20, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.0)
    
    # Add value labels on top
    for bar, v in zip(bars, bin_means.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.03, f'{v:.2f}',
               ha='center', fontsize=16, fontweight='bold')
    
    # Add arrow showing trend
    ax.annotate('', xy=(2.3, 0.85), xytext=(-0.3, 0.55),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(1, 0.15, 'More similar → More accurate', fontsize=14, 
            ha='center', style='italic')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "SIMPLE_3_similarity.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ {output_path.name}")


# =============================================================================
# BONUS: One combined figure
# =============================================================================

def plot_combined_simple(df):
    """All 3 plots in one figure."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # --- Plot 1: Curvature ---
    ax = axes[0]
    adamson = df[df['dataset'] == 'adamson']
    
    for bl_id, label, color in [('lpm_selftrained', 'PCA', GREEN),
                                 ('lpm_scgptGeneEmb', 'scGPT', GRAY),
                                 ('lpm_randomGeneEmb', 'Random', RED)]:
        bl_data = adamson[adamson['baseline'] == bl_id]
        if len(bl_data) == 0:
            continue
        k_means = bl_data.groupby('k')['performance_local_pearson_r'].mean()
        ax.plot(k_means.index, k_means.values, 'o-', 
                markersize=8, linewidth=2.5, color=color, label=label)
    
    ax.set_xlabel('Neighbors (k)')
    ax.set_ylabel('Accuracy (r)')
    ax.set_title('1. Small k is Best', fontsize=16, fontweight='bold')
    ax.set_ylim(0.5, 1.02)
    ax.set_xscale('log')
    ax.legend(loc='lower left', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # --- Plot 2: Baseline ---
    ax = axes[1]
    k5 = df[df['k'] == 5]
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
    labels = ['PCA', 'scGPT', 'Random']
    values = [k5[k5['baseline'] == bl]['performance_local_pearson_r'].mean() for bl in baselines]
    colors = [GREEN, GRAY, RED]
    
    ax.barh(range(3), values, color=colors, height=0.5)
    ax.set_yticks(range(3))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Accuracy (r)')
    ax.set_title('2. PCA Wins', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1.0)
    for i, v in enumerate(values):
        ax.text(v - 0.05, i, f'{v:.2f}', va='center', ha='right', 
               fontsize=12, fontweight='bold', color='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # --- Plot 3: Similarity ---
    ax = axes[2]
    pca_k5 = df[(df['k'] == 5) & (df['baseline'] == 'lpm_selftrained')].copy()
    if 'local_mean_similarity' in pca_k5.columns:
        pca_k5['sim_bin'] = pd.qcut(pca_k5['local_mean_similarity'], 3, labels=['Low', 'Med', 'High'])
        bin_means = pca_k5.groupby('sim_bin')['performance_local_pearson_r'].mean()
        colors = [RED, GRAY, GREEN]
        ax.bar(range(3), bin_means.values, color=colors, width=0.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(['Low', 'Med', 'High'])
        for i, v in enumerate(bin_means.values):
            ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Neighbor Similarity')
    ax.set_ylabel('Accuracy (r)')
    ax.set_title('3. Similarity Matters', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.suptitle('The Manifold Law: Local Neighborhoods Explain Everything',
                fontsize=20, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "SIMPLE_combined.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ {output_path.name}")


def main():
    print("=" * 50)
    print("GENERATING ULTRA-SIMPLE PLOTS")
    print("=" * 50)
    print()
    
    df = load_data()
    print(f"Loaded {len(df)} rows")
    print()
    
    plot_simple_curvature(df)
    plot_simple_baseline(df)
    plot_simple_similarity(df)
    plot_combined_simple(df)
    
    print()
    print("=" * 50)
    print("✅ DONE! Check SIMPLE_*.png files")
    print("=" * 50)


if __name__ == "__main__":
    main()


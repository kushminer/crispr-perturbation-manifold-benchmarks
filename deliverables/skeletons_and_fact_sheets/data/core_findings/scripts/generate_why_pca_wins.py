#!/usr/bin/env python3
"""
WHY PCA WINS: The Key Explanatory Plots

The answer: CONSISTENCY across functional classes.
- PCA never fails (all r > 0.80)
- Deep learning sometimes fails catastrophically (r < 0 or r ~ 0.2)
- Random embeddings often predict the WRONG direction (r < 0)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent.parent
OUTPUT_DIR = SCRIPT_DIR

GREEN = '#27ae60'
BLUE = '#3498db'
ORANGE = '#f39c12'
GRAY = '#7f8c8d'
RED = '#e74c3c'

plt.rcParams['font.size'] = 14


def load_logo_data():
    """Load LOGO results."""
    logo_path = BASE_DIR / "results" / "goal_3_prediction" / "functional_class_holdout_resampling" / "adamson" / "logo_adamson_transcription_results.csv"
    return pd.read_csv(logo_path)


def plot_consistency_is_key(logo_df):
    """
    THE KEY PLOT: Show that PCA is consistent while others fail.
    Uses box/violin plots to show variance.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Order by mean performance
    baselines = ['lpm_selftrained', 'lpm_rpe1PertEmb', 'lpm_k562PertEmb', 
                 'lpm_scgptGeneEmb', 'lpm_gearsPertEmb', 
                 'lpm_scFoundationGeneEmb', 'lpm_randomGeneEmb', 'lpm_randomPertEmb']
    labels = ['PCA\n(self-trained)', 'RPE1\nEmb', 'K562\nEmb', 
              'scGPT', 'GEARS', 'scFoundation', 'Random\nGene', 'Random\nPert']
    
    # Filter to available baselines
    baselines = [b for b in baselines if b in logo_df['baseline'].values]
    labels = labels[:len(baselines)]
    
    data = [logo_df[logo_df['baseline'] == bl]['pearson_r'].values for bl in baselines]
    
    # Create box plots
    positions = range(len(baselines))
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    
    # Color the boxes
    colors = [GREEN, GREEN, GREEN, BLUE, ORANGE, BLUE, GRAY, RED][:len(baselines)]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add individual points
    for i, d in enumerate(data):
        x = np.random.normal(i, 0.04, size=len(d))
        ax.scatter(x, d, color='black', alpha=0.6, s=50, zorder=3)
    
    # Add zero line
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero (wrong direction)')
    
    # Highlight the "danger zone"
    ax.axhspan(-1, 0, alpha=0.1, color='red')
    ax.text(len(baselines)-0.5, -0.3, 'WRONG\nDIRECTION', ha='right', fontsize=12, 
            color='red', fontweight='bold', style='italic')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('LOGO Pearson r', fontsize=14, fontweight='bold')
    ax.set_title('Why PCA Wins: Consistency Across Functional Classes\n(Leave-One-GO-Class-Out Evaluation)',
                fontsize=16, fontweight='bold', pad=15)
    ax.set_ylim(-0.7, 1.0)
    
    # Add annotation for PCA
    ax.annotate('PCA: Always works\n(min r = 0.81)', xy=(0, 0.85), 
                xytext=(2, 0.65), fontsize=11, color=GREEN,
                arrowprops=dict(arrowstyle='->', color=GREEN),
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Add annotation for deep learning
    ax.annotate('Deep learning:\nSometimes fails\n(min r = -0.14)', xy=(5, -0.14), 
                xytext=(6.5, -0.5), fontsize=11, color=BLUE,
                arrowprops=dict(arrowstyle='->', color=BLUE),
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "WHY_consistency_is_key.png", 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ WHY_consistency_is_key.png")


def plot_failure_rate(logo_df):
    """
    Show proportion of "failures" (r < 0.5) by baseline.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baselines = ['lpm_selftrained', 'lpm_k562PertEmb', 'lpm_rpe1PertEmb',
                 'lpm_scgptGeneEmb', 'lpm_scFoundationGeneEmb', 
                 'lpm_gearsPertEmb', 'lpm_randomGeneEmb', 'lpm_randomPertEmb']
    labels = ['PCA', 'K562 Emb', 'RPE1 Emb', 'scGPT', 'scFoundation', 
              'GEARS', 'Random Gene', 'Random Pert']
    
    failure_rates = []
    for bl in baselines:
        bl_data = logo_df[logo_df['baseline'] == bl]['pearson_r']
        if len(bl_data) > 0:
            failure_rate = (bl_data < 0.5).sum() / len(bl_data)
            failure_rates.append(failure_rate * 100)
        else:
            failure_rates.append(0)
    
    colors = [GREEN if r == 0 else (RED if r > 50 else ORANGE) for r in failure_rates]
    
    bars = ax.barh(range(len(baselines)), failure_rates, color=colors, height=0.6)
    
    ax.set_yticks(range(len(baselines)))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel('Failure Rate (% with r < 0.5)', fontsize=14, fontweight='bold')
    ax.set_title('PCA Never Fails, Deep Learning Sometimes Does',
                fontsize=16, fontweight='bold', pad=15)
    ax.set_xlim(0, 100)
    
    # Add value labels
    for i, (bar, v) in enumerate(zip(bars, failure_rates)):
        ax.text(v + 2, i, f'{v:.0f}%', va='center', fontsize=12, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "WHY_failure_rate.png", 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ WHY_failure_rate.png")


def plot_variance_comparison(logo_df):
    """
    Compare variance (std) across baselines.
    Lower variance = more reliable.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baselines = ['lpm_selftrained', 'lpm_k562PertEmb', 'lpm_rpe1PertEmb',
                 'lpm_scgptGeneEmb', 'lpm_scFoundationGeneEmb', 
                 'lpm_gearsPertEmb', 'lpm_randomGeneEmb', 'lpm_randomPertEmb']
    labels = ['PCA', 'K562 Emb', 'RPE1 Emb', 'scGPT', 'scFoundation', 
              'GEARS', 'Random Gene', 'Random Pert']
    
    stds = []
    means = []
    for bl in baselines:
        bl_data = logo_df[logo_df['baseline'] == bl]['pearson_r']
        if len(bl_data) > 0:
            stds.append(bl_data.std())
            means.append(bl_data.mean())
        else:
            stds.append(0)
            means.append(0)
    
    x = np.arange(len(baselines))
    width = 0.35
    
    colors = [GREEN if s < 0.1 else (RED if s > 0.5 else ORANGE) for s in stds]
    
    ax.bar(x, stds, width, color=colors, label='Std Dev (lower = more reliable)')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, rotation=30, ha='right')
    ax.set_ylabel('Standard Deviation of r', fontsize=14, fontweight='bold')
    ax.set_title('PCA Is Reliable, Deep Learning Is Variable',
                fontsize=16, fontweight='bold', pad=15)
    ax.set_ylim(0, 0.8)
    
    # Add threshold line
    ax.axhline(y=0.1, color=GREEN, linestyle='--', alpha=0.7)
    ax.text(len(baselines)-0.5, 0.12, 'Low variance\n(reliable)', 
            ha='right', fontsize=10, color=GREEN)
    
    ax.axhline(y=0.5, color=RED, linestyle='--', alpha=0.7)
    ax.text(len(baselines)-0.5, 0.52, 'High variance\n(unreliable)', 
            ha='right', fontsize=10, color=RED)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "WHY_variance_comparison.png", 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ WHY_variance_comparison.png")


def plot_summary_why_pca_wins():
    """
    Combined summary plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Load data
    logo_df = load_logo_data()
    
    # Left: Mean ± Std
    ax = axes[0]
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 
                 'lpm_scFoundationGeneEmb', 'lpm_randomGeneEmb']
    labels = ['PCA', 'scGPT', 'scFoundation', 'Random']
    colors = [GREEN, BLUE, BLUE, GRAY]
    
    means = []
    stds = []
    for bl in baselines:
        bl_data = logo_df[logo_df['baseline'] == bl]['pearson_r']
        means.append(bl_data.mean())
        stds.append(bl_data.std())
    
    x = np.arange(len(baselines))
    bars = ax.bar(x, means, yerr=stds, capsize=10, color=colors, width=0.6)
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('LOGO Pearson r', fontsize=12)
    ax.set_title('Mean ± Std Across Classes', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.3, 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Right: Min vs Max
    ax = axes[1]
    mins = [logo_df[logo_df['baseline'] == bl]['pearson_r'].min() for bl in baselines]
    maxs = [logo_df[logo_df['baseline'] == bl]['pearson_r'].max() for bl in baselines]
    
    ax.scatter(x, mins, s=100, marker='v', color=colors, label='Min', zorder=3)
    ax.scatter(x, maxs, s=100, marker='^', color=colors, label='Max', zorder=3)
    
    for i in range(len(baselines)):
        ax.plot([i, i], [mins[i], maxs[i]], color=colors[i], linewidth=2, alpha=0.5)
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('LOGO Pearson r', fontsize=12)
    ax.set_title('Range (Min to Max)', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.7, 1.0)
    ax.legend(loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Highlight danger zone
    ax.axhspan(-0.7, 0, alpha=0.1, color='red')
    
    plt.suptitle('Why PCA Wins: Consistent Performance Across Functional Classes',
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "WHY_summary.png", 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ WHY_summary.png")


def main():
    print("=" * 50)
    print("GENERATING 'WHY PCA WINS' PLOTS")
    print("=" * 50)
    
    logo_df = load_logo_data()
    print(f"Loaded {len(logo_df)} LOGO results\n")
    
    plot_consistency_is_key(logo_df)
    plot_failure_rate(logo_df)
    plot_variance_comparison(logo_df)
    plot_summary_why_pca_wins()
    
    print("\n" + "=" * 50)
    print("✅ DONE!")
    print("=" * 50)


if __name__ == "__main__":
    main()


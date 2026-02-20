#!/usr/bin/env python3
"""
The Full Story: Raw Baseline → LSFT → LOGO

Key insight: LSFT rescues weak embeddings locally, but only PCA generalizes.

"Local similarity — not giant AI models — predicts gene knockout effects."
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR

GREEN = '#27ae60'
BLUE = '#3498db'
GRAY = '#95a5a6'
RED = '#e74c3c'
ORANGE = '#f39c12'

plt.rcParams['font.size'] = 12


def load_data():
    """Load all data."""
    raw = pd.read_csv(DATA_DIR / "LSFT_raw_per_perturbation.csv")
    logo = pd.read_csv(DATA_DIR / "LOGO_results.csv")
    lsft = pd.read_csv(DATA_DIR / "LSFT_resampling.csv")
    return {'raw': raw, 'logo': logo, 'lsft': lsft}


def get_baseline_performance(raw):
    """Extract raw baseline performance (no LSFT)."""
    k5 = raw[raw['top_pct'] == 0.05]
    
    results = []
    for dataset in ['adamson', 'k562', 'rpe1']:
        ds_data = k5[k5['dataset'] == dataset]
        for bl in ds_data['baseline'].unique():
            bl_data = ds_data[ds_data['baseline'] == bl]
            results.append({
                'dataset': dataset,
                'baseline': bl,
                'raw_r': bl_data['performance_baseline_pearson_r'].mean(),
                'raw_l2': bl_data['performance_baseline_l2'].mean(),
                'lsft_r': bl_data['performance_local_pearson_r'].mean(),
                'lsft_l2': bl_data['performance_local_l2'].mean(),
            })
    
    return pd.DataFrame(results)


# =============================================================================
# PLOT 1: THE LIFT - Raw vs LSFT
# =============================================================================

def plot_1_the_lift(data):
    """Show how LSFT lifts weak embeddings."""
    
    baseline_perf = get_baseline_performance(data['raw'])
    
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
    labels = ['PCA', 'scGPT', 'Random']
    colors = [GREEN, BLUE, GRAY]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    for ax, dataset in zip(axes, ['adamson', 'k562', 'rpe1']):
        ds_data = baseline_perf[baseline_perf['dataset'] == dataset]
        
        x = np.arange(len(baselines))
        width = 0.35
        
        raw_values = [ds_data[ds_data['baseline'] == bl]['raw_r'].values[0] 
                     for bl in baselines]
        lsft_values = [ds_data[ds_data['baseline'] == bl]['lsft_r'].values[0] 
                      for bl in baselines]
        
        bars1 = ax.bar(x - width/2, raw_values, width, label='Raw Baseline',
                      color=[c for c in colors], alpha=0.5)
        bars2 = ax.bar(x + width/2, lsft_values, width, label='After LSFT',
                      color=[c for c in colors], edgecolor='black', linewidth=2)
        
        # Add arrows showing improvement
        for i, (r, l) in enumerate(zip(raw_values, lsft_values)):
            improvement = l - r
            arrow_color = 'green' if improvement > 0 else 'red'
            ax.annotate('', xy=(i + width/2, l), xytext=(i - width/2, r),
                       arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
            ax.text(i, max(r, l) + 0.03, f'{improvement:+.2f}',
                   ha='center', fontsize=10, fontweight='bold',
                   color='green' if improvement > 0 else 'red')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel('Pearson r' if dataset == 'adamson' else '')
        ax.set_ylim(0, 1.1)
        ax.set_title(f'{dataset.upper()}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        if dataset == 'adamson':
            ax.legend(loc='lower right')
    
    fig.suptitle('LSFT Rescues Weak Embeddings:\nLocal Similarity Lifts Random to Match Deep Learning',
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "FULL_1_the_lift.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ FULL_1_the_lift.png")


# =============================================================================
# PLOT 2: THE CONVERGENCE - All methods converge with LSFT
# =============================================================================

def plot_2_convergence(data):
    """Show that LSFT makes all embeddings perform similarly."""
    
    lsft = data['lsft']
    baseline_perf = get_baseline_performance(data['raw'])
    
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
    labels = ['PCA', 'scGPT', 'Random']
    colors = [GREEN, BLUE, GRAY]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Raw baseline (big spread)
    ax = axes[0]
    raw_means = [baseline_perf[baseline_perf['baseline'] == bl]['raw_r'].mean() 
                for bl in baselines]
    bars = ax.bar(range(3), raw_means, color=colors, width=0.5)
    for bar, v in zip(bars, raw_means):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
               f'{v:.2f}', ha='center', fontsize=14, fontweight='bold')
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel('Pearson r', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_title('Raw Baseline\n(Big differences)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add spread annotation
    spread = max(raw_means) - min(raw_means)
    ax.annotate(f'Spread: {spread:.2f}', xy=(1, 0.3), fontsize=12, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    # Right: After LSFT (converged)
    ax = axes[1]
    lsft_means = [lsft[lsft['baseline'] == bl]['r_mean'].mean() for bl in baselines]
    bars = ax.bar(range(3), lsft_means, color=colors, width=0.5)
    for bar, v in zip(bars, lsft_means):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
               f'{v:.2f}', ha='center', fontsize=14, fontweight='bold')
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.set_title('After LSFT\n(All converge!)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add spread annotation
    spread = max(lsft_means) - min(lsft_means)
    ax.annotate(f'Spread: {spread:.2f}', xy=(1, 0.3), fontsize=12, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    fig.suptitle('Local Similarity Makes All Embeddings Equivalent',
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "FULL_2_convergence.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ FULL_2_convergence.png")


# =============================================================================
# PLOT 3: THE GENERALIZATION TEST - But only PCA generalizes
# =============================================================================

def plot_3_generalization(data):
    """Show that only PCA generalizes to out-of-distribution."""
    
    logo = data['logo']
    baseline_perf = get_baseline_performance(data['raw'])
    lsft = data['lsft']
    
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
    labels = ['PCA', 'scGPT', 'Random']
    colors = [GREEN, BLUE, GRAY]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    # Raw, LSFT, LOGO side by side
    titles = ['Raw Baseline', 'After LSFT (Local)', 'LOGO (Generalization)']
    
    for ax, title, data_source in zip(axes, titles, ['raw', 'lsft', 'logo']):
        if data_source == 'raw':
            means = [baseline_perf[baseline_perf['baseline'] == bl]['raw_r'].mean() 
                    for bl in baselines]
        elif data_source == 'lsft':
            means = [lsft[lsft['baseline'] == bl]['r_mean'].mean() for bl in baselines]
        else:
            means = [logo[logo['baseline'] == bl]['r_mean'].mean() for bl in baselines]
        
        bars = ax.bar(range(3), means, color=colors, width=0.5)
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                   f'{v:.2f}', ha='center', fontsize=14, fontweight='bold')
        
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel('Pearson r' if data_source == 'raw' else '')
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Highlight winner
        if data_source == 'logo':
            ax.annotate('PCA wins\ngeneralization!', xy=(0, 0.55), fontsize=11,
                       ha='center', color=GREEN, fontweight='bold')
    
    fig.suptitle('Only PCA Generalizes to Out-of-Distribution:\nLocal Similarity Wins Locally, But Needs Good Embeddings to Generalize',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "FULL_3_generalization.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ FULL_3_generalization.png")


# =============================================================================
# PLOT 4: THE HEADLINE - Single summary
# =============================================================================

def plot_4_headline(data):
    """The headline plot for the poster."""
    
    logo = data['logo']
    lsft = data['lsft']
    baseline_perf = get_baseline_performance(data['raw'])
    
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
    labels = ['PCA\n(Local Similarity)', 'scGPT\n(Billion params)', 'Random']
    colors = [GREEN, BLUE, GRAY]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Average across datasets
    lsft_means = [lsft[lsft['baseline'] == bl]['r_mean'].mean() for bl in baselines]
    
    bars = ax.bar(range(3), lsft_means, color=colors, width=0.5, edgecolor='white', linewidth=2)
    
    for bar, v in zip(bars, lsft_means):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
               f'{v:.2f}', ha='center', fontsize=22, fontweight='bold')
    
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=16)
    ax.set_ylabel('Prediction Accuracy (Pearson r)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.0)
    
    # The headline
    ax.text(0.5, 0.95, 'Local Similarity — Not Giant AI Models —\nPredicts Gene Knockout Effects',
           transform=ax.transAxes, fontsize=20, fontweight='bold',
           ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2))
    
    # Key insight
    ax.text(0.5, 0.15, 'Using just 5% nearest neighbors,\nall embeddings achieve similar accuracy.',
           transform=ax.transAxes, fontsize=14, ha='center', style='italic',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "FULL_4_headline.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ FULL_4_headline.png")


# =============================================================================
# PLOT 5: THE COMPLETE STORY - 3 panels
# =============================================================================

def plot_5_complete(data):
    """The complete story in one figure."""
    
    logo = data['logo']
    lsft = data['lsft']
    baseline_perf = get_baseline_performance(data['raw'])
    
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
    labels = ['PCA', 'scGPT', 'Random']
    colors = [GREEN, BLUE, GRAY]
    
    fig = plt.figure(figsize=(18, 6))
    
    # Panel 1: Raw baseline
    ax1 = fig.add_subplot(131)
    raw_means = [baseline_perf[baseline_perf['baseline'] == bl]['raw_r'].mean() 
                for bl in baselines]
    bars = ax1.bar(range(3), raw_means, color=colors, width=0.5)
    for bar, v in zip(bars, raw_means):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.2f}',
                ha='center', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Pearson r', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.set_title('① Raw Prediction\n(Without LSFT)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: After LSFT
    ax2 = fig.add_subplot(132)
    lsft_means = [lsft[lsft['baseline'] == bl]['r_mean'].mean() for bl in baselines]
    bars = ax2.bar(range(3), lsft_means, color=colors, width=0.5)
    for bar, v in zip(bars, lsft_means):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.2f}',
                ha='center', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 1.0)
    ax2.set_title('② With Local Similarity\n(LSFT: 5% neighbors)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.annotate('All converge!', xy=(1, 0.65), fontsize=12, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Panel 3: LOGO
    ax3 = fig.add_subplot(133)
    logo_means = [logo[logo['baseline'] == bl]['r_mean'].mean() for bl in baselines]
    bars = ax3.bar(range(3), logo_means, color=colors, width=0.5)
    for bar, v in zip(bars, logo_means):
        ax3.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.2f}',
                ha='center', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(labels)
    ax3.set_ylim(0, 1.0)
    ax3.set_title('③ Generalization\n(Functional Holdout)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.annotate('PCA wins!', xy=(0, logo_means[0] - 0.15), fontsize=12, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    fig.suptitle('Local Similarity — Not Giant AI Models — Predicts Gene Knockout Effects',
                fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "FULL_5_complete.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ FULL_5_complete.png")


def main():
    print("=" * 60)
    print("BUILDING THE FULL STORY")
    print("Raw Baseline → LSFT → LOGO")
    print("=" * 60)
    print()
    
    data = load_data()
    
    plot_1_the_lift(data)
    plot_2_convergence(data)
    plot_3_generalization(data)
    plot_4_headline(data)
    plot_5_complete(data)
    
    print()
    print("=" * 60)
    print("✅ FULL STORY COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()


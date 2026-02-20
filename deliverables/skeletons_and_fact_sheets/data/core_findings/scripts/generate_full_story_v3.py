#!/usr/bin/env python3
"""
The Full Story V3 - Refined

Updates:
- FULL_4_headline: Uses RAW BASELINE numbers, includes scGPT and GEARS
- FULL_5_complete: Vertical layout, more models

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
PURPLE = '#9b59b6'
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
# UPDATED PLOT 4: THE HEADLINE - Raw baseline with more models
# =============================================================================

def plot_4_headline_updated(data):
    """The headline plot - RAW BASELINE, includes scGPT and GEARS."""
    
    baseline_perf = get_baseline_performance(data['raw'])
    
    # More baselines, ordered by expected performance
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_scFoundationGeneEmb', 
                 'lpm_gearsPertEmb', 'lpm_randomGeneEmb']
    labels = ['PCA\n(Self-trained)', 'scGPT\n(Pretrained)', 'scFoundation\n(Pretrained)',
              'GEARS\n(Graph-based)', 'Random\n(Control)']
    colors = [GREEN, BLUE, PURPLE, ORANGE, GRAY]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Average raw baseline across datasets
    means = []
    for bl in baselines:
        bl_data = baseline_perf[baseline_perf['baseline'] == bl]
        if len(bl_data) > 0:
            means.append(bl_data['raw_r'].mean())
        else:
            means.append(0)
    
    bars = ax.bar(range(len(baselines)), means, color=colors, width=0.6, 
                  edgecolor='white', linewidth=2)
    
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
               f'{v:.2f}', ha='center', fontsize=18, fontweight='bold')
    
    ax.set_xticks(range(len(baselines)))
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel('Prediction Accuracy (Pearson r)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.0)
    
    # The headline
    ax.text(0.5, 0.95, 'Local Similarity — Not Giant AI Models —\nPredicts Gene Knockout Effects',
           transform=ax.transAxes, fontsize=22, fontweight='bold',
           ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='black', linewidth=2))
    
    # Key insight - updated for raw baseline
    ax.text(0.5, 0.15, 'Raw baseline prediction accuracy\n(Before local similarity training)',
           transform=ax.transAxes, fontsize=13, ha='center', style='italic',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "FULL_4_headline.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ FULL_4_headline.png (updated with raw baseline + more models)")


# =============================================================================
# UPDATED PLOT 5: THE COMPLETE STORY - Vertical, more models
# =============================================================================

def plot_5_complete_updated(data):
    """The complete story - VERTICAL layout, more models."""
    
    logo = data['logo']
    lsft = data['lsft']
    baseline_perf = get_baseline_performance(data['raw'])
    
    # More baselines
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_scFoundationGeneEmb',
                 'lpm_gearsPertEmb', 'lpm_randomGeneEmb']
    labels = ['PCA', 'scGPT', 'scFoundation', 'GEARS', 'Random']
    colors = [GREEN, BLUE, PURPLE, ORANGE, GRAY]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    
    # Panel 1: Raw baseline
    ax = axes[0]
    raw_means = []
    for bl in baselines:
        bl_data = baseline_perf[baseline_perf['baseline'] == bl]
        raw_means.append(bl_data['raw_r'].mean() if len(bl_data) > 0 else 0)
    
    bars = ax.bar(range(len(baselines)), raw_means, color=colors, width=0.6)
    for bar, v in zip(bars, raw_means):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.2f}',
                ha='center', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(baselines)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Pearson r', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_title('① Raw Baseline Prediction\n(Without local similarity training)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Highlight spread
    spread = max(raw_means) - min(raw_means)
    ax.annotate(f'Spread: {spread:.2f}', xy=(2, 0.35), fontsize=12, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Panel 2: After LSFT
    ax = axes[1]
    lsft_means = []
    for bl in baselines:
        bl_data = lsft[lsft['baseline'] == bl]
        lsft_means.append(bl_data['r_mean'].mean() if len(bl_data) > 0 else 0)
    
    bars = ax.bar(range(len(baselines)), lsft_means, color=colors, width=0.6)
    for bar, v in zip(bars, lsft_means):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.2f}',
                ha='center', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(baselines)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Pearson r', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_title('② With Local Similarity Training\n(LSFT: 5% nearest neighbors)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Highlight convergence
    spread = max(lsft_means) - min([m for m in lsft_means if m > 0])
    ax.annotate(f'All converge!\nSpread: {spread:.2f}', xy=(2, 0.35), fontsize=12, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Panel 3: LOGO (Generalization)
    ax = axes[2]
    logo_means = []
    for bl in baselines:
        bl_data = logo[logo['baseline'] == bl]
        logo_means.append(bl_data['r_mean'].mean() if len(bl_data) > 0 else 0)
    
    bars = ax.bar(range(len(baselines)), logo_means, color=colors, width=0.6)
    for bar, v in zip(bars, logo_means):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.2f}',
                ha='center', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(baselines)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Pearson r', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_title('③ Generalization Test\n(Functional class holdout - predict unseen gene functions)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Highlight PCA winning
    ax.annotate('PCA wins\ngeneralization!', xy=(0, logo_means[0] - 0.15), fontsize=12, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontweight='bold')
    
    fig.suptitle('Local Similarity — Not Giant AI Models —\nPredicts Gene Knockout Effects',
                fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "FULL_5_complete.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ FULL_5_complete.png (updated: vertical + more models)")


def main():
    print("=" * 60)
    print("UPDATING FULL STORY PLOTS")
    print("=" * 60)
    print()
    
    data = load_data()
    
    plot_4_headline_updated(data)
    plot_5_complete_updated(data)
    
    print()
    print("=" * 60)
    print("✅ UPDATES COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()


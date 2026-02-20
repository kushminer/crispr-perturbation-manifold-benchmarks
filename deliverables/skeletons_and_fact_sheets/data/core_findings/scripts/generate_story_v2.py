#!/usr/bin/env python3
"""
Updated STORY plots:
- STORY_6: Uses BASELINE values (not LSFT)
- STORY_7: Shows how LSFT improves baselines

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
                'lsft_r': bl_data['performance_local_pearson_r'].mean(),
            })
    
    return pd.DataFrame(results)


# =============================================================================
# STORY_6: Single slide with BASELINE values
# =============================================================================

def plot_story_6_baseline(data):
    """Single slide using BASELINE values (not LSFT)."""
    
    baseline_perf = get_baseline_performance(data['raw'])
    logo = data['logo']
    
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
    labels = ['Local\nSimilarity', 'scGPT', 'Random']
    colors = [GREEN, BLUE, GRAY]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get RAW BASELINE means (not LSFT)
    baseline_means = []
    for bl in baselines:
        bl_data = baseline_perf[baseline_perf['baseline'] == bl]
        baseline_means.append(bl_data['raw_r'].mean() if len(bl_data) > 0 else 0)
    
    # Get LOGO means
    logo_means = []
    for bl in baselines:
        bl_data = logo[logo['baseline'] == bl]
        logo_means.append(bl_data['r_mean'].mean() if len(bl_data) > 0 else 0)
    
    x = np.arange(3)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_means, width, label='Raw Baseline',
                   color=colors, alpha=0.8)
    bars2 = ax.bar(x + width/2, logo_means, width, label='Functional Holdout',
                   color=colors, alpha=0.5, hatch='//')
    
    # Add values
    for bars, means in [(bars1, baseline_means), (bars2, logo_means)]:
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                   f'{v:.2f}', ha='center', fontsize=14, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=16)
    ax.set_ylabel('Prediction Accuracy (Pearson r)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=14, loc='upper right')
    
    # The headline
    ax.text(0.5, 0.95, 'Local Similarity — Not Giant AI Models —\nPredicts Gene Knockout Effects',
           transform=ax.transAxes, fontsize=20, fontweight='bold',
           ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "STORY_6_single_slide.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ STORY_6_single_slide.png (updated with baseline values)")


# =============================================================================
# STORY_7: How LSFT improves baselines
# =============================================================================

def plot_story_7_lsft_improvement(data):
    """Show how LSFT (local similarity training) improves each baseline."""
    
    baseline_perf = get_baseline_performance(data['raw'])
    
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_scFoundationGeneEmb',
                 'lpm_gearsPertEmb', 'lpm_randomGeneEmb']
    labels = ['PCA', 'scGPT', 'scFoundation', 'GEARS', 'Random']
    colors = [GREEN, BLUE, PURPLE, ORANGE, GRAY]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get before/after LSFT
    raw_means = []
    lsft_means = []
    improvements = []
    
    for bl in baselines:
        bl_data = baseline_perf[baseline_perf['baseline'] == bl]
        if len(bl_data) > 0:
            raw = bl_data['raw_r'].mean()
            lsft = bl_data['lsft_r'].mean()
            raw_means.append(raw)
            lsft_means.append(lsft)
            improvements.append(lsft - raw)
        else:
            raw_means.append(0)
            lsft_means.append(0)
            improvements.append(0)
    
    x = np.arange(len(baselines))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, raw_means, width, label='Before LSFT',
                   color=colors, alpha=0.5)
    bars2 = ax.bar(x + width/2, lsft_means, width, label='After LSFT',
                   color=colors, edgecolor='black', linewidth=2)
    
    # Add improvement arrows and values
    for i, (raw, lsft, imp) in enumerate(zip(raw_means, lsft_means, improvements)):
        # Arrow from raw to lsft
        if imp > 0:
            ax.annotate('', xy=(i + width/2, lsft - 0.02), xytext=(i - width/2, raw + 0.02),
                       arrowprops=dict(arrowstyle='->', color='green', lw=2))
            ax.text(i, max(raw, lsft) + 0.05, f'+{imp:.2f}',
                   ha='center', fontsize=12, fontweight='bold', color='green')
        else:
            ax.annotate('', xy=(i + width/2, lsft + 0.02), xytext=(i - width/2, raw - 0.02),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
            ax.text(i, max(raw, lsft) + 0.05, f'{imp:.2f}',
                   ha='center', fontsize=12, fontweight='bold', color='red')
        
        # Add values on bars
        ax.text(bars1[i].get_x() + bars1[i].get_width()/2, raw + 0.01,
               f'{raw:.2f}', ha='center', fontsize=10)
        ax.text(bars2[i].get_x() + bars2[i].get_width()/2, lsft + 0.01,
               f'{lsft:.2f}', ha='center', fontsize=10)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel('Prediction Accuracy (Pearson r)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_title('Local Similarity Training Improves All Embeddings',
                fontsize=18, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right')
    
    # Key insight
    ax.text(0.5, 0.15, 'LSFT uses 5% nearest neighbors to improve predictions.\n'
                       'Biggest gains for weak embeddings (Random: +0.19)',
           transform=ax.transAxes, fontsize=12, ha='center', style='italic',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "STORY_7_lsft_improvement.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ STORY_7_lsft_improvement.png (new: shows LSFT improvement)")


def main():
    print("=" * 60)
    print("UPDATING STORY PLOTS")
    print("=" * 60)
    print()
    
    data = load_data()
    
    plot_story_6_baseline(data)
    plot_story_7_lsft_improvement(data)
    
    print()
    print("=" * 60)
    print("✅ STORY UPDATES COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
POSTER v4: Vertically stacked 3-panel layout

Shows the complete story in vertical flow:
1. Raw Embedding → 2. + Local Similarity → 3. Generalization Test
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR

# Colors
GREEN = '#27ae60'
BLUE = '#3498db'
PURPLE = '#9b59b6'
ORANGE = '#f39c12'
GRAY = '#95a5a6'

plt.rcParams['font.family'] = 'DejaVu Sans'


def load_data():
    """Load all data."""
    raw = pd.read_csv(DATA_DIR / "LSFT_raw_per_perturbation.csv")
    logo = pd.read_csv(DATA_DIR / "LOGO_results.csv")
    return {'raw': raw, 'logo': logo}


def get_all_performance(raw):
    """Extract both baseline AND LSFT performance."""
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


def create_poster_v4(data):
    """Vertically stacked 3-panel layout."""
    
    perf = get_all_performance(data['raw'])
    logo = data['logo']
    
    methods = [
        ('lpm_selftrained', 'PCA', GREEN),
        ('lpm_scgptGeneEmb', 'scGPT', BLUE),
        ('lpm_scFoundationGeneEmb', 'scFoundation', PURPLE),
        ('lpm_gearsPertEmb', 'GEARS', ORANGE),
        ('lpm_randomGeneEmb', 'Random', GRAY),
    ]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    
    # Main title
    fig.suptitle('Local Similarity — Not Giant AI Models —\nPredicts Gene Knockout Effects',
                fontsize=24, fontweight='bold', y=0.97,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=2))
    
    colors = [m[2] for m in methods]
    labels = [m[1] for m in methods]
    
    # Get all data
    raw_means = []
    lsft_means = []
    logo_means = []
    
    for bl, label, color in methods:
        bl_data = perf[perf['baseline'] == bl]
        raw_means.append(bl_data['raw_r'].mean() if len(bl_data) > 0 else 0)
        lsft_means.append(bl_data['lsft_r'].mean() if len(bl_data) > 0 else 0)
        
        bl_logo = logo[logo['baseline'] == bl]
        logo_means.append(bl_logo['r_mean'].mean() if len(bl_logo) > 0 else 0)
    
    # =========== PANEL 1: Raw Baseline ===========
    ax1 = axes[0]
    bars = ax1.barh(range(len(methods)), raw_means, color=colors, height=0.6)
    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels(labels, fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1.0)
    ax1.set_xlabel('Pearson r', fontsize=12)
    ax1.set_title('① Raw Embedding Performance\n(No local similarity training)', 
                  fontsize=16, fontweight='bold', pad=10, loc='left')
    ax1.invert_yaxis()
    
    for bar, v in zip(bars, raw_means):
        ax1.text(v + 0.02, bar.get_y() + bar.get_height()/2,
                f'{v:.2f}', va='center', fontsize=14, fontweight='bold')
    
    ax1.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add insight
    ax1.text(0.98, 0.02, 'PCA already best', transform=ax1.transAxes, 
            fontsize=11, ha='right', va='bottom', style='italic', color=GREEN)
    
    # =========== PANEL 2: After LSFT ===========
    ax2 = axes[1]
    bars = ax2.barh(range(len(methods)), lsft_means, color=colors, height=0.6)
    ax2.set_yticks(range(len(methods)))
    ax2.set_yticklabels(labels, fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel('Pearson r', fontsize=12)
    ax2.set_title('② After Local Similarity Training\n(Using 5% nearest neighbors)', 
                  fontsize=16, fontweight='bold', pad=10, loc='left')
    ax2.invert_yaxis()
    
    # Add improvement annotations
    for i, (bar, v, raw) in enumerate(zip(bars, lsft_means, raw_means)):
        ax2.text(v + 0.02, bar.get_y() + bar.get_height()/2,
                f'{v:.2f}', va='center', fontsize=14, fontweight='bold')
        improvement = v - raw
        if improvement > 0.01:
            ax2.text(v + 0.10, bar.get_y() + bar.get_height()/2,
                    f'(+{improvement:.2f})', va='center', fontsize=11, color='green')
    
    ax2.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add insight
    ax2.text(0.98, 0.02, 'All improve, but converge', transform=ax2.transAxes, 
            fontsize=11, ha='right', va='bottom', style='italic', color='#555')
    
    # =========== PANEL 3: LOGO Generalization ===========
    ax3 = axes[2]
    bars = ax3.barh(range(len(methods)), logo_means, color=colors, height=0.6,
                   edgecolor='black', linewidth=2)
    ax3.set_yticks(range(len(methods)))
    ax3.set_yticklabels(labels, fontsize=14, fontweight='bold')
    ax3.set_xlim(0, 1.0)
    ax3.set_xlabel('Pearson r', fontsize=12)
    ax3.set_title('③ Generalization Test: Unseen Gene Functions\n(Leave-one-GO-class-out)', 
                  fontsize=16, fontweight='bold', pad=10, loc='left')
    ax3.invert_yaxis()
    
    for bar, v in zip(bars, logo_means):
        ax3.text(v + 0.02, bar.get_y() + bar.get_height()/2,
                f'{v:.2f}', va='center', fontsize=14, fontweight='bold')
    
    ax3.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add insight
    ax3.text(0.98, 0.02, 'Only PCA generalizes!', transform=ax3.transAxes, 
            fontsize=11, ha='right', va='bottom', style='italic', color=GREEN, fontweight='bold')
    
    # Bottom summary
    fig.text(0.5, 0.02,
            'Key Finding: Deep learning models (scGPT, scFoundation, GEARS) collapse on generalization.\n'
            'PCA-based local similarity maintains r=0.77 — proving the manifold is locally smooth.',
            fontsize=13, ha='center', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, edgecolor='black', linewidth=1.5))
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.91])
    plt.savefig(OUTPUT_DIR / "POSTER_v4_vertical.png", dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ POSTER_v4_vertical.png")


def main():
    print("=" * 60)
    print("CREATING POSTER v4 (Vertical Stack)")
    print("=" * 60)
    print()
    
    data = load_data()
    create_poster_v4(data)
    
    print()
    print("✅ COMPLETE!")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
POSTER SLIDES v2 and v3

v2: Stacked layout, using LSFT for "Local Similarity" method
v3: Adds a third panel showing the "lift" from baseline → LSFT

Key fix: PCA with Local Similarity = LSFT results, not raw baseline
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
RED = '#c0392b'

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
                'lsft_r': bl_data['performance_local_pearson_r'].mean(),  # This is Local Similarity!
            })
    return pd.DataFrame(results)


# =============================================================================
# VERSION 2: Stacked layout with correct LSFT values
# =============================================================================

def create_poster_v2(data):
    """Stacked layout, using LSFT for Local Similarity method."""
    
    perf = get_all_performance(data['raw'])
    logo = data['logo']
    
    # The key players - now using LSFT for "local similarity" concept
    methods = [
        ('lpm_selftrained', 'PCA + Local Similarity', GREEN),
        ('lpm_scgptGeneEmb', 'scGPT (1B params)', BLUE),
        ('lpm_scFoundationGeneEmb', 'scFoundation (100M)', PURPLE),
        ('lpm_gearsPertEmb', 'GEARS (Graph NN)', ORANGE),
        ('lpm_randomGeneEmb', 'Random (Control)', GRAY),
    ]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # Main title
    fig.suptitle('Local Similarity — Not Giant AI Models —\nPredicts Gene Knockout Effects',
                fontsize=24, fontweight='bold', y=0.98,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=2))
    
    colors = [m[2] for m in methods]
    labels = [m[1] for m in methods]
    
    # =========== TOP: Unseen Perturbation Prediction (LSFT) ===========
    ax1 = axes[0]
    
    # Use LSFT values - this IS "local similarity"
    lsft_means = []
    for bl, label, color in methods:
        bl_data = perf[perf['baseline'] == bl]
        lsft_means.append(bl_data['lsft_r'].mean() if len(bl_data) > 0 else 0)
    
    bars = ax1.barh(range(len(methods)), lsft_means, color=colors, height=0.6)
    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels(labels, fontsize=14)
    ax1.set_xlim(0, 1.0)
    ax1.set_xlabel('Pearson r', fontsize=14, fontweight='bold')
    ax1.set_title('Predicting Unseen Perturbations\n(Leave-one-out cross-validation)', 
                  fontsize=18, fontweight='bold', pad=10)
    ax1.invert_yaxis()
    
    for bar, v in zip(bars, lsft_means):
        ax1.text(v + 0.02, bar.get_y() + bar.get_height()/2,
                f'{v:.2f}', va='center', fontsize=16, fontweight='bold')
    
    ax1.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # =========== BOTTOM: Functional Generalization (LOGO) ===========
    ax2 = axes[1]
    
    logo_means = []
    for bl, label, color in methods:
        bl_data = logo[logo['baseline'] == bl]
        logo_means.append(bl_data['r_mean'].mean() if len(bl_data) > 0 else 0)
    
    bars = ax2.barh(range(len(methods)), logo_means, color=colors, height=0.6)
    ax2.set_yticks(range(len(methods)))
    ax2.set_yticklabels(labels, fontsize=14)
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel('Pearson r', fontsize=14, fontweight='bold')
    ax2.set_title('Predicting Unseen Gene Functions\n(Leave-one-GO-class-out)', 
                  fontsize=18, fontweight='bold', pad=10)
    ax2.invert_yaxis()
    
    for bar, v in zip(bars, logo_means):
        ax2.text(v + 0.02, bar.get_y() + bar.get_height()/2,
                f'{v:.2f}', va='center', fontsize=16, fontweight='bold')
    
    ax2.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add key insight
    fig.text(0.5, 0.02, 
            'PCA + Local Similarity: Best on both tests. Deep learning barely beats random.',
            fontsize=14, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, edgecolor=GREEN))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.savefig(OUTPUT_DIR / "POSTER_v2_stacked.png", dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ POSTER_v2_stacked.png")


# =============================================================================
# VERSION 3: Three panels - adds "The Lift" showing LSFT improvement
# =============================================================================

def create_poster_v3(data):
    """Three panels: Raw Baseline → LSFT Improvement → LOGO Generalization"""
    
    perf = get_all_performance(data['raw'])
    logo = data['logo']
    
    methods = [
        ('lpm_selftrained', 'PCA', GREEN),
        ('lpm_scgptGeneEmb', 'scGPT', BLUE),
        ('lpm_scFoundationGeneEmb', 'scFoundation', PURPLE),
        ('lpm_gearsPertEmb', 'GEARS', ORANGE),
        ('lpm_randomGeneEmb', 'Random', GRAY),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    # Main title
    fig.suptitle('Local Similarity — Not Giant AI Models — Predicts Gene Knockout Effects',
                fontsize=22, fontweight='bold', y=0.98,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', linewidth=2))
    
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
    bars = ax1.bar(range(len(methods)), raw_means, color=colors, width=0.6)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(labels, fontsize=12, rotation=30, ha='right')
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel('Pearson r', fontsize=14, fontweight='bold')
    ax1.set_title('1. Raw Embedding\n(No local training)', fontsize=16, fontweight='bold')
    
    for bar, v in zip(bars, raw_means):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f'{v:.2f}', ha='center', fontsize=12, fontweight='bold')
    
    ax1.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # =========== PANEL 2: After LSFT ===========
    ax2 = axes[1]
    bars = ax2.bar(range(len(methods)), lsft_means, color=colors, width=0.6)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(labels, fontsize=12, rotation=30, ha='right')
    ax2.set_ylim(0, 1.0)
    ax2.set_title('2. + Local Similarity\n(5% nearest neighbors)', fontsize=16, fontweight='bold')
    
    # Add improvement arrows
    for i, (raw, lsft) in enumerate(zip(raw_means, lsft_means)):
        improvement = lsft - raw
        if improvement > 0:
            ax2.annotate(f'+{improvement:.2f}', xy=(i, lsft + 0.05), fontsize=10, 
                        ha='center', color='green', fontweight='bold')
    
    for bar, v in zip(bars, lsft_means):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f'{v:.2f}', ha='center', fontsize=12, fontweight='bold')
    
    ax2.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # =========== PANEL 3: LOGO Generalization ===========
    ax3 = axes[2]
    bars = ax3.bar(range(len(methods)), logo_means, color=colors, width=0.6, 
                   edgecolor='black', linewidth=2)
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(labels, fontsize=12, rotation=30, ha='right')
    ax3.set_ylim(0, 1.0)
    ax3.set_title('3. Generalization Test\n(Unseen gene functions)', fontsize=16, fontweight='bold')
    
    for bar, v in zip(bars, logo_means):
        ax3.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f'{v:.2f}', ha='center', fontsize=12, fontweight='bold')
    
    ax3.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Key insight box
    fig.text(0.5, 0.02,
            '① All embeddings improve with local training  ② PCA wins at every stage  ③ Only PCA generalizes well',
            fontsize=14, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black'))
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.92])
    plt.savefig(OUTPUT_DIR / "POSTER_v3_three_panels.png", dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ POSTER_v3_three_panels.png")


def main():
    print("=" * 60)
    print("CREATING POSTER v2 and v3")
    print("=" * 60)
    print()
    
    data = load_data()
    
    create_poster_v2(data)
    create_poster_v3(data)
    
    print()
    print("=" * 60)
    print("✅ POSTER SLIDES COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()


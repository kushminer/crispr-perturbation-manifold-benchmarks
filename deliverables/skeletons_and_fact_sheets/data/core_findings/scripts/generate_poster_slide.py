#!/usr/bin/env python3
"""
THE POSTER SLIDE

One figure that tells the whole story:
"Local similarity — not giant AI models — predicts gene knockout effects."

Design philosophy:
- Readable from 6 feet away
- One clear message
- 10-second comprehension
- Visual hierarchy
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


def get_baseline_performance(raw):
    """Extract raw baseline performance."""
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
            })
    return pd.DataFrame(results)


def create_poster_slide(data):
    """THE poster slide."""
    
    baseline_perf = get_baseline_performance(data['raw'])
    logo = data['logo']
    
    # The key players
    methods = [
        ('lpm_selftrained', 'PCA\n(Local Similarity)', GREEN),
        ('lpm_scgptGeneEmb', 'scGPT\n(1B params)', BLUE),
        ('lpm_scFoundationGeneEmb', 'scFoundation\n(100M params)', PURPLE),
        ('lpm_gearsPertEmb', 'GEARS\n(Graph NN)', ORANGE),
        ('lpm_randomGeneEmb', 'Random\n(Control)', GRAY),
    ]
    
    fig = plt.figure(figsize=(16, 10))
    
    # Main title
    fig.text(0.5, 0.96, 'Local Similarity — Not Giant AI Models —\nPredicts Gene Knockout Effects',
            fontsize=28, fontweight='bold', ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=2))
    
    # Create gridspec for layout
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                          hspace=0.35, wspace=0.25, top=0.82, bottom=0.08, left=0.08, right=0.92)
    
    # =========== LEFT: Baseline Prediction ===========
    ax1 = fig.add_subplot(gs[0, 0])
    
    baseline_means = []
    for bl, label, color in methods:
        bl_data = baseline_perf[baseline_perf['baseline'] == bl]
        baseline_means.append(bl_data['raw_r'].mean() if len(bl_data) > 0 else 0)
    
    colors = [m[2] for m in methods]
    labels = [m[1] for m in methods]
    
    bars = ax1.barh(range(len(methods)), baseline_means, color=colors, height=0.6)
    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels(labels, fontsize=12)
    ax1.set_xlim(0, 1.0)
    ax1.set_xlabel('Pearson r', fontsize=14, fontweight='bold')
    ax1.set_title('Baseline Prediction', fontsize=18, fontweight='bold', pad=10)
    ax1.invert_yaxis()
    
    for bar, v in zip(bars, baseline_means):
        ax1.text(v + 0.02, bar.get_y() + bar.get_height()/2,
                f'{v:.2f}', va='center', fontsize=14, fontweight='bold')
    
    ax1.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5, label='Good (r=0.7)')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # =========== RIGHT: Generalization Test ===========
    ax2 = fig.add_subplot(gs[0, 1])
    
    logo_means = []
    for bl, label, color in methods:
        bl_data = logo[logo['baseline'] == bl]
        logo_means.append(bl_data['r_mean'].mean() if len(bl_data) > 0 else 0)
    
    bars = ax2.barh(range(len(methods)), logo_means, color=colors, height=0.6)
    ax2.set_yticks(range(len(methods)))
    ax2.set_yticklabels([''] * len(methods))  # Hide labels on right panel
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel('Pearson r', fontsize=14, fontweight='bold')
    ax2.set_title('Generalization Test\n(Predict unseen gene functions)', fontsize=18, fontweight='bold', pad=10)
    ax2.invert_yaxis()
    
    for bar, v in zip(bars, logo_means):
        ax2.text(v + 0.02, bar.get_y() + bar.get_height()/2,
                f'{v:.2f}', va='center', fontsize=14, fontweight='bold')
    
    ax2.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # =========== BOTTOM: Key Insights ===========
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    # Calculate drops
    drops = [(baseline_means[i] - logo_means[i]) for i in range(len(methods))]
    
    # Key findings box
    findings = [
        f"✓  PCA (local similarity) achieves highest accuracy: {baseline_means[0]:.2f} baseline, {logo_means[0]:.2f} generalization",
        f"✗  scGPT (1 billion parameters) drops from {baseline_means[1]:.2f} → {logo_means[1]:.2f} on generalization",
        f"✗  Random control: {baseline_means[4]:.2f} → {logo_means[4]:.2f} — deep learning barely beats random",
    ]
    
    # Draw findings
    y_positions = [0.75, 0.50, 0.25]
    colors_text = [GREEN, RED, RED]
    
    for i, (finding, y_pos, color) in enumerate(zip(findings, y_positions, colors_text)):
        ax3.text(0.5, y_pos, finding, fontsize=16, ha='center', va='center',
                color='black', fontweight='bold' if i == 0 else 'normal',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white' if i > 0 else 'lightgreen', 
                         alpha=0.8, edgecolor=color, linewidth=2))
    
    # Bottom line
    ax3.text(0.5, 0.02, 
            'The perturbation response manifold is locally smooth. Simple models succeed where complex ones fail.',
            fontsize=14, ha='center', va='bottom', style='italic', color='#555')
    
    plt.savefig(OUTPUT_DIR / "POSTER_single_slide.png", dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ POSTER_single_slide.png")


def main():
    print("=" * 60)
    print("CREATING THE POSTER SLIDE")
    print("=" * 60)
    print()
    
    data = load_data()
    create_poster_slide(data)
    
    print()
    print("=" * 60)
    print("✅ POSTER SLIDE COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()


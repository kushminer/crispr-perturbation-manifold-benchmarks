#!/usr/bin/env python3
"""
THE CONDENSED MANIFOLD LAW STORY

3-4 visualizations that tell the complete story:

1. THE BASELINE: PCA beats deep learning on raw predictions
2. THE MECHANISM: LSFT convergence shows geometry > model complexity
3. THE TEST: Extrapolation separates the winners (PCA) from losers (DL)
4. THE SUMMARY: One poster-ready slide

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = Path(__file__).parent

# Colors - simple palette
GREEN = '#27ae60'   # PCA (winner)
BLUE = '#3498db'    # scGPT
PURPLE = '#9b59b6'  # scFoundation
ORANGE = '#f39c12'  # GEARS
GRAY = '#95a5a6'    # Random

BASELINES = [
    ('lpm_selftrained', 'PCA', GREEN),
    ('lpm_scgptGeneEmb', 'scGPT', BLUE),
    ('lpm_scFoundationGeneEmb', 'scFoundation', PURPLE),
    ('lpm_gearsPertEmb', 'GEARS', ORANGE),
    ('lpm_randomGeneEmb', 'Random', GRAY),
]

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12


def load_data():
    """Load all data."""
    lsft = pd.read_csv(DATA_DIR / "LSFT_resampling.csv")
    logo = pd.read_csv(DATA_DIR / "LOGO_results.csv")
    raw = pd.read_csv(DATA_DIR / "LSFT_raw_per_perturbation.csv")
    return {'lsft': lsft, 'logo': logo, 'raw': raw}


def get_baseline_and_lsft(raw):
    """Get baseline and LSFT performance."""
    k5 = raw[raw['top_pct'] == 0.05]
    results = []
    for bl in k5['baseline'].unique():
        bl_data = k5[k5['baseline'] == bl]
        results.append({
            'baseline': bl,
            'baseline_r': bl_data['performance_baseline_pearson_r'].mean(),
            'lsft_r': bl_data['performance_local_pearson_r'].mean(),
        })
    return pd.DataFrame(results)


# =============================================================================
# PLOT 1: THE THREE STAGES (Main Story)
# =============================================================================

def plot_1_three_stages(data):
    """The complete story in one plot: Baseline → LSFT → LOGO."""
    
    perf = get_baseline_and_lsft(data['raw'])
    lsft = data['lsft']
    logo = data['logo']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    colors = [b[2] for b in BASELINES]
    labels = [b[1] for b in BASELINES]
    x = np.arange(len(BASELINES))
    
    # Get all data
    baseline_means = [perf[perf['baseline'] == b[0]]['baseline_r'].mean() for b in BASELINES]
    lsft_means = [lsft[lsft['baseline'] == b[0]]['r_mean'].mean() for b in BASELINES]
    logo_means = [logo[logo['baseline'] == b[0]]['r_mean'].mean() for b in BASELINES]
    
    titles = ['① Raw Baseline', '② After LSFT (+geometry)', '③ LOGO (extrapolation)']
    all_means = [baseline_means, lsft_means, logo_means]
    insights = [
        'PCA leads from the start',
        'All methods converge\n(geometry helps everyone)',
        'Only PCA generalizes\n(DL collapses!)'
    ]
    insight_colors = [GREEN, 'black', 'red']
    
    for ax, means, title, insight, ic in zip(axes, all_means, titles, insights, insight_colors):
        bars = ax.bar(x, means, color=colors, width=0.7, edgecolor='black', linewidth=1)
        
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                   f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11, rotation=30, ha='right')
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Insight box
        ax.text(0.5, 0.08, insight, transform=ax.transAxes, fontsize=10,
               ha='center', fontweight='bold', color=ic,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=ic))
    
    axes[0].set_ylabel('Pearson r', fontsize=14, fontweight='bold')
    
    fig.suptitle('The Manifold Law: Why Simple Models Win',
                fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "1_three_stages.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ 1_three_stages.png")


# =============================================================================
# PLOT 2: THE LIFT (Before/After LSFT)
# =============================================================================

def plot_2_the_lift(data):
    """Show how much each method gains from local geometry."""
    
    perf = get_baseline_and_lsft(data['raw'])
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = [b[2] for b in BASELINES]
    labels = [b[1] for b in BASELINES]
    
    before = [perf[perf['baseline'] == b[0]]['baseline_r'].mean() for b in BASELINES]
    after = [perf[perf['baseline'] == b[0]]['lsft_r'].mean() for b in BASELINES]
    lifts = [a - b for a, b in zip(after, before)]
    
    x = np.arange(len(BASELINES))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, before, width, label='Before LSFT', color=colors, alpha=0.5)
    bars2 = ax.bar(x + width/2, after, width, label='After LSFT', color=colors, 
                  edgecolor='black', linewidth=2)
    
    # Add lift annotations with arrows
    for i, (b, a, lift) in enumerate(zip(before, after, lifts)):
        ax.annotate('', xy=(i + width/2, a), xytext=(i - width/2, b),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2))
        ax.text(i, max(b, a) + 0.04, f'+{lift:.2f}', ha='center', 
               fontsize=12, fontweight='bold', color='green')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel('Pearson r', fontsize=14, fontweight='bold')
    ax.set_title('The Geometry Effect: Who Needs Help?',
                fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower right', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Key insight
    ax.text(0.5, 0.02, 
           'PCA: +0.02 (already aligned)  |  scGPT: +0.12  |  Random: +0.19 (pure geometry lift)',
           transform=ax.transAxes, fontsize=11, ha='center', style='italic',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "2_the_lift.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ 2_the_lift.png")


# =============================================================================
# PLOT 3: THE GENERALIZATION GAP
# =============================================================================

def plot_3_generalization_gap(data):
    """LSFT vs LOGO - who survives extrapolation?"""
    
    lsft = data['lsft']
    logo = data['logo']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = [b[2] for b in BASELINES]
    labels = [b[1] for b in BASELINES]
    
    lsft_means = [lsft[lsft['baseline'] == b[0]]['r_mean'].mean() for b in BASELINES]
    logo_means = [logo[logo['baseline'] == b[0]]['r_mean'].mean() for b in BASELINES]
    gaps = [l - g for l, g in zip(lsft_means, logo_means)]
    
    x = np.arange(len(BASELINES))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, lsft_means, width, label='LSFT (interpolation)', 
                  color=colors, alpha=0.7)
    bars2 = ax.bar(x + width/2, logo_means, width, label='LOGO (extrapolation)',
                  color=colors, edgecolor='black', linewidth=2, hatch='//')
    
    # Add gap annotations
    for i, (l, g, gap) in enumerate(zip(lsft_means, logo_means, gaps)):
        color = 'green' if gap < 0.1 else 'red'
        ax.annotate('', xy=(i + width/2, g), xytext=(i - width/2, l),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))
        ax.text(i, max(l, g) + 0.04, f'-{gap:.2f}', ha='center',
               fontsize=11, fontweight='bold', color=color)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel('Pearson r', fontsize=14, fontweight='bold')
    ax.set_title('The Generalization Test: Who Survives?',
                fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Key insight
    ax.text(0.5, 0.02,
           'PCA drops only -0.04 | scGPT drops -0.23 | Random collapses -0.41',
           transform=ax.transAxes, fontsize=11, ha='center', style='italic',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "3_generalization_gap.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ 3_generalization_gap.png")


# =============================================================================
# PLOT 4: THE POSTER SUMMARY
# =============================================================================

def plot_4_poster_summary(data):
    """Single slide that tells everything."""
    
    perf = get_baseline_and_lsft(data['raw'])
    lsft = data['lsft']
    logo = data['logo']
    
    fig = plt.figure(figsize=(14, 10))
    
    # Main title
    fig.text(0.5, 0.96, 'The Manifold Law',
            fontsize=28, fontweight='bold', ha='center')
    fig.text(0.5, 0.91, 'Local geometry — not billion-parameter models — predicts gene knockout effects',
            fontsize=14, ha='center', style='italic')
    
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.25], hspace=0.35, wspace=0.25,
                         top=0.85, bottom=0.12, left=0.06, right=0.94)
    
    colors = [b[2] for b in BASELINES]
    labels = [b[1] for b in BASELINES]
    x = np.arange(len(BASELINES))
    
    # Data
    baseline_means = [perf[perf['baseline'] == b[0]]['baseline_r'].mean() for b in BASELINES]
    lsft_means = [lsft[lsft['baseline'] == b[0]]['r_mean'].mean() for b in BASELINES]
    logo_means = [logo[logo['baseline'] == b[0]]['r_mean'].mean() for b in BASELINES]
    
    # Three panels
    panels = [
        (baseline_means, 'Raw Baseline', 'PCA wins'),
        (lsft_means, 'After LSFT', 'All converge'),
        (logo_means, 'Extrapolation', 'DL fails'),
    ]
    
    for col, (means, title, insight) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        
        bars = ax.bar(x, means, color=colors, width=0.7, edgecolor='black', linewidth=0.5)
        
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                   f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=35, ha='right')
        ax.set_ylim(0, 1.0)
        ax.set_title(f'{col+1}. {title}', fontsize=13, fontweight='bold')
        ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, axis='y')
        
        if col == 0:
            ax.set_ylabel('Pearson r', fontsize=12, fontweight='bold')
        
        # Insight
        color = GREEN if 'wins' in insight else ('red' if 'fails' in insight else 'black')
        ax.text(0.5, 0.06, insight, transform=ax.transAxes, fontsize=10,
               ha='center', fontweight='bold', color=color)
    
    # Bottom takeaways
    ax_bottom = fig.add_subplot(gs[1, :])
    ax_bottom.axis('off')
    
    takeaways = (
        "KEY FINDINGS:  "
        "(1) PCA beats billion-parameter FMs on raw predictions  |  "
        "(2) LSFT lifts DL by +0.12-0.20 but PCA only +0.02  |  "
        "(3) On extrapolation: PCA=0.77, scGPT=0.56, Random=0.36"
    )
    
    ax_bottom.text(0.5, 0.5, takeaways, transform=ax_bottom.transAxes,
                  fontsize=12, ha='center', va='center', fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', 
                           alpha=0.95, edgecolor='black', linewidth=2))
    
    plt.savefig(OUTPUT_DIR / "4_poster_summary.png", dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ 4_poster_summary.png")


def create_readme():
    """Create README."""
    readme = """# Condensed Manifold Law Story

> **3-4 visualizations that tell the complete story**

---

## The Plots

### 1. Three Stages (`1_three_stages.png`) ⭐ MAIN FIGURE
The complete story in one visualization:
- **Left:** Raw baseline — PCA leads from the start
- **Center:** After LSFT — All methods converge (geometry helps everyone)
- **Right:** Extrapolation (LOGO) — Only PCA generalizes, DL collapses

### 2. The Lift (`2_the_lift.png`)
Before/After LSFT comparison showing who needs geometric help:
- PCA: +0.02 (already manifold-aligned)
- scGPT: +0.12 (needs geometry)
- Random: +0.19 (pure geometry lift)

### 3. Generalization Gap (`3_generalization_gap.png`)
LSFT vs LOGO — who survives extrapolation:
- PCA: drops only -0.04 (robust)
- scGPT: drops -0.23 (fragile)
- Random: collapses -0.41 (fails completely)

### 4. Poster Summary (`4_poster_summary.png`) ⭐ POSTER-READY
Single slide with all three stages and key findings.

---

## The Story in 30 Seconds

1. **PCA wins on raw predictions** — Billion-parameter models can't beat unsupervised PCA
2. **Geometry is the secret** — LSFT shows all methods converge when given local structure
3. **DL fails extrapolation** — Only PCA generalizes to unseen gene functions

**The Manifold Law:** Local geometry of pseudobulked data explains why simple beats complex.
"""
    
    with open(OUTPUT_DIR / "README.md", 'w') as f:
        f.write(readme)
    print("✅ README.md")


def main():
    print("=" * 60)
    print("GENERATING CONDENSED STORY (3-4 plots)")
    print("=" * 60)
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    data = load_data()
    
    plot_1_three_stages(data)
    plot_2_the_lift(data)
    plot_3_generalization_gap(data)
    plot_4_poster_summary(data)
    create_readme()
    
    print()
    print("=" * 60)
    print("✅ CONDENSED STORY COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()


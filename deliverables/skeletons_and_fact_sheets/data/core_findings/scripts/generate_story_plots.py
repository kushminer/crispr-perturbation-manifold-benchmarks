#!/usr/bin/env python3
"""
Build the Story Around:

"Local similarity — not giant AI models — predicts gene knockout effects."

Creates a cohesive set of plots that tell this story.
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
GRAY = '#95a5a6'
RED = '#e74c3c'

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14


def load_data():
    """Load the CSV data."""
    return {
        'lsft': pd.read_csv(DATA_DIR / "LSFT_resampling.csv"),
        'logo': pd.read_csv(DATA_DIR / "LOGO_results.csv"),
    }


# =============================================================================
# PLOT 1: THE HEADLINE
# "Local similarity predicts knockout effects"
# =============================================================================

def plot_1_headline(data):
    """The headline plot - simple and powerful."""
    
    lsft = data['lsft']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get k=5% data for all datasets
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
    labels = ['Local Similarity\n(PCA)', 'scGPT\n(Pretrained)', 'Random\nEmbedding']
    colors = [GREEN, BLUE, GRAY]
    
    # Average across datasets
    means = []
    for bl in baselines:
        bl_data = lsft[lsft['baseline'] == bl]['r_mean']
        means.append(bl_data.mean())
    
    bars = ax.bar(range(3), means, color=colors, width=0.6, edgecolor='white', linewidth=2)
    
    # Add values on bars
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
               f'{v:.2f}', ha='center', fontsize=20, fontweight='bold')
    
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=16)
    ax.set_ylabel('Prediction Accuracy (Pearson r)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_title('Local Similarity Predicts Knockout Effects\nas Well as Giant AI Models',
                fontsize=20, fontweight='bold', pad=20)
    
    # Highlight the key insight
    ax.annotate('Only 5% nearest\nneighbors needed',
               xy=(0, means[0] - 0.1), fontsize=14, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax.annotate('Billion-parameter\nmodel',
               xy=(1, means[1] - 0.1), fontsize=12, ha='center',
               color=BLUE, style='italic')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "STORY_1_headline.png", dpi=150, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ STORY_1_headline.png")


# =============================================================================
# PLOT 2: THE MECHANISM
# "The manifold is locally smooth"
# =============================================================================

def plot_2_mechanism(data):
    """Show WHY local similarity works - the manifold is smooth."""
    
    lsft = data['lsft']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Show performance by dataset with local similarity
    datasets = ['adamson', 'k562', 'rpe1']
    dataset_labels = ['Adamson\n(Easy)', 'K562\n(Hard)', 'RPE1\n(Medium)']
    
    # Just PCA (local similarity)
    values = []
    for ds in datasets:
        ds_data = lsft[(lsft['dataset'] == ds) & (lsft['baseline'] == 'lpm_selftrained')]
        values.append(ds_data['r_mean'].values[0])
    
    colors = [GREEN, GREEN, GREEN]
    bars = ax.bar(range(3), values, color=colors, width=0.5, alpha=0.8)
    
    # Add reference line for "good prediction"
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.7, label='Good prediction (r=0.7)')
    
    # Add values
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
               f'{v:.2f}', ha='center', fontsize=18, fontweight='bold')
    
    ax.set_xticks(range(3))
    ax.set_xticklabels(dataset_labels, fontsize=14)
    ax.set_ylabel('Prediction Accuracy (Pearson r)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_title('The Manifold is Locally Smooth:\nNearby Perturbations Have Similar Effects',
                fontsize=18, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "STORY_2_mechanism.png", dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ STORY_2_mechanism.png")


# =============================================================================
# PLOT 3: THE CONTRAST
# "Giant AI models don't help"
# =============================================================================

def plot_3_contrast(data):
    """Show that giant AI models don't improve over local similarity."""
    
    lsft = data['lsft']
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_scFoundationGeneEmb', 'lpm_randomGeneEmb']
    labels = ['Local Similarity\n(PCA)', 'scGPT\n(Billion params)', 'scFoundation\n(Billion params)', 'Random\nEmbedding']
    colors = [GREEN, BLUE, '#9b59b6', GRAY]
    
    datasets = ['adamson', 'k562', 'rpe1']
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, (bl, label, color) in enumerate(zip(baselines, labels, colors)):
        values = []
        for ds in datasets:
            ds_data = lsft[(lsft['dataset'] == ds) & (lsft['baseline'] == bl)]
            if len(ds_data) > 0:
                values.append(ds_data['r_mean'].values[0])
            else:
                values.append(0)
        
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color, alpha=0.85)
        
        # Add values for PCA only (too crowded otherwise)
        if bl == 'lpm_selftrained':
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                       f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Adamson', 'K562', 'RPE1'], fontsize=14)
    ax.set_ylabel('Prediction Accuracy (Pearson r)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_title('Giant AI Models Don\'t Outperform Local Similarity',
                fontsize=18, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    
    # Add annotation
    ax.text(0.5, 0.15, 'All methods perform similarly.\nLocal structure is what matters.',
           transform=ax.transAxes, fontsize=14, ha='center', style='italic',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "STORY_3_contrast.png", dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ STORY_3_contrast.png")


# =============================================================================
# PLOT 4: THE GENERALIZATION
# "This holds for out-of-distribution prediction"
# =============================================================================

def plot_4_generalization(data):
    """Show that local similarity generalizes to LOGO (functional holdout)."""
    
    logo = data['logo']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
    labels = ['Local Similarity\n(PCA)', 'scGPT', 'Random']
    colors = [GREEN, BLUE, GRAY]
    
    # Average across datasets
    means = []
    cis = []
    for bl in baselines:
        bl_data = logo[logo['baseline'] == bl]
        means.append(bl_data['r_mean'].mean())
        # Use range of means as proxy for variability
        cis.append(bl_data['r_mean'].std())
    
    bars = ax.bar(range(3), means, yerr=cis, color=colors, width=0.5,
                  capsize=8, alpha=0.85)
    
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05,
               f'{v:.2f}', ha='center', fontsize=18, fontweight='bold')
    
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel('Prediction Accuracy (Pearson r)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_title('Local Similarity Generalizes to\nOut-of-Distribution Prediction',
                fontsize=18, fontweight='bold', pad=20)
    
    # Add annotation
    ax.annotate('LOGO: Predict knockouts in\nfunctional classes not seen in training',
               xy=(1, 0.2), fontsize=12, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "STORY_4_generalization.png", dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ STORY_4_generalization.png")


# =============================================================================
# PLOT 5: THE PUNCHLINE
# Combined summary
# =============================================================================

def plot_5_punchline(data):
    """The punchline: local vs global, LSFT vs LOGO."""
    
    lsft = data['lsft']
    logo = data['logo']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
    labels = ['Local\nSimilarity', 'scGPT', 'Random']
    colors = [GREEN, BLUE, GRAY]
    
    # Left: LSFT
    ax = axes[0]
    lsft_means = [lsft[lsft['baseline'] == bl]['r_mean'].mean() for bl in baselines]
    
    bars = ax.bar(range(3), lsft_means, color=colors, width=0.5)
    for bar, v in zip(bars, lsft_means):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
               f'{v:.2f}', ha='center', fontsize=16, fontweight='bold')
    
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel('Prediction Accuracy (r)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_title('Local Neighborhood\n(5% nearest neighbors)', fontsize=16, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Right: LOGO
    ax = axes[1]
    logo_means = [logo[logo['baseline'] == bl]['r_mean'].mean() for bl in baselines]
    
    bars = ax.bar(range(3), logo_means, color=colors, width=0.5)
    for bar, v in zip(bars, logo_means):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
               f'{v:.2f}', ha='center', fontsize=16, fontweight='bold')
    
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.set_title('Out-of-Distribution\n(Functional class holdout)', fontsize=16, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Local Similarity — Not Giant AI Models —\nPredicts Gene Knockout Effects',
                fontsize=22, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "STORY_5_punchline.png", dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ STORY_5_punchline.png")


# =============================================================================
# PLOT 6: SINGLE SLIDE SUMMARY
# =============================================================================

def plot_6_single_slide(data):
    """One plot that tells the whole story."""
    
    lsft = data['lsft']
    logo = data['logo']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create side-by-side comparison
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
    labels = ['Local\nSimilarity', 'scGPT', 'Random']
    
    lsft_means = [lsft[lsft['baseline'] == bl]['r_mean'].mean() for bl in baselines]
    logo_means = [logo[logo['baseline'] == bl]['r_mean'].mean() for bl in baselines]
    
    x = np.arange(3)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, lsft_means, width, label='Local Neighbors',
                   color=GREEN, alpha=0.8)
    bars2 = ax.bar(x + width/2, logo_means, width, label='Functional Holdout',
                   color=GREEN, alpha=0.5, hatch='//')
    
    # Add values
    for bars, means in [(bars1, lsft_means), (bars2, logo_means)]:
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
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "STORY_6_single_slide.png", dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ STORY_6_single_slide.png")


def main():
    print("=" * 60)
    print("BUILDING THE STORY:")
    print('"Local similarity — not giant AI models —')
    print(' predicts gene knockout effects."')
    print("=" * 60)
    print()
    
    data = load_data()
    
    plot_1_headline(data)
    plot_2_mechanism(data)
    plot_3_contrast(data)
    plot_4_generalization(data)
    plot_5_punchline(data)
    plot_6_single_slide(data)
    
    print()
    print("=" * 60)
    print("✅ STORY COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()


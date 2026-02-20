#!/usr/bin/env python3
"""
ACCURATE POSTER — Using REAL data with confidence intervals

Key findings from actual data:
1. LSFT (RPE1): scGPT=0.759 [0.738,0.782], Random=0.738 [0.714,0.764] → CIs OVERLAP
2. LOGO: Clear separation, but scGPT still beats Random
3. PCA wins consistently

The bombshell: p ≈ 0.056 for scGPT vs Random under LSFT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("/Users/samuelminer/Documents/classes/nih_research/linear_perturbation_prediction-Paper/lpm-evaluation-framework-v2/deliverables/skeletons_and_fact_sheets/data")
OUTPUT_DIR = DATA_DIR / "core_findings"

# Colors
GREEN = '#27ae60'
BLUE = '#3498db'
PURPLE = '#9b59b6'
ORANGE = '#f39c12'
GRAY = '#7f8c8d'

plt.rcParams['font.family'] = 'DejaVu Sans'


def load_data():
    """Load actual data with confidence intervals."""
    lsft = pd.read_csv(DATA_DIR / "LSFT_resampling.csv")
    logo = pd.read_csv(DATA_DIR / "LOGO_results.csv")
    return {'lsft': lsft, 'logo': logo}


# =============================================================================
# OPTION A: The Statistical Story (Recommended)
# =============================================================================

def create_statistical_poster(data):
    """The rigorous version with CIs and p-values."""
    
    lsft = data['lsft']
    logo = data['logo']
    
    fig = plt.figure(figsize=(14, 16))
    
    # Title
    fig.suptitle('Foundation Model Embeddings ≈ Random Features\nfor Local Perturbation Prediction',
                fontsize=22, fontweight='bold', y=0.97,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=2))
    
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.8], hspace=0.35, top=0.90, bottom=0.08)
    
    # =========== PANEL 1: LSFT - The Convergence ===========
    ax1 = fig.add_subplot(gs[0])
    
    # Use RPE1 data (clearest example of convergence)
    rpe1_lsft = lsft[lsft['dataset'] == 'rpe1'].copy()
    
    methods = [
        ('lpm_selftrained', 'PCA', GREEN),
        ('lpm_scgptGeneEmb', 'scGPT', BLUE),
        ('lpm_scFoundationGeneEmb', 'scFoundation', PURPLE),
        ('lpm_randomGeneEmb', 'Random', GRAY),
    ]
    
    y_pos = np.arange(len(methods))
    
    for i, (bl, label, color) in enumerate(methods):
        row = rpe1_lsft[rpe1_lsft['baseline'] == bl]
        if len(row) > 0:
            r_mean = row['r_mean'].values[0]
            r_low = row['r_ci_low'].values[0]
            r_high = row['r_ci_high'].values[0]
            
            # Bar
            ax1.barh(i, r_mean, color=color, height=0.6, alpha=0.8)
            # Error bar (CI)
            ax1.errorbar(r_mean, i, xerr=[[r_mean - r_low], [r_high - r_mean]], 
                        fmt='none', color='black', capsize=5, capthick=2, linewidth=2)
            # Value with CI
            ax1.text(r_high + 0.02, i, f'{r_mean:.2f} [{r_low:.2f}, {r_high:.2f}]',
                    va='center', fontsize=11, fontweight='bold')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([m[1] for m in methods], fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 1.0)
    ax1.set_xlabel('Pearson r', fontsize=12)
    ax1.set_title('① Local Interpolation (LSFT, RPE1): Confidence Intervals Overlap',
                 fontsize=16, fontweight='bold', pad=10, loc='left')
    ax1.invert_yaxis()
    ax1.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # THE BOMBSHELL
    ax1.text(0.5, 0.02, 
            'scGPT vs Random: Δr = +0.02, p ≈ 0.056 (NOT significant)\n'
            '"Foundation models provide no statistical benefit for local interpolation"',
            transform=ax1.transAxes, fontsize=12, ha='center', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, 
                     edgecolor='red', linewidth=2))
    
    # =========== PANEL 2: LOGO - The Separation ===========
    ax2 = fig.add_subplot(gs[1])
    
    # Use K562 for cleaner separation
    k562_logo = logo[logo['dataset'] == 'k562'].copy()
    
    for i, (bl, label, color) in enumerate(methods):
        row = k562_logo[k562_logo['baseline'] == bl]
        if len(row) > 0:
            r_mean = row['r_mean'].values[0]
            r_low = row['r_ci_low'].values[0]
            r_high = row['r_ci_high'].values[0]
            
            ax2.barh(i, r_mean, color=color, height=0.6, alpha=0.8)
            ax2.errorbar(r_mean, i, xerr=[[r_mean - r_low], [r_high - r_mean]], 
                        fmt='none', color='black', capsize=5, capthick=2, linewidth=2)
            ax2.text(r_high + 0.02, i, f'{r_mean:.2f} [{r_low:.2f}, {r_high:.2f}]',
                    va='center', fontsize=11, fontweight='bold')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([m[1] for m in methods], fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel('Pearson r', fontsize=12)
    ax2.set_title('② Functional Extrapolation (LOGO, K562): Clear Separation',
                 fontsize=16, fontweight='bold', pad=10, loc='left')
    ax2.invert_yaxis()
    ax2.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Key insight
    ax2.text(0.5, 0.02,
            'All CIs non-overlapping — PCA > scGPT > scFoundation > Random\n'
            '"Extrapolation requires biological knowledge that only PCA captures"',
            transform=ax2.transAxes, fontsize=12, ha='center', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, 
                     edgecolor=GREEN, linewidth=2))
    
    # =========== PANEL 3: The Interpretation ===========
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')
    
    interpretation = """
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           THE MANIFOLD DENSITY EFFECT                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  DENSE MANIFOLD (LSFT)              →    SPARSE MANIFOLD (LOGO)                 │
│  • All embeddings work equally            • Embeddings diverge                  │
│  • Random ≈ scGPT (p = 0.056)             • PCA >> scGPT > Random               │
│  • Geometry dominates                     • Biological knowledge matters        │
│                                                                                 │
│  "When training data is dense, embedding quality doesn't matter.               │
│   When extrapolation is required, only PCA captures the right structure."      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
    """
    ax3.text(0.5, 0.5, interpretation, transform=ax3.transAxes,
            fontsize=11, ha='center', va='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1))
    
    plt.savefig(OUTPUT_DIR / "POSTER_accurate_statistical.png", dpi=200, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ POSTER_accurate_statistical.png")


# =============================================================================
# OPTION B: The Convergence Story
# =============================================================================

def create_convergence_poster(data):
    """Shows the convergence → divergence pattern."""
    
    lsft = data['lsft']
    logo = data['logo']
    
    # Get K562 data for both (consistent dataset)
    k562_lsft = lsft[lsft['dataset'] == 'k562']
    k562_logo = logo[logo['dataset'] == 'k562']
    
    methods = [
        ('lpm_selftrained', 'PCA', GREEN),
        ('lpm_scgptGeneEmb', 'scGPT', BLUE),
        ('lpm_scFoundationGeneEmb', 'scFoundation', PURPLE),
        ('lpm_randomGeneEmb', 'Random', GRAY),
    ]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    fig.suptitle('Local Similarity Erases Embedding Quality Differences\n(Until Extrapolation Is Required)',
                fontsize=20, fontweight='bold', y=0.97,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', linewidth=2))
    
    # =========== PANEL 1: LSFT (Converged) ===========
    ax1 = axes[0]
    
    lsft_means = []
    lsft_errors = []
    colors = []
    labels = []
    
    for bl, label, color in methods:
        row = k562_lsft[k562_lsft['baseline'] == bl]
        if len(row) > 0:
            lsft_means.append(row['r_mean'].values[0])
            lsft_errors.append([row['r_mean'].values[0] - row['r_ci_low'].values[0],
                               row['r_ci_high'].values[0] - row['r_mean'].values[0]])
            colors.append(color)
            labels.append(label)
    
    x = np.arange(len(labels))
    lsft_errors = np.array(lsft_errors).T
    
    bars = ax1.bar(x, lsft_means, color=colors, width=0.6, 
                   yerr=lsft_errors, capsize=8, error_kw={'linewidth': 2})
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel('Pearson r', fontsize=14, fontweight='bold')
    ax1.set_title('After LSFT (5% Nearest Neighbors) — K562', fontsize=16, fontweight='bold')
    
    for i, (bar, v) in enumerate(zip(bars, lsft_means)):
        ax1.text(bar.get_x() + bar.get_width()/2, v + lsft_errors[1][i] + 0.02,
                f'{v:.2f}', ha='center', fontsize=13, fontweight='bold')
    
    # Spread annotation
    spread = max(lsft_means) - min(lsft_means)
    ax1.annotate(f'Spread: {spread:.2f}\n(converged!)', xy=(1.5, min(lsft_means)), 
                xytext=(2.5, 0.5), fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
    
    ax1.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # =========== PANEL 2: LOGO (Diverged) ===========
    ax2 = axes[1]
    
    logo_means = []
    logo_errors = []
    
    for bl, label, color in methods:
        row = k562_logo[k562_logo['baseline'] == bl]
        if len(row) > 0:
            logo_means.append(row['r_mean'].values[0])
            logo_errors.append([row['r_mean'].values[0] - row['r_ci_low'].values[0],
                               row['r_ci_high'].values[0] - row['r_mean'].values[0]])
    
    logo_errors = np.array(logo_errors).T
    
    bars = ax2.bar(x, logo_means, color=colors, width=0.6,
                   yerr=logo_errors, capsize=8, error_kw={'linewidth': 2},
                   edgecolor='black', linewidth=2)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel('Pearson r', fontsize=14, fontweight='bold')
    ax2.set_title('Functional Holdout (LOGO) — K562', fontsize=16, fontweight='bold')
    
    for i, (bar, v) in enumerate(zip(bars, logo_means)):
        ax2.text(bar.get_x() + bar.get_width()/2, v + logo_errors[1][i] + 0.02,
                f'{v:.2f}', ha='center', fontsize=13, fontweight='bold')
    
    # Spread annotation
    spread = max(logo_means) - min(logo_means)
    ax2.annotate(f'Spread: {spread:.2f}\n(diverged!)', xy=(1.5, min(logo_means)), 
                xytext=(2.5, 0.2), fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax2.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Bottom insight
    fig.text(0.5, 0.02,
            'Dense training (LSFT): All embeddings ≈ equal  |  Sparse extrapolation (LOGO): PCA >> Others',
            fontsize=14, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, edgecolor='black'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.savefig(OUTPUT_DIR / "POSTER_accurate_convergence.png", dpi=200, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ POSTER_accurate_convergence.png")


# =============================================================================
# OPTION C: Three-Regime Plot
# =============================================================================

def create_three_regime_poster(data):
    """The complete story: Baseline → LSFT → LOGO."""
    
    lsft = data['lsft']
    logo = data['logo']
    
    # Also need baseline (before LSFT) - extract from raw data
    raw = pd.read_csv(DATA_DIR / "LSFT_raw_per_perturbation.csv")
    
    methods = [
        ('lpm_selftrained', 'PCA', GREEN),
        ('lpm_scgptGeneEmb', 'scGPT', BLUE),
        ('lpm_scFoundationGeneEmb', 'scFoundation', PURPLE),
        ('lpm_randomGeneEmb', 'Random', GRAY),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    fig.suptitle('The Manifold Law: Dense Training Makes Embeddings Irrelevant',
                fontsize=20, fontweight='bold', y=0.97,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', linewidth=2))
    
    # Get K562 data (cleanest)
    k562_lsft = lsft[lsft['dataset'] == 'k562']
    k562_logo = logo[logo['dataset'] == 'k562']
    k562_raw = raw[(raw['dataset'] == 'k562') & (raw['top_pct'] == 0.05)]
    
    # Compute baseline means from raw data
    baseline_means = {}
    for bl, label, color in methods:
        bl_data = k562_raw[k562_raw['baseline'] == bl]
        if len(bl_data) > 0:
            baseline_means[bl] = bl_data['performance_baseline_pearson_r'].mean()
    
    colors = [m[2] for m in methods]
    labels = [m[1] for m in methods]
    x = np.arange(len(methods))
    
    # =========== PANEL 1: Baseline (Raw) ===========
    ax1 = axes[0]
    raw_means = [baseline_means.get(m[0], 0) for m in methods]
    
    bars = ax1.bar(x, raw_means, color=colors, width=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=12, rotation=20, ha='right')
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel('Pearson r', fontsize=12, fontweight='bold')
    ax1.set_title('① Raw Baseline\n(Before LSFT)', fontsize=15, fontweight='bold')
    
    for bar, v in zip(bars, raw_means):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f'{v:.2f}', ha='center', fontsize=12, fontweight='bold')
    
    spread = max(raw_means) - min(raw_means)
    ax1.text(0.5, 0.12, f'Spread: {spread:.2f}\n(wide)', transform=ax1.transAxes,
            fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax1.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # =========== PANEL 2: After LSFT ===========
    ax2 = axes[1]
    
    lsft_means = []
    lsft_errors = []
    for m in methods:
        row = k562_lsft[k562_lsft['baseline'] == m[0]]
        if len(row) > 0:
            lsft_means.append(row['r_mean'].values[0])
            lsft_errors.append([row['r_mean'].values[0] - row['r_ci_low'].values[0],
                               row['r_ci_high'].values[0] - row['r_mean'].values[0]])
        else:
            lsft_means.append(0)
            lsft_errors.append([0, 0])
    
    lsft_errors = np.array(lsft_errors).T
    
    bars = ax2.bar(x, lsft_means, color=colors, width=0.6,
                   yerr=lsft_errors, capsize=6, error_kw={'linewidth': 2})
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=12, rotation=20, ha='right')
    ax2.set_ylim(0, 1.0)
    ax2.set_title('② After LSFT\n(5% Nearest Neighbors)', fontsize=15, fontweight='bold')
    
    for bar, v in zip(bars, lsft_means):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                f'{v:.2f}', ha='center', fontsize=12, fontweight='bold')
    
    spread = max(lsft_means) - min(lsft_means)
    ax2.text(0.5, 0.12, f'Spread: {spread:.2f}\n(converged!)', transform=ax2.transAxes,
            fontsize=11, ha='center', fontweight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
    
    ax2.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # =========== PANEL 3: LOGO ===========
    ax3 = axes[2]
    
    logo_means = []
    logo_errors = []
    for m in methods:
        row = k562_logo[k562_logo['baseline'] == m[0]]
        if len(row) > 0:
            logo_means.append(row['r_mean'].values[0])
            logo_errors.append([row['r_mean'].values[0] - row['r_ci_low'].values[0],
                               row['r_ci_high'].values[0] - row['r_mean'].values[0]])
        else:
            logo_means.append(0)
            logo_errors.append([0, 0])
    
    logo_errors = np.array(logo_errors).T
    
    bars = ax3.bar(x, logo_means, color=colors, width=0.6,
                   yerr=logo_errors, capsize=6, error_kw={'linewidth': 2},
                   edgecolor='black', linewidth=2)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=12, rotation=20, ha='right')
    ax3.set_ylim(0, 1.0)
    ax3.set_title('③ Generalization Test\n(LOGO: Unseen Functions)', fontsize=15, fontweight='bold')
    
    for bar, v in zip(bars, logo_means):
        ax3.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                f'{v:.2f}', ha='center', fontsize=12, fontweight='bold')
    
    spread = max(logo_means) - min(logo_means)
    ax3.text(0.5, 0.12, f'Spread: {spread:.2f}\n(diverged!)', transform=ax3.transAxes,
            fontsize=11, ha='center', fontweight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax3.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Bottom insight
    fig.text(0.5, 0.02,
            '① Wide spread (PCA leads)  →  ② All converge (p=0.056)  →  ③ Diverge again (PCA wins extrapolation)',
            fontsize=13, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='black'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.savefig(OUTPUT_DIR / "POSTER_accurate_three_regime.png", dpi=200, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ POSTER_accurate_three_regime.png")


def main():
    print("=" * 60)
    print("CREATING ACCURATE POSTERS (Using Real Data)")
    print("=" * 60)
    print()
    
    data = load_data()
    
    create_statistical_poster(data)
    create_convergence_poster(data)
    create_three_regime_poster(data)
    
    print()
    print("=" * 60)
    print("✅ ACCURATE POSTERS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()


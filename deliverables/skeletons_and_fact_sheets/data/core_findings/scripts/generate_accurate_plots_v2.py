#!/usr/bin/env python3
"""
Generate Accurate Plots from CSV Data - V2

Uses the CORRECT data from:
- LSFT_resampling.csv (with bootstrap CIs)
- LOGO_results.csv (with bootstrap CIs)

Creates both Pearson R and L2 versions of each plot.
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
GRAY = '#7f8c8d'
RED = '#e74c3c'
ORANGE = '#f39c12'

# Baseline labels
LABELS = {
    'lpm_selftrained': 'PCA (Self-trained)',
    'lpm_scgptGeneEmb': 'scGPT',
    'lpm_scFoundationGeneEmb': 'scFoundation',
    'lpm_randomGeneEmb': 'Random Gene',
    'lpm_randomPertEmb': 'Random Pert',
    'lpm_gearsPertEmb': 'GEARS',
    'lpm_k562PertEmb': 'K562 Emb',
    'lpm_rpe1PertEmb': 'RPE1 Emb',
    'mean_response': 'Mean Response',
}

COLORS = {
    'lpm_selftrained': GREEN,
    'lpm_scgptGeneEmb': BLUE,
    'lpm_scFoundationGeneEmb': PURPLE,
    'lpm_randomGeneEmb': GRAY,
    'lpm_randomPertEmb': RED,
    'lpm_gearsPertEmb': ORANGE,
}

plt.rcParams['font.size'] = 12


def load_data():
    """Load the correct CSV data."""
    data = {}
    
    # LSFT with bootstrap CIs (correct data)
    data['lsft'] = pd.read_csv(DATA_DIR / "LSFT_resampling.csv")
    
    # LOGO with bootstrap CIs
    data['logo'] = pd.read_csv(DATA_DIR / "LOGO_results.csv")
    
    # Raw per-perturbation for additional analysis
    data['lsft_raw'] = pd.read_csv(DATA_DIR / "LSFT_raw_per_perturbation.csv")
    data['logo_raw'] = pd.read_csv(DATA_DIR / "LOGO_raw_per_perturbation.csv")
    
    return data


# =============================================================================
# PLOT 1: LSFT Performance Comparison (R and L2)
# =============================================================================

def plot_lsft_comparison(data):
    """Compare LSFT performance across baselines - both R and L2."""
    
    lsft = data['lsft']
    
    # Core baselines (ordered by expected performance)
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_scFoundationGeneEmb',
                 'lpm_randomGeneEmb', 'lpm_gearsPertEmb', 'lpm_randomPertEmb']
    
    datasets = ['adamson', 'k562', 'rpe1']
    
    for metric, ylabel, title_suffix, higher_better in [
        ('r_mean', 'Pearson r', 'R', True),
        ('l2_mean', 'L2 Distance', 'L2', False)
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        
        for ax, dataset in zip(axes, datasets):
            ds_data = lsft[lsft['dataset'] == dataset]
            
            values = []
            ci_lows = []
            ci_highs = []
            labels = []
            colors = []
            
            for bl in baselines:
                row = ds_data[ds_data['baseline'] == bl]
                if len(row) > 0:
                    values.append(row[metric].values[0])
                    ci_lows.append(row[f'{metric.split("_")[0]}_ci_low'].values[0])
                    ci_highs.append(row[f'{metric.split("_")[0]}_ci_high'].values[0])
                    labels.append(LABELS.get(bl, bl))
                    colors.append(COLORS.get(bl, GRAY))
            
            # Calculate error bars
            yerr_low = [v - l for v, l in zip(values, ci_lows)]
            yerr_high = [h - v for v, h in zip(values, ci_highs)]
            
            x = np.arange(len(labels))
            bars = ax.bar(x, values, yerr=[yerr_low, yerr_high], 
                         color=colors, capsize=4, alpha=0.85)
            
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            ax.set_title(f'{dataset.upper()}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, v in zip(bars, values):
                y_pos = v + 0.02 if higher_better else v - 0.1
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                       f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')
        
        axes[0].set_ylabel(ylabel, fontsize=12, fontweight='bold')
        
        fig.suptitle(f'LSFT Performance: {ylabel} (top 5% neighbors)',
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"ACCURATE_lsft_comparison_{title_suffix}.png",
                   dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ ACCURATE_lsft_comparison_{title_suffix}.png")


# =============================================================================
# PLOT 2: LOGO Performance Comparison (R and L2)
# =============================================================================

def plot_logo_comparison(data):
    """Compare LOGO performance across baselines - both R and L2."""
    
    logo = data['logo']
    
    # Core baselines
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_scFoundationGeneEmb',
                 'lpm_randomGeneEmb', 'lpm_gearsPertEmb', 'lpm_randomPertEmb']
    
    datasets = ['adamson', 'k562', 'rpe1']
    
    for metric, ci_low_col, ci_high_col, ylabel, title_suffix, higher_better in [
        ('r_mean', 'r_ci_low', 'r_ci_high', 'Pearson r', 'R', True),
        ('l2_mean', 'l2_ci_low', 'l2_ci_high', 'L2 Distance', 'L2', False)
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        
        for ax, dataset in zip(axes, datasets):
            ds_data = logo[logo['dataset'] == dataset]
            
            values = []
            ci_lows = []
            ci_highs = []
            labels = []
            colors = []
            
            for bl in baselines:
                row = ds_data[ds_data['baseline'] == bl]
                if len(row) > 0:
                    values.append(row[metric].values[0])
                    ci_lows.append(row[ci_low_col].values[0])
                    ci_highs.append(row[ci_high_col].values[0])
                    labels.append(LABELS.get(bl, bl))
                    colors.append(COLORS.get(bl, GRAY))
            
            yerr_low = [v - l for v, l in zip(values, ci_lows)]
            yerr_high = [h - v for v, h in zip(values, ci_highs)]
            
            x = np.arange(len(labels))
            bars = ax.bar(x, values, yerr=[yerr_low, yerr_high],
                         color=colors, capsize=4, alpha=0.85)
            
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            ax.set_title(f'{dataset.upper()}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                       f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')
        
        axes[0].set_ylabel(ylabel, fontsize=12, fontweight='bold')
        
        fig.suptitle(f'LOGO Performance: {ylabel} (Functional Class Holdout)',
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"ACCURATE_logo_comparison_{title_suffix}.png",
                   dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ ACCURATE_logo_comparison_{title_suffix}.png")


# =============================================================================
# PLOT 3: PCA vs Deep Learning Summary (R and L2)
# =============================================================================

def plot_pca_vs_deep_learning(data):
    """Direct comparison: PCA vs scGPT vs Random across all tests."""
    
    lsft = data['lsft']
    logo = data['logo']
    
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
    labels = ['PCA', 'scGPT', 'Random']
    colors_list = [GREEN, BLUE, GRAY]
    
    for metric_type in ['R', 'L2']:
        if metric_type == 'R':
            lsft_col = 'r_mean'
            logo_col = 'r_mean'
            ylabel = 'Pearson r'
        else:
            lsft_col = 'l2_mean'
            logo_col = 'l2_mean'
            ylabel = 'L2 Distance'
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1: LSFT
        for col, dataset in enumerate(['adamson', 'k562', 'rpe1']):
            ax = axes[0, col]
            ds_data = lsft[lsft['dataset'] == dataset]
            
            values = [ds_data[ds_data['baseline'] == bl][lsft_col].values[0] 
                     for bl in baselines if len(ds_data[ds_data['baseline'] == bl]) > 0]
            
            bars = ax.bar(range(len(values)), values, color=colors_list[:len(values)])
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_title(f'LSFT: {dataset.upper()}', fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel if col == 0 else '')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                       f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
        
        # Row 2: LOGO
        for col, dataset in enumerate(['adamson', 'k562', 'rpe1']):
            ax = axes[1, col]
            ds_data = logo[logo['dataset'] == dataset]
            
            values = []
            for bl in baselines:
                row = ds_data[ds_data['baseline'] == bl]
                if len(row) > 0:
                    values.append(row[logo_col].values[0])
            
            bars = ax.bar(range(len(values)), values, color=colors_list[:len(values)])
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_title(f'LOGO: {dataset.upper()}', fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel if col == 0 else '')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                       f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
        
        fig.suptitle(f'PCA vs Deep Learning: {ylabel}',
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"ACCURATE_pca_vs_deep_{metric_type}.png",
                   dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ ACCURATE_pca_vs_deep_{metric_type}.png")


# =============================================================================
# PLOT 4: LSFT vs LOGO Gap (Generalization)
# =============================================================================

def plot_generalization_gap(data):
    """Show generalization gap between LSFT and LOGO."""
    
    lsft = data['lsft']
    logo = data['logo']
    
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
    labels = ['PCA', 'scGPT', 'Random']
    colors_list = [GREEN, BLUE, GRAY]
    
    for metric_type in ['R', 'L2']:
        if metric_type == 'R':
            lsft_col = 'r_mean'
            logo_col = 'r_mean'
            ylabel = 'LSFT - LOGO (Pearson r)'
        else:
            lsft_col = 'l2_mean'
            logo_col = 'l2_mean'
            ylabel = 'LSFT - LOGO (L2)'
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for ax, dataset in zip(axes, ['adamson', 'k562', 'rpe1']):
            lsft_ds = lsft[lsft['dataset'] == dataset]
            logo_ds = logo[logo['dataset'] == dataset]
            
            gaps = []
            for bl in baselines:
                lsft_val = lsft_ds[lsft_ds['baseline'] == bl][lsft_col]
                logo_val = logo_ds[logo_ds['baseline'] == bl][logo_col]
                
                if len(lsft_val) > 0 and len(logo_val) > 0:
                    gaps.append(lsft_val.values[0] - logo_val.values[0])
            
            bars = ax.bar(range(len(gaps)), gaps, color=colors_list[:len(gaps)])
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_title(f'{dataset.upper()}', fontsize=14, fontweight='bold')
            ax.set_ylabel(ylabel if dataset == 'adamson' else '')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, v in zip(bars, gaps):
                ax.text(bar.get_x() + bar.get_width()/2, 
                       v + 0.01 if v >= 0 else v - 0.02,
                       f'{v:+.2f}', ha='center', fontsize=11, fontweight='bold')
        
        fig.suptitle(f'Generalization Gap: {metric_type}\n(Positive = Local > Global)',
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"ACCURATE_generalization_gap_{metric_type}.png",
                   dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ ACCURATE_generalization_gap_{metric_type}.png")


# =============================================================================
# PLOT 5: Summary Table as Plot
# =============================================================================

def plot_summary_table(data):
    """Create a visual summary table of all metrics."""
    
    lsft = data['lsft']
    logo = data['logo']
    
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb', 
                 'lpm_randomPertEmb']
    bl_labels = ['PCA', 'scGPT', 'Random Gene', 'Random Pert']
    
    datasets = ['adamson', 'k562', 'rpe1']
    
    # Collect data for table
    for metric_type in ['R', 'L2']:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('off')
        
        # Build table data
        if metric_type == 'R':
            lsft_col = 'r_mean'
            logo_col = 'r_mean'
            title = 'Summary: Pearson r Values'
        else:
            lsft_col = 'l2_mean'
            logo_col = 'l2_mean'
            title = 'Summary: L2 Distance Values'
        
        # Headers
        headers = ['Baseline'] + [f'LSFT\n{ds}' for ds in datasets] + [f'LOGO\n{ds}' for ds in datasets]
        
        table_data = []
        for bl, bl_label in zip(baselines, bl_labels):
            row = [bl_label]
            
            # LSFT values
            for ds in datasets:
                ds_data = lsft[(lsft['dataset'] == ds) & (lsft['baseline'] == bl)]
                if len(ds_data) > 0:
                    row.append(f"{ds_data[lsft_col].values[0]:.3f}")
                else:
                    row.append('-')
            
            # LOGO values
            for ds in datasets:
                ds_data = logo[(logo['dataset'] == ds) & (logo['baseline'] == bl)]
                if len(ds_data) > 0:
                    row.append(f"{ds_data[logo_col].values[0]:.3f}")
                else:
                    row.append('-')
            
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colColours=['lightblue'] * len(headers))
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2.0)
        
        # Color the PCA row green
        for j in range(len(headers)):
            table[(1, j)].set_facecolor('lightgreen')
        
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        
        plt.savefig(OUTPUT_DIR / f"ACCURATE_summary_table_{metric_type}.png",
                   dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ ACCURATE_summary_table_{metric_type}.png")


# =============================================================================
# PLOT 6: Simple Headline Plot
# =============================================================================

def plot_simple_headline(data):
    """Super simple headline plot for poster."""
    
    lsft = data['lsft']
    logo = data['logo']
    
    for metric_type in ['R', 'L2']:
        if metric_type == 'R':
            lsft_col = 'r_mean'
            logo_col = 'r_mean'
            ylabel = 'Pearson r'
            ylim = (0, 1.0)
        else:
            lsft_col = 'l2_mean'
            logo_col = 'l2_mean'
            ylabel = 'L2 Distance'
            ylim = (0, 10)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Average across all datasets
        baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
        labels = ['PCA\n(Self-trained)', 'scGPT\n(Pretrained)', 'Random']
        colors_list = [GREEN, BLUE, GRAY]
        
        # LSFT averages
        lsft_means = []
        for bl in baselines:
            bl_data = lsft[lsft['baseline'] == bl]
            lsft_means.append(bl_data[lsft_col].mean())
        
        # LOGO averages
        logo_means = []
        for bl in baselines:
            bl_data = logo[logo['baseline'] == bl]
            logo_means.append(bl_data[logo_col].mean())
        
        x = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, lsft_means, width, label='LSFT (Local)',
                      color=[c for c in colors_list], alpha=0.7)
        bars2 = ax.bar(x + width/2, logo_means, width, label='LOGO (Holdout)',
                      color=[c for c in colors_list], edgecolor='black', linewidth=2)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
        ax.set_title(f'Linear Model (PCA) Wins on {ylabel}',
                    fontsize=18, fontweight='bold')
        ax.set_ylim(ylim)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add values
        for bars, means in [(bars1, lsft_means), (bars2, logo_means)]:
            for bar, v in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                       f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"ACCURATE_headline_{metric_type}.png",
                   dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ ACCURATE_headline_{metric_type}.png")


def main():
    print("=" * 60)
    print("GENERATING ACCURATE PLOTS FROM CSV DATA")
    print("=" * 60)
    print()
    
    data = load_data()
    print(f"LSFT data: {len(data['lsft'])} rows")
    print(f"LOGO data: {len(data['logo'])} rows")
    print()
    
    # Generate all plots
    plot_lsft_comparison(data)
    plot_logo_comparison(data)
    plot_pca_vs_deep_learning(data)
    plot_generalization_gap(data)
    plot_summary_table(data)
    plot_simple_headline(data)
    
    print()
    print("=" * 60)
    print("✅ ALL ACCURATE PLOTS GENERATED!")
    print("=" * 60)


if __name__ == "__main__":
    main()


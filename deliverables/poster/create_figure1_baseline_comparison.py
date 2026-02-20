#!/usr/bin/env python3
"""
Figure 1: Baseline Performance Comparison
Bar plot of perturbation-level r for each embedding by dataset.

Shows:
- Self-trained PCA is the top baseline at both pseudobulk and single-cell resolution
- Foundation models (scGPT, scFoundation) perform modestly better than random
- GEARS yields weaker performance than PCA
- Random Gene and Random Perturbation define the noise floor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Data from verified results
# Pseudobulk data (from actual LSFT resampling results)
pseudobulk_data = {
    'Adamson': {
        'Self-trained PCA': 0.9465,
        'scGPT': 0.8107,
        'scFoundation': 0.7767,
        'GEARS': 0.7485,
        'Random Gene': 0.7214,
        'Random Pert': 0.7075,
    },
    'K562': {
        'Self-trained PCA': 0.6638,
        'scGPT': 0.5127,
        'scFoundation': 0.4293,
        'GEARS': 0.4456,
        'Random Gene': 0.3882,
        'Random Pert': 0.3838,
    },
    'RPE1': {
        'Self-trained PCA': 0.7678,
        'scGPT': 0.6672,
        'scFoundation': 0.6359,
        'GEARS': 0.6278,
        'Random Gene': 0.6295,
        'Random Pert': 0.6286,
    },
}

# Single-cell data (from baseline_results_all.csv)
single_cell_data = {
    'Adamson': {
        'Self-trained PCA': 0.396,
        'scGPT': 0.312,
        'scFoundation': 0.257,
        'GEARS': 0.207,
        'Random Gene': 0.205,
        'Random Pert': 0.204,
    },
    'K562': {
        'Self-trained PCA': 0.262,
        'scGPT': 0.194,
        'scFoundation': 0.115,
        'GEARS': 0.086,
        'Random Gene': 0.074,
        'Random Pert': 0.074,
    },
    'RPE1': {
        'Self-trained PCA': 0.395,
        'scGPT': 0.316,
        'scFoundation': 0.233,
        'GEARS': 0.203,
        'Random Gene': 0.203,
        'Random Pert': 0.203,
    },
}

# Colors - distinctive palette
colors = {
    'Self-trained PCA': '#2E86AB',  # Strong blue
    'scGPT': '#A23B72',             # Magenta
    'scFoundation': '#F18F01',      # Orange
    'GEARS': '#C73E1D',             # Red
    'Random Gene': '#95A5A6',       # Gray
    'Random Pert': '#BDC3C7',       # Light gray
}

baselines = ['Self-trained PCA', 'scGPT', 'scFoundation', 'GEARS', 'Random Gene', 'Random Pert']

def create_grouped_bar_plot():
    """Create a grouped bar plot comparing baselines across datasets and resolutions."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Pseudobulk Panel ---
    ax1 = axes[0]
    datasets_pb = ['Adamson', 'K562', 'RPE1']
    x = np.arange(len(datasets_pb))
    width = 0.12
    
    for i, baseline in enumerate(baselines):
        values = [pseudobulk_data[d][baseline] for d in datasets_pb]
        offset = (i - len(baselines)/2 + 0.5) * width
        bars = ax1.bar(x + offset, values, width, label=baseline, color=colors[baseline], 
                       edgecolor='white', linewidth=0.5)
    
    ax1.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Pearson r', fontsize=13, fontweight='bold')
    ax1.set_title('Pseudobulk Resolution', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets_pb, fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value annotations for PCA
    for i, d in enumerate(datasets_pb):
        val = pseudobulk_data[d]['Self-trained PCA']
        ax1.annotate(f'{val:.2f}', xy=(i - 2.5*width, val + 0.02), 
                     fontsize=9, ha='center', fontweight='bold', color=colors['Self-trained PCA'])
    
    # --- Single-cell Panel ---
    ax2 = axes[1]
    datasets_sc = ['Adamson', 'K562', 'RPE1']
    x = np.arange(len(datasets_sc))
    
    for i, baseline in enumerate(baselines):
        values = [single_cell_data[d].get(baseline, 0) for d in datasets_sc]
        offset = (i - len(baselines)/2 + 0.5) * width
        bars = ax2.bar(x + offset, values, width, label=baseline, color=colors[baseline],
                       edgecolor='white', linewidth=0.5)
    
    ax2.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Pearson r', fontsize=13, fontweight='bold')
    ax2.set_title('Single-Cell Resolution', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets_sc, fontsize=12)
    ax2.set_ylim(0, 0.5)
    ax2.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value annotations for PCA
    for i, d in enumerate(datasets_sc):
        val = single_cell_data[d]['Self-trained PCA']
        ax2.annotate(f'{val:.2f}', xy=(i - 2.5*width, val + 0.01), 
                     fontsize=9, ha='center', fontweight='bold', color=colors['Self-trained PCA'])
    
    # Shared legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=10, 
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    
    # Add overall title
    fig.suptitle('Baseline Performance Comparison: Pseudobulk vs Single-Cell', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig


def create_horizontal_bar_plot():
    """Alternative: Horizontal bar plot for cleaner comparison."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    datasets = ['Adamson', 'K562', 'RPE1']
    
    for col, dataset in enumerate(datasets):
        # Pseudobulk
        ax_pb = axes[0, col]
        values_pb = [pseudobulk_data[dataset][b] for b in baselines]
        y_pos = np.arange(len(baselines))
        bars = ax_pb.barh(y_pos, values_pb, color=[colors[b] for b in baselines],
                         edgecolor='white', linewidth=0.5, height=0.7)
        ax_pb.set_yticks(y_pos)
        ax_pb.set_yticklabels(baselines if col == 0 else [''] * len(baselines), fontsize=10)
        ax_pb.set_xlim(0, 1.05)
        ax_pb.set_title(f'{dataset}\n(Pseudobulk)', fontsize=12, fontweight='bold')
        ax_pb.axvline(x=0, color='black', linewidth=0.5)
        ax_pb.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add values
        for i, (bar, val) in enumerate(zip(bars, values_pb)):
            ax_pb.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                      va='center', fontsize=9)
        
        # Single-cell (if available)
        ax_sc = axes[1, col]
        if dataset in single_cell_data:
            values_sc = [single_cell_data[dataset].get(b, 0) for b in baselines]
            bars = ax_sc.barh(y_pos, values_sc, color=[colors[b] for b in baselines],
                             edgecolor='white', linewidth=0.5, height=0.7)
            ax_sc.set_yticks(y_pos)
            ax_sc.set_yticklabels(baselines if col == 0 else [''] * len(baselines), fontsize=10)
            ax_sc.set_xlim(0, 0.5)
            ax_sc.set_title(f'{dataset}\n(Single-Cell)', fontsize=12, fontweight='bold')
            ax_sc.axvline(x=0, color='black', linewidth=0.5)
            ax_sc.grid(axis='x', alpha=0.3, linestyle='--')
            
            for i, (bar, val) in enumerate(zip(bars, values_sc)):
                ax_sc.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                          va='center', fontsize=9)
        else:
            ax_sc.text(0.5, 0.5, 'Not yet\navailable', ha='center', va='center',
                      fontsize=12, color='gray', transform=ax_sc.transAxes)
            ax_sc.set_title(f'{dataset}\n(Single-Cell)', fontsize=12, fontweight='bold')
            ax_sc.axis('off')
    
    # Add xlabel
    for ax in axes[1, :]:
        ax.set_xlabel('Pearson r', fontsize=11, fontweight='bold')
    
    # Add overall title
    fig.suptitle('Baseline Performance Comparison by Dataset and Resolution', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def create_summary_figure():
    """Create a clean summary figure for the poster."""
    
    fig = plt.figure(figsize=(14, 8))
    
    # Create custom gridspec for asymmetric layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1.5, 1], 
                          hspace=0.35, wspace=0.25)
    
    # --- Main panel: Pseudobulk comparison ---
    ax_main = fig.add_subplot(gs[:, 0])
    
    datasets = ['Adamson', 'K562', 'RPE1']
    x = np.arange(len(datasets))
    width = 0.12
    
    for i, baseline in enumerate(baselines):
        values = [pseudobulk_data[d][baseline] for d in datasets]
        offset = (i - len(baselines)/2 + 0.5) * width
        ax_main.bar(x + offset, values, width, label=baseline, color=colors[baseline],
                   edgecolor='white', linewidth=0.5)
    
    ax_main.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax_main.set_ylabel('Pearson r (Pseudobulk)', fontsize=13, fontweight='bold')
    ax_main.set_title('A. Baseline Performance Comparison', fontsize=14, fontweight='bold', 
                      loc='left', pad=10)
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(datasets, fontsize=12)
    ax_main.set_ylim(0, 1.1)
    ax_main.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax_main.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add annotation
    ax_main.annotate('Self-trained PCA\nconsistently wins', 
                    xy=(0, 0.95), xytext=(0.5, 0.85),
                    fontsize=10, ha='center',
                    arrowprops=dict(arrowstyle='->', color=colors['Self-trained PCA'], lw=1.5),
                    color=colors['Self-trained PCA'], fontweight='bold')
    
    # --- Single-cell panels ---
    ax_sc1 = fig.add_subplot(gs[0, 1])
    ax_sc2 = fig.add_subplot(gs[1, 1])
    ax_sc3 = fig.add_subplot(gs[2, 1])
    
    for ax, dataset in zip([ax_sc1, ax_sc2, ax_sc3], ['Adamson', 'K562', 'RPE1']):
        values = [single_cell_data[dataset][b] for b in baselines]
        y_pos = np.arange(len(baselines))
        bars = ax.barh(y_pos, values, color=[colors[b] for b in baselines],
                      edgecolor='white', linewidth=0.5, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(baselines, fontsize=9)
        ax.set_xlim(0, 0.5)
        ax.set_title(f'B. {dataset} (Single-Cell)', fontsize=11, fontweight='bold', loc='left')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        for bar, val in zip(bars, values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                   va='center', fontsize=8)
    
    ax_sc3.set_xlabel('Pearson r', fontsize=11, fontweight='bold')
    
    # Add overall title
    fig.suptitle('Baseline Performance: Pseudobulk Overview and Single-Cell Details', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    
    # Create grouped bar plot
    fig1 = create_grouped_bar_plot()
    fig1.savefig(output_dir / 'figure1_baseline_comparison_grouped.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    fig1.savefig(output_dir / 'figure1_baseline_comparison_grouped.pdf', 
                 bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / 'figure1_baseline_comparison_grouped.png'}")
    
    # Create horizontal bar plot
    fig2 = create_horizontal_bar_plot()
    fig2.savefig(output_dir / 'figure1_baseline_comparison_horizontal.png',
                 dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / 'figure1_baseline_comparison_horizontal.png'}")
    
    # Create summary figure
    fig3 = create_summary_figure()
    fig3.savefig(output_dir / 'figure1_baseline_comparison_summary.png',
                 dpi=300, bbox_inches='tight', facecolor='white')
    fig3.savefig(output_dir / 'figure1_baseline_comparison_summary.pdf',
                 bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / 'figure1_baseline_comparison_summary.png'}")
    
    plt.close('all')
    print("\nAll figures generated successfully!")


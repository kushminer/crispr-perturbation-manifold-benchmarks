#!/usr/bin/env python3
"""
Figure 4: LOGO r per embedding for pseudobulk and single-cell
Shows extrapolation performance (Leave-One-GO-Out) across embedding types.

Key message: Self-trained PCA generalizes well to new functional classes,
while random embeddings fail. Pretrained embeddings are intermediate.
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

project_root = Path(__file__).parent.parent.parent

# Baseline mapping
baselines_map = {
    'lpm_selftrained': 'Self-trained PCA',
    'lpm_scgptGeneEmb': 'scGPT',
    'lpm_scFoundationGeneEmb': 'scFoundation',
    'lpm_gearsPertEmb': 'GEARS',
    'lpm_randomGeneEmb': 'Random Gene',
    'lpm_randomPertEmb': 'Random Pert',
}

baselines = ['Self-trained PCA', 'scGPT', 'scFoundation', 'GEARS', 'Random Gene', 'Random Pert']

# Colors matching Figure 1
colors = {
    'Self-trained PCA': '#2E86AB',
    'scGPT': '#A23B72',
    'scFoundation': '#F18F01',
    'GEARS': '#C73E1D',
    'Random Gene': '#95A5A6',
    'Random Pert': '#BDC3C7',
}


def load_pseudobulk_logo():
    """Load pseudobulk LOGO results."""
    logo_file = project_root / 'skeletons_and_fact_sheets/data/LOGO_results.csv'
    df = pd.read_csv(logo_file)
    
    pseudobulk_logo = {
        'Adamson': {},
        'K562': {},
        'RPE1': {},
    }
    
    for _, row in df.iterrows():
        dataset = row['dataset'].capitalize()
        if dataset == 'Adamson':
            dataset = 'Adamson'
        elif dataset == 'K562':
            dataset = 'K562'
        elif dataset == 'Rpe1':
            dataset = 'RPE1'
        
        baseline_key = row['baseline']
        if baseline_key in baselines_map:
            baseline_name = baselines_map[baseline_key]
            pseudobulk_logo[dataset][baseline_name] = row['r_mean']
    
    return pseudobulk_logo


def load_single_cell_logo():
    """Load single-cell LOGO results."""
    single_cell_logo = {
        'Adamson': {},
        'K562': {},
        'RPE1': {},
    }
    
    # Load Adamson (fixed version)
    adamson_file = project_root / 'results/single_cell_analysis/adamson/logo_fixed/logo_single_cell_summary_adamson_Transcription.csv'
    if adamson_file.exists():
        df = pd.read_csv(adamson_file)
        for _, row in df.iterrows():
            baseline_key = row['baseline_type']
            if baseline_key in baselines_map:
                baseline_name = baselines_map[baseline_key]
                single_cell_logo['Adamson'][baseline_name] = row['pert_mean_pearson_r']
    
    # Load K562 and RPE1 from actual result files
    for dataset in ['k562', 'rpe1']:
        dataset_key = dataset.upper() if dataset == 'k562' else 'RPE1'
        logo_file = project_root / f'results/single_cell_analysis/{dataset}/logo/logo_single_cell_summary_{dataset}_Transcription.csv'
        if logo_file.exists():
            df = pd.read_csv(logo_file)
            for _, row in df.iterrows():
                baseline_key = row['baseline_type']
                if baseline_key in baselines_map:
                    baseline_name = baselines_map[baseline_key]
                    single_cell_logo[dataset_key][baseline_name] = row['pert_mean_pearson_r']
    
    return single_cell_logo


def create_dual_resolution_figure():
    """Create figure showing both pseudobulk and single-cell LOGO r values."""
    
    pseudobulk_logo = load_pseudobulk_logo()
    single_cell_logo = load_single_cell_logo()
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    datasets = ['Adamson', 'K562', 'RPE1']
    
    for col, dataset in enumerate(datasets):
        # Pseudobulk panel (top row)
        ax_pb = axes[0, col]
        x = np.arange(1)  # Single dataset
        width = 0.12
        
        values_pb = [pseudobulk_logo[dataset].get(b, None) for b in baselines]
        
        for i, (baseline, val) in enumerate(zip(baselines, values_pb)):
            if val is None:
                continue
            offset = (i - len(baselines)/2 + 0.5) * width
            color = colors[baseline]
            bars = ax_pb.bar(x + offset, val, width, label=baseline, color=color,
                           edgecolor='white', linewidth=0.5)
            
            # Add value annotation
            y_pos = val + 0.03 if val >= 0 else val - 0.03
            ax_pb.annotate(f'{val:.3f}', xy=(x + offset, y_pos),
                          fontsize=8, ha='center', va='bottom' if val >= 0 else 'top',
                          fontweight='bold' if val > 0.5 else 'normal')
        
        ax_pb.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax_pb.set_ylabel('LOGO r', fontsize=12, fontweight='bold')
        ax_pb.set_title(f'{dataset}\n(Pseudobulk)', fontsize=13, fontweight='bold', pad=10)
        ax_pb.set_xticks(x)
        ax_pb.set_xticklabels([dataset], fontsize=11)
        ax_pb.axhline(y=0, color='black', linewidth=1, linestyle='-', zorder=0)
        ax_pb.set_ylim(-0.1, 1.0)
        ax_pb.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
        
        # Single-cell panel (bottom row)
        ax_sc = axes[1, col]
        values_sc = [single_cell_logo[dataset].get(b, None) for b in baselines]
        
        for i, (baseline, val) in enumerate(zip(baselines, values_sc)):
            if val is None:
                continue
            offset = (i - len(baselines)/2 + 0.5) * width
            color = colors[baseline]
            bars = ax_sc.bar(x + offset, val, width, label=baseline, color=color,
                           edgecolor='white', linewidth=0.5)
            
            # Add value annotation
            y_pos = val + 0.03 if val >= 0 else val - 0.03
            ax_sc.annotate(f'{val:.3f}', xy=(x + offset, y_pos),
                          fontsize=8, ha='center', va='bottom' if val >= 0 else 'top',
                          fontweight='bold' if val > 0.3 else 'normal')
        
        ax_sc.set_ylim(-0.1, 0.5)
        
        ax_sc.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax_sc.set_ylabel('LOGO r', fontsize=12, fontweight='bold')
        ax_sc.set_title(f'{dataset}\n(Single-Cell)', fontsize=13, fontweight='bold', pad=10)
        ax_sc.set_xticks(x)
        ax_sc.set_xticklabels([dataset], fontsize=11)
        ax_sc.axhline(y=0, color='black', linewidth=1, linestyle='-', zorder=0)
        ax_sc.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    
    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=10,
               bbox_to_anchor=(0.5, 0.99), frameon=False)
    
    # Add overall title
    fig.suptitle('LOGO Performance by Embedding Type and Resolution\n(Extrapolation to Held-Out Functional Classes)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig


def create_combined_comparison_figure():
    """Create side-by-side comparison of pseudobulk vs single-cell."""
    
    pseudobulk_logo = load_pseudobulk_logo()
    single_cell_logo = load_single_cell_logo()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    datasets = ['Adamson', 'K562', 'RPE1']
    x = np.arange(len(datasets))
    width = 0.12
    
    # Pseudobulk panel
    ax1 = axes[0]
    for i, baseline in enumerate(baselines):
        values = [pseudobulk_logo[d].get(baseline, None) for d in datasets]
        # Filter out None values
        valid_indices = [j for j, v in enumerate(values) if v is not None]
        if not valid_indices:
            continue
        
        valid_x = x[valid_indices]
        valid_values = [values[j] for j in valid_indices]
        
        offset = (i - len(baselines)/2 + 0.5) * width
        color = colors[baseline]
        
        bars = ax1.bar(valid_x + offset, valid_values, width, label=baseline, color=color,
                      edgecolor='white', linewidth=0.5)
        
        # Add annotations
        for bar, val in zip(bars, valid_values):
            y_pos = val + 0.03 if val >= 0 else val - 0.03
            ax1.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, y_pos),
                         fontsize=8, ha='center', va='bottom' if val >= 0 else 'top',
                         fontweight='bold' if val > 0.5 else 'normal')
    
    ax1.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax1.set_ylabel('LOGO r', fontsize=13, fontweight='bold')
    ax1.set_title('Pseudobulk Resolution', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=12)
    ax1.axhline(y=0, color='black', linewidth=1.2, linestyle='-', zorder=0)
    ax1.set_ylim(-0.1, 1.0)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    
    # Single-cell panel
    ax2 = axes[1]
    for i, baseline in enumerate(baselines):
        values = [single_cell_logo[d].get(baseline, None) for d in datasets]
        valid_indices = [j for j, v in enumerate(values) if v is not None]
        if not valid_indices:
            continue
        
        valid_x = x[valid_indices]
        valid_values = [values[j] for j in valid_indices]
        
        offset = (i - len(baselines)/2 + 0.5) * width
        color = colors[baseline]
        
        bars = ax2.bar(valid_x + offset, valid_values, width, label=baseline, color=color,
                      edgecolor='white', linewidth=0.5)
        
        # Add annotations
        for bar, val in zip(bars, valid_values):
            y_pos = val + 0.03 if val >= 0 else val - 0.03
            ax2.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, y_pos),
                         fontsize=8, ha='center', va='bottom' if val >= 0 else 'top',
                         fontweight='bold' if val > 0.3 else 'normal')
    
    ax2.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax2.set_ylabel('LOGO r', fontsize=13, fontweight='bold')
    ax2.set_title('Single-Cell Resolution', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, fontsize=12)
    ax2.axhline(y=0, color='black', linewidth=1.2, linestyle='-', zorder=0)
    ax2.set_ylim(-0.1, 0.5)
    ax2.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    
    # Shared legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=10,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig


def create_horizontal_comparison():
    """Create horizontal bar plot comparing pseudobulk vs single-cell."""
    
    pseudobulk_logo = load_pseudobulk_logo()
    single_cell_logo = load_single_cell_logo()
    
    # Average across datasets (only where data exists)
    baseline_means_pb = {}
    baseline_means_sc = {}
    
    for baseline in baselines:
        pb_values = []
        sc_values = []
        for dataset in ['Adamson', 'K562', 'RPE1']:
            pb_val = pseudobulk_logo[dataset].get(baseline)
            sc_val = single_cell_logo[dataset].get(baseline)
            if pb_val is not None:
                pb_values.append(pb_val)
            if sc_val is not None:
                sc_values.append(sc_val)
        
        if pb_values:
            baseline_means_pb[baseline] = np.mean(pb_values)
        if sc_values:
            baseline_means_sc[baseline] = np.mean(sc_values)
    
    # Sort by pseudobulk r
    sorted_baselines = sorted(baseline_means_pb.items(), key=lambda x: x[1], reverse=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    y_pos = np.arange(len(sorted_baselines))
    pb_values = [baseline_means_pb[b] for b, _ in sorted_baselines]
    sc_values = [baseline_means_sc.get(b, None) for b, _ in sorted_baselines]
    labels = [b for b, _ in sorted_baselines]
    bar_colors = [colors[b] for b in labels]
    
    width = 0.35
    
    # Pseudobulk bars
    bars_pb = ax.barh(y_pos - width/2, pb_values, width, label='Pseudobulk', 
                     color=bar_colors, edgecolor='white', linewidth=0.5, alpha=0.8)
    
    # Single-cell bars (only Adamson available)
    bars_sc = []
    for i, (sc_val, label) in enumerate(zip(sc_values, labels)):
        if sc_val is not None:
            bar = ax.barh(y_pos[i] + width/2, sc_val, width, label='Single-Cell' if i == 0 else '',
                         color=bar_colors[i], edgecolor='white', linewidth=0.5, 
                         alpha=0.6, hatch='...')
            bars_sc.append(bar)
    
    # Add value labels
    for i, (bar_pb, val_pb, val_sc) in enumerate(zip(bars_pb, pb_values, sc_values)):
        x_pos_pb = val_pb + 0.02 if val_pb >= 0 else val_pb - 0.02
        ax.text(x_pos_pb, bar_pb.get_y() + bar_pb.get_height()/2, f'{val_pb:.3f}',
               va='center', ha='left' if val_pb >= 0 else 'right',
               fontsize=9, fontweight='bold' if val_pb > 0.5 else 'normal')
        
        if val_sc is not None:
            x_pos_sc = val_sc + 0.02 if val_sc >= 0 else val_sc - 0.02
            ax.text(x_pos_sc, bar_pb.get_y() + bar_pb.get_height()/2 + width, f'{val_sc:.3f}',
                   va='center', ha='left' if val_sc >= 0 else 'right',
                   fontsize=9, fontweight='bold' if val_sc > 0.3 else 'normal', 
                   style='italic')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('LOGO r', fontsize=13, fontweight='bold')
    ax.set_title('LOGO Performance: Pseudobulk vs Single-Cell\n(Average across Adamson, K562, RPE1)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.axvline(x=0, color='black', linewidth=1.2, linestyle='-', zorder=0)
    ax.set_xlim(-0.1, 1.0)
    ax.grid(axis='x', alpha=0.3, linestyle='--', zorder=0)
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.8, label='Pseudobulk'),
        Patch(facecolor='gray', alpha=0.6, hatch='...', label='Single-Cell'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    
    # Create dual resolution figure (6 panels: 3 datasets × 2 resolutions)
    fig1 = create_dual_resolution_figure()
    fig1.savefig(output_dir / 'figure4_logo_comparison_dual_resolution.png',
                 dpi=300, bbox_inches='tight', facecolor='white')
    fig1.savefig(output_dir / 'figure4_logo_comparison_dual_resolution.pdf',
                 bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / 'figure4_logo_comparison_dual_resolution.png'}")
    
    # Create side-by-side comparison
    fig2 = create_combined_comparison_figure()
    fig2.savefig(output_dir / 'figure4_logo_comparison_combined.png',
                 dpi=300, bbox_inches='tight', facecolor='white')
    fig2.savefig(output_dir / 'figure4_logo_comparison_combined.pdf',
                 bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / 'figure4_logo_comparison_combined.png'}")
    
    # Create horizontal comparison
    fig3 = create_horizontal_comparison()
    fig3.savefig(output_dir / 'figure4_logo_comparison_horizontal.png',
                 dpi=300, bbox_inches='tight', facecolor='white')
    fig3.savefig(output_dir / 'figure4_logo_comparison_horizontal.pdf',
                 bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / 'figure4_logo_comparison_horizontal.png'}")
    
    plt.close('all')
    print("\nAll Figure 4 variants generated successfully!")
    print("\nKey finding: Self-trained PCA generalizes well to new functional classes,")
    print("while random embeddings fail. Pattern holds across both resolutions.")


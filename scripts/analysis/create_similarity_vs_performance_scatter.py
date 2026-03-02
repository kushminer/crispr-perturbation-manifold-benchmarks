"""
Create scatter plot of cosine similarity vs Pearson r performance.

Shows the relationship between similarity (hardness) and performance.
"""

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Publication-quality color palette
COLORS = {
    'primary': '#3498DB',
    'secondary': '#9B59B6',
    'accent': '#E74C3C',
    'success': '#27AE60',
    'baseline_colors': {
        'lpm_selftrained': '#3498DB',
        'lpm_scgptGeneEmb': '#9B59B6',
        'lpm_randomGeneEmb': '#E67E22',
        'lpm_scFoundationGeneEmb': '#1ABC9C',
        'lpm_k562PertEmb': '#E74C3C',
        'lpm_rpe1PertEmb': '#F39C12',
        'lpm_gearsPertEmb': '#95A5A6',
        'lpm_randomPertEmb': '#34495E',
    },
    'dataset_colors': {
        'Adamson': '#3498DB',
        'K562': '#9B59B6',
        'RPE1': '#E67E22',
    },
}

BASELINE_NAMES = {
    'lpm_selftrained': 'Self-trained (PCA)',
    'lpm_scgptGeneEmb': 'scGPT Gene Emb.',
    'lpm_scFoundationGeneEmb': 'scFoundation Gene Emb.',
    'lpm_randomGeneEmb': 'Random Gene Emb.',
    'lpm_k562PertEmb': 'K562 Pert Emb.',
    'lpm_rpe1PertEmb': 'RPE1 Pert Emb.',
    'lpm_gearsPertEmb': 'GEARS Pert Emb.',
    'lpm_randomPertEmb': 'Random Pert Emb.',
}

DATASET_NAMES = {
    'adamson': 'Adamson',
    'k562': 'K562',
    'rpe1': 'RPE1',
}

# Clean, modern styling
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 17,
    'figure.titleweight': 'bold',
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.2,
})


def load_lsft_data(results_dir: Path, dataset: str) -> pd.DataFrame:
    """Load combined LSFT standardized data for a dataset."""
    combined_file = results_dir / dataset / f"lsft_{dataset}_all_baselines_combined.csv"
    
    if not combined_file.exists():
        return pd.DataFrame()
    
    return pd.read_csv(combined_file)


def create_similarity_vs_performance_scatter(
    results_dir: Path,
    output_path: Path,
    datasets: List[str] = None,
    baselines: List[str] = None,
    top_pct: float = 0.05,
    color_by: str = 'baseline',  # 'baseline', 'dataset', or 'none'
    show_regression: bool = True,
    figsize: Tuple[float, float] = (12, 8),
) -> None:
    """
    Create scatter plot of cosine similarity vs Pearson r performance.
    
    Args:
        color_by: How to color points ('baseline', 'dataset', or 'none')
        show_regression: Whether to show regression line and R²
    """
    if datasets is None:
        datasets = ['adamson', 'k562', 'rpe1']
    
    if baselines is None:
        # All 8 baselines
        baselines = [
            'lpm_selftrained',
            'lpm_scgptGeneEmb',
            'lpm_scFoundationGeneEmb',
            'lpm_randomGeneEmb',
            'lpm_k562PertEmb',
            'lpm_rpe1PertEmb',
            'lpm_gearsPertEmb',
            'lpm_randomPertEmb',
        ]
    
    # Load and combine data
    all_data = []
    for dataset in datasets:
        df = load_lsft_data(results_dir, dataset)
        if len(df) == 0:
            continue
        
        df_filtered = df[(df['top_pct'] == top_pct) & (df['baseline_type'].isin(baselines))].copy()
        
        # Extract similarity and performance
        df_filtered['dataset_name'] = DATASET_NAMES.get(dataset, dataset)
        df_filtered['baseline_name'] = df_filtered['baseline_type'].map(BASELINE_NAMES)
        
        all_data.append(df_filtered)
    
    if not all_data:
        print("  ⚠️  No data to plot")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Filter out NaN values
    combined_df = combined_df.dropna(subset=['hardness', 'pearson_r'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Color points based on choice
    if color_by == 'baseline':
        unique_baselines = combined_df['baseline_type'].unique()
        for baseline in unique_baselines:
            baseline_data = combined_df[combined_df['baseline_type'] == baseline]
            color = COLORS['baseline_colors'].get(baseline, COLORS['primary'])
            label = BASELINE_NAMES.get(baseline, baseline.replace('lpm_', ''))
            
            ax.scatter(
                baseline_data['hardness'],
                baseline_data['pearson_r'],
                alpha=0.6,
                s=50,
                color=color,
                label=label,
                edgecolors='white',
                linewidths=0.5,
            )
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True)
        
    elif color_by == 'dataset':
        for dataset in datasets:
            dataset_name = DATASET_NAMES.get(dataset, dataset)
            dataset_data = combined_df[combined_df['dataset_name'] == dataset_name]
            color = COLORS['dataset_colors'].get(dataset_name, COLORS['primary'])
            
            ax.scatter(
                dataset_data['hardness'],
                dataset_data['pearson_r'],
                alpha=0.6,
                s=50,
                color=color,
                label=dataset_name,
                edgecolors='white',
                linewidths=0.5,
            )
        
        ax.legend(loc='upper left', frameon=True, fancybox=True)
        
    else:  # color_by == 'none'
        ax.scatter(
            combined_df['hardness'],
            combined_df['pearson_r'],
            alpha=0.6,
            s=50,
            color=COLORS['primary'],
            edgecolors='white',
            linewidths=0.5,
        )
    
    # Add regression line if requested
    if show_regression:
        x = combined_df['hardness'].values
        y = combined_df['pearson_r'].values
        
        # Remove any remaining NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) > 2:
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
            
            # Plot regression line
            x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
            y_line = slope * x_line + intercept
            
            ax.plot(
                x_line,
                y_line,
                '--',
                color='red',
                linewidth=2.5,
                alpha=0.8,
                label=f'Regression (R² = {r_value**2:.3f}, p < 0.001)',
                zorder=1,
            )
            
            # Add text annotation with statistics
            ax.text(
                0.05,
                0.95,
                f'R² = {r_value**2:.3f}\np < 0.001\nn = {len(x_clean)}',
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
            )
    
    # Labels and title
    ax.set_xlabel('Cosine Similarity (Hardness)', fontweight='bold', fontsize=13, labelpad=10)
    ax.set_ylabel('Performance (Pearson r)', fontweight='bold', fontsize=13, labelpad=10)
    ax.set_title(
        f'Cosine Similarity vs Performance (Top {int(top_pct*100)}%)\n'
        f'Relationship Between Similarity and Prediction Accuracy',
        fontweight='bold',
        pad=15,
    )
    
    # Grid and limits
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Created: {output_path.name}")


def create_multi_panel_scatter(
    results_dir: Path,
    output_path: Path,
    datasets: List[str] = None,
    baselines: List[str] = None,
    top_pct: float = 0.05,
    figsize: Tuple[float, float] = (18, 6),
) -> None:
    """Create multi-panel scatter plot (one panel per dataset)."""
    if datasets is None:
        datasets = ['adamson', 'k562', 'rpe1']
    
    if baselines is None:
        baselines = [
            'lpm_selftrained',
            'lpm_scgptGeneEmb',
            'lpm_randomGeneEmb',
        ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(datasets), figsize=figsize, sharey=True, facecolor='white')
    
    if len(datasets) == 1:
        axes = [axes]
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        
        # Load data
        df = load_lsft_data(results_dir, dataset)
        if len(df) == 0:
            continue
        
        df_filtered = df[(df['top_pct'] == top_pct) & (df['baseline_type'].isin(baselines))].copy()
        df_filtered = df_filtered.dropna(subset=['hardness', 'pearson_r'])
        
        # Color by baseline
        for baseline in baselines:
            baseline_data = df_filtered[df_filtered['baseline_type'] == baseline]
            if len(baseline_data) == 0:
                continue
            
            color = COLORS['baseline_colors'].get(baseline, COLORS['primary'])
            label = BASELINE_NAMES.get(baseline, baseline.replace('lpm_', ''))
            
            ax.scatter(
                baseline_data['hardness'],
                baseline_data['pearson_r'],
                alpha=0.6,
                s=50,
                color=color,
                label=label,
                edgecolors='white',
                linewidths=0.5,
            )
        
        # Regression line per dataset
        x = df_filtered['hardness'].values
        y = df_filtered['pearson_r'].values
        
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
            
            x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
            y_line = slope * x_line + intercept
            
            ax.plot(
                x_line,
                y_line,
                '--',
                color='red',
                linewidth=2,
                alpha=0.7,
                zorder=1,
            )
            
            ax.text(
                0.05,
                0.95,
                f'R² = {r_value**2:.3f}',
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            )
        
        # Styling
        ax.set_xlabel('Cosine Similarity (Hardness)', fontweight='bold', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Performance (Pearson r)', fontweight='bold', fontsize=12)
        
        ax.set_title(f'{DATASET_NAMES.get(dataset, dataset)}', fontweight='bold', pad=10)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim([0, 1.05])
        ax.set_ylim([0, 1.05])
        
        if idx == 0:
            ax.legend(loc='upper left', frameon=True, fancybox=True, fontsize=9)
    
    fig.suptitle(
        f'Cosine Similarity vs Performance (Top {int(top_pct*100)}%)\n'
        f'Relationship Across Datasets',
        fontweight='bold',
        fontsize=16,
        y=1.02,
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Created: {output_path.name}")


def main():
    """Generate scatter plots."""
    base_dir = Path(__file__).parent.parent.parent
    results_dir = base_dir / "results" / "goal_3_prediction" / "lsft_resampling"
    output_dir = base_dir / "publication_figures"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("Creating Cosine Similarity vs Performance Scatter Plots")
    print("=" * 70)
    print()
    
    datasets = ['adamson', 'k562', 'rpe1']
    all_baselines = [
        'lpm_selftrained',
        'lpm_scgptGeneEmb',
        'lpm_scFoundationGeneEmb',
        'lpm_randomGeneEmb',
        'lpm_k562PertEmb',
        'lpm_rpe1PertEmb',
        'lpm_gearsPertEmb',
        'lpm_randomPertEmb',
    ]
    key_baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
    top_pct = 0.05
    
    # 1. Scatter colored by baseline (all baselines)
    print("1. Creating scatter plot (colored by baseline, all baselines)...")
    create_similarity_vs_performance_scatter(
        results_dir=results_dir,
        output_path=output_dir / "similarity_vs_performance_scatter_all_baselines.png",
        datasets=datasets,
        baselines=all_baselines,
        top_pct=top_pct,
        color_by='baseline',
        show_regression=True,
        figsize=(14, 8),
    )
    
    # 2. Scatter colored by dataset (all baselines)
    print("\n2. Creating scatter plot (colored by dataset, all baselines)...")
    create_similarity_vs_performance_scatter(
        results_dir=results_dir,
        output_path=output_dir / "similarity_vs_performance_scatter_by_dataset.png",
        datasets=datasets,
        baselines=all_baselines,
        top_pct=top_pct,
        color_by='dataset',
        show_regression=True,
        figsize=(12, 8),
    )
    
    # 3. Multi-panel scatter (one per dataset, key baselines)
    print("\n3. Creating multi-panel scatter plot (one panel per dataset)...")
    create_multi_panel_scatter(
        results_dir=results_dir,
        output_path=output_dir / "similarity_vs_performance_multi_panel.png",
        datasets=datasets,
        baselines=key_baselines,
        top_pct=top_pct,
        figsize=(18, 6),
    )
    
    # 4. Simple scatter (no colors, all data)
    print("\n4. Creating simple scatter plot (all data combined)...")
    create_similarity_vs_performance_scatter(
        results_dir=results_dir,
        output_path=output_dir / "similarity_vs_performance_simple.png",
        datasets=datasets,
        baselines=all_baselines,
        top_pct=top_pct,
        color_by='none',
        show_regression=True,
        figsize=(10, 8),
    )
    
    print()
    print("=" * 70)
    print("✅ All scatter plots generated successfully!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()


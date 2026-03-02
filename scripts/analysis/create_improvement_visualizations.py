"""
Create multiple high-impact visualizations of LSFT improvement (LSFT - Baseline).

Focus: Improvement = LSFT - Baseline across all 8 baselines and 3 datasets (top 5%).
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Publication-quality color palette
COLORS = {
    'positive': '#27AE60',      # Green - positive improvement
    'negative': '#E74C3C',      # Red - negative improvement
    'neutral': '#95A5A6',       # Gray - neutral
    'accent': '#3498DB',        # Blue - accent
    'background': '#FFFFFF',    # White
}

BASELINE_NAMES = {
    'lpm_selftrained': 'Self-trained\n(PCA)',
    'lpm_scgptGeneEmb': 'scGPT\nGene Emb.',
    'lpm_scFoundationGeneEmb': 'scFoundation\nGene Emb.',
    'lpm_randomGeneEmb': 'Random\nGene Emb.',
    'lpm_k562PertEmb': 'K562\nPert Emb.',
    'lpm_rpe1PertEmb': 'RPE1\nPert Emb.',
    'lpm_gearsPertEmb': 'GEARS\nPert Emb.',
    'lpm_randomPertEmb': 'Random\nPert Emb.',
}

BASELINE_ORDER = [
    'lpm_selftrained',
    'lpm_scgptGeneEmb',
    'lpm_scFoundationGeneEmb',
    'lpm_randomGeneEmb',
    'lpm_k562PertEmb',
    'lpm_rpe1PertEmb',
    'lpm_gearsPertEmb',
    'lpm_randomPertEmb',
]

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
    'xtick.labelsize': 10,
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


def prepare_improvement_data(
    results_dir: Path,
    datasets: List[str],
    baselines: List[str],
    top_pct: float = 0.05,
) -> pd.DataFrame:
    """Prepare improvement data across all datasets and baselines."""
    all_results = []
    
    for dataset in datasets:
        df = load_lsft_data(results_dir, dataset)
        if len(df) == 0:
            continue
        
        df_filtered = df[(df['top_pct'] == top_pct) & (df['baseline_type'].isin(baselines))].copy()
        
        for baseline in baselines:
            baseline_data = df_filtered[df_filtered['baseline_type'] == baseline]
            
            if len(baseline_data) == 0:
                continue
            
            # Per-perturbation improvements
            improvements = baseline_data['improvement_pearson_r'].values
            
            # Aggregate statistics
            mean_improvement = np.mean(improvements)
            std_improvement = np.std(improvements)
            n = len(improvements)
            se_improvement = std_improvement / np.sqrt(n) if n > 0 else 0
            
            # Bootstrap CI (approximate using t-distribution)
            ci_lower = mean_improvement - 1.96 * se_improvement
            ci_upper = mean_improvement + 1.96 * se_improvement
            
            all_results.append({
                'dataset': DATASET_NAMES.get(dataset, dataset),
                'baseline': baseline,
                'baseline_name': BASELINE_NAMES.get(baseline, baseline.replace('lpm_', '')),
                'mean_improvement': mean_improvement,
                'std_improvement': std_improvement,
                'se_improvement': se_improvement,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_perturbations': n,
                'improvements': improvements,  # Store raw values for distribution plots
            })
    
    return pd.DataFrame(all_results)


def create_heatmap_improvement(
    improvement_df: pd.DataFrame,
    output_path: Path,
    figsize: Tuple[float, float] = (10, 8),
) -> None:
    """Create heatmap showing improvement values across baselines and datasets."""
    # Pivot data for heatmap
    pivot_data = improvement_df.pivot(
        index='baseline_name',
        columns='dataset',
        values='mean_improvement'
    )
    
    # Reorder baselines
    baseline_order = [BASELINE_NAMES.get(b, b) for b in BASELINE_ORDER if b in improvement_df['baseline'].values]
    pivot_data = pivot_data.reindex([name for name in baseline_order if name in pivot_data.index])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Create heatmap with diverging colormap
    vmax = max(abs(pivot_data.min().min()), abs(pivot_data.max().max()))
    vmin = -vmax
    
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=0,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Improvement (LSFT - Baseline)'},
        linewidths=1,
        linecolor='white',
        ax=ax,
        square=False,
        annot_kws={'fontsize': 9, 'fontweight': 'bold'},
    )
    
    ax.set_title(
        f'LSFT Improvement Heatmap (Top 5%)\nImprovement = LSFT - Baseline Performance',
        fontweight='bold',
        pad=15,
    )
    ax.set_xlabel('Dataset', fontweight='bold', labelpad=10)
    ax.set_ylabel('Baseline Method', fontweight='bold', labelpad=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Created: {output_path.name}")


def create_diverging_bar_chart(
    improvement_df: pd.DataFrame,
    output_path: Path,
    figsize: Tuple[float, float] = (14, 8),
) -> None:
    """Create diverging bar chart showing improvements diverging from zero."""
    # Aggregate across datasets for each baseline
    aggregated = improvement_df.groupby('baseline').agg({
        'mean_improvement': 'mean',
        'se_improvement': lambda x: np.sqrt(np.sum(x**2)) / len(x),
        'baseline_name': 'first',
    }).reset_index()
    
    # Reorder by baseline
    baseline_order = [BASELINE_NAMES.get(b, b) for b in BASELINE_ORDER if b in aggregated['baseline'].values]
    aggregated['baseline_name'] = pd.Categorical(aggregated['baseline_name'], categories=baseline_order, ordered=True)
    aggregated = aggregated.sort_values('baseline_name')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Colors based on improvement sign
    colors = [COLORS['positive'] if x >= 0 else COLORS['negative'] for x in aggregated['mean_improvement']]
    
    # Plot diverging bars
    bars = ax.barh(
        aggregated['baseline_name'],
        aggregated['mean_improvement'],
        color=colors,
        alpha=0.8,
        edgecolor='white',
        linewidth=1.5,
    )
    
    # Add error bars
    ax.errorbar(
        aggregated['mean_improvement'],
        aggregated['baseline_name'],
        xerr=aggregated['se_improvement'],
        fmt='none',
        color='black',
        capsize=4,
        capthick=1.5,
        elinewidth=1.5,
        alpha=0.7,
    )
    
    # Add zero line
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=0)
    
    # Add value labels
    for bar, val, err in zip(bars, aggregated['mean_improvement'], aggregated['se_improvement']):
        width = bar.get_width()
        ax.text(
            width + (err if width >= 0 else -err) + 0.01,
            bar.get_y() + bar.get_height()/2,
            f'{val:.3f}',
            ha='left' if width >= 0 else 'right',
            va='center',
            fontsize=10,
            fontweight='bold',
        )
    
    ax.set_xlabel('Improvement (LSFT - Baseline) Pearson r', fontweight='bold', fontsize=13)
    ax.set_title(
        f'LSFT Improvement Across All Baselines (Top 5%)\nMean Improvement Aggregated Across Datasets',
        fontweight='bold',
        pad=15,
    )
    
    ax.grid(True, alpha=0.2, axis='x', linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Created: {output_path.name}")


def create_dot_plot_with_ci(
    improvement_df: pd.DataFrame,
    output_path: Path,
    figsize: Tuple[float, float] = (14, 10),
) -> None:
    """Create clean dot plot with confidence intervals."""
    # Prepare data - show all baseline × dataset combinations
    plot_data = []
    for _, row in improvement_df.iterrows():
        plot_data.append({
            'baseline': row['baseline_name'],
            'dataset': row['dataset'],
            'mean': row['mean_improvement'],
            'ci_lower': row['ci_lower'],
            'ci_upper': row['ci_upper'],
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Reorder baselines
    baseline_order = [BASELINE_NAMES.get(b, b) for b in BASELINE_ORDER if b in improvement_df['baseline'].values]
    plot_df['baseline'] = pd.Categorical(plot_df['baseline'], categories=baseline_order, ordered=True)
    plot_df = plot_df.sort_values(['baseline', 'dataset'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Create positions for grouping
    n_baselines = len(plot_df['baseline'].unique())
    n_datasets = len(plot_df['dataset'].unique())
    
    dataset_colors = {'Adamson': '#3498DB', 'K562': '#9B59B6', 'RPE1': '#E67E22'}
    
    y_positions = []
    y_labels = []
    colors_list = []
    
    pos = 0
    for baseline in plot_df['baseline'].unique():
        baseline_data = plot_df[plot_df['baseline'] == baseline]
        for _, row in baseline_data.iterrows():
            y_positions.append(pos)
            colors_list.append(dataset_colors.get(row['dataset'], COLORS['neutral']))
            pos += 1
        
        pos += 0.5  # Space between baseline groups
    
    # Plot dots with error bars
    for i, (_, row) in enumerate(plot_df.iterrows()):
        color = colors_list[i]
        y_pos = y_positions[i]
        
        ax.plot(
            row['mean'],
            y_pos,
            'o',
            color=color,
            markersize=10,
            alpha=0.8,
            zorder=3,
        )
        
        ax.plot(
            [row['ci_lower'], row['ci_upper']],
            [y_pos, y_pos],
            '-',
            color=color,
            linewidth=2,
            alpha=0.6,
            zorder=2,
        )
        
        ax.plot(
            [row['ci_lower'], row['ci_lower']],
            [y_pos - 0.1, y_pos + 0.1],
            '-',
            color=color,
            linewidth=2,
            alpha=0.6,
            zorder=2,
        )
        
        ax.plot(
            [row['ci_upper'], row['ci_upper']],
            [y_pos - 0.1, y_pos + 0.1],
            '-',
            color=color,
            linewidth=2,
            alpha=0.6,
            zorder=2,
        )
    
    # Zero line
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=0)
    
    # Labels
    ax.set_yticks(y_positions)
    labels = []
    for baseline in plot_df['baseline'].unique():
        baseline_data = plot_df[plot_df['baseline'] == baseline]
        for _, row in baseline_data.iterrows():
            labels.append(f"{row['baseline']} - {row['dataset']}")
    ax.set_yticklabels(labels, fontsize=9)
    
    ax.set_xlabel('Improvement (LSFT - Baseline) Pearson r', fontweight='bold', fontsize=13)
    ax.set_title(
        f'LSFT Improvement with 95% Confidence Intervals (Top 5%)\nAll Baselines × All Datasets',
        fontweight='bold',
        pad=15,
    )
    
    # Legend for datasets
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=dataset_colors['Adamson'], label='Adamson'),
        Patch(facecolor=dataset_colors['K562'], label='K562'),
        Patch(facecolor=dataset_colors['RPE1'], label='RPE1'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True)
    
    ax.grid(True, alpha=0.2, axis='x', linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Created: {output_path.name}")


def create_violin_plot(
    improvement_df: pd.DataFrame,
    output_path: Path,
    figsize: Tuple[float, float] = (14, 8),
) -> None:
    """Create violin plot showing distribution of improvements."""
    # Flatten improvement arrays
    plot_data = []
    for _, row in improvement_df.iterrows():
        for imp in row['improvements']:
            plot_data.append({
                'baseline': row['baseline_name'],
                'dataset': row['dataset'],
                'improvement': imp,
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Reorder baselines
    baseline_order = [BASELINE_NAMES.get(b, b) for b in BASELINE_ORDER if b in improvement_df['baseline'].values]
    plot_df['baseline'] = pd.Categorical(plot_df['baseline'], categories=baseline_order, ordered=True)
    plot_df = plot_df.sort_values('baseline')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Create violin plot
    positions = np.arange(len(plot_df['baseline'].unique()))
    datasets = plot_df['dataset'].unique()
    n_datasets = len(datasets)
    width = 0.8 / n_datasets
    
    for i, dataset in enumerate(datasets):
        dataset_data = plot_df[plot_df['dataset'] == dataset]
        baseline_positions = []
        values_list = []
        
        for j, baseline in enumerate(plot_df['baseline'].unique()):
            baseline_data = dataset_data[dataset_data['baseline'] == baseline]
            if len(baseline_data) > 0:
                baseline_positions.append(positions[j] - width*(n_datasets-1)/2 + i*width)
                values_list.append(baseline_data['improvement'].values)
        
        if len(values_list) > 0:
            parts = ax.violinplot(
                values_list,
                positions=baseline_positions,
                widths=width*0.8,
                showmeans=True,
                showmedians=True,
            )
            
            # Color violins
            for pc in parts['bodies']:
                pc.set_facecolor(COLORS['accent'])
                pc.set_alpha(0.6)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=0)
    ax.set_xticks(positions)
    ax.set_xticklabels(plot_df['baseline'].unique(), rotation=45, ha='right')
    ax.set_ylabel('Improvement (LSFT - Baseline) Pearson r', fontweight='bold', fontsize=13)
    ax.set_title(
        f'Distribution of LSFT Improvements (Top 5%)\nPer-Perturbation Improvements Across All Baselines',
        fontweight='bold',
        pad=15,
    )
    
    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Created: {output_path.name}")


def create_lollipop_chart(
    improvement_df: pd.DataFrame,
    output_path: Path,
    figsize: Tuple[float, float] = (14, 8),
) -> None:
    """Create modern lollipop chart showing improvements."""
    # Aggregate across datasets
    aggregated = improvement_df.groupby('baseline').agg({
        'mean_improvement': 'mean',
        'se_improvement': lambda x: np.sqrt(np.sum(x**2)) / len(x),
        'baseline_name': 'first',
    }).reset_index()
    
    # Reorder by improvement value
    aggregated = aggregated.sort_values('mean_improvement', ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Colors based on improvement
    colors = [COLORS['positive'] if x >= 0 else COLORS['negative'] for x in aggregated['mean_improvement']]
    
    # Plot lollipops
    y_pos = np.arange(len(aggregated))
    
    # Lines
    for i, (mean, se, color) in enumerate(zip(aggregated['mean_improvement'], aggregated['se_improvement'], colors)):
        ax.plot(
            [0, mean],
            [i, i],
            '-',
            color=color,
            linewidth=3,
            alpha=0.8,
            zorder=2,
        )
        
        # Error bars
        ax.plot(
            [mean - se, mean + se],
            [i, i],
            '-',
            color=color,
            linewidth=4,
            alpha=0.6,
            zorder=1,
        )
    
    # Dots
    ax.scatter(
        aggregated['mean_improvement'],
        y_pos,
        s=200,
        c=colors,
        alpha=0.9,
        edgecolors='white',
        linewidths=2,
        zorder=3,
    )
    
    # Zero line
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=0)
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(aggregated['baseline_name'], fontsize=10)
    
    # Add value labels
    for i, (mean, se) in enumerate(zip(aggregated['mean_improvement'], aggregated['se_improvement'])):
        ax.text(
            mean + se + 0.01 if mean >= 0 else mean - se - 0.01,
            i,
            f'{mean:.3f}',
            ha='left' if mean >= 0 else 'right',
            va='center',
            fontsize=10,
            fontweight='bold',
        )
    
    ax.set_xlabel('Improvement (LSFT - Baseline) Pearson r', fontweight='bold', fontsize=13)
    ax.set_title(
        f'LSFT Improvement Lollipop Chart (Top 5%)\nSorted by Improvement Magnitude',
        fontweight='bold',
        pad=15,
    )
    
    ax.grid(True, alpha=0.2, axis='x', linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Created: {output_path.name}")


def main():
    """Generate all improvement visualizations."""
    base_dir = Path(__file__).parent.parent.parent
    results_dir = base_dir / "results" / "goal_3_prediction" / "lsft_resampling"
    output_dir = base_dir / "publication_figures"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("Creating LSFT Improvement Visualizations (Top 5%)")
    print("=" * 70)
    print()
    
    # All 8 baselines
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
    
    datasets = ['adamson', 'k562', 'rpe1']
    top_pct = 0.05
    
    # Prepare data
    print("Loading and preparing data...")
    improvement_df = prepare_improvement_data(
        results_dir=results_dir,
        datasets=datasets,
        baselines=all_baselines,
        top_pct=top_pct,
    )
    
    if len(improvement_df) == 0:
        print("  ⚠️  No data found!")
        return
    
    print(f"  Loaded data for {len(improvement_df)} baseline×dataset combinations")
    print()
    
    # Create all visualizations
    print("1. Creating heatmap...")
    create_heatmap_improvement(
        improvement_df=improvement_df,
        output_path=output_dir / "improvement_heatmap_top5pct.png",
    )
    
    print("\n2. Creating diverging bar chart...")
    create_diverging_bar_chart(
        improvement_df=improvement_df,
        output_path=output_dir / "improvement_diverging_bars_top5pct.png",
    )
    
    print("\n3. Creating dot plot with CI...")
    create_dot_plot_with_ci(
        improvement_df=improvement_df,
        output_path=output_dir / "improvement_dot_plot_top5pct.png",
    )
    
    print("\n4. Creating violin plot...")
    create_violin_plot(
        improvement_df=improvement_df,
        output_path=output_dir / "improvement_violin_top5pct.png",
    )
    
    print("\n5. Creating lollipop chart...")
    create_lollipop_chart(
        improvement_df=improvement_df,
        output_path=output_dir / "improvement_lollipop_top5pct.png",
    )
    
    print()
    print("=" * 70)
    print("✅ All visualizations generated successfully!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()


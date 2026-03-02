"""
Create a single, visually appealing grouped bar chart comparing baseline vs LSFT (top 5%).

Groups by model type, with Baseline and LSFT bars side-by-side for each model.
Averages across datasets for cleaner visualization.
"""

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Publication-quality color palette - modern and clean
COLORS = {
    'baseline': '#95A5A6',      # Light gray - baseline (NOT LSFT)
    'lsft': '#3498DB',          # Vibrant blue - LSFT
    'accent': '#E74C3C',        # Red for threshold line
    'grid': '#ECF0F1',          # Light gray for grid
}

BASELINE_NAMES = {
    'lpm_selftrained': 'Self-trained\n(PCA)',
    'lpm_scgptGeneEmb': 'scGPT\nGene Emb.',
    'lpm_randomGeneEmb': 'Random\nGene Emb.',
    'lpm_scFoundationGeneEmb': 'scFoundation\nGene Emb.',
}

# Clean, modern styling
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.titleweight': 'bold',
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.2,
})


def load_lsft_data(results_dir: Path, dataset: str) -> pd.DataFrame:
    """Load combined LSFT standardized data for a dataset."""
    combined_file = results_dir / dataset / f"lsft_{dataset}_all_baselines_combined.csv"
    
    if not combined_file.exists():
        return pd.DataFrame()
    
    return pd.read_csv(combined_file)


def create_grouped_bar_chart(
    results_dir: Path,
    output_path: Path,
    datasets: List[str] = None,
    baselines: List[str] = None,
    top_pct: float = 0.05,
    figsize: Tuple[float, float] = (12, 8),
    aggregate_datasets: bool = True,
) -> None:
    """
    Create a clean, visually appealing grouped bar chart.
    
    Groups by model type, with Baseline (NOT LSFT) and LSFT bars side-by-side.
    """
    if datasets is None:
        datasets = ['adamson', 'k562', 'rpe1']
    
    if baselines is None:
        baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
    
    # Load and aggregate data
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
            
            baseline_perf = baseline_data['performance_baseline_pearson_r'].mean()
            lsft_perf = baseline_data['performance_local_pearson_r'].mean()
            
            # Compute standard errors
            n = len(baseline_data)
            baseline_se = baseline_data['performance_baseline_pearson_r'].std() / np.sqrt(n) if n > 0 else 0
            lsft_se = baseline_data['performance_local_pearson_r'].std() / np.sqrt(n) if n > 0 else 0
            
            all_results.append({
                'dataset': dataset,
                'baseline': baseline,
                'baseline_mean': baseline_perf,
                'baseline_se': baseline_se,
                'lsft_mean': lsft_perf,
                'lsft_se': lsft_se,
            })
    
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) == 0:
        print("  ⚠️  No data to plot")
        return
    
    # Aggregate across datasets if requested
    if aggregate_datasets:
        aggregated = results_df.groupby('baseline').agg({
            'baseline_mean': 'mean',
            'baseline_se': lambda x: np.sqrt(np.sum(x**2)) / len(x),  # Combine SEs
            'lsft_mean': 'mean',
            'lsft_se': lambda x: np.sqrt(np.sum(x**2)) / len(x),
        }).reset_index()
    else:
        aggregated = results_df
    
    # Create figure with clean white background
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Set up positions
    n_groups = len(baselines)
    group_spacing = 1.0
    bar_width = 0.38
    
    # Positions for each group
    x_pos = np.arange(n_groups) * group_spacing
    
    # Prepare data
    baseline_values = []
    baseline_errors = []
    lsft_values = []
    lsft_errors = []
    group_labels = []
    
    for baseline in baselines:
        if aggregate_datasets:
            subset = aggregated[aggregated['baseline'] == baseline]
        else:
            # Average across datasets
            subset = results_df[results_df['baseline'] == baseline]
            if len(subset) > 0:
                subset = pd.DataFrame({
                    'baseline_mean': [subset['baseline_mean'].mean()],
                    'baseline_se': [np.sqrt(np.sum(subset['baseline_se']**2)) / len(subset)],
                    'lsft_mean': [subset['lsft_mean'].mean()],
                    'lsft_se': [np.sqrt(np.sum(subset['lsft_se']**2)) / len(subset)],
                })
        
        if len(subset) > 0:
            baseline_values.append(subset['baseline_mean'].values[0])
            baseline_errors.append(subset['baseline_se'].values[0])
            lsft_values.append(subset['lsft_mean'].values[0])
            lsft_errors.append(subset['lsft_se'].values[0])
        else:
            baseline_values.append(np.nan)
            baseline_errors.append(0)
            lsft_values.append(np.nan)
            lsft_errors.append(0)
        
        group_labels.append(BASELINE_NAMES.get(baseline, baseline.replace('lpm_', '')))
    
    # Plot bars with modern styling
    baseline_bars = ax.bar(
        x_pos - bar_width/2,
        baseline_values,
        bar_width,
        label='Baseline (All Data)',
        color=COLORS['baseline'],
        alpha=0.85,
        edgecolor='white',
        linewidth=2,
        zorder=2,
    )
    
    lsft_bars = ax.bar(
        x_pos + bar_width/2,
        lsft_values,
        bar_width,
        label=f'LSFT (Top {int(top_pct*100)}%)',
        color=COLORS['lsft'],
        alpha=0.95,
        edgecolor='white',
        linewidth=2,
        zorder=2,
    )
    
    # Add error bars with clean styling
    ax.errorbar(
        x_pos - bar_width/2,
        baseline_values,
        yerr=baseline_errors,
        fmt='none',
        color='#34495E',
        capsize=5,
        capthick=2,
        elinewidth=2,
        alpha=0.8,
        zorder=3,
    )
    
    ax.errorbar(
        x_pos + bar_width/2,
        lsft_values,
        yerr=lsft_errors,
        fmt='none',
        color='#2C3E50',
        capsize=5,
        capthick=2,
        elinewidth=2,
        alpha=0.8,
        zorder=3,
    )
    
    # Add value labels on top of bars (clean and subtle)
    for bars, values, errors in [(baseline_bars, baseline_values, baseline_errors),
                                  (lsft_bars, lsft_values, lsft_errors)]:
        for bar, val, err in zip(bars, values, errors):
            if not np.isnan(val) and val > 0:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + err + 0.015,
                    f'{val:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold',
                    color='#2C3E50',
                )
    
    # Customize axes with clean styling
    ax.set_ylabel('Performance (Pearson r)', fontweight='bold', fontsize=14, labelpad=12)
    ax.set_xlabel('Model Type', fontweight='bold', fontsize=14, labelpad=12)
    ax.set_title(
        f'Baseline vs LSFT Performance Comparison\n(Top {int(top_pct*100)}% Most Similar Perturbations)',
        fontweight='bold',
        fontsize=16,
        pad=25,
    )
    
    # Set x-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(group_labels, rotation=0, ha='center', fontsize=11)
    
    # Add subtle grid
    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.8, color=COLORS['grid'])
    ax.set_axisbelow(True)
    
    # Set limits
    ax.set_ylim([0, max(max(baseline_values) + max(baseline_errors) if baseline_values else 0,
                        max(lsft_values) + max(lsft_errors) if lsft_values else 0) + 0.1])
    ax.set_xlim([x_pos[0] - 0.6, x_pos[-1] + 0.6])
    
    # Add reference line at r=0.9
    max_val = max([v for v in baseline_values + lsft_values if not np.isnan(v)], default=1.0)
    if max_val > 0.9:
        ax.axhline(y=0.9, color=COLORS['accent'], linestyle=':', linewidth=2.5, 
                   alpha=0.6, zorder=0, label='r = 0.9')
    
    # Clean legend
    ax.legend(
        loc='upper left',
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=11,
        framealpha=0.95,
        edgecolor='gray',
        facecolor='white',
    )
    
    # Clean layout
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.2)
    plt.close()
    
    print(f"  ✅ Created: {output_path.name}")


def main():
    """Generate the grouped bar chart."""
    base_dir = Path(__file__).parent.parent.parent
    results_dir = base_dir / "results" / "goal_3_prediction" / "lsft_resampling"
    output_dir = base_dir / "publication_figures"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("Creating Grouped Baseline vs LSFT Bar Chart (Top 5%)")
    print("=" * 70)
    print()
    
    key_baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
    datasets = ['adamson', 'k562', 'rpe1']
    
    output_path = output_dir / "grouped_baseline_vs_lsft_top5pct_clean.png"
    
    create_grouped_bar_chart(
        results_dir=results_dir,
        output_path=output_path,
        datasets=datasets,
        baselines=key_baselines,
        top_pct=0.05,
        figsize=(12, 8),
        aggregate_datasets=True,
    )
    
    print()
    print("=" * 70)
    print("✅ Figure generated successfully!")
    print("=" * 70)
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()


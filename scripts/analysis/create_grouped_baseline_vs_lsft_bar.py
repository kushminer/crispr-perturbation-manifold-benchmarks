"""
Create a single, visually appealing grouped bar chart comparing baseline vs LSFT (top 5%).

Groups model types together, with one bar for Baseline and one for LSFT.
"""

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Publication-quality color palette
COLORS = {
    'baseline': '#95A5A6',      # Light gray - baseline
    'lsft': '#3498DB',          # Blue - LSFT
    'accent': '#E74C3C',        # Red for emphasis
    'background': '#FFFFFF',    # White background
}

BASELINE_NAMES = {
    'lpm_selftrained': 'Self-trained\n(PCA)',
    'lpm_scgptGeneEmb': 'scGPT\nGene Emb.',
    'lpm_randomGeneEmb': 'Random\nGene Emb.',
    'lpm_scFoundationGeneEmb': 'scFoundation\nGene Emb.',
}

DATASET_NAMES = {
    'adamson': 'Adamson',
    'k562': 'K562',
    'rpe1': 'RPE1',
}

# Modern, clean styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Custom styling for maximum visual appeal
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'axes.labelsize': 13,
    'axes.titlesize': 16,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.titleweight': 'bold',
    'axes.linewidth': 1.5,
    'grid.linewidth': 0.6,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'patch.linewidth': 0,
})


def load_lsft_data(results_dir: Path, dataset: str) -> pd.DataFrame:
    """Load combined LSFT standardized data for a dataset."""
    combined_file = results_dir / dataset / f"lsft_{dataset}_all_baselines_combined.csv"
    
    if not combined_file.exists():
        print(f"  ⚠️  Combined file not found: {combined_file}")
        return pd.DataFrame()
    
    df = pd.read_csv(combined_file)
    return df


def create_grouped_bar_chart(
    results_dir: Path,
    output_path: Path,
    datasets: List[str] = None,
    baselines: List[str] = None,
    top_pct: float = 0.05,
    figsize: Tuple[float, float] = (14, 8),
) -> None:
    """
    Create a single, visually appealing grouped bar chart.
    
    Groups by model type, with Baseline and LSFT bars side-by-side.
    """
    if datasets is None:
        datasets = ['adamson', 'k562', 'rpe1']
    
    if baselines is None:
        baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
    
    # Load data for all datasets
    all_data = []
    for dataset in datasets:
        df = load_lsft_data(results_dir, dataset)
        if len(df) == 0:
            continue
        
        df_filtered = df[(df['top_pct'] == top_pct) & (df['baseline_type'].isin(baselines))].copy()
        
        # Compute means for each baseline
        for baseline in baselines:
            baseline_data = df_filtered[df_filtered['baseline_type'] == baseline]
            
            if len(baseline_data) == 0:
                continue
            
            baseline_perf = baseline_data['performance_baseline_pearson_r'].mean()
            lsft_perf = baseline_data['performance_local_pearson_r'].mean()
            
            # Compute standard errors
            baseline_std = baseline_data['performance_baseline_pearson_r'].std()
            lsft_std = baseline_data['performance_local_pearson_r'].std()
            n = len(baseline_data)
            
            baseline_se = baseline_std / np.sqrt(n) if n > 0 else 0
            lsft_se = lsft_std / np.sqrt(n) if n > 0 else 0
            
            all_data.append({
                'dataset': DATASET_NAMES.get(dataset, dataset),
                'baseline': BASELINE_NAMES.get(baseline, baseline.replace('lpm_', '')),
                'baseline_type': baseline,
                'baseline_mean': baseline_perf,
                'baseline_se': baseline_se,
                'lsft_mean': lsft_perf,
                'lsft_se': lsft_se,
            })
    
    results_df = pd.DataFrame(all_data)
    
    if len(results_df) == 0:
        print("  ⚠️  No data to plot")
        return
    
    # Create figure with custom styling
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Set up grouping
    n_datasets = len(datasets)
    n_baselines = len(baselines)
    
    # X positions for groups
    group_width = 0.8
    bar_width = 0.35
    
    # Create positions for groups
    x_positions = []
    group_labels = []
    baseline_positions = []
    lsft_positions = []
    
    pos = 0
    for baseline in baselines:
        baseline_name = BASELINE_NAMES.get(baseline, baseline.replace('lpm_', ''))
        for dataset in [DATASET_NAMES.get(d, d) for d in datasets]:
            baseline_x = pos - bar_width/2
            lsft_x = pos + bar_width/2
            
            baseline_positions.append(baseline_x)
            lsft_positions.append(lsft_x)
            x_positions.append(pos)
            group_labels.append(f'{baseline_name}\n{dataset}')
            
            pos += 1
        
        # Add spacing between baseline groups
        pos += 0.5
    
    # Prepare data for plotting
    baseline_values = []
    baseline_errors = []
    lsft_values = []
    lsft_errors = []
    
    for baseline in baselines:
        for dataset in [DATASET_NAMES.get(d, d) for d in datasets]:
            subset = results_df[
                (results_df['baseline_type'] == baseline) & 
                (results_df['dataset'] == dataset)
            ]
            
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
    
    # Plot bars
    bars1 = ax.bar(
        baseline_positions,
        baseline_values,
        bar_width,
        label='Baseline (All Data)',
        color=COLORS['baseline'],
        alpha=0.85,
        edgecolor='white',
        linewidth=1.5,
        capsize=5,
    )
    
    bars2 = ax.bar(
        lsft_positions,
        lsft_values,
        bar_width,
        label=f'LSFT (Top {int(top_pct*100)}%)',
        color=COLORS['lsft'],
        alpha=0.9,
        edgecolor='white',
        linewidth=1.5,
        capsize=5,
    )
    
    # Add error bars
    ax.errorbar(
        baseline_positions,
        baseline_values,
        yerr=baseline_errors,
        fmt='none',
        color='black',
        capsize=4,
        capthick=1.5,
        elinewidth=1.5,
        alpha=0.7,
    )
    
    ax.errorbar(
        lsft_positions,
        lsft_values,
        yerr=lsft_errors,
        fmt='none',
        color='black',
        capsize=4,
        capthick=1.5,
        elinewidth=1.5,
        alpha=0.7,
    )
    
    # Add subtle value labels on top of bars (only if significant)
    for bars, values, errors in [(bars1, baseline_values, baseline_errors), 
                                  (bars2, lsft_values, lsft_errors)]:
        for bar, val, err in zip(bars, values, errors):
            if not np.isnan(val) and val > 0:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + err + 0.02,
                    f'{val:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    fontweight='bold',
                    alpha=0.8,
                )
    
    # Customize axes
    ax.set_ylabel('Performance (Pearson r)', fontweight='bold', fontsize=13, labelpad=10)
    ax.set_xlabel('Model Type × Dataset', fontweight='bold', fontsize=13, labelpad=10)
    ax.set_title(
        f'Baseline vs LSFT Performance Comparison\n(Top {int(top_pct*100)}% Most Similar Perturbations)',
        fontweight='bold',
        fontsize=16,
        pad=20,
    )
    
    # Set x-axis labels
    # Group labels by baseline, showing dataset under each
    tick_positions = []
    tick_labels = []
    
    pos = 0
    for i, baseline in enumerate(baselines):
        baseline_name = BASELINE_NAMES.get(baseline, baseline.replace('lpm_', ''))
        
        # First occurrence shows baseline name
        dataset_labels = []
        for dataset in [DATASET_NAMES.get(d, d) for d in datasets]:
            dataset_labels.append(dataset)
            tick_positions.append(pos)
            pos += 1
        
        pos += 0.5  # Spacing between groups
    
    # Set ticks and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(group_labels, rotation=45, ha='right', fontsize=9)
    
    # Add subtle vertical lines to separate baseline groups
    sep_pos = n_datasets - 0.5
    for i in range(len(baselines) - 1):
        x_sep = (i + 1) * (n_datasets + 0.5)
        ax.axvline(x=x_sep, color='gray', linestyle='--', linewidth=1, alpha=0.3, zorder=0)
    
    # Legend
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
    
    # Grid and limits
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
    ax.set_ylim([0, 1.05])
    ax.set_xlim([-0.5, x_positions[-1] + 1])
    
    # Add reference line at r=0.9
    ax.axhline(y=0.9, color=COLORS['accent'], linestyle=':', linewidth=2, alpha=0.5, 
               label='r = 0.9 threshold', zorder=0)
    
    # Tight layout
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
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
    
    output_path = output_dir / "grouped_baseline_vs_lsft_top5pct.png"
    
    create_grouped_bar_chart(
        results_dir=results_dir,
        output_path=output_path,
        datasets=datasets,
        baselines=key_baselines,
        top_pct=0.05,
        figsize=(16, 9),
    )
    
    print()
    print("=" * 70)
    print("✅ Figure generated successfully!")
    print("=" * 70)
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()


"""
Create professional-grade figures comparing baseline vs LSFT performance.

Visualization types:
1. Bar chart: Baseline vs LSFT side-by-side bars
2. Box plot: Distribution of improvements
3. Histogram: Distribution of improvement values
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Publication-quality styling
PUBLICATION_COLORS = {
    'baseline': '#6C757D',       # Gray - baseline performance
    'lsft_1pct': '#2E86AB',      # Blue - LSFT 1%
    'lsft_5pct': '#A23B72',      # Purple - LSFT 5%
    'lsft_10pct': '#F18F01',     # Orange - LSFT 10%
    'improvement': '#06A77D',    # Green - improvement
}

BASELINE_NAMES = {
    'lpm_selftrained': 'Self-trained\n(PCA)',
    'lpm_scgptGeneEmb': 'scGPT\nGene Emb.',
    'lpm_randomGeneEmb': 'Random\nGene Emb.',
    'lpm_scFoundationGeneEmb': 'scFoundation\nGene Emb.',
    'lpm_k562PertEmb': 'K562\nPert Emb.',
    'lpm_rpe1PertEmb': 'RPE1\nPert Emb.',
    'lpm_gearsPertEmb': 'GEARS\nPert Emb.',
    'lpm_randomPertEmb': 'Random\nPert Emb.',
}

DATASET_NAMES = {
    'adamson': 'Adamson',
    'k562': 'K562',
    'rpe1': 'RPE1',
}

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.titleweight': 'bold',
    'axes.linewidth': 1.5,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2,
    'patch.linewidth': 1.5,
})


def load_lsft_data(results_dir: Path, dataset: str) -> pd.DataFrame:
    """Load combined LSFT standardized data for a dataset."""
    combined_file = results_dir / dataset / f"lsft_{dataset}_all_baselines_combined.csv"
    
    if not combined_file.exists():
        print(f"  ⚠️  Combined file not found: {combined_file}")
        return pd.DataFrame()
    
    df = pd.read_csv(combined_file)
    return df


def create_bar_chart_baseline_vs_lsft(
    results_dir: Path,
    output_path: Path,
    dataset: str,
    top_pct: float = 0.05,
    baselines: List[str] = None,
    figsize: Tuple[float, float] = (12, 7),
) -> None:
    """
    Create bar chart comparing baseline vs LSFT performance side-by-side.
    """
    df = load_lsft_data(results_dir, dataset)
    if len(df) == 0:
        print(f"  ⚠️  No data for {dataset}")
        return
    
    if baselines is None:
        # Get all unique baselines
        baselines = sorted(df['baseline_type'].unique().tolist())
        # Filter to key baselines if they exist
        key_baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
        baselines = [b for b in baselines if b in key_baselines] or baselines[:3]
    
    # Filter to specific top_pct
    df_filtered = df[df['top_pct'] == top_pct].copy()
    
    # Compute means for each baseline
    results = []
    for baseline in baselines:
        baseline_data = df_filtered[df_filtered['baseline_type'] == baseline]
        
        baseline_perf = baseline_data['performance_baseline_pearson_r'].mean()
        lsft_perf = baseline_data['performance_local_pearson_r'].mean()
        
        # Compute CIs (simple std-based for now, or use summary files)
        baseline_std = baseline_data['performance_baseline_pearson_r'].std()
        lsft_std = baseline_data['performance_local_pearson_r'].std()
        n = len(baseline_data)
        
        baseline_ci = 1.96 * baseline_std / np.sqrt(n)
        lsft_ci = 1.96 * lsft_std / np.sqrt(n)
        
        results.append({
            'baseline': baseline,
            'baseline_mean': baseline_perf,
            'baseline_ci': baseline_ci,
            'lsft_mean': lsft_perf,
            'lsft_ci': lsft_ci,
        })
    
    results_df = pd.DataFrame(results)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    n_baselines = len(results_df)
    x = np.arange(n_baselines)
    width = 0.35
    
    # Get baseline names
    baseline_labels = [BASELINE_NAMES.get(b, b.replace('lpm_', '')) for b in results_df['baseline']]
    
    # Plot bars
    bars1 = ax.bar(
        x - width/2,
        results_df['baseline_mean'],
        width,
        label='Baseline (All Data)',
        color=PUBLICATION_COLORS['baseline'],
        alpha=0.8,
        edgecolor='black',
        linewidth=1,
    )
    
    bars2 = ax.bar(
        x + width/2,
        results_df['lsft_mean'],
        width,
        label=f'LSFT (Top {int(top_pct*100)}%)',
        color=PUBLICATION_COLORS[f'lsft_{int(top_pct*100)}pct'],
        alpha=0.8,
        edgecolor='black',
        linewidth=1,
    )
    
    # Add error bars
    ax.errorbar(
        x - width/2,
        results_df['baseline_mean'],
        yerr=results_df['baseline_ci'],
        fmt='none',
        color='black',
        capsize=4,
        capthick=1.5,
        elinewidth=1.5,
    )
    
    ax.errorbar(
        x + width/2,
        results_df['lsft_mean'],
        yerr=results_df['lsft_ci'],
        fmt='none',
        color='black',
        capsize=4,
        capthick=1.5,
        elinewidth=1.5,
    )
    
    # Labels
    ax.set_xlabel('Baseline Method', fontweight='bold')
    ax.set_ylabel('Performance (Pearson r)', fontweight='bold')
    ax.set_title(
        f'Baseline vs LSFT Performance: {DATASET_NAMES.get(dataset, dataset)}\n'
        f'(Top {int(top_pct*100)}% Most Similar Perturbations)',
        fontweight='bold',
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(baseline_labels, rotation=0, ha='center')
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim([0, 1.05])
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold'
        )
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold'
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Created: {output_path.name}")


def create_box_plot_improvements(
    results_dir: Path,
    output_path: Path,
    dataset: str,
    baselines: List[str] = None,
    figsize: Tuple[float, float] = (12, 7),
) -> None:
    """
    Create box plot showing distribution of improvements (LSFT - Baseline) across perturbations.
    """
    df = load_lsft_data(results_dir, dataset)
    if len(df) == 0:
        print(f"  ⚠️  No data for {dataset}")
        return
    
    if baselines is None:
        key_baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
        baselines = [b for b in key_baselines if b in df['baseline_type'].unique()]
    
    # Filter to key baselines and extract improvements
    df_filtered = df[df['baseline_type'].isin(baselines)].copy()
    
    # Prepare data for box plot
    plot_data = []
    for baseline in baselines:
        baseline_data = df_filtered[df_filtered['baseline_type'] == baseline]
        
        for top_pct in [0.01, 0.05, 0.10]:
            top_pct_data = baseline_data[baseline_data['top_pct'] == top_pct]
            improvements = top_pct_data['improvement_pearson_r'].values
            
            for imp in improvements:
                plot_data.append({
                    'baseline': BASELINE_NAMES.get(baseline, baseline.replace('lpm_', '')),
                    'top_pct': f'Top {int(top_pct*100)}%',
                    'improvement': imp,
                })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create box plot
    box_data = []
    labels = []
    positions = []
    pos = 0
    
    for baseline in [BASELINE_NAMES.get(b, b.replace('lpm_', '')) for b in baselines]:
        for top_pct in ['Top 1%', 'Top 5%', 'Top 10%']:
            subset = plot_df[(plot_df['baseline'] == baseline) & (plot_df['top_pct'] == top_pct)]
            if len(subset) > 0:
                box_data.append(subset['improvement'].values)
                labels.append(f'{baseline}\n{top_pct}')
                positions.append(pos)
                pos += 1
    
    bp = ax.boxplot(
        box_data,
        positions=positions,
        tick_labels=labels,
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        meanline=True,
    )
    
    # Color boxes
    colors = [PUBLICATION_COLORS['lsft_1pct'], PUBLICATION_COLORS['lsft_5pct'], PUBLICATION_COLORS['lsft_10pct']] * len(baselines)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # Styling
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='No Change')
    ax.set_ylabel('Improvement (LSFT - Baseline)', fontweight='bold')
    ax.set_title(
        f'Distribution of LSFT Improvements: {DATASET_NAMES.get(dataset, dataset)}\n'
        f'(Per-Perturbation Improvements)',
        fontweight='bold',
        pad=15,
    )
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Created: {output_path.name}")


def create_histogram_improvements(
    results_dir: Path,
    output_path: Path,
    dataset: str,
    top_pct: float = 0.05,
    baselines: List[str] = None,
    figsize: Tuple[float, float] = (12, 7),
) -> None:
    """
    Create histogram showing distribution of improvement values.
    """
    df = load_lsft_data(results_dir, dataset)
    if len(df) == 0:
        print(f"  ⚠️  No data for {dataset}")
        return
    
    if baselines is None:
        key_baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
        baselines = [b for b in key_baselines if b in df['baseline_type'].unique()]
    
    # Filter to specific top_pct and baselines
    df_filtered = df[(df['top_pct'] == top_pct) & (df['baseline_type'].isin(baselines))].copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create histogram for each baseline
    for i, baseline in enumerate(baselines):
        baseline_data = df_filtered[df_filtered['baseline_type'] == baseline]
        improvements = baseline_data['improvement_pearson_r'].values
        
        baseline_name = BASELINE_NAMES.get(baseline, baseline.replace('lpm_', ''))
        color = [PUBLICATION_COLORS['lsft_1pct'], PUBLICATION_COLORS['lsft_5pct'], 
                 PUBLICATION_COLORS['lsft_10pct']][i % 3]
        
        ax.hist(
            improvements,
            bins=30,
            alpha=0.6,
            label=baseline_name,
            color=color,
            edgecolor='black',
            linewidth=1,
        )
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='No Change')
    ax.set_xlabel('Improvement (LSFT - Baseline) Pearson r', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title(
        f'Distribution of LSFT Improvements: {DATASET_NAMES.get(dataset, dataset)}\n'
        f'(Top {int(top_pct*100)}% Most Similar Perturbations)',
        fontweight='bold',
        pad=15,
    )
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Created: {output_path.name}")


def main():
    """Generate all baseline vs LSFT comparison figures."""
    base_dir = Path(__file__).parent.parent.parent
    results_dir = base_dir / "results" / "goal_3_prediction" / "lsft_resampling"
    output_dir = base_dir / "publication_figures"
    output_dir.mkdir(exist_ok=True)
    
    datasets = ['adamson', 'k562', 'rpe1']
    key_baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
    
    print("=" * 70)
    print("Creating Baseline vs LSFT Comparison Figures")
    print("=" * 70)
    print()
    
    for dataset in datasets:
        print(f"Processing {dataset}...")
        
        # 1. Bar chart - Baseline vs LSFT (top 5%)
        print(f"  1. Creating bar chart (top 5%)...")
        output_path = output_dir / f"baseline_vs_lsft_bar_{dataset}_top5pct.png"
        create_bar_chart_baseline_vs_lsft(
            results_dir=results_dir,
            output_path=output_path,
            dataset=dataset,
            top_pct=0.05,
            baselines=key_baselines,
        )
        
        # 2. Bar chart - Baseline vs LSFT (top 1%)
        print(f"  2. Creating bar chart (top 1%)...")
        output_path = output_dir / f"baseline_vs_lsft_bar_{dataset}_top1pct.png"
        create_bar_chart_baseline_vs_lsft(
            results_dir=results_dir,
            output_path=output_path,
            dataset=dataset,
            top_pct=0.01,
            baselines=key_baselines,
        )
        
        # 3. Box plot - Improvements distribution
        print(f"  3. Creating box plot...")
        output_path = output_dir / f"baseline_vs_lsft_boxplot_{dataset}.png"
        create_box_plot_improvements(
            results_dir=results_dir,
            output_path=output_path,
            dataset=dataset,
            baselines=key_baselines,
        )
        
        # 4. Histogram - Improvements distribution
        print(f"  4. Creating histogram (top 5%)...")
        output_path = output_dir / f"baseline_vs_lsft_histogram_{dataset}_top5pct.png"
        create_histogram_improvements(
            results_dir=results_dir,
            output_path=output_path,
            dataset=dataset,
            top_pct=0.05,
            baselines=key_baselines,
        )
        
        print()
    
    print("=" * 70)
    print("✅ All figures generated successfully!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  - baseline_vs_lsft_bar_{dataset}_top{1,5}pct.png (bar charts)")
    print("  - baseline_vs_lsft_boxplot_{dataset}.png (box plots)")
    print("  - baseline_vs_lsft_histogram_{dataset}_top5pct.png (histograms)")


if __name__ == "__main__":
    main()


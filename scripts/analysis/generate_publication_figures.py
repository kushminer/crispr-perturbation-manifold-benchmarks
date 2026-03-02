#!/usr/bin/env python3
"""
Generate publication-quality figures for LSFT resampling results.

This script creates enhanced visualizations with:
- Publication-ready styling (consistent fonts, colors, formatting)
- High resolution (300+ DPI)
- Clear labels, legends, and annotations
- Figure captions and metadata
- Consistent formatting across all figures
"""

import sys
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Set publication-style matplotlib parameters
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Publication color palette
PUBLICATION_COLORS = {
    'primary': '#2E86AB',  # Blue
    'secondary': '#A23B72',  # Purple
    'accent': '#F18F01',  # Orange
    'success': '#06A77D',  # Green
    'warning': '#F18F01',  # Orange
    'error': '#D63447',  # Red
    'neutral': '#6C757D',  # Gray
    'light_gray': '#E9ECEF',
    'dark_gray': '#495057',
}

# Baseline color mapping
BASELINE_COLORS = {
    'lpm_selftrained': '#2E86AB',
    'lpm_scgptGeneEmb': '#A23B72',
    'lpm_scFoundationGeneEmb': '#8E44AD',
    'lpm_randomGeneEmb': '#95A5A6',
    'lpm_k562PertEmb': '#06A77D',
    'lpm_rpe1PertEmb': '#16A085',
    'lpm_gearsPertEmb': '#F39C12',
    'lpm_randomPertEmb': '#E74C3C',
}

# Baseline display names
BASELINE_NAMES = {
    'lpm_selftrained': 'Self-trained',
    'lpm_scgptGeneEmb': 'scGPT Gene',
    'lpm_scFoundationGeneEmb': 'scFoundation Gene',
    'lpm_randomGeneEmb': 'Random Gene',
    'lpm_k562PertEmb': 'K562 Perturbation',
    'lpm_rpe1PertEmb': 'RPE1 Perturbation',
    'lpm_gearsPertEmb': 'GEARS Perturbation',
    'lpm_randomPertEmb': 'Random Perturbation',
}


def load_lsft_summary(summary_path: Path) -> Dict:
    """Load LSFT summary with CIs from JSON file."""
    with open(summary_path, "r") as f:
        return json.load(f)


def create_performance_comparison_bar(
    summaries: Dict[str, Dict],
    output_path: Path,
    metric: str = "pearson_r",
    top_pct: float = 0.05,
    figsize: Tuple[float, float] = (10, 6),
) -> None:
    """
    Create publication-quality bar chart comparing baseline performance across datasets.
    
    Parameters
    ----------
    summaries : Dict[str, Dict]
        Dictionary mapping dataset names to summary dictionaries
    output_path : Path
        Output file path
    metric : str
        Metric to plot: "pearson_r" or "l2"
    top_pct : float
        Top percentage to use (e.g., 0.05 for top 5%)
    figsize : Tuple[float, float]
        Figure size (width, height)
    """
    # Extract data
    data = []
    key_suffix = f"_top{int(top_pct*100)}pct"
    
    for dataset, summary in summaries.items():
        for key, baseline_data in summary.items():
            # Handle both structures: nested baseline dict or direct summary keys
            if isinstance(baseline_data, dict) and "pearson_r" in baseline_data:
                # Direct summary entry
                if metric in baseline_data:
                    baseline = baseline_data.get("baseline_type", key.split("_top")[0])
                    mean_val = baseline_data[metric]["mean"]
                    ci_lower = baseline_data[metric]["ci_lower"]
                    ci_upper = baseline_data[metric]["ci_upper"]
                    
                    data.append({
                        'Dataset': dataset.replace('replogle_', '').capitalize(),
                        'Baseline': BASELINE_NAMES.get(baseline, baseline.replace('lpm_', '')),
                        'Baseline_Code': baseline,
                        'Mean': mean_val,
                        'CI_Lower': ci_lower,
                        'CI_Upper': ci_upper,
                        'CI_Width': ci_upper - ci_lower,
                    })
            elif isinstance(baseline_data, dict):
                # Nested structure - check for summary keys
                for summary_key, summary_data in baseline_data.items():
                    if key_suffix in summary_key and isinstance(summary_data, dict):
                        if metric in summary_data:
                            baseline = summary_key.split("_top")[0]
                            mean_val = summary_data[metric]["mean"]
                            ci_lower = summary_data[metric]["ci_lower"]
                            ci_upper = summary_data[metric]["ci_upper"]
                            
                            data.append({
                                'Dataset': dataset.replace('replogle_', '').capitalize(),
                                'Baseline': BASELINE_NAMES.get(baseline, baseline.replace('lpm_', '')),
                                'Baseline_Code': baseline,
                                'Mean': mean_val,
                                'CI_Lower': ci_lower,
                                'CI_Upper': ci_upper,
                                'CI_Width': ci_upper - ci_lower,
                            })
    
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        print(f"  ⚠️  No data to plot for {output_path}")
        return
    
    # Sort by performance (mean)
    df_sorted = df.sort_values('Mean', ascending=(metric == 'l2'))  # Lower is better for L2
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique baselines and datasets
    baselines = df_sorted['Baseline'].unique()
    datasets = df_sorted['Dataset'].unique()
    
    # Position of bars
    x = np.arange(len(datasets))
    width = 0.85 / len(baselines)
    
    # Plot bars
    for i, baseline in enumerate(baselines):
        baseline_data = df_sorted[df_sorted['Baseline'] == baseline]
        means = []
        ci_lowers = []
        ci_uppers = []
        
        for dataset in datasets:
            dataset_data = baseline_data[baseline_data['Dataset'] == dataset]
            if len(dataset_data) > 0:
                means.append(dataset_data.iloc[0]['Mean'])
                ci_lowers.append(dataset_data.iloc[0]['CI_Lower'])
                ci_uppers.append(dataset_data.iloc[0]['CI_Upper'])
            else:
                means.append(np.nan)
                ci_lowers.append(np.nan)
                ci_uppers.append(np.nan)
        
        offset = (i - len(baselines)/2 + 0.5) * width
        pos = x + offset
        
        # Get color
        baseline_code = df_sorted[df_sorted['Baseline'] == baseline]['Baseline_Code'].iloc[0]
        color = BASELINE_COLORS.get(baseline_code, PUBLICATION_COLORS['neutral'])
        
        # Plot bars with error bars
        bars = ax.bar(pos, means, width, label=baseline, color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Error bars
        yerr_lower = np.array(means) - np.array(ci_lowers)
        yerr_upper = np.array(ci_uppers) - np.array(means)
        ax.errorbar(pos, means, yerr=[yerr_lower, yerr_upper], fmt='none', color='black', 
                   capsize=3, capthick=1, elinewidth=1)
    
    # Labels and formatting
    ax.set_xlabel('Dataset', fontweight='bold')
    metric_label = 'Pearson Correlation (r)' if metric == 'pearson_r' else 'L2 Distance'
    ax.set_ylabel(metric_label, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Title
    title = f'Baseline Performance Comparison (Top {int(top_pct*100)}% Similarity)'
    ax.set_title(title, fontweight='bold', pad=10)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Created: {output_path}")


def create_cross_dataset_ranking(
    summaries: Dict[str, Dict],
    output_path: Path,
    metric: str = "pearson_r",
    top_pct: float = 0.05,
    figsize: Tuple[float, float] = (12, 8),
) -> None:
    """
    Create publication-quality heatmap showing baseline rankings across datasets.
    """
    # Extract data and compute rankings
    data = []
    key_suffix = f"_top{int(top_pct*100)}pct"
    
    for dataset, summary in summaries.items():
        dataset_scores = []
        for key, baseline_data in summary.items():
            baseline = None
            mean_val = None
            
            # Handle both structures
            if isinstance(baseline_data, dict) and "pearson_r" in baseline_data:
                # Direct summary entry
                if metric in baseline_data:
                    baseline = baseline_data.get("baseline_type", key.split("_top")[0])
                    mean_val = baseline_data[metric]["mean"]
            elif isinstance(baseline_data, dict):
                # Nested structure
                for summary_key, summary_data in baseline_data.items():
                    if key_suffix in summary_key and isinstance(summary_data, dict):
                        if metric in summary_data:
                            baseline = summary_key.split("_top")[0]
                            mean_val = summary_data[metric]["mean"]
                            break
            
            if baseline and mean_val is not None:
                dataset_scores.append({
                    'baseline': baseline,
                    'score': mean_val,
                })
        
        # Sort and assign ranks
        dataset_scores.sort(key=lambda x: x['score'], reverse=(metric != 'l2'))
        for rank, item in enumerate(dataset_scores, 1):
            data.append({
                'Dataset': dataset.replace('replogle_', '').capitalize(),
                'Baseline': BASELINE_NAMES.get(item['baseline'], item['baseline'].replace('lpm_', '')),
                'Baseline_Code': item['baseline'],
                'Rank': rank,
                'Score': item['score'],
            })
    
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        print(f"  ⚠️  No data to plot for {output_path}")
        return
    
    # Create pivot table
    pivot = df.pivot(index='Baseline', columns='Dataset', values='Rank')
    pivot_scores = df.pivot(index='Baseline', columns='Dataset', values='Score')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(pivot, annot=pivot_scores.round(3), fmt='.3f', cmap='RdYlGn_r' if metric == 'l2' else 'RdYlGn',
                cbar_kws={'label': f'Rank (1=best)'}, ax=ax, linewidths=0.5, linecolor='white',
                square=False, vmin=1, vmax=len(pivot.index))
    
    # Labels
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('Baseline', fontweight='bold')
    ax.set_title(f'Baseline Rankings Across Datasets (Top {int(top_pct*100)}% Similarity)', 
                fontweight='bold', pad=15)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Created: {output_path}")


def create_hardness_performance_curve_pub(
    regression_df: pd.DataFrame,
    output_path: Path,
    metric: str = "pearson_r",
    baseline: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> None:
    """
    Create publication-quality hardness-performance curve with CI bands.
    """
    # Filter data
    plot_df = regression_df.copy()
    if baseline:
        plot_df = plot_df[plot_df['baseline_type'] == baseline]
    
    if len(plot_df) == 0:
        print(f"  ⚠️  No data to plot for {output_path}")
        return
    
    # Sort by top_pct
    plot_df = plot_df.sort_values('top_pct')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data
    top_pcts = plot_df['top_pct'].values
    slopes = plot_df['slope'].values
    intercepts = plot_df['intercept'].values
    rs = plot_df['r'].values
    
    # Get CI columns if they exist
    ci_lower_slopes = plot_df['slope_ci_lower'].values if 'slope_ci_lower' in plot_df.columns else [None] * len(plot_df)
    ci_upper_slopes = plot_df['slope_ci_upper'].values if 'slope_ci_upper' in plot_df.columns else [None] * len(plot_df)
    
    # Create hardness values for plotting (0 to 1)
    hardness = np.linspace(0, 1, 100)
    
    # Plot regression lines with CI bands
    baseline_name = BASELINE_NAMES.get(baseline, baseline.replace('lpm_', '')) if baseline else 'All Baselines'
    color = BASELINE_COLORS.get(baseline, PUBLICATION_COLORS['primary']) if baseline else PUBLICATION_COLORS['primary']
    
    # For each top_pct, plot regression line
    for i, top_pct in enumerate(top_pcts):
        slope = slopes[i]
        intercept = intercepts[i]
        r = rs[i]
        
        # Performance prediction
        performance = intercept + slope * hardness
        
        # CI band (if available)
        if ci_lower_slopes[i] is not None and ci_upper_slopes[i] is not None:
            perf_lower = intercept + ci_lower_slopes[i] * hardness
            perf_upper = intercept + ci_upper_slopes[i] * hardness
            ax.fill_between(hardness, perf_lower, perf_upper, alpha=0.2, color=color)
        
        # Regression line
        label = f'Top {int(top_pct*100)}% (r={r:.3f})'
        ax.plot(hardness, performance, label=label, color=color, linewidth=2, alpha=0.8)
    
    # Labels
    ax.set_xlabel('Hardness (Cosine Similarity)', fontweight='bold')
    metric_label = 'Pearson Correlation (r)' if metric == 'pearson_r' else 'L2 Distance'
    ax.set_ylabel(metric_label, fontweight='bold')
    ax.set_title(f'Hardness-Performance Relationship: {baseline_name}', fontweight='bold', pad=10)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Created: {output_path}")


def main():
    """Generate all publication-quality figures."""
    print("=" * 70)
    print("Generating Publication-Quality Figures")
    print("=" * 70)
    print()
    
    # Create output directory
    pub_dir = Path("publication_figures")
    pub_dir.mkdir(exist_ok=True)
    
    # Load summaries for all datasets
    datasets = ['adamson', 'k562', 'rpe1']
    summaries = {}
    
    print("Loading data...")
    for dataset in datasets:
        dataset_dir = Path(f"results/goal_3_prediction/lsft_resampling/{dataset}")
        
        # Check if directory exists
        if not dataset_dir.exists():
            print(f"  ⚠️  Directory not found: {dataset_dir}")
            continue
        
        # Load all baseline summaries
        dataset_summaries = {}
        baseline_files = list(dataset_dir.glob(f"lsft_{dataset}_*_summary.json"))
        
        if len(baseline_files) == 0:
            print(f"  ⚠️  No summary files found in {dataset_dir}")
            continue
        
        for baseline_file in baseline_files:
            baseline = baseline_file.stem.replace(f"lsft_{dataset}_", "").replace("_summary", "")
            try:
                with open(baseline_file) as f:
                    baseline_data = json.load(f)
                    # Store the entire summary structure (contains all top_pct entries)
                    dataset_summaries[baseline] = baseline_data
            except Exception as e:
                print(f"  ⚠️  Error loading {baseline_file}: {e}")
        
        if dataset_summaries:
            summaries[dataset] = dataset_summaries
            print(f"  ✅ Loaded {dataset}: {len(dataset_summaries)} baselines")
        else:
            print(f"  ⚠️  No valid summaries for {dataset}")
    
    if not summaries:
        print("\n❌ No summaries found. Please run LSFT evaluation first.")
        return
    
    print()
    print("Generating figures...")
    print()
    
    # 1. Performance comparison bar chart
    print("1. Creating performance comparison bar charts...")
    for metric in ['pearson_r', 'l2']:
        output_path = pub_dir / f"figure1_performance_comparison_{metric}.png"
        create_performance_comparison_bar(
            summaries=summaries,
            output_path=output_path,
            metric=metric,
            top_pct=0.05,
        )
    
    # 2. Cross-dataset ranking heatmap
    print("\n2. Creating cross-dataset ranking heatmaps...")
    for metric in ['pearson_r', 'l2']:
        output_path = pub_dir / f"figure2_cross_dataset_ranking_{metric}.png"
        create_cross_dataset_ranking(
            summaries=summaries,
            output_path=output_path,
            metric=metric,
            top_pct=0.05,
        )
    
    # 3. Hardness-performance curves (for key baselines)
    print("\n3. Creating hardness-performance curves...")
    key_baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
    
    for dataset in datasets:
        for baseline in key_baselines:
            regression_path = Path(f"results/goal_3_prediction/lsft_resampling/{dataset}/lsft_{dataset}_{baseline}_hardness_regressions.csv")
            if regression_path.exists():
                regression_df = pd.read_csv(regression_path)
                
                # Check if baseline has regression data
                baseline_regression = regression_df[regression_df['baseline_type'] == baseline]
                if len(baseline_regression) > 0:
                    output_path = pub_dir / f"figure3_hardness_curve_{dataset}_{baseline.replace('lpm_', '')}_pearson_r.png"
                    create_hardness_performance_curve_pub(
                        regression_df=regression_df,
                        output_path=output_path,
                        metric='pearson_r',
                        baseline=baseline,
                    )
    
    print()
    print("=" * 70)
    print("✅ Publication figures generated successfully!")
    print("=" * 70)
    print()
    print(f"Figures saved to: {pub_dir.absolute()}")
    print()
    print("Generated figures:")
    for fig_file in sorted(pub_dir.glob("*.png")):
        print(f"  - {fig_file.name}")


if __name__ == "__main__":
    main()


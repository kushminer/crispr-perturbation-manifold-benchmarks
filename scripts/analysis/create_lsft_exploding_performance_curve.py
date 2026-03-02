"""
Create professional-grade figure showing LSFT Exploding Performance Curve.

This figure demonstrates how local predictability skyrockets when models train
on only the top 1-5% most similar perturbations, with r → 0.9+ across all
datasets for PCA, scGPT, and even random embeddings.
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
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Purple
    'accent': '#F18F01',       # Orange
    'success': '#06A77D',      # Green
    'gray': '#6C757D',         # Gray
    'dark': '#212529',         # Dark
}

BASELINE_COLORS = {
    'lpm_selftrained': PUBLICATION_COLORS['primary'],      # Blue - PCA/Self-trained
    'lpm_scgptGeneEmb': PUBLICATION_COLORS['secondary'],    # Purple - scGPT
    'lpm_randomGeneEmb': PUBLICATION_COLORS['accent'],      # Orange - Random
    'lpm_scFoundationGeneEmb': PUBLICATION_COLORS['success'], # Green - scFoundation
}

BASELINE_NAMES = {
    'lpm_selftrained': 'Self-trained (PCA)',
    'lpm_scgptGeneEmb': 'scGPT Gene Emb.',
    'lpm_randomGeneEmb': 'Random Gene Emb.',
    'lpm_scFoundationGeneEmb': 'scFoundation Gene Emb.',
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
    'font.size': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.titleweight': 'bold',
    'axes.linewidth': 1.5,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'patch.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.minor.width': 1,
    'ytick.minor.width': 1,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def load_lsft_summaries(results_dir: Path, dataset: str) -> Dict:
    """Load all LSFT summary JSON files for a dataset."""
    summaries = {}
    dataset_dir = results_dir / dataset
    
    if not dataset_dir.exists():
        print(f"  ⚠️  Dataset directory not found: {dataset_dir}")
        return summaries
    
    # Find all summary JSON files
    summary_files = list(dataset_dir.glob(f"lsft_{dataset}_*_summary.json"))
    
    for summary_file in summary_files:
        # Extract baseline type from filename
        # Format: lsft_{dataset}_{baseline}_summary.json
        parts = summary_file.stem.split('_')
        if len(parts) >= 4:
            baseline = '_'.join(parts[2:-1])  # Everything between dataset and "summary"
            
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    summaries[baseline] = data
            except (json.JSONDecodeError, IOError) as e:
                print(f"  ⚠️  Error loading {summary_file}: {e}")
    
    return summaries


def extract_performance_by_top_pct(summaries: Dict, baseline: str) -> pd.DataFrame:
    """Extract performance (Pearson r) for each top_pct value."""
    if baseline not in summaries:
        return pd.DataFrame()
    
    data = summaries[baseline]
    results = []
    
    # Iterate through all keys in the summary (e.g., "lpm_selftrained_top1pct")
    for _, value in data.items():
        if isinstance(value, dict) and 'top_pct' in value:
            top_pct = value['top_pct']
            if 'pearson_r' in value and isinstance(value['pearson_r'], dict):
                pearson_r = value['pearson_r']
                results.append({
                    'top_pct': top_pct,
                    'pearson_r_mean': pearson_r.get('mean', np.nan),
                    'pearson_r_ci_lower': pearson_r.get('ci_lower', np.nan),
                    'pearson_r_ci_upper': pearson_r.get('ci_upper', np.nan),
                    'n_perturbations': value.get('n_perturbations', np.nan),
                })
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    df = df.sort_values('top_pct')
    return df


def create_exploding_performance_curve(
    results_dir: Path,
    output_path: Path,
    baselines: List[str] = None,
    datasets: List[str] = None,
    figsize: Tuple[float, float] = (14, 8),
) -> None:
    """
    Create the LSFT Exploding Performance Curve figure.
    
    Shows how performance (Pearson r) increases dramatically as we filter to
    more similar perturbations (lower top_pct values).
    """
    if baselines is None:
        baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
    
    if datasets is None:
        datasets = ['adamson', 'k562', 'rpe1']
    
    # Load all summaries
    all_data = {}
    for dataset in datasets:
        summaries = load_lsft_summaries(results_dir, dataset)
        all_data[dataset] = summaries
    
    # Create figure with subplots (one per dataset)
    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=figsize, sharey=True)
    
    if n_datasets == 1:
        axes = [axes]
    
    # Plot each dataset
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        summaries = all_data[dataset]
        
        # Plot each baseline
        for baseline in baselines:
            if baseline not in summaries:
                continue
            
            df = extract_performance_by_top_pct(summaries, baseline)
            if len(df) == 0:
                continue
            
            # Get color and name
            color = BASELINE_COLORS.get(baseline, PUBLICATION_COLORS['gray'])
            name = BASELINE_NAMES.get(baseline, baseline.replace('lpm_', ''))
            
            # Plot line with error bars
            top_pcts = df['top_pct'].values * 100  # Convert to percentage
            means = df['pearson_r_mean'].values
            ci_lower = df['pearson_r_ci_lower'].values
            ci_upper = df['pearson_r_ci_upper'].values
            
            # Main line
            ax.plot(
                top_pcts,
                means,
                marker='o',
                markersize=8,
                linewidth=2.5,
                color=color,
                label=name,
                alpha=0.9,
                zorder=3,
            )
            
            # Error bars (CI)
            ax.errorbar(
                top_pcts,
                means,
                yerr=[means - ci_lower, ci_upper - means],
                fmt='none',
                color=color,
                alpha=0.6,
                capsize=4,
                capthick=1.5,
                elinewidth=1.5,
                zorder=2,
            )
            
            # Fill area between CI bounds
            ax.fill_between(
                top_pcts,
                ci_lower,
                ci_upper,
                color=color,
                alpha=0.15,
                zorder=1,
            )
        
        # Styling
        ax.set_xlabel('Training Set Size\n(% Most Similar Perturbations)', fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Performance (Pearson r)', fontweight='bold')
        
        ax.set_title(f'{DATASET_NAMES.get(dataset, dataset)}', fontweight='bold', pad=15)
        ax.set_xlim([0.5, 10.5])
        ax.set_ylim([0.5, 1.0])
        ax.set_xticks([1, 5, 10])
        ax.set_xticklabels(['1%', '5%', '10%'])
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.axhline(y=0.9, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='r = 0.9 threshold')
        
        # Add legend only to first subplot
        if idx == 0:
            ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    # Overall title
    fig.suptitle(
        'LSFT Exploding Performance Curve:\nLocal Predictability Skyrockets with Top 1-5% Similar Perturbations',
        fontsize=18,
        fontweight='bold',
        y=0.98,
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Created: {output_path.name}")


def create_single_panel_exploding_curve(
    results_dir: Path,
    output_path: Path,
    baselines: List[str] = None,
    datasets: List[str] = None,
    figsize: Tuple[float, float] = (10, 7),
) -> None:
    """
    Create a single-panel version showing all datasets together.
    """
    if baselines is None:
        baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
    
    if datasets is None:
        datasets = ['adamson', 'k562', 'rpe1']
    
    # Load all summaries
    all_data = {}
    for dataset in datasets:
        summaries = load_lsft_summaries(results_dir, dataset)
        all_data[dataset] = summaries
    
    # Create single figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each baseline across all datasets
    for baseline in baselines:
        color = BASELINE_COLORS.get(baseline, PUBLICATION_COLORS['gray'])
        name = BASELINE_NAMES.get(baseline, baseline.replace('lpm_', ''))
        
        # Collect data across all datasets
        all_top_pcts = []
        all_means = []
        all_ci_lower = []
        all_ci_upper = []
        
        for dataset in datasets:
            summaries = all_data[dataset]
            if baseline not in summaries:
                continue
            
            df = extract_performance_by_top_pct(summaries, baseline)
            if len(df) == 0:
                continue
            
            all_top_pcts.extend(df['top_pct'].values * 100)
            all_means.extend(df['pearson_r_mean'].values)
            all_ci_lower.extend(df['pearson_r_ci_lower'].values)
            all_ci_upper.extend(df['pearson_r_ci_upper'].values)
        
        if len(all_top_pcts) == 0:
            continue
        
        # Group by top_pct and compute mean across datasets
        df_combined = pd.DataFrame({
            'top_pct': all_top_pcts,
            'mean': all_means,
            'ci_lower': all_ci_lower,
            'ci_upper': all_ci_upper,
        })
        
        df_grouped = df_combined.groupby('top_pct').agg({
            'mean': 'mean',
            'ci_lower': 'mean',
            'ci_upper': 'mean',
        }).reset_index()
        
        df_grouped = df_grouped.sort_values('top_pct')
        
        top_pcts = df_grouped['top_pct'].values
        means = df_grouped['mean'].values
        ci_lower = df_grouped['ci_lower'].values
        ci_upper = df_grouped['ci_upper'].values
        
        # Plot line with error bars
        ax.plot(
            top_pcts,
            means,
            marker='o',
            markersize=10,
            linewidth=3,
            color=color,
            label=name,
            alpha=0.9,
            zorder=3,
        )
        
        # Error bars
        ax.errorbar(
            top_pcts,
            means,
            yerr=[means - ci_lower, ci_upper - means],
            fmt='none',
            color=color,
            alpha=0.6,
            capsize=5,
            capthick=2,
            elinewidth=2,
            zorder=2,
        )
        
        # Fill area
        ax.fill_between(
            top_pcts,
            ci_lower,
            ci_upper,
            color=color,
            alpha=0.2,
            zorder=1,
        )
    
    # Styling
    ax.set_xlabel('Training Set Size (% Most Similar Perturbations)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Performance (Pearson r)', fontweight='bold', fontsize=14)
    ax.set_title(
        'LSFT Exploding Performance Curve:\nLocal Predictability Skyrockets with Top 1-5% Similar Perturbations',
        fontweight='bold',
        fontsize=16,
        pad=20,
    )
    ax.set_xlim([0.5, 10.5])
    ax.set_ylim([0.5, 1.0])
    ax.set_xticks([1, 5, 10])
    ax.set_xticklabels(['1%', '5%', '10%'])
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.axhline(y=0.9, color='gray', linestyle=':', linewidth=2, alpha=0.6, label='r = 0.9 threshold')
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Created: {output_path.name}")


def main():
    """Generate LSFT Exploding Performance Curve figures."""
    base_dir = Path(__file__).parent.parent.parent
    results_dir = base_dir / "results" / "goal_3_prediction" / "lsft_resampling"
    output_dir = base_dir / "publication_figures"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("Creating LSFT Exploding Performance Curve Figures")
    print("=" * 70)
    print()
    
    # Key baselines to highlight
    key_baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
    
    # 1. Multi-panel version (one panel per dataset)
    print("1. Creating multi-panel version (one panel per dataset)...")
    output_path = output_dir / "lsft_exploding_performance_curve_multi_panel.png"
    create_exploding_performance_curve(
        results_dir=results_dir,
        output_path=output_path,
        baselines=key_baselines,
        datasets=['adamson', 'k562', 'rpe1'],
    )
    
    # 2. Single-panel version (all datasets combined)
    print("\n2. Creating single-panel version (all datasets combined)...")
    output_path = output_dir / "lsft_exploding_performance_curve_single_panel.png"
    create_single_panel_exploding_curve(
        results_dir=results_dir,
        output_path=output_path,
        baselines=key_baselines,
        datasets=['adamson', 'k562', 'rpe1'],
    )
    
    print()
    print("=" * 70)
    print("✅ All figures generated successfully!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  - lsft_exploding_performance_curve_multi_panel.png")
    print("  - lsft_exploding_performance_curve_single_panel.png")


if __name__ == "__main__":
    main()


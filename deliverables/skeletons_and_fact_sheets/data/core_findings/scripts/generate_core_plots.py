#!/usr/bin/env python3
"""
Generate Core Finding Plots for Publication/Poster

This script creates polished versions of the 3 key plots:
1. Curvature Sweep (multiple alternatives)
2. Baseline Comparison (all datasets)
3. Similarity vs Performance (cleaned up)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Setup paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "manifold_law_diagnostics"
OUTPUT_DIR = SCRIPT_DIR

# Core baselines only (exclude cross-dataset)
CORE_BASELINES = [
    'lpm_selftrained',
    'lpm_randomGeneEmb',
    'lpm_randomPertEmb',
    'lpm_scgptGeneEmb',
    'lpm_scFoundationGeneEmb',
    'lpm_gearsPertEmb',
]

COLORS = {
    'lpm_selftrained': '#2ecc71',      # Green
    'lpm_randomGeneEmb': '#95a5a6',    # Gray
    'lpm_randomPertEmb': '#e74c3c',    # Red
    'lpm_scgptGeneEmb': '#3498db',     # Blue
    'lpm_scFoundationGeneEmb': '#9b59b6',  # Purple
    'lpm_gearsPertEmb': '#f39c12',     # Orange
}

LABELS = {
    'lpm_selftrained': 'PCA (Self-trained)',
    'lpm_randomGeneEmb': 'Random Gene',
    'lpm_randomPertEmb': 'Random Pert.',
    'lpm_scgptGeneEmb': 'scGPT',
    'lpm_scFoundationGeneEmb': 'scFoundation',
    'lpm_gearsPertEmb': 'GEARS',
}

DATASET_INFO = {
    'adamson': {'label': 'Adamson (Easy)', 'n': 12, 'difficulty': 'easy'},
    'k562': {'label': 'K562 (Hard)', 'n': 163, 'difficulty': 'hard'},
    'rpe1': {'label': 'RPE1 (Medium)', 'n': 231, 'difficulty': 'medium'},
}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11


def normalize_baseline(name):
    """Normalize baseline name."""
    if pd.isna(name):
        return None
    for prefix in ['adamson_', 'k562_', 'rpe1_']:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name if name in CORE_BASELINES else None


def load_curvature_data():
    """Load all curvature sweep data."""
    epic1_dir = RESULTS_DIR / "epic1_curvature"
    all_data = []
    
    for csv_file in epic1_dir.glob("lsft_k_sweep_*.csv"):
        parts = csv_file.stem.replace("lsft_k_sweep_", "").split("_", 1)
        if len(parts) != 2:
            continue
        dataset, baseline = parts
        baseline = normalize_baseline(baseline)
        if baseline is None:
            continue
        
        df = pd.read_csv(csv_file)
        df['dataset'] = dataset
        df['baseline'] = baseline
        all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)


# =============================================================================
# PLOT 1: CURVATURE SWEEP ALTERNATIVES
# =============================================================================

def plot_curvature_style_A_stratified(df, output_dir):
    """Style A: 3-panel stratified by dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    for ax, dataset in zip(axes, ['adamson', 'k562', 'rpe1']):
        subset = df[df['dataset'] == dataset]
        
        for baseline in CORE_BASELINES:
            bl_data = subset[subset['baseline'] == baseline]
            if len(bl_data) == 0:
                continue
            
            k_means = bl_data.groupby('k')['performance_local_pearson_r'].agg(['mean', 'std'])
            ax.plot(k_means.index, k_means['mean'], 'o-',
                   color=COLORS[baseline], label=LABELS[baseline],
                   linewidth=2, markersize=5)
            ax.fill_between(k_means.index,
                           k_means['mean'] - k_means['std'],
                           k_means['mean'] + k_means['std'],
                           alpha=0.15, color=COLORS[baseline])
        
        info = DATASET_INFO[dataset]
        ax.set_title(f"{info['label']}\n(n={info['n']} perturbations)", fontsize=12, fontweight='bold')
        ax.set_xlabel('Neighborhood Size (k)')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    axes[0].set_ylabel('LSFT Pearson r', fontsize=12)
    axes[2].legend(loc='lower right', fontsize=9)
    
    fig.suptitle('Curvature Sweep: Local Prediction Accuracy by Dataset Difficulty',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path = output_dir / "curvature_stratified_3panel.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {output_path.name}")
    return output_path


def plot_curvature_style_B_easy_vs_hard(df, output_dir):
    """Style B: 2-panel - Easy (Adamson) vs Hard (K562)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    for ax, dataset in zip(axes, ['adamson', 'k562']):
        subset = df[df['dataset'] == dataset]
        
        for baseline in CORE_BASELINES:
            bl_data = subset[subset['baseline'] == baseline]
            if len(bl_data) == 0:
                continue
            
            k_means = bl_data.groupby('k')['performance_local_pearson_r'].mean()
            ax.plot(k_means.index, k_means.values, 'o-',
                   color=COLORS[baseline], label=LABELS[baseline],
                   linewidth=2.5, markersize=7)
        
        info = DATASET_INFO[dataset]
        difficulty_color = '#27ae60' if dataset == 'adamson' else '#e74c3c'
        ax.set_title(f"{'🟢 EASY' if dataset == 'adamson' else '🔴 HARD'}: {info['label']}",
                    fontsize=13, fontweight='bold', color=difficulty_color)
        ax.set_xlabel('Neighborhood Size (k)', fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    axes[0].set_ylabel('LSFT Pearson r', fontsize=12)
    axes[1].legend(loc='lower right', fontsize=10)
    
    fig.suptitle('The Manifold Law Holds Across Dataset Difficulty',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path = output_dir / "curvature_easy_vs_hard.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {output_path.name}")
    return output_path


def plot_curvature_style_C_single_clean(df, output_dir):
    """Style C: Single panel, all datasets averaged, clean design."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for baseline in CORE_BASELINES:
        bl_data = df[df['baseline'] == baseline]
        if len(bl_data) == 0:
            continue
        
        # Average across datasets
        k_means = bl_data.groupby('k')['performance_local_pearson_r'].agg(['mean', 'sem'])
        
        ax.plot(k_means.index, k_means['mean'], 'o-',
               color=COLORS[baseline], label=LABELS[baseline],
               linewidth=2.5, markersize=8)
        ax.fill_between(k_means.index,
                       k_means['mean'] - k_means['sem'],
                       k_means['mean'] + k_means['sem'],
                       alpha=0.2, color=COLORS[baseline])
    
    ax.set_xlabel('Neighborhood Size (k)', fontsize=12)
    ax.set_ylabel('LSFT Pearson r', fontsize=12)
    ax.set_title('Curvature Sweep: How Accuracy Varies with Neighborhood Size',
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
    
    # Add annotation
    ax.annotate('Optimal: k=5-10\nLocal manifold is smooth',
               xy=(7, 0.85), fontsize=10, style='italic',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    output_path = output_dir / "curvature_clean_single.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {output_path.name}")
    return output_path


def plot_curvature_style_D_lsft_lift(df, output_dir):
    """Style D: Show how LSFT lifts performance over baseline."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for baseline in CORE_BASELINES:
        bl_data = df[df['baseline'] == baseline]
        if len(bl_data) == 0 or 'improvement_pearson_r' not in bl_data.columns:
            continue
        
        # Compute lift: local_r - baseline_r
        k_means = bl_data.groupby('k')['improvement_pearson_r'].agg(['mean', 'sem'])
        
        ax.plot(k_means.index, k_means['mean'], 'o-',
               color=COLORS[baseline], label=LABELS[baseline],
               linewidth=2.5, markersize=8)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Neighborhood Size (k)', fontsize=12)
    ax.set_ylabel('LSFT Improvement (Δr = local - baseline)', fontsize=12)
    ax.set_title('How Much Does LSFT Improve Over Raw Similarity?',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    output_path = output_dir / "curvature_lsft_lift.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {output_path.name}")
    return output_path


# =============================================================================
# PLOT 2: BASELINE COMPARISON
# =============================================================================

def plot_baseline_comparison(df, output_dir):
    """Baseline comparison across all datasets."""
    
    # Get k=5 performance
    k5_data = df[df['k'] == 5].copy()
    
    # Aggregate
    summary = k5_data.groupby(['dataset', 'baseline'])['performance_local_pearson_r'].agg(['mean', 'std']).reset_index()
    summary.columns = ['dataset', 'baseline', 'mean_r', 'std_r']
    
    # Filter to core baselines
    summary = summary[summary['baseline'].isin(CORE_BASELINES)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(CORE_BASELINES))
    width = 0.25
    
    for i, dataset in enumerate(['adamson', 'k562', 'rpe1']):
        ds_data = summary[summary['dataset'] == dataset]
        
        # Ensure order matches CORE_BASELINES
        values = []
        errors = []
        for bl in CORE_BASELINES:
            row = ds_data[ds_data['baseline'] == bl]
            if len(row) > 0:
                values.append(row['mean_r'].iloc[0])
                errors.append(row['std_r'].iloc[0])
            else:
                values.append(0)
                errors.append(0)
        
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=DATASET_INFO[dataset]['label'],
                     yerr=errors, capsize=3, alpha=0.85)
    
    ax.set_xlabel('Embedding Method', fontsize=12)
    ax.set_ylabel('LSFT Pearson r (k=5)', fontsize=12)
    ax.set_title('Baseline Comparison Across Datasets (k=5 neighbors)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[bl] for bl in CORE_BASELINES], rotation=30, ha='right')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal line for reference
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    output_path = output_dir / "baseline_comparison_all_datasets.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {output_path.name}")
    return output_path


def plot_baseline_adamson_only(df, output_dir):
    """Baseline comparison - Adamson only (simplified)."""
    
    k5_adamson = df[(df['k'] == 5) & (df['dataset'] == 'adamson')].copy()
    
    summary = k5_adamson.groupby('baseline')['performance_local_pearson_r'].mean().sort_values(ascending=False)
    summary = summary[summary.index.isin(CORE_BASELINES)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [COLORS.get(bl, '#888888') for bl in summary.index]
    bars = ax.barh(range(len(summary)), summary.values, color=colors)
    
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels([LABELS.get(bl, bl) for bl in summary.index])
    ax.set_xlabel('LSFT Pearson r (k=5)', fontsize=12)
    ax.set_title('Adamson Dataset: Baseline Performance\n(Top 5% neighbors → Near-perfect prediction)',
                fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add values on bars
    for i, (bl, v) in enumerate(summary.items()):
        ax.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=11, fontweight='bold')
    
    # Highlight winner
    ax.get_children()[0].set_edgecolor('black')
    ax.get_children()[0].set_linewidth(2)
    
    plt.tight_layout()
    output_path = output_dir / "baseline_adamson_simple.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {output_path.name}")
    return output_path


# =============================================================================
# PLOT 3: SIMILARITY VS PERFORMANCE (CLEANED)
# =============================================================================

def plot_similarity_vs_performance_clean(df, output_dir):
    """Clean similarity vs performance plot - fixed k=5, PCA only, binned."""
    
    # Filter: k=5, PCA only
    pca_k5 = df[(df['k'] == 5) & (df['baseline'] == 'lpm_selftrained')].copy()
    
    if len(pca_k5) == 0 or 'local_mean_similarity' not in pca_k5.columns:
        print("⚠️ No data for similarity plot")
        return None
    
    x = pca_k5['local_mean_similarity']
    y = pca_k5['performance_local_pearson_r']
    
    # Compute correlation
    r, p = stats.pearsonr(x, y)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter with density coloring
    scatter = ax.scatter(x, y, c=pca_k5['dataset'].map({'adamson': 0, 'k562': 1, 'rpe1': 2}),
                        cmap='viridis', alpha=0.6, s=50, edgecolor='white', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(x, y, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p_line(x_line), 'r--', linewidth=2, label=f'Trend (r={r:.2f})')
    
    # Highlight high-similarity region
    ax.axvspan(0.8, 1.0, alpha=0.1, color='green', label='High similarity region')
    
    ax.set_xlabel('Mean Cosine Similarity to Neighbors', fontsize=12)
    ax.set_ylabel('LSFT Pearson r', fontsize=12)
    ax.set_title(f'Similarity Predicts Performance (PCA, k=5)\nCorrelation: r = {r:.3f}',
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Add color legend for datasets
    cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Adamson', 'K562', 'RPE1'])
    cbar.set_label('Dataset')
    
    plt.tight_layout()
    output_path = output_dir / "similarity_vs_performance_clean.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {output_path.name}")
    return output_path


def plot_similarity_vs_performance_binned(df, output_dir):
    """Binned version - aggregate by similarity percentile."""
    
    # All baselines, k=5
    k5_data = df[df['k'] == 5].copy()
    
    if 'local_mean_similarity' not in k5_data.columns:
        print("⚠️ No similarity data")
        return None
    
    # Create percentile bins
    k5_data['sim_percentile'] = pd.qcut(k5_data['local_mean_similarity'], 10, labels=False) * 10 + 5
    
    # Aggregate by baseline and percentile
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for baseline in ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']:
        bl_data = k5_data[k5_data['baseline'] == baseline]
        if len(bl_data) == 0:
            continue
        
        binned = bl_data.groupby('sim_percentile')['performance_local_pearson_r'].agg(['mean', 'sem'])
        
        ax.plot(binned.index, binned['mean'], 'o-',
               color=COLORS[baseline], label=LABELS[baseline],
               linewidth=2.5, markersize=8)
        ax.fill_between(binned.index,
                       binned['mean'] - binned['sem'],
                       binned['mean'] + binned['sem'],
                       alpha=0.2, color=COLORS[baseline])
    
    ax.set_xlabel('Similarity Percentile', fontsize=12)
    ax.set_ylabel('LSFT Pearson r', fontsize=12)
    ax.set_title('Higher Similarity → Better Predictions\n(Binned by similarity percentile, k=5)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Highlight top percentile
    ax.axvspan(85, 100, alpha=0.1, color='green')
    ax.annotate('Top 15%:\nNear-perfect\npredictions',
               xy=(92, 0.5), fontsize=10, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    output_path = output_dir / "similarity_vs_performance_binned.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {output_path.name}")
    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("GENERATING CORE FINDING PLOTS")
    print("=" * 60)
    print()
    
    # Load data
    print("Loading curvature data...")
    df = load_curvature_data()
    print(f"  Loaded {len(df)} rows from {df['dataset'].nunique()} datasets, {df['baseline'].nunique()} baselines")
    print()
    
    if len(df) == 0:
        print("❌ No data found!")
        return
    
    # Generate curvature plots
    print("Generating curvature sweep alternatives...")
    plot_curvature_style_A_stratified(df, OUTPUT_DIR)
    plot_curvature_style_B_easy_vs_hard(df, OUTPUT_DIR)
    plot_curvature_style_C_single_clean(df, OUTPUT_DIR)
    plot_curvature_style_D_lsft_lift(df, OUTPUT_DIR)
    print()
    
    # Generate baseline comparison
    print("Generating baseline comparison plots...")
    plot_baseline_comparison(df, OUTPUT_DIR)
    plot_baseline_adamson_only(df, OUTPUT_DIR)
    print()
    
    # Generate similarity plots
    print("Generating similarity vs performance plots...")
    plot_similarity_vs_performance_clean(df, OUTPUT_DIR)
    plot_similarity_vs_performance_binned(df, OUTPUT_DIR)
    print()
    
    print("=" * 60)
    print("✅ ALL PLOTS GENERATED!")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()


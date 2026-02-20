#!/usr/bin/env python3
"""
Cross-Epic Meta-Analysis for Manifold Law Diagnostic Suite

Analyzes correlations between metrics across all 5 epics to validate
that all geometric probes tell the same story.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "manifold_law_diagnostics"
OUTPUT_DIR = Path(__file__).parent / "cross_epic_analysis"

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11

BASELINE_LABELS = {
    'lpm_selftrained': 'PCA',
    'lpm_randomGeneEmb': 'Rand.Gene',
    'lpm_randomPertEmb': 'Rand.Pert',
    'lpm_scgptGeneEmb': 'scGPT',
    'lpm_scFoundationGeneEmb': 'scFound.',
    'lpm_gearsPertEmb': 'GEARS',
    'lpm_k562PertEmb': 'K562',
    'lpm_rpe1PertEmb': 'RPE1',
}


def load_epic_summaries() -> Dict[str, pd.DataFrame]:
    """Load summary CSVs from final_tables if available."""
    tables_dir = Path(__file__).parent / "final_tables"
    
    data = {}
    
    for epic in ['epic1_curvature_metrics', 'epic2_alignment_summary', 
                 'epic3_lipschitz_summary', 'epic4_flip_summary', 
                 'epic5_alignment_summary', 'baseline_summary']:
        path = tables_dir / f"{epic}.csv"
        if path.exists():
            data[epic] = pd.read_csv(path)
    
    return data


def compute_baseline_metrics(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate metrics per baseline across all epics."""
    all_baselines = set()
    for df in data.values():
        if 'baseline' in df.columns:
            all_baselines.update(df['baseline'].unique())
    
    results = []
    
    for baseline in all_baselines:
        row = {'baseline': baseline}
        
        # Epic 1: Peak accuracy
        if 'epic1_curvature_metrics' in data:
            e1 = data['epic1_curvature_metrics']
            subset = e1[e1['baseline'] == baseline]
            row['peak_r'] = subset['peak_r'].mean() if len(subset) > 0 else np.nan
            row['curvature_index'] = subset['curvature_index'].mean() if len(subset) > 0 else np.nan
        
        # Epic 3: Lipschitz
        if 'epic3_lipschitz_summary' in data:
            e3 = data['epic3_lipschitz_summary']
            subset = e3[e3['baseline'] == baseline]
            row['lipschitz'] = subset['lipschitz_constant'].mean() if len(subset) > 0 else np.nan
            row['baseline_r'] = subset['baseline_r'].mean() if len(subset) > 0 else np.nan
        
        # Epic 4: Flip rate
        if 'epic4_flip_summary' in data:
            e4 = data['epic4_flip_summary']
            subset = e4[e4['baseline'] == baseline]
            row['flip_rate'] = subset['mean_adversarial_rate'].mean() if len(subset) > 0 else np.nan
        
        # Epic 5: Tangent alignment
        if 'epic5_alignment_summary' in data:
            e5 = data['epic5_alignment_summary']
            subset = e5[e5['baseline'] == baseline]
            row['tas'] = subset['mean_tas'].mean() if len(subset) > 0 else np.nan
        
        results.append(row)
    
    return pd.DataFrame(results)


def compute_metric_correlations(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise correlations between epic metrics."""
    numeric_cols = [c for c in metrics_df.columns if c != 'baseline']
    
    corr_matrix = metrics_df[numeric_cols].corr(method='spearman')
    
    return corr_matrix


def plot_metric_correlation_heatmap(corr_matrix: pd.DataFrame, output_dir: Path):
    """Plot correlation heatmap between epic metrics."""
    if len(corr_matrix) == 0:
        return
    
    # Rename for display
    col_labels = {
        'peak_r': 'Peak r\n(E1)',
        'curvature_index': 'Curvature\n(E1)',
        'lipschitz': 'Lipschitz\n(E3)',
        'baseline_r': 'Baseline r\n(E3)',
        'flip_rate': 'Flip Rate\n(E4)',
        'tas': 'TAS\n(E5)',
    }
    
    plot_df = corr_matrix.copy()
    plot_df.columns = [col_labels.get(c, c) for c in plot_df.columns]
    plot_df.index = [col_labels.get(c, c) for c in plot_df.index]
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    mask = np.triu(np.ones_like(plot_df, dtype=bool), k=1)
    
    sns.heatmap(plot_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
               vmin=-1, vmax=1, ax=ax, mask=mask, square=True,
               cbar_kws={'label': 'Spearman ρ'})
    
    ax.set_title('Cross-Epic Metric Correlations\n(Do all probes tell the same story?)', 
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / "metric_correlation_heatmap.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_baseline_clustering(metrics_df: pd.DataFrame, output_dir: Path):
    """Create dendrogram showing baseline clustering across metrics."""
    if len(metrics_df) < 3:
        return
    
    # Prepare data
    numeric_cols = [c for c in metrics_df.columns if c != 'baseline']
    plot_data = metrics_df.dropna(subset=numeric_cols, how='all')
    
    if len(plot_data) < 3:
        return
    
    # Normalize each metric
    X = plot_data[numeric_cols].fillna(0).values
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    
    # Compute linkage
    linkage = hierarchy.linkage(X_norm, method='ward')
    
    # Create dendrogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = [BASELINE_LABELS.get(b, b) for b in plot_data['baseline']]
    
    hierarchy.dendrogram(linkage, labels=labels, ax=ax, leaf_rotation=45,
                        color_threshold=0.7*max(linkage[:,2]))
    
    ax.set_title('Baseline Clustering Across All Epic Metrics', fontsize=12, fontweight='bold')
    ax.set_xlabel('Baseline')
    ax.set_ylabel('Ward Distance')
    
    plt.tight_layout()
    output_path = output_dir / "baseline_clustering_dendrogram.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_pca_of_baselines(metrics_df: pd.DataFrame, output_dir: Path):
    """Create PCA projection of baselines across all metrics."""
    if len(metrics_df) < 3:
        return
    
    from sklearn.decomposition import PCA
    
    numeric_cols = [c for c in metrics_df.columns if c != 'baseline']
    plot_data = metrics_df.dropna(subset=numeric_cols, how='all')
    
    if len(plot_data) < 3:
        return
    
    X = plot_data[numeric_cols].fillna(0).values
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_norm)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#0077BB', '#33BBEE', '#EE7733', '#CC3311', '#009988', 
              '#EE3377', '#BBBBBB', '#666666']
    
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        baseline = row['baseline']
        label = BASELINE_LABELS.get(baseline, baseline)
        color = colors[i % len(colors)]
        
        ax.scatter(X_pca[i, 0], X_pca[i, 1], s=200, c=color, label=label, 
                  edgecolor='black', linewidth=1, alpha=0.8)
        ax.annotate(label, (X_pca[i, 0], X_pca[i, 1]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    ax.set_title('PCA of Baselines Across All Epic Metrics', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_path = output_dir / "baseline_pca_projection.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_scatter_tas_vs_accuracy(metrics_df: pd.DataFrame, output_dir: Path):
    """Scatter plot: Tangent Alignment vs LSFT Accuracy."""
    if 'tas' not in metrics_df.columns or 'peak_r' not in metrics_df.columns:
        return
    
    plot_data = metrics_df.dropna(subset=['tas', 'peak_r'])
    
    if len(plot_data) < 2:
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        baseline = row['baseline']
        label = BASELINE_LABELS.get(baseline, baseline)
        
        colors = ['#0077BB', '#33BBEE', '#EE7733', '#CC3311', '#009988', 
                  '#EE3377', '#BBBBBB', '#666666']
        color = colors[i % len(colors)]
        
        ax.scatter(row['tas'], row['peak_r'], s=150, c=color, 
                  label=label, edgecolor='black', linewidth=1, alpha=0.8)
    
    # Compute correlation
    if len(plot_data) > 2:
        r, p = stats.spearmanr(plot_data['tas'], plot_data['peak_r'])
        ax.text(0.05, 0.95, f'ρ = {r:.3f} (p = {p:.3f})', transform=ax.transAxes,
               fontsize=10, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Tangent Alignment Score (TAS)', fontsize=11)
    ax.set_ylabel('Peak LSFT Pearson r', fontsize=11)
    ax.set_title('Tangent Alignment Predicts LSFT Accuracy', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / "scatter_tas_vs_accuracy.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_scatter_flip_vs_curvature(metrics_df: pd.DataFrame, output_dir: Path):
    """Scatter plot: Flip Rate vs Curvature Index."""
    if 'flip_rate' not in metrics_df.columns or 'curvature_index' not in metrics_df.columns:
        return
    
    plot_data = metrics_df.dropna(subset=['flip_rate', 'curvature_index'])
    
    if len(plot_data) < 2:
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        baseline = row['baseline']
        label = BASELINE_LABELS.get(baseline, baseline)
        
        colors = ['#0077BB', '#33BBEE', '#EE7733', '#CC3311', '#009988', 
                  '#EE3377', '#BBBBBB', '#666666']
        color = colors[i % len(colors)]
        
        ax.scatter(row['flip_rate'], row['curvature_index'], s=150, c=color, 
                  label=label, edgecolor='black', linewidth=1, alpha=0.8)
    
    # Compute correlation
    if len(plot_data) > 2:
        r, p = stats.spearmanr(plot_data['flip_rate'], plot_data['curvature_index'])
        ax.text(0.05, 0.95, f'ρ = {r:.3f} (p = {p:.3f})', transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Adversarial Flip Rate', fontsize=11)
    ax.set_ylabel('Curvature Index', fontsize=11)
    ax.set_title('Direction Flips Correlate with Manifold Curvature', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / "scatter_flip_vs_curvature.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved: {output_path}")


def main():
    print("=" * 60)
    print("CROSS-EPIC META-ANALYSIS")
    print("=" * 60)
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading epic summaries...")
    data = load_epic_summaries()
    
    if len(data) == 0:
        print("⚠️  No summary tables found. Run generate_publication_reports.py first.")
        return
    
    print(f"Loaded {len(data)} summary tables")
    for name, df in data.items():
        print(f"  - {name}: {len(df)} rows")
    print()
    
    # Compute baseline metrics
    print("Computing baseline metrics...")
    if 'baseline_summary' in data:
        metrics_df = data['baseline_summary']
    else:
        metrics_df = compute_baseline_metrics(data)
    
    if len(metrics_df) == 0:
        print("⚠️  No baseline metrics computed")
        return
    
    print(f"  {len(metrics_df)} baselines with metrics")
    metrics_df.to_csv(OUTPUT_DIR / "cross_epic_baseline_metrics.csv", index=False)
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # 1. Metric correlation heatmap
    corr_matrix = compute_metric_correlations(metrics_df)
    if len(corr_matrix) > 0:
        plot_metric_correlation_heatmap(corr_matrix, OUTPUT_DIR)
    
    # 2. Baseline clustering dendrogram
    try:
        plot_baseline_clustering(metrics_df, OUTPUT_DIR)
    except Exception as e:
        print(f"  ⚠️ Clustering failed: {e}")
    
    # 3. PCA projection
    try:
        plot_pca_of_baselines(metrics_df, OUTPUT_DIR)
    except ImportError:
        print("  ⚠️ sklearn not available for PCA")
    except Exception as e:
        print(f"  ⚠️ PCA failed: {e}")
    
    # 4. Scatter: TAS vs Accuracy
    plot_scatter_tas_vs_accuracy(metrics_df, OUTPUT_DIR)
    
    # 5. Scatter: Flip rate vs Curvature
    plot_scatter_flip_vs_curvature(metrics_df, OUTPUT_DIR)
    
    print()
    print("=" * 60)
    print("✅ CROSS-EPIC ANALYSIS COMPLETE!")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()


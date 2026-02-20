#!/usr/bin/env python3
"""
Generate Publication-Ready Reports for Manifold Law Diagnostic Suite

This script creates:
1. Per-epic summary reports and figures
2. Cross-epic meta-analysis
3. Final data tables
4. Poster-ready figures
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "manifold_law_diagnostics"
OUTPUT_DIR = Path(__file__).parent

# Style configuration
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

# Color palette for baselines
BASELINE_COLORS = {
    'lpm_selftrained': '#2E86AB',      # Blue - best performer
    'lpm_randomGeneEmb': '#A23B72',    # Pink
    'lpm_randomPertEmb': '#F18F01',    # Orange
    'lpm_scgptGeneEmb': '#C73E1D',     # Red
    'lpm_scFoundationGeneEmb': '#3B1F2B',  # Dark
    'lpm_gearsPertEmb': '#44AF69',     # Green
    'lpm_k562PertEmb': '#6B4E71',      # Purple
    'lpm_rpe1PertEmb': '#9BC53D',      # Lime
}

BASELINE_LABELS = {
    'lpm_selftrained': 'PCA (Self-trained)',
    'lpm_randomGeneEmb': 'Random Gene Emb.',
    'lpm_randomPertEmb': 'Random Pert. Emb.',
    'lpm_scgptGeneEmb': 'scGPT Gene Emb.',
    'lpm_scFoundationGeneEmb': 'scFoundation Gene Emb.',
    'lpm_gearsPertEmb': 'GEARS (GO Graph)',
    'lpm_k562PertEmb': 'K562 Cross-Dataset',
    'lpm_rpe1PertEmb': 'RPE1 Cross-Dataset',
}

DATASET_LABELS = {
    'adamson': 'Adamson',
    'k562': 'K562',
    'rpe1': 'RPE1',
}

# ==============================================================================
# DATA LOADING FUNCTIONS
# ==============================================================================

def load_epic1_data() -> pd.DataFrame:
    """Load all Epic 1 (Curvature Sweep) data."""
    epic1_dir = RESULTS_DIR / "epic1_curvature"
    all_data = []
    
    # Load k-sweep files
    for file in epic1_dir.glob("lsft_k_sweep_*.csv"):
        try:
            df = pd.read_csv(file)
            if len(df) == 0:
                continue
            parts = file.stem.replace("lsft_k_sweep_", "").split("_", 1)
            if len(parts) == 2:
                df["dataset"] = parts[0]
                df["baseline"] = parts[1]
                all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file.name}: {e}")
    
    # Also load summary files
    for file in epic1_dir.glob("curvature_sweep_summary_*.csv"):
        try:
            df = pd.read_csv(file)
            if len(df) == 0:
                continue
            parts = file.stem.replace("curvature_sweep_summary_", "").split("_", 1)
            if len(parts) == 2:
                df["dataset"] = parts[0]
                df["baseline"] = parts[1]
                df["source"] = "summary"
                all_data.append(df)
        except Exception as e:
            pass
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def load_epic2_data() -> pd.DataFrame:
    """Load all Epic 2 (Mechanism Ablation) data."""
    epic2_dir = RESULTS_DIR / "epic2_mechanism_ablation"
    all_data = []
    
    for file in epic2_dir.glob("mechanism_ablation_*.csv"):
        if file.parent != epic2_dir:  # Skip subdirectories
            continue
        try:
            df = pd.read_csv(file)
            if len(df) == 0:
                continue
            parts = file.stem.replace("mechanism_ablation_", "").split("_", 1)
            if len(parts) == 2:
                df["dataset"] = parts[0]
                df["baseline"] = parts[1]
                all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file.name}: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def load_epic3_data() -> pd.DataFrame:
    """Load all Epic 3 (Noise Injection) data."""
    epic3_dir = RESULTS_DIR / "epic3_noise_injection"
    all_data = []
    
    for file in epic3_dir.glob("noise_injection_*.csv"):
        try:
            df = pd.read_csv(file)
            if len(df) == 0:
                continue
            parts = file.stem.replace("noise_injection_", "").split("_", 1)
            if len(parts) == 2:
                df["dataset"] = parts[0]
                df["baseline"] = parts[1]
                all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file.name}: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def load_epic4_data() -> pd.DataFrame:
    """Load all Epic 4 (Direction-Flip Probe) data."""
    epic4_dir = RESULTS_DIR / "epic4_direction_flip"
    all_data = []
    
    for file in epic4_dir.glob("direction_flip_*.csv"):
        try:
            df = pd.read_csv(file)
            if len(df) == 0:
                continue
            parts = file.stem.replace("direction_flip_", "").split("_", 1)
            if len(parts) == 2:
                df["dataset"] = parts[0]
                df["baseline"] = parts[1]
                all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file.name}: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def load_epic5_data() -> pd.DataFrame:
    """Load all Epic 5 (Tangent Alignment) data."""
    epic5_dir = RESULTS_DIR / "epic5_tangent_alignment"
    all_data = []
    
    for file in epic5_dir.glob("tangent_alignment_*.csv"):
        try:
            df = pd.read_csv(file)
            if len(df) == 0:
                continue
            parts = file.stem.replace("tangent_alignment_", "").split("_", 1)
            if len(parts) == 2:
                df["dataset"] = parts[0]
                df["baseline"] = parts[1]
                all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file.name}: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


# ==============================================================================
# EPIC 1: CURVATURE SWEEP ANALYSIS
# ==============================================================================

def analyze_epic1_curvature(df: pd.DataFrame) -> pd.DataFrame:
    """Compute curvature metrics per baseline × dataset."""
    if len(df) == 0:
        return pd.DataFrame()
    
    # Get r column
    r_col = None
    for col in ['performance_local_pearson_r', 'mean_r', 'pearson_r']:
        if col in df.columns:
            r_col = col
            break
    
    if r_col is None:
        print("No r column found in Epic 1 data")
        return pd.DataFrame()
    
    results = []
    
    for (dataset, baseline), group in df.groupby(['dataset', 'baseline']):
        if 'k' not in group.columns:
            continue
            
        # Get mean r per k
        k_stats = group.groupby('k')[r_col].agg(['mean', 'std', 'count']).reset_index()
        k_stats.columns = ['k', 'mean_r', 'std_r', 'n']
        k_stats = k_stats.sort_values('k')
        
        if len(k_stats) < 2:
            continue
        
        # Find peak k and r
        peak_idx = k_stats['mean_r'].idxmax()
        peak_k = k_stats.loc[peak_idx, 'k']
        peak_r = k_stats.loc[peak_idx, 'mean_r']
        
        # Compute curvature index (second derivative estimate)
        r_values = k_stats['mean_r'].values
        if len(r_values) >= 3:
            curvature = np.diff(r_values, 2).mean() if len(r_values) > 2 else 0
        else:
            curvature = 0
        
        # Check if U-shaped (drops then rises)
        mid_idx = len(r_values) // 2
        is_u_shaped = (r_values[:mid_idx].mean() > r_values[mid_idx:].mean() * 1.05 
                      if mid_idx > 0 and len(r_values) > mid_idx else False)
        
        # Stability (std of r across k)
        stability = k_stats['mean_r'].std()
        
        results.append({
            'dataset': dataset,
            'baseline': baseline,
            'peak_k': peak_k,
            'peak_r': peak_r,
            'mean_r': k_stats['mean_r'].mean(),
            'curvature_index': curvature,
            'is_u_shaped': is_u_shaped,
            'stability': stability,
            'n_k_values': len(k_stats),
        })
    
    return pd.DataFrame(results)


def plot_epic1_curvature_grid(df: pd.DataFrame, output_dir: Path):
    """Create curvature sweep grid plot."""
    if len(df) == 0:
        print("No Epic 1 data to plot")
        return
    
    r_col = None
    for col in ['performance_local_pearson_r', 'mean_r', 'pearson_r']:
        if col in df.columns:
            r_col = col
            break
    
    if r_col is None:
        return
    
    datasets = sorted(df['dataset'].unique())
    baselines = sorted(df['baseline'].unique())
    
    fig, axes = plt.subplots(len(baselines), len(datasets), 
                            figsize=(4*len(datasets), 3*len(baselines)))
    
    if len(baselines) == 1:
        axes = axes.reshape(1, -1)
    if len(datasets) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, baseline in enumerate(baselines):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            
            subset = df[(df['baseline'] == baseline) & (df['dataset'] == dataset)]
            if len(subset) == 0 or 'k' not in subset.columns:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{BASELINE_LABELS.get(baseline, baseline)}\n{DATASET_LABELS.get(dataset, dataset)}', fontsize=10)
                continue
            
            # Aggregate by k
            k_stats = subset.groupby('k')[r_col].agg(['mean', 'std']).reset_index()
            k_stats.columns = ['k', 'mean_r', 'std_r']
            k_stats = k_stats.sort_values('k')
            
            color = BASELINE_COLORS.get(baseline, '#333333')
            
            if 'std_r' in k_stats.columns and not k_stats['std_r'].isna().all():
                ax.errorbar(k_stats['k'], k_stats['mean_r'], yerr=k_stats['std_r'],
                           fmt='o-', color=color, linewidth=2, markersize=5, capsize=3)
            else:
                ax.plot(k_stats['k'], k_stats['mean_r'], 'o-', color=color, 
                       linewidth=2, markersize=5)
            
            ax.set_xscale('log')
            ax.set_xlabel('k', fontsize=9)
            ax.set_ylabel('Pearson r', fontsize=9)
            ax.set_title(f'{BASELINE_LABELS.get(baseline, baseline)}\n{DATASET_LABELS.get(dataset, dataset)}', fontsize=10)
            ax.set_ylim([0, 1.0])
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "epic1_curvature_sweep_grid.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_epic1_curvature_heatmap(metrics_df: pd.DataFrame, output_dir: Path):
    """Create curvature heatmap across baselines."""
    if len(metrics_df) == 0:
        return
    
    # Pivot for heatmap
    pivot = metrics_df.pivot_table(index='baseline', columns='dataset', values='peak_r', aggfunc='mean')
    
    # Reorder baselines
    baseline_order = ['lpm_selftrained', 'lpm_gearsPertEmb', 'lpm_scgptGeneEmb', 
                     'lpm_scFoundationGeneEmb', 'lpm_randomGeneEmb', 'lpm_randomPertEmb',
                     'lpm_k562PertEmb', 'lpm_rpe1PertEmb']
    pivot = pivot.reindex([b for b in baseline_order if b in pivot.index])
    
    # Rename for display
    pivot.index = [BASELINE_LABELS.get(b, b) for b in pivot.index]
    pivot.columns = [DATASET_LABELS.get(d, d) for d in pivot.columns]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
               vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Peak Pearson r'})
    ax.set_title('Epic 1: Peak LSFT Accuracy by Baseline × Dataset', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    plt.tight_layout()
    output_path = output_dir / "epic1_curvature_heatmap.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved: {output_path}")


# ==============================================================================
# EPIC 2: MECHANISM ABLATION ANALYSIS
# ==============================================================================

def analyze_epic2_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mechanism ablation metrics."""
    if len(df) == 0:
        return pd.DataFrame()
    
    results = []
    
    for (dataset, baseline), group in df.groupby(['dataset', 'baseline']):
        # Get original r
        r_col = 'original_pearson_r' if 'original_pearson_r' in group.columns else None
        if r_col is None:
            continue
        
        mean_original_r = group[r_col].mean()
        
        # Get functional class breakdown if available
        n_classes = group['functional_class'].nunique() if 'functional_class' in group.columns else 0
        
        # Get delta_r if available
        delta_r = group['delta_r'].mean() if 'delta_r' in group.columns and group['delta_r'].notna().any() else np.nan
        
        results.append({
            'dataset': dataset,
            'baseline': baseline,
            'mean_original_r': mean_original_r,
            'delta_r': delta_r,
            'n_functional_classes': n_classes,
            'n_perturbations': len(group),
        })
    
    return pd.DataFrame(results)


def plot_epic2_baseline_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot original_r comparison across baselines."""
    if len(df) == 0:
        return
    
    r_col = 'original_pearson_r' if 'original_pearson_r' in df.columns else None
    if r_col is None:
        return
    
    # Aggregate by baseline × dataset
    agg = df.groupby(['dataset', 'baseline'])[r_col].agg(['mean', 'std']).reset_index()
    
    datasets = sorted(agg['dataset'].unique())
    baselines = sorted(agg['baseline'].unique())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(baselines))
    width = 0.25
    
    for i, dataset in enumerate(datasets):
        subset = agg[agg['dataset'] == dataset]
        means = [subset[subset['baseline'] == b]['mean'].values[0] if len(subset[subset['baseline'] == b]) > 0 else 0 
                for b in baselines]
        stds = [subset[subset['baseline'] == b]['std'].values[0] if len(subset[subset['baseline'] == b]) > 0 else 0 
               for b in baselines]
        
        ax.bar(x + i*width, means, width, yerr=stds, label=DATASET_LABELS.get(dataset, dataset),
              capsize=3, alpha=0.8)
    
    ax.set_xlabel('Baseline')
    ax.set_ylabel('Original LSFT Pearson r')
    ax.set_title('Epic 2: Mechanism Ablation - Original LSFT Accuracy', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([BASELINE_LABELS.get(b, b) for b in baselines], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "epic2_baseline_comparison.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved: {output_path}")


# ==============================================================================
# EPIC 3: NOISE INJECTION ANALYSIS
# ==============================================================================

def analyze_epic3_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Compute noise stability metrics including Lipschitz constants."""
    if len(df) == 0:
        return pd.DataFrame()
    
    results = []
    
    for (dataset, baseline), group in df.groupby(['dataset', 'baseline']):
        if 'noise_level' not in group.columns or 'mean_r' not in group.columns:
            continue
        
        # Get baseline (noise=0) data
        baseline_data = group[group['noise_level'] == 0.0]
        noisy_data = group[group['noise_level'] > 0]
        
        if len(baseline_data) == 0:
            continue
        
        baseline_r = baseline_data['mean_r'].mean()
        
        # Compute Lipschitz constant
        if len(noisy_data) > 0 and noisy_data['mean_r'].notna().any():
            noisy_data = noisy_data[noisy_data['mean_r'].notna()]
            if len(noisy_data) > 0:
                delta_r = np.abs(noisy_data['mean_r'].values - baseline_r)
                noise_levels = noisy_data['noise_level'].values
                sensitivity = delta_r / noise_levels
                lipschitz = np.max(sensitivity) if len(sensitivity) > 0 else np.nan
                mean_sensitivity = np.mean(sensitivity) if len(sensitivity) > 0 else np.nan
            else:
                lipschitz = np.nan
                mean_sensitivity = np.nan
        else:
            lipschitz = np.nan
            mean_sensitivity = np.nan
        
        # Classify stability
        if pd.isna(lipschitz):
            stability_class = 'Unknown'
        elif lipschitz < 0.5:
            stability_class = 'Robust'
        elif lipschitz < 1.0:
            stability_class = 'Semi-robust'
        elif lipschitz < 2.0:
            stability_class = 'Fragile'
        else:
            stability_class = 'Hyper-fragile'
        
        results.append({
            'dataset': dataset,
            'baseline': baseline,
            'baseline_r': baseline_r,
            'lipschitz_constant': lipschitz,
            'mean_sensitivity': mean_sensitivity,
            'stability_class': stability_class,
            'n_noise_levels': noisy_data['noise_level'].nunique() if len(noisy_data) > 0 else 0,
        })
    
    return pd.DataFrame(results)


def plot_epic3_lipschitz_barplot(metrics_df: pd.DataFrame, output_dir: Path):
    """Create Lipschitz constant barplot."""
    if len(metrics_df) == 0:
        return
    
    # Aggregate by baseline
    agg = metrics_df.groupby('baseline')['lipschitz_constant'].agg(['mean', 'std']).reset_index()
    agg = agg.dropna(subset=['mean'])
    
    if len(agg) == 0:
        print("No valid Lipschitz data to plot")
        return
    
    # Sort by Lipschitz (lower = more stable = better)
    agg = agg.sort_values('mean')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [BASELINE_COLORS.get(b, '#333333') for b in agg['baseline']]
    bars = ax.barh(range(len(agg)), agg['mean'], xerr=agg['std'], color=colors, 
                   capsize=3, alpha=0.8)
    
    ax.set_yticks(range(len(agg)))
    ax.set_yticklabels([BASELINE_LABELS.get(b, b) for b in agg['baseline']])
    ax.set_xlabel('Lipschitz Constant (lower = more stable)')
    ax.set_title('Epic 3: Noise Stability - Lipschitz Constants', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Fragility threshold')
    ax.legend()
    
    plt.tight_layout()
    output_path = output_dir / "epic3_lipschitz_barplot.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved: {output_path}")


# ==============================================================================
# EPIC 4: DIRECTION-FLIP ANALYSIS
# ==============================================================================

def analyze_epic4_flips(df: pd.DataFrame) -> pd.DataFrame:
    """Compute direction-flip metrics."""
    if len(df) == 0:
        return pd.DataFrame()
    
    results = []
    
    for (dataset, baseline), group in df.groupby(['dataset', 'baseline']):
        n_adversarial = group['n_adversarial'].sum() if 'n_adversarial' in group.columns else 0
        n_top5 = group['n_top_5pct'].sum() if 'n_top_5pct' in group.columns else 0
        
        adversarial_rate = group['adversarial_rate'].mean() if 'adversarial_rate' in group.columns else np.nan
        
        results.append({
            'dataset': dataset,
            'baseline': baseline,
            'n_adversarial_pairs': int(n_adversarial),
            'n_top5_neighbors': int(n_top5),
            'mean_adversarial_rate': adversarial_rate,
            'n_test_perturbations': len(group),
        })
    
    return pd.DataFrame(results)


def plot_epic4_flip_rates(metrics_df: pd.DataFrame, output_dir: Path):
    """Create direction-flip rate barplot."""
    if len(metrics_df) == 0:
        return
    
    # Aggregate by baseline
    agg = metrics_df.groupby('baseline')['mean_adversarial_rate'].agg(['mean', 'std']).reset_index()
    agg = agg.dropna(subset=['mean'])
    
    if len(agg) == 0:
        return
    
    # Sort by flip rate (lower = better)
    agg = agg.sort_values('mean')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [BASELINE_COLORS.get(b, '#333333') for b in agg['baseline']]
    bars = ax.barh(range(len(agg)), agg['mean'], xerr=agg['std'], color=colors, 
                   capsize=3, alpha=0.8)
    
    ax.set_yticks(range(len(agg)))
    ax.set_yticklabels([BASELINE_LABELS.get(b, b) for b in agg['baseline']])
    ax.set_xlabel('Mean Adversarial Rate (lower = better)')
    ax.set_title('Epic 4: Direction-Flip Probe - Adversarial Neighbor Rates', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = output_dir / "epic4_flip_rates_barplot.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved: {output_path}")


# ==============================================================================
# EPIC 5: TANGENT ALIGNMENT ANALYSIS
# ==============================================================================

def analyze_epic5_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """Compute tangent alignment metrics."""
    if len(df) == 0:
        return pd.DataFrame()
    
    results = []
    
    for (dataset, baseline), group in df.groupby(['dataset', 'baseline']):
        tas_col = None
        for col in ['tangent_alignment_score', 'tas', 'alignment_score']:
            if col in group.columns:
                tas_col = col
                break
        
        if tas_col is None:
            continue
        
        mean_tas = group[tas_col].mean()
        std_tas = group[tas_col].std()
        
        results.append({
            'dataset': dataset,
            'baseline': baseline,
            'mean_tas': mean_tas,
            'std_tas': std_tas,
            'n_perturbations': len(group),
        })
    
    return pd.DataFrame(results)


def plot_epic5_alignment_barplot(metrics_df: pd.DataFrame, output_dir: Path):
    """Create tangent alignment barplot."""
    if len(metrics_df) == 0:
        return
    
    # Aggregate by baseline
    agg = metrics_df.groupby('baseline')['mean_tas'].agg(['mean', 'std']).reset_index()
    agg = agg.dropna(subset=['mean'])
    
    if len(agg) == 0:
        return
    
    # Sort by TAS (higher = better)
    agg = agg.sort_values('mean', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [BASELINE_COLORS.get(b, '#333333') for b in agg['baseline']]
    bars = ax.barh(range(len(agg)), agg['mean'], xerr=agg['std'], color=colors, 
                   capsize=3, alpha=0.8)
    
    ax.set_yticks(range(len(agg)))
    ax.set_yticklabels([BASELINE_LABELS.get(b, b) for b in agg['baseline']])
    ax.set_xlabel('Mean Tangent Alignment Score (higher = better)')
    ax.set_title('Epic 5: Tangent Alignment - Subspace Agreement', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "epic5_alignment_barplot.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved: {output_path}")


# ==============================================================================
# CROSS-EPIC META-ANALYSIS
# ==============================================================================

def create_cross_epic_summary(
    epic1_metrics: pd.DataFrame,
    epic2_metrics: pd.DataFrame,
    epic3_metrics: pd.DataFrame,
    epic4_metrics: pd.DataFrame,
    epic5_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Create unified baseline summary across all epics."""
    
    baselines = set()
    for df in [epic1_metrics, epic2_metrics, epic3_metrics, epic4_metrics, epic5_metrics]:
        if len(df) > 0 and 'baseline' in df.columns:
            baselines.update(df['baseline'].unique())
    
    results = []
    
    for baseline in baselines:
        row = {'baseline': baseline}
        
        # Epic 1: Peak r
        if len(epic1_metrics) > 0:
            e1 = epic1_metrics[epic1_metrics['baseline'] == baseline]
            row['epic1_peak_r'] = e1['peak_r'].mean() if len(e1) > 0 else np.nan
            row['epic1_curvature'] = e1['curvature_index'].mean() if len(e1) > 0 else np.nan
        
        # Epic 2: Original r
        if len(epic2_metrics) > 0:
            e2 = epic2_metrics[epic2_metrics['baseline'] == baseline]
            row['epic2_original_r'] = e2['mean_original_r'].mean() if len(e2) > 0 else np.nan
        
        # Epic 3: Lipschitz
        if len(epic3_metrics) > 0:
            e3 = epic3_metrics[epic3_metrics['baseline'] == baseline]
            row['epic3_lipschitz'] = e3['lipschitz_constant'].mean() if len(e3) > 0 else np.nan
            row['epic3_baseline_r'] = e3['baseline_r'].mean() if len(e3) > 0 else np.nan
        
        # Epic 4: Flip rate
        if len(epic4_metrics) > 0:
            e4 = epic4_metrics[epic4_metrics['baseline'] == baseline]
            row['epic4_flip_rate'] = e4['mean_adversarial_rate'].mean() if len(e4) > 0 else np.nan
        
        # Epic 5: TAS
        if len(epic5_metrics) > 0:
            e5 = epic5_metrics[epic5_metrics['baseline'] == baseline]
            row['epic5_tas'] = e5['mean_tas'].mean() if len(e5) > 0 else np.nan
        
        results.append(row)
    
    return pd.DataFrame(results)


def plot_cross_epic_heatmap(summary_df: pd.DataFrame, output_dir: Path):
    """Create cross-epic comparison heatmap."""
    if len(summary_df) == 0:
        return
    
    # Select metrics to display
    metric_cols = ['epic1_peak_r', 'epic3_baseline_r', 'epic4_flip_rate', 'epic5_tas']
    available_cols = [c for c in metric_cols if c in summary_df.columns]
    
    if len(available_cols) == 0:
        return
    
    # Prepare data
    plot_df = summary_df[['baseline'] + available_cols].dropna(subset=available_cols, how='all')
    
    if len(plot_df) == 0:
        return
    
    # Normalize each metric to [0,1] for comparison
    for col in available_cols:
        if col == 'epic4_flip_rate':
            # Lower is better, invert
            plot_df[col] = 1 - (plot_df[col] - plot_df[col].min()) / (plot_df[col].max() - plot_df[col].min() + 1e-10)
        else:
            plot_df[col] = (plot_df[col] - plot_df[col].min()) / (plot_df[col].max() - plot_df[col].min() + 1e-10)
    
    plot_df = plot_df.set_index('baseline')
    plot_df.index = [BASELINE_LABELS.get(b, b) for b in plot_df.index]
    
    col_labels = {
        'epic1_peak_r': 'Peak LSFT r\n(Curvature)',
        'epic3_baseline_r': 'Baseline r\n(Noise)',
        'epic4_flip_rate': 'Low Flip Rate\n(Direction)',
        'epic5_tas': 'Tangent Align.\n(TAS)',
    }
    plot_df.columns = [col_labels.get(c, c) for c in plot_df.columns]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(plot_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5,
               vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized Score (higher = better)'})
    ax.set_title('Cross-Epic Meta-Analysis: Manifold Quality by Baseline', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / "cross_epic_heatmap.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_5epic_winner_grid(summary_df: pd.DataFrame, output_dir: Path):
    """Create the 5-epic 'winner' summary grid."""
    if len(summary_df) == 0:
        return
    
    # Define metrics and their "good" direction
    metrics = [
        ('epic1_peak_r', 'Epic 1: Curvature', True),  # higher is better
        ('epic3_baseline_r', 'Epic 3: Noise Stability', True),  # higher is better
        ('epic4_flip_rate', 'Epic 4: Direction-Flip', False),  # lower is better
        ('epic5_tas', 'Epic 5: Tangent Alignment', True),  # higher is better
    ]
    
    available_metrics = [(m, l, d) for m, l, d in metrics if m in summary_df.columns]
    
    if len(available_metrics) == 0:
        return
    
    fig, axes = plt.subplots(1, len(available_metrics), figsize=(4*len(available_metrics), 5))
    
    if len(available_metrics) == 1:
        axes = [axes]
    
    for idx, (metric, label, higher_better) in enumerate(available_metrics):
        ax = axes[idx]
        
        data = summary_df[['baseline', metric]].dropna()
        if higher_better:
            data = data.sort_values(metric, ascending=True)
        else:
            data = data.sort_values(metric, ascending=False)
        
        colors = [BASELINE_COLORS.get(b, '#333333') for b in data['baseline']]
        
        ax.barh(range(len(data)), data[metric], color=colors, alpha=0.8)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels([BASELINE_LABELS.get(b, b)[:15] for b in data['baseline']], fontsize=8)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Highlight winner
        if higher_better:
            winner_idx = len(data) - 1
        else:
            winner_idx = 0
        ax.get_children()[winner_idx].set_edgecolor('gold')
        ax.get_children()[winner_idx].set_linewidth(2)
    
    plt.suptitle('Manifold Law: Winner Summary Across All Epics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = output_dir / "5epic_winner_grid.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved: {output_path}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("=" * 70)
    print("GENERATING PUBLICATION REPORTS")
    print("=" * 70)
    print()
    
    # Create output directories
    for subdir in ['epic1_curvature', 'epic2_mechanism_ablation', 'epic3_noise_injection',
                   'epic4_direction_flip', 'epic5_tangent_alignment', 'cross_epic_analysis',
                   'poster_figures', 'final_tables']:
        (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)
    
    # Load all data
    print("Loading Epic 1 data...")
    epic1_df = load_epic1_data()
    print(f"  Loaded {len(epic1_df)} records")
    
    print("Loading Epic 2 data...")
    epic2_df = load_epic2_data()
    print(f"  Loaded {len(epic2_df)} records")
    
    print("Loading Epic 3 data...")
    epic3_df = load_epic3_data()
    print(f"  Loaded {len(epic3_df)} records")
    
    print("Loading Epic 4 data...")
    epic4_df = load_epic4_data()
    print(f"  Loaded {len(epic4_df)} records")
    
    print("Loading Epic 5 data...")
    epic5_df = load_epic5_data()
    print(f"  Loaded {len(epic5_df)} records")
    
    print()
    
    # ==== EPIC 1 ====
    print("=" * 40)
    print("EPIC 1: Curvature Sweep")
    print("=" * 40)
    
    epic1_metrics = analyze_epic1_curvature(epic1_df)
    if len(epic1_metrics) > 0:
        epic1_metrics.to_csv(OUTPUT_DIR / "final_tables" / "epic1_curvature_metrics.csv", index=False)
        print(f"  Saved metrics table ({len(epic1_metrics)} rows)")
        
        plot_epic1_curvature_grid(epic1_df, OUTPUT_DIR / "epic1_curvature")
        plot_epic1_curvature_heatmap(epic1_metrics, OUTPUT_DIR / "epic1_curvature")
    
    # ==== EPIC 2 ====
    print()
    print("=" * 40)
    print("EPIC 2: Mechanism Ablation")
    print("=" * 40)
    
    epic2_metrics = analyze_epic2_ablation(epic2_df)
    if len(epic2_metrics) > 0:
        epic2_metrics.to_csv(OUTPUT_DIR / "final_tables" / "epic2_alignment_summary.csv", index=False)
        print(f"  Saved metrics table ({len(epic2_metrics)} rows)")
        
        plot_epic2_baseline_comparison(epic2_df, OUTPUT_DIR / "epic2_mechanism_ablation")
    
    # ==== EPIC 3 ====
    print()
    print("=" * 40)
    print("EPIC 3: Noise Injection")
    print("=" * 40)
    
    epic3_metrics = analyze_epic3_noise(epic3_df)
    if len(epic3_metrics) > 0:
        epic3_metrics.to_csv(OUTPUT_DIR / "final_tables" / "epic3_lipschitz_summary.csv", index=False)
        print(f"  Saved metrics table ({len(epic3_metrics)} rows)")
        
        plot_epic3_lipschitz_barplot(epic3_metrics, OUTPUT_DIR / "epic3_noise_injection")
    
    # ==== EPIC 4 ====
    print()
    print("=" * 40)
    print("EPIC 4: Direction-Flip Probe")
    print("=" * 40)
    
    epic4_metrics = analyze_epic4_flips(epic4_df)
    if len(epic4_metrics) > 0:
        epic4_metrics.to_csv(OUTPUT_DIR / "final_tables" / "epic4_flip_summary.csv", index=False)
        print(f"  Saved metrics table ({len(epic4_metrics)} rows)")
        
        plot_epic4_flip_rates(epic4_metrics, OUTPUT_DIR / "epic4_direction_flip")
    
    # ==== EPIC 5 ====
    print()
    print("=" * 40)
    print("EPIC 5: Tangent Alignment")
    print("=" * 40)
    
    epic5_metrics = analyze_epic5_alignment(epic5_df)
    if len(epic5_metrics) > 0:
        epic5_metrics.to_csv(OUTPUT_DIR / "final_tables" / "epic5_alignment_summary.csv", index=False)
        print(f"  Saved metrics table ({len(epic5_metrics)} rows)")
        
        plot_epic5_alignment_barplot(epic5_metrics, OUTPUT_DIR / "epic5_tangent_alignment")
    
    # ==== CROSS-EPIC ====
    print()
    print("=" * 40)
    print("CROSS-EPIC META-ANALYSIS")
    print("=" * 40)
    
    cross_epic_summary = create_cross_epic_summary(
        epic1_metrics, epic2_metrics, epic3_metrics, epic4_metrics, epic5_metrics
    )
    
    if len(cross_epic_summary) > 0:
        cross_epic_summary.to_csv(OUTPUT_DIR / "final_tables" / "baseline_summary.csv", index=False)
        print(f"  Saved cross-epic summary ({len(cross_epic_summary)} baselines)")
        
        plot_cross_epic_heatmap(cross_epic_summary, OUTPUT_DIR / "cross_epic_analysis")
        plot_5epic_winner_grid(cross_epic_summary, OUTPUT_DIR / "poster_figures")
    
    # ==== POSTER FIGURES ====
    print()
    print("=" * 40)
    print("POSTER FIGURES")
    print("=" * 40)
    
    # Copy key figures to poster_figures
    import shutil
    key_figures = [
        ("epic1_curvature", "epic1_curvature_heatmap.png"),
        ("epic3_noise_injection", "epic3_lipschitz_barplot.png"),
        ("epic4_direction_flip", "epic4_flip_rates_barplot.png"),
    ]
    
    for subdir, filename in key_figures:
        src = OUTPUT_DIR / subdir / filename
        if src.exists():
            dst = OUTPUT_DIR / "poster_figures" / filename
            shutil.copy(src, dst)
            print(f"  Copied: {filename}")
    
    print()
    print("=" * 70)
    print("✅ PUBLICATION REPORTS COMPLETE!")
    print("=" * 70)
    print()
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("Generated:")
    for subdir in ['epic1_curvature', 'epic2_mechanism_ablation', 'epic3_noise_injection',
                   'epic4_direction_flip', 'epic5_tangent_alignment', 'cross_epic_analysis',
                   'poster_figures', 'final_tables']:
        path = OUTPUT_DIR / subdir
        n_files = len(list(path.glob("*.*")))
        print(f"  - {subdir}/: {n_files} files")


if __name__ == "__main__":
    main()


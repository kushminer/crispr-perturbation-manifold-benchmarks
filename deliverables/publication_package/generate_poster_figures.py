#!/usr/bin/env python3
"""
Generate Poster-Ready Figures for Manifold Law Diagnostic Suite

Creates high-quality, publication-ready visualizations for posters and papers.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "manifold_law_diagnostics"
OUTPUT_DIR = Path(__file__).parent / "poster_figures"

# Publication-quality settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Color palette optimized for color-blind accessibility
BASELINE_COLORS = {
    'lpm_selftrained': '#0077BB',      # Blue - Primary (best)
    'lpm_gearsPertEmb': '#33BBEE',     # Cyan - GEARS
    'lpm_scgptGeneEmb': '#EE7733',     # Orange - scGPT
    'lpm_scFoundationGeneEmb': '#CC3311',  # Red - scFoundation
    'lpm_randomGeneEmb': '#009988',    # Teal - Random Gene
    'lpm_randomPertEmb': '#EE3377',    # Magenta - Random Pert
    'lpm_k562PertEmb': '#BBBBBB',      # Gray - K562
    'lpm_rpe1PertEmb': '#666666',      # Dark gray - RPE1
}

BASELINE_SHORT_LABELS = {
    'lpm_selftrained': 'PCA',
    'lpm_gearsPertEmb': 'GEARS',
    'lpm_scgptGeneEmb': 'scGPT',
    'lpm_scFoundationGeneEmb': 'scFound.',
    'lpm_randomGeneEmb': 'Rand.Gene',
    'lpm_randomPertEmb': 'Rand.Pert',
    'lpm_k562PertEmb': 'K562',
    'lpm_rpe1PertEmb': 'RPE1',
}

DATASET_COLORS = {
    'adamson': '#1f77b4',
    'k562': '#ff7f0e',
    'rpe1': '#2ca02c',
}


def load_all_epic_data() -> Dict[str, pd.DataFrame]:
    """Load data from all epics."""
    data = {}
    
    # Epic 1 - Use corrected directory if available
    epic1_dir = RESULTS_DIR / "epic1_curvature_corrected"
    if not epic1_dir.exists():
        epic1_dir = RESULTS_DIR / "epic1_curvature"  # Fallback to original
    all_e1 = []
    # Look for both lsft_k_sweep files and curvature_sweep_summary files
    for pattern in ["lsft_k_sweep_*.csv", "curvature_sweep_summary_*.csv"]:
        for f in epic1_dir.glob(pattern):
            try:
                df = pd.read_csv(f)
                # Handle different file naming conventions
                if "curvature_sweep_summary" in f.stem:
                    # Format: curvature_sweep_summary_{dataset}_{baseline}.csv
                    parts = f.stem.replace("curvature_sweep_summary_", "").split("_", 1)
                    if len(parts) == 2:
                        df["dataset"] = parts[0]
                        df["baseline"] = parts[1]
                        all_e1.append(df)
                else:
                    # Format: lsft_k_sweep_{dataset}_{baseline}.csv
                    parts = f.stem.replace("lsft_k_sweep_", "").split("_", 1)
                    if len(parts) == 2:
                        df["dataset"] = parts[0]
                        df["baseline"] = parts[1]
                        all_e1.append(df)
            except:
                pass
    data['epic1'] = pd.concat(all_e1, ignore_index=True) if all_e1 else pd.DataFrame()
    
    # Epic 3
    epic3_dir = RESULTS_DIR / "epic3_noise_injection"
    all_e3 = []
    for f in epic3_dir.glob("noise_injection_*.csv"):
        try:
            df = pd.read_csv(f)
            parts = f.stem.replace("noise_injection_", "").split("_", 1)
            if len(parts) == 2:
                df["dataset"] = parts[0]
                df["baseline"] = parts[1]
                all_e3.append(df)
        except:
            pass
    data['epic3'] = pd.concat(all_e3, ignore_index=True) if all_e3 else pd.DataFrame()
    
    # Epic 4
    epic4_dir = RESULTS_DIR / "epic4_direction_flip"
    all_e4 = []
    for f in epic4_dir.glob("direction_flip_*.csv"):
        try:
            df = pd.read_csv(f)
            parts = f.stem.replace("direction_flip_", "").split("_", 1)
            if len(parts) == 2:
                df["dataset"] = parts[0]
                df["baseline"] = parts[1]
                all_e4.append(df)
        except:
            pass
    data['epic4'] = pd.concat(all_e4, ignore_index=True) if all_e4 else pd.DataFrame()
    
    # Epic 5
    epic5_dir = RESULTS_DIR / "epic5_tangent_alignment"
    all_e5 = []
    for f in epic5_dir.glob("tangent_alignment_*.csv"):
        try:
            df = pd.read_csv(f)
            parts = f.stem.replace("tangent_alignment_", "").split("_", 1)
            if len(parts) == 2:
                df["dataset"] = parts[0]
                df["baseline"] = parts[1]
                all_e5.append(df)
        except:
            pass
    data['epic5'] = pd.concat(all_e5, ignore_index=True) if all_e5 else pd.DataFrame()
    
    return data


def create_manifold_law_diagram(output_dir: Path):
    """Create the conceptual Manifold Law diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Background
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(6, 7.5, 'The Manifold Law of Perturbation Responses', 
           fontsize=16, fontweight='bold', ha='center', va='top')
    
    # Subtitle
    ax.text(6, 6.8, '"Biological responses live on locally smooth manifolds"',
           fontsize=11, style='italic', ha='center', va='top', color='#555555')
    
    # Draw smooth manifold (left side)
    theta = np.linspace(0, 2*np.pi, 100)
    x_manifold = 2 + 0.8*np.cos(theta) + 0.2*np.sin(2*theta)
    y_manifold = 4 + 1.2*np.sin(theta) + 0.1*np.cos(3*theta)
    ax.fill(x_manifold, y_manifold, alpha=0.3, color='#0077BB')
    ax.plot(x_manifold, y_manifold, 'k-', linewidth=2)
    ax.text(2, 2.2, 'True Manifold\n(Smooth, Low-Dim)', ha='center', fontsize=9, fontweight='bold')
    
    # Arrow from manifold to embeddings
    ax.annotate('', xy=(4.5, 4), xytext=(3, 4),
               arrowprops=dict(arrowstyle='->', lw=2, color='#333333'))
    ax.text(3.75, 4.4, 'Embed', fontsize=9, ha='center')
    
    # Draw different embedding outcomes (right side)
    # 1. PCA - preserves geometry (top)
    circle1 = plt.Circle((6, 5.5), 0.6, fill=True, alpha=0.3, color='#0077BB')
    ax.add_patch(circle1)
    ax.plot([5.4, 6.6], [5.5, 5.5], 'k-', linewidth=1)
    ax.text(6, 6.4, 'PCA', fontsize=10, fontweight='bold', ha='center', color='#0077BB')
    ax.text(7.2, 5.5, '✓ Preserves geometry', fontsize=9, va='center', color='#0077BB')
    
    # 2. GEARS - partial (middle-top)
    ellipse1 = mpatches.Ellipse((6, 4.2), 1.2, 0.5, angle=15, fill=True, alpha=0.3, color='#33BBEE')
    ax.add_patch(ellipse1)
    ax.text(6, 4.9, 'GEARS', fontsize=10, fontweight='bold', ha='center', color='#33BBEE')
    ax.text(7.2, 4.2, '~ Partial preservation', fontsize=9, va='center', color='#33BBEE')
    
    # 3. Deep models - folds manifold (middle)
    x_folded = 6 + 0.5*np.sin(np.linspace(0, 4*np.pi, 50))
    y_folded = 3 + 0.3*np.linspace(0, 1, 50)
    ax.plot(x_folded, y_folded, '-', color='#EE7733', linewidth=2, alpha=0.7)
    ax.text(6, 3.7, 'scGPT/scFound.', fontsize=10, fontweight='bold', ha='center', color='#EE7733')
    ax.text(7.5, 3.15, '✗ Folds manifold', fontsize=9, va='center', color='#EE7733')
    
    # 4. Random Pert - breaks manifold (bottom)
    np.random.seed(42)
    x_random = 5.5 + 1.0*np.random.randn(30)
    y_random = 2.0 + 0.3*np.random.randn(30)
    ax.scatter(x_random, y_random, s=15, alpha=0.5, color='#EE3377')
    ax.text(6, 1.5, 'Random Pert.', fontsize=10, fontweight='bold', ha='center', color='#EE3377')
    ax.text(7.5, 2.0, '✗ Destroys structure', fontsize=9, va='center', color='#EE3377')
    
    # Key insight box
    rect = mpatches.FancyBboxPatch((8.5, 4.5), 3.2, 2.2, boxstyle="round,pad=0.1",
                                    facecolor='#f0f0f0', edgecolor='#333333', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(10.1, 6.4, 'Key Insight', fontsize=11, fontweight='bold', ha='center')
    ax.text(10.1, 5.9, 'LSFT succeeds because', fontsize=9, ha='center')
    ax.text(10.1, 5.5, 'local neighborhoods are', fontsize=9, ha='center')
    ax.text(10.1, 5.1, 'smooth & well-aligned.', fontsize=9, ha='center')
    ax.text(10.1, 4.7, 'Deep embeddings', fontsize=9, ha='center', color='#EE7733')
    ax.text(10.1, 4.3, 'break this structure.', fontsize=9, ha='center', color='#EE7733')
    
    plt.tight_layout()
    output_path = output_dir / "manifold_law_diagram.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_path}")


def create_curvature_comparison_poster(data: Dict[str, pd.DataFrame], output_dir: Path):
    """Create top-line curvature comparison figure."""
    df = data.get('epic1', pd.DataFrame())
    if len(df) == 0:
        print("No Epic 1 data for curvature comparison")
        return
    
    # Find r column
    r_col = None
    for col in ['performance_local_pearson_r', 'mean_r', 'pearson_r']:
        if col in df.columns:
            r_col = col
            break
    
    if r_col is None or 'k' not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baselines = sorted(df['baseline'].unique())
    
    for baseline in baselines:
        subset = df[df['baseline'] == baseline]
        k_stats = subset.groupby('k')[r_col].agg(['mean', 'sem']).reset_index()
        k_stats.columns = ['k', 'mean', 'sem']
        k_stats = k_stats.sort_values('k')
        
        color = BASELINE_COLORS.get(baseline, '#888888')
        label = BASELINE_SHORT_LABELS.get(baseline, baseline)
        
        ax.plot(k_stats['k'], k_stats['mean'], 'o-', color=color, 
               label=label, linewidth=2, markersize=6)
        ax.fill_between(k_stats['k'], 
                       k_stats['mean'] - k_stats['sem'],
                       k_stats['mean'] + k_stats['sem'],
                       alpha=0.2, color=color)
    
    ax.set_xlabel('Neighborhood Size (k)', fontsize=12)
    ax.set_ylabel('LSFT Pearson r', fontsize=12)
    ax.set_title('Curvature Sweep: How Accuracy Varies with Neighborhood Size', 
                fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', ncol=2, framealpha=0.9)
    
    plt.tight_layout()
    output_path = output_dir / "curvature_comparison_poster.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved: {output_path}")


def create_direction_flip_poster(data: Dict[str, pd.DataFrame], output_dir: Path):
    """Create direction-flip barplot poster figure."""
    df = data.get('epic4', pd.DataFrame())
    if len(df) == 0:
        print("No Epic 4 data for direction flip")
        return
    
    # Calculate mean adversarial rate per baseline
    agg = df.groupby('baseline')['adversarial_rate'].agg(['mean', 'sem']).reset_index()
    agg = agg.dropna()
    
    if len(agg) == 0:
        return
    
    # Sort by flip rate (lower = better)
    agg = agg.sort_values('mean')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [BASELINE_COLORS.get(b, '#888888') for b in agg['baseline']]
    labels = [BASELINE_SHORT_LABELS.get(b, b) for b in agg['baseline']]
    
    bars = ax.barh(range(len(agg)), agg['mean'], xerr=agg['sem'], 
                   color=colors, capsize=4, alpha=0.85, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(agg)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Adversarial Neighbor Rate (lower = better)', fontsize=12)
    ax.set_title('Direction-Flip Probe: Finding "Misleading" Neighbors', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (val, err) in enumerate(zip(agg['mean'], agg['sem'])):
        ax.text(val + err + 0.005, i, f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / "direction_flip_poster.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved: {output_path}")


def create_tangent_alignment_poster(data: Dict[str, pd.DataFrame], output_dir: Path):
    """Create tangent alignment poster figure."""
    df = data.get('epic5', pd.DataFrame())
    if len(df) == 0:
        print("No Epic 5 data for tangent alignment")
        return
    
    # Find alignment score column
    score_col = None
    for col in ['tangent_alignment_score', 'tas', 'alignment_score', 'procrustes_similarity']:
        if col in df.columns:
            score_col = col
            break
    
    if score_col is None:
        print(f"No alignment score column found. Columns: {df.columns.tolist()}")
        return
    
    # Calculate mean alignment per baseline
    agg = df.groupby('baseline')[score_col].agg(['mean', 'sem']).reset_index()
    agg = agg.dropna()
    
    if len(agg) == 0:
        return
    
    # Sort by alignment (higher = better)
    agg = agg.sort_values('mean', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [BASELINE_COLORS.get(b, '#888888') for b in agg['baseline']]
    labels = [BASELINE_SHORT_LABELS.get(b, b) for b in agg['baseline']]
    
    bars = ax.barh(range(len(agg)), agg['mean'], xerr=agg['sem'],
                   color=colors, capsize=4, alpha=0.85, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(agg)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Tangent Alignment Score (higher = better)', fontsize=12)
    ax.set_title('Tangent Alignment: Do Train/Test Live in Same Subspace?', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "tangent_alignment_poster.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved: {output_path}")


def create_5epic_thumbnail_grid(data: Dict[str, pd.DataFrame], output_dir: Path):
    """Create 5-epic thumbnail grid for poster."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    
    epic_titles = [
        'E1: Curvature',
        'E2: Mechanism',
        'E3: Noise',
        'E4: Direction',
        'E5: Alignment'
    ]
    
    for idx, (ax, title) in enumerate(zip(axes, epic_titles)):
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add placeholder or simplified visualization
        if idx == 0 and 'epic1' in data and len(data['epic1']) > 0:
            # Mini curvature plot
            df = data['epic1']
            r_col = None
            for col in ['performance_local_pearson_r', 'mean_r']:
                if col in df.columns:
                    r_col = col
                    break
            if r_col and 'k' in df.columns:
                for baseline in ['lpm_selftrained', 'lpm_gearsPertEmb', 'lpm_scgptGeneEmb']:
                    if baseline in df['baseline'].values:
                        subset = df[df['baseline'] == baseline]
                        k_stats = subset.groupby('k')[r_col].mean().reset_index()
                        ax.plot(k_stats['k'], k_stats[r_col], '-', 
                               color=BASELINE_COLORS.get(baseline, '#888'),
                               label=BASELINE_SHORT_LABELS.get(baseline, baseline)[:6], linewidth=2)
                ax.set_xscale('log')
                ax.set_ylim([0, 1])
                ax.legend(fontsize=7, loc='lower left')
        
        elif idx == 3 and 'epic4' in data and len(data['epic4']) > 0:
            # Mini flip rate bar
            df = data['epic4']
            if 'adversarial_rate' in df.columns:
                agg = df.groupby('baseline')['adversarial_rate'].mean().sort_values()
                colors = [BASELINE_COLORS.get(b, '#888') for b in agg.index]
                ax.barh(range(len(agg)), agg.values, color=colors, alpha=0.8)
                ax.set_yticks([])
        
        elif idx == 4 and 'epic5' in data and len(data['epic5']) > 0:
            # Mini alignment bar
            df = data['epic5']
            score_col = None
            for col in ['tangent_alignment_score', 'tas', 'procrustes_similarity']:
                if col in df.columns:
                    score_col = col
                    break
            if score_col:
                agg = df.groupby('baseline')[score_col].mean().sort_values(ascending=False)
                colors = [BASELINE_COLORS.get(b, '#888') for b in agg.index]
                ax.barh(range(len(agg)), agg.values, color=colors, alpha=0.8)
                ax.set_yticks([])
        
        else:
            ax.text(0.5, 0.5, '📊', fontsize=40, ha='center', va='center', transform=ax.transAxes)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle('Manifold Law Diagnostic Suite: 5 Tests of Geometric Quality', 
                fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    output_path = output_dir / "5epic_thumbnail_grid.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved: {output_path}")


def main():
    print("=" * 60)
    print("GENERATING POSTER FIGURES")
    print("=" * 60)
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading all epic data...")
    data = load_all_epic_data()
    for epic, df in data.items():
        print(f"  {epic}: {len(df)} records")
    print()
    
    # Generate figures
    print("Creating figures...")
    
    create_manifold_law_diagram(OUTPUT_DIR)
    create_curvature_comparison_poster(data, OUTPUT_DIR)
    create_direction_flip_poster(data, OUTPUT_DIR)
    create_tangent_alignment_poster(data, OUTPUT_DIR)
    create_5epic_thumbnail_grid(data, OUTPUT_DIR)
    
    print()
    print("=" * 60)
    print("✅ POSTER FIGURES COMPLETE!")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()


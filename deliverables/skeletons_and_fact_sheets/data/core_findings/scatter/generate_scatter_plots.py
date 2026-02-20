#!/usr/bin/env python3
"""
Scatter plots: Local Mean Similarity (X) vs Performance (Y)

For top_pct = 0.05 (5% nearest neighbors):
- X-axis: local_mean_similarity
- Y-axis: Pearson r or L2
- Each dataset in a different color
- Labels include dataset name, # targets, and difficulty level
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = Path(__file__).parent

# Dataset metadata
DATASET_INFO = {
    'adamson': {'n_targets': 87, 'difficulty': 'Easy', 'color': '#27ae60'},      # Green
    'k562': {'n_targets': 1067, 'difficulty': 'Medium', 'color': '#3498db'},     # Blue
    'rpe1': {'n_targets': 1047, 'difficulty': 'Hard', 'color': '#e74c3c'},       # Red
}

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11


def load_data():
    """Load LSFT raw per-perturbation data."""
    df = pd.read_csv(DATA_DIR / "LSFT_raw_per_perturbation.csv")
    # Filter to top_pct = 0.05
    df = df[df['top_pct'] == 0.05].copy()
    return df


def create_scatter_pearson_r(df):
    """Scatter: local_mean_similarity (X) vs Pearson r (Y)."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for dataset, info in DATASET_INFO.items():
        ds_data = df[df['dataset'] == dataset]
        
        x = ds_data['local_mean_similarity'].values
        y = ds_data['performance_local_pearson_r'].values
        
        label = f"{dataset.upper()} (n={info['n_targets']}, {info['difficulty']})"
        
        ax.scatter(x, y, s=40, c=info['color'], alpha=0.6, label=label, 
                  edgecolors='none')
    
    # Add trend line (overall)
    all_x = df['local_mean_similarity'].values
    all_y = df['performance_local_pearson_r'].values
    z = np.polyfit(all_x, all_y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(all_x.min(), all_x.max(), 100)
    ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=2, label=f'Trend (slope={z[0]:.2f})')
    
    # Correlation
    corr = np.corrcoef(all_x, all_y)[0, 1]
    
    ax.set_xlabel('Local Mean Similarity', fontsize=14, fontweight='bold')
    ax.set_ylabel('Pearson r (LSFT prediction)', fontsize=14, fontweight='bold')
    ax.set_title(f'Similarity vs Prediction Accuracy\n(top 5% neighbors, r={corr:.3f})',
                fontsize=16, fontweight='bold')
    ax.set_ylim(-0.5, 1.05)
    ax.set_xlim(0, 1.05)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Good (r=0.7)')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scatter_similarity_vs_pearson_r.png", dpi=150, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ scatter_similarity_vs_pearson_r.png")


def create_scatter_l2(df):
    """Scatter: local_mean_similarity (X) vs L2 (Y)."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for dataset, info in DATASET_INFO.items():
        ds_data = df[df['dataset'] == dataset]
        
        x = ds_data['local_mean_similarity'].values
        y = ds_data['performance_local_l2'].values
        
        label = f"{dataset.upper()} (n={info['n_targets']}, {info['difficulty']})"
        
        ax.scatter(x, y, s=40, c=info['color'], alpha=0.6, label=label,
                  edgecolors='none')
    
    # Add trend line (overall)
    all_x = df['local_mean_similarity'].values
    all_y = df['performance_local_l2'].values
    z = np.polyfit(all_x, all_y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(all_x.min(), all_x.max(), 100)
    ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=2, label=f'Trend (slope={z[0]:.2f})')
    
    # Correlation
    corr = np.corrcoef(all_x, all_y)[0, 1]
    
    ax.set_xlabel('Local Mean Similarity', fontsize=14, fontweight='bold')
    ax.set_ylabel('L2 Distance (LSFT prediction error)', fontsize=14, fontweight='bold')
    ax.set_title(f'Similarity vs Prediction Error\n(top 5% neighbors, r={corr:.3f})',
                fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scatter_similarity_vs_l2.png", dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ scatter_similarity_vs_l2.png")


def create_combined_scatter(df):
    """Side-by-side scatter plots."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # === LEFT: Pearson r ===
    ax1 = axes[0]
    for dataset, info in DATASET_INFO.items():
        ds_data = df[df['dataset'] == dataset]
        x = ds_data['local_mean_similarity'].values
        y = ds_data['performance_local_pearson_r'].values
        label = f"{dataset.upper()} (n={info['n_targets']}, {info['difficulty']})"
        ax1.scatter(x, y, s=30, c=info['color'], alpha=0.5, label=label, edgecolors='none')
    
    # Trend line
    all_x = df['local_mean_similarity'].values
    all_y = df['performance_local_pearson_r'].values
    z = np.polyfit(all_x, all_y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(all_x.min(), all_x.max(), 100)
    ax1.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=2)
    corr = np.corrcoef(all_x, all_y)[0, 1]
    
    ax1.set_xlabel('Local Mean Similarity', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Pearson r', fontsize=12, fontweight='bold')
    ax1.set_title(f'Prediction Accuracy (r={corr:.3f})', fontsize=14, fontweight='bold')
    ax1.set_ylim(-0.5, 1.05)
    ax1.set_xlim(0, 1.05)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # === RIGHT: L2 ===
    ax2 = axes[1]
    for dataset, info in DATASET_INFO.items():
        ds_data = df[df['dataset'] == dataset]
        x = ds_data['local_mean_similarity'].values
        y = ds_data['performance_local_l2'].values
        label = f"{dataset.upper()} (n={info['n_targets']}, {info['difficulty']})"
        ax2.scatter(x, y, s=30, c=info['color'], alpha=0.5, label=label, edgecolors='none')
    
    # Trend line
    all_y = df['performance_local_l2'].values
    z = np.polyfit(all_x, all_y, 1)
    p = np.poly1d(z)
    ax2.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=2)
    corr = np.corrcoef(all_x, all_y)[0, 1]
    
    ax2.set_xlabel('Local Mean Similarity', fontsize=12, fontweight='bold')
    ax2.set_ylabel('L2 Distance', fontsize=12, fontweight='bold')
    ax2.set_title(f'Prediction Error (r={corr:.3f})', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1.05)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    fig.suptitle('Local Similarity vs LSFT Performance (top 5% neighbors)',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scatter_combined_similarity.png", dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ scatter_combined_similarity.png")


def main():
    print("=" * 60)
    print("CREATING SCATTER PLOTS: Similarity vs Performance")
    print("=" * 60)
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df = load_data()
    print(f"Loaded {len(df)} perturbations from LSFT_raw_per_perturbation.csv\n")
    
    create_scatter_pearson_r(df)
    create_scatter_l2(df)
    create_combined_scatter(df)
    
    print()
    print("=" * 60)
    print("✅ SCATTER PLOTS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Figure 2 v2: Similarity → Error (By Model)

Reorganized version where:
- Each subplot represents a model/baseline
- Colors encode dataset (Adamson, K562, RPE1)
- LOESS smoothed global trendline per model

Data source: LSFT (Local Similarity-Filtered Training) predictions
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Try to import statsmodels for LOESS, fallback to scipy if not available
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available, using scipy-based smoothing")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Set style matching other poster figures
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica Neue']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Load data - USE CORRECTED SINGLE-CELL DATA instead of buggy pseudobulk
print("Creating Figure 2 v2 (By Model): Similarity → Error")
print("Using CORRECTED single-cell LSFT data (centering bug fixed)")

# Load corrected single-cell LSFT data
lsft_data = []
for dataset in ['adamson', 'k562', 'rpe1']:
    lsft_dir = project_root / 'results' / 'single_cell_analysis' / dataset / 'lsft'
    if not lsft_dir.exists():
        continue
    
    for baseline_file in lsft_dir.glob(f'lsft_single_cell_{dataset}_lpm_*.csv'):
        if 'summary' in baseline_file.name:
            continue
        
        try:
            df = pd.read_csv(baseline_file)
            if 'mean_similarity' in df.columns and 'lsft_l2' in df.columns:
                df_05 = df[df['top_pct'] == 0.05].copy() if 'top_pct' in df.columns else df.copy()
                baseline = baseline_file.stem.replace(f'lsft_single_cell_{dataset}_', '')
                df_05['dataset'] = dataset
                df_05['baseline'] = baseline
                df_05['local_mean_similarity'] = df_05['mean_similarity']
                df_05['performance_local_l2'] = df_05['lsft_l2']
                lsft_data.append(df_05)
        except Exception as e:
            print(f"Warning: Could not load {baseline_file}: {e}")

if lsft_data:
    lsft_raw = pd.concat(lsft_data, ignore_index=True)
    print(f"Loaded {len(lsft_raw)} corrected single-cell LSFT results")
else:
    lsft_raw_path = project_root / "skeletons_and_fact_sheets" / "data" / "LSFT_raw_per_perturbation.csv"
    print(f"WARNING: Using old pseudobulk data (may contain buggy values)")
    lsft_raw = pd.read_csv(lsft_raw_path)
    print(f"Loaded {len(lsft_raw)} LSFT raw results (OLD DATA - may be buggy)")

# Include all key baselines
key_baselines = [
    'lpm_selftrained',
    'lpm_scgptGeneEmb',
    'lpm_scFoundationGeneEmb',
    'lpm_randomGeneEmb',
    'lpm_randomPertEmb',
    'lpm_gearsPertEmb'
]

lsft_filtered = lsft_raw[
    (lsft_raw['baseline'].isin(key_baselines)) & 
    (lsft_raw['top_pct'] == 0.05)
].copy()

# Dataset color scheme - distinct colors for each dataset
dataset_colors = {
    'adamson': '#1f77b4',   # Deep blue
    'k562': '#ff7f0e',      # Orange
    'rpe1': '#2ca02c',      # Green
}

# Baseline label mapping
baseline_labels = {
    'lpm_selftrained': 'Self-trained PCA',
    'lpm_scgptGeneEmb': 'scGPT',
    'lpm_scFoundationGeneEmb': 'scFoundation',
    'lpm_gearsPertEmb': 'GEARS',
    'lpm_randomGeneEmb': 'Random Gene',
    'lpm_randomPertEmb': 'Random Pert',
}

datasets = ['adamson', 'k562', 'rpe1']

def compute_loess(x, y, frac=0.5):
    """
    Compute LOESS smoothing.
    
    Args:
        x: x values (similarity)
        y: y values (error)
        frac: Fraction of data used for local regression (0.4-0.6 recommended)
    
    Returns:
        x_smooth, y_smooth: Smoothed values
    """
    if HAS_STATSMODELS:
        # Use statsmodels LOESS
        lowess = sm.nonparametric.lowess
        result = lowess(y, x, frac=frac, it=3)
        return result[:, 0], result[:, 1]
    else:
        # Fallback: Use scipy's UnivariateSpline for smoothing
        from scipy.interpolate import UnivariateSpline
        # Sort by x
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        # Use spline with smoothing factor
        spline = UnivariateSpline(x_sorted, y_sorted, s=len(x) * np.var(y) * 0.1)
        x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 200)
        y_smooth = spline(x_smooth)
        return x_smooth, y_smooth

# Create figure with subplots for each model (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Publication-grade title
fig.suptitle('LSFT: Higher Neighbor Similarity → Lower Prediction Error', 
             fontsize=16, fontweight='bold', y=0.995)

# Flatten axes for easier indexing
axes_flat = axes.flatten()

for baseline_idx, baseline in enumerate(key_baselines):
    ax = axes_flat[baseline_idx]
    
    baseline_data_all = lsft_filtered[lsft_filtered['baseline'] == baseline]
    
    if len(baseline_data_all) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title(baseline_labels[baseline], fontsize=13, fontweight='bold')
        continue
    
    # Collect all data for global LOESS trendline per model
    all_similarity = []
    all_error = []
    
    # Plot each dataset with different colors
    for dataset in datasets:
        dataset_data = baseline_data_all[baseline_data_all['dataset'] == dataset]
        
        if len(dataset_data) == 0:
            continue
        
        # Extract similarity and error
        # Handle both old pseudobulk format and new single-cell format
        if 'local_mean_similarity' in dataset_data.columns:
            similarity = dataset_data['local_mean_similarity'].values
        elif 'mean_similarity' in dataset_data.columns:
            similarity = dataset_data['mean_similarity'].values
        else:
            continue
            
        if 'performance_local_l2' in dataset_data.columns:
            error = dataset_data['performance_local_l2'].values
        elif 'lsft_l2' in dataset_data.columns:
            error = dataset_data['lsft_l2'].values
        else:
            continue
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(similarity) | np.isnan(error))
        similarity = similarity[valid_mask]
        error = error[valid_mask]
        
        if len(similarity) == 0:
            continue
        
        # Collect for global trendline
        all_similarity.extend(similarity)
        all_error.extend(error)
        
        color = dataset_colors[dataset]
        label = dataset.upper()
        
        # Plot scatter - all same shape (circles)
        ax.scatter(similarity, error, alpha=0.7, s=50, label=label, 
                  edgecolors='black', linewidth=0.5, color=color, marker='o')
    
    # Add global LOESS trendline for this model
    if len(all_similarity) > 10:  # Need enough points for LOESS
        all_similarity = np.array(all_similarity)
        all_error = np.array(all_error)
        
        # Compute LOESS with moderate span (0.5)
        try:
            x_smooth, y_smooth = compute_loess(all_similarity, all_error, frac=0.5)
            # Plot trendline - thick, prominent line
            ax.plot(x_smooth, y_smooth, color='black', linewidth=3, 
                   linestyle='-', alpha=0.8, zorder=10, label='Trend')
        except Exception as e:
            print(f"Warning: Could not compute LOESS for {baseline_labels[baseline]}: {e}")
    
    # Set labels and title
    ax.set_xlabel('Mean Neighbor Similarity', fontsize=11, fontweight='bold')
    ax.set_ylabel('Prediction Error (L2)', fontsize=11, fontweight='bold')
    ax.set_title(baseline_labels[baseline], fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.98])

# Save figure
output_path = project_root / "poster" / "figure2_similarity_error_v2_by_model.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved figure to: {output_path}")

# Also save as PDF
output_path_pdf = project_root / "poster" / "figure2_similarity_error_v2_by_model.pdf"
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved figure to: {output_path_pdf}")

plt.close()

print("Figure 2 v2 (By Model) created successfully!")
print("\nLayout:")
print("  ✓ 6 subplots (one per model/baseline)")
print("  ✓ Colors encode dataset (Adamson=blue, K562=orange, RPE1=green)")
print("  ✓ LOESS smoothed global trendline per model")
print("  ✓ Uniform marker shapes (all circles)")
print("\nNote: This figure uses LSFT (Local Similarity-Filtered Training) predictions,")
print("not baseline predictions. The error values are from LSFT evaluations.")


#!/usr/bin/env python3
"""
Figure 1: Neighborhood Smoothness Curve

Mean Pearson r vs. k with 95% confidence intervals.
Shows how performance changes as neighborhood size increases.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Set style matching other poster figures
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Load data - use raw per-perturbation data directly (NOT corrupted LSFT_results.csv)
lsft_raw_path = project_root / "skeletons_and_fact_sheets" / "data" / "LSFT_raw_per_perturbation.csv"
lsft_resampling_path = project_root / "skeletons_and_fact_sheets" / "data" / "LSFT_resampling.csv"

lsft_raw = pd.read_csv(lsft_raw_path)
lsft_resampling = pd.read_csv(lsft_resampling_path)

print("Creating Figure 1: Neighborhood Smoothness Curve")
print(f"Loaded {len(lsft_raw)} LSFT raw per-perturbation results")
print(f"Loaded {len(lsft_resampling)} LSFT resampling results")

# Compute mean r values for each dataset, baseline, and top_pct from raw data
lsft_results = lsft_raw.groupby(['dataset', 'baseline', 'top_pct']).agg({
    'performance_local_pearson_r': 'mean',
    'local_train_size': 'mean',
}).reset_index()
lsft_results = lsft_results.rename(columns={
    'performance_local_pearson_r': 'local_r',
    'top_pct': 'top_k',
    'local_train_size': 'mean_k'
})
lsft_results['top_k'] = lsft_results['top_k'].astype(float)

# Focus on key baselines for clarity
key_baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb', 'lpm_gearsPertEmb']
lsft_results_filtered = lsft_results[lsft_results['baseline'].isin(key_baselines)].copy()

# Create figure with subplots for each dataset
datasets = ['adamson', 'k562', 'rpe1']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Neighborhood Smoothness Curve: Mean Pearson r vs. Neighborhood Size', 
             fontsize=16, fontweight='bold', y=1.02)

# Color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

for idx, dataset in enumerate(datasets):
    ax = axes[idx]
    
    dataset_data = lsft_results_filtered[lsft_results_filtered['dataset'] == dataset]
    
    # Plot each baseline
    for baseline_idx, baseline in enumerate(key_baselines):
        baseline_data = dataset_data[dataset_data['baseline'] == baseline].sort_values('top_k')
        
        if len(baseline_data) == 0:
            continue
        
        # Use mean_k if available, otherwise use top_k as percentage
        if 'mean_k' in baseline_data.columns and not baseline_data['mean_k'].isna().all():
            x_values = baseline_data['mean_k'].values
            x_label = 'Neighborhood Size (k)'
        else:
            x_values = baseline_data['top_k'].values * 100
            x_label = 'Top-K Percentage'
        
        # Get mean r values
        y_values = baseline_data['local_r'].values
        
        # Get CIs from resampling data (for top_k=0.05)
        resampling_data = lsft_resampling[
            (lsft_resampling['dataset'] == dataset) & 
            (lsft_resampling['baseline'] == baseline)
        ]
        
        # Plot line
        label = baseline.replace('lpm_', '').replace('GeneEmb', '').replace('PertEmb', '')
        if label == 'selftrained':
            label = 'Self-trained PCA'
        elif label == 'scgptGeneEmb':
            label = 'scGPT'
        elif label == 'randomGeneEmb':
            label = 'Random Gene'
        elif label == 'gearsPertEmb':
            label = 'GEARS'
        
        ax.plot(x_values, y_values, marker='o', linewidth=2, markersize=8, 
               label=label, alpha=0.8, color=colors[baseline_idx])
        
        # Add CI for top_k=0.05 if available
        if len(resampling_data) > 0:
            ci_data = resampling_data.iloc[0]
            # Find the point at top_k=0.05
            k_05_idx = np.where(baseline_data['top_k'].values == 0.05)[0]
            if len(k_05_idx) > 0:
                k_05_x = x_values[k_05_idx[0]]
                # Use CI values directly from resampling data
                ci_low = ci_data['r_ci_low']
                ci_high = ci_data['r_ci_high']
                ci_mean = ci_data['r_mean']
                # Errorbar format
                yerr_lower = max(0, ci_mean - ci_low)
                yerr_upper = max(0, ci_high - ci_mean)
                if yerr_lower > 0 or yerr_upper > 0:
                    ax.errorbar(k_05_x, ci_mean, yerr=[[yerr_lower], [yerr_upper]], 
                               fmt='none', capsize=5, capthick=2, alpha=0.7, color=colors[baseline_idx])
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Mean Pearson r', fontsize=12)
    ax.set_title(f'{dataset.upper()}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

plt.tight_layout()

# Save figure
output_path = project_root / "poster" / "figure1_neighborhood_smoothness.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved figure to: {output_path}")

# Also save as PDF
output_path_pdf = project_root / "poster" / "figure1_neighborhood_smoothness.pdf"
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"Saved figure to: {output_path_pdf}")

plt.close()

print("Figure 1 created successfully!")

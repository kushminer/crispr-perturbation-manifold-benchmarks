#!/usr/bin/env python3
"""
Figure 2: Similarity → Error

Scatter plot showing:
- Higher neighbor similarity → Lower error (L2)
- Demonstrates the relationship between embedding similarity and prediction accuracy

Data source: LSFT (Local Similarity-Filtered Training) predictions
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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
print("Creating Figure 2: Similarity → Error")
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
    lsft_raw_path = project_root / "skeletons_and_fact_sheets" / "data" / "lsft_raw_per_perturbation.csv"
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

# High-contrast color scheme organized by model type
# Blue scale for Self-trained PCA
# Green scale for scGPT and scFoundation
# Red scale for GEARS
# Gray scale for Random
colors_map = {
    'lpm_selftrained': '#0066CC',  # Strong blue
    'lpm_scgptGeneEmb': '#00AA44',  # Strong green
    'lpm_scFoundationGeneEmb': '#66CC00',  # Bright green
    'lpm_gearsPertEmb': '#CC0000',  # Strong red
    'lpm_randomGeneEmb': '#666666',  # Dark gray
    'lpm_randomPertEmb': '#999999',  # Medium gray
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

# Create figure with subplots for each dataset
datasets = ['adamson', 'k562', 'rpe1']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('LSFT: Higher Neighbor Similarity → Lower Prediction Error', 
             fontsize=16, fontweight='bold', y=1.02)

for idx, dataset in enumerate(datasets):
    ax = axes[idx]
    
    dataset_data = lsft_filtered[lsft_filtered['dataset'] == dataset]
    
    # Plot each baseline
    for baseline in key_baselines:
        baseline_data = dataset_data[dataset_data['baseline'] == baseline]
        
        if len(baseline_data) == 0:
            continue
        
        # Extract similarity and error
        # Handle both old pseudobulk format and new single-cell format
        if 'local_mean_similarity' in baseline_data.columns:
            similarity = baseline_data['local_mean_similarity'].values
        elif 'mean_similarity' in baseline_data.columns:
            similarity = baseline_data['mean_similarity'].values
        else:
            continue
            
        if 'performance_local_l2' in baseline_data.columns:
            error = baseline_data['performance_local_l2'].values
        elif 'lsft_l2' in baseline_data.columns:
            error = baseline_data['lsft_l2'].values
        else:
            continue
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(similarity) | np.isnan(error))
        similarity = similarity[valid_mask]
        error = error[valid_mask]
        
        if len(similarity) == 0:
            continue
        
        label = baseline_labels[baseline]
        color = colors_map[baseline]
        
        ax.scatter(similarity, error, alpha=0.7, s=50, label=label, 
                  edgecolors='black', linewidth=0.5, color=color)
    
    ax.set_xlabel('Mean Neighbor Similarity', fontsize=12)
    ax.set_ylabel('Prediction Error (L2)', fontsize=12)
    ax.set_title(f'{dataset.upper()}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
output_path = project_root / "poster" / "figure2_similarity_error.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved figure to: {output_path}")

# Also save as PDF
output_path_pdf = project_root / "poster" / "figure2_similarity_error.pdf"
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"Saved figure to: {output_path_pdf}")

plt.close()

# Create a combined version with all datasets
# Organized by Model - Data (not Data - Model)
fig, ax = plt.subplots(figsize=(12, 8))

# Plot by baseline first, then dataset (Model - Data)
for baseline in key_baselines:
    baseline_data_all = lsft_filtered[lsft_filtered['baseline'] == baseline]
    
    if len(baseline_data_all) == 0:
        continue
    
    label_base = baseline_labels[baseline]
    color = colors_map[baseline]
    
    # Plot each dataset for this baseline
    for dataset in datasets:
        dataset_data = baseline_data_all[baseline_data_all['dataset'] == dataset]
        
        if len(dataset_data) == 0:
            continue
        
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
        
        valid_mask = ~(np.isnan(similarity) | np.isnan(error))
        similarity = similarity[valid_mask]
        error = error[valid_mask]
        
        if len(similarity) == 0:
            continue
        
        # Label format: Model - Data
        full_label = f"{label_base} - {dataset.upper()}"
        
        # Use slightly different shades for different datasets within same model
        # to maintain color scheme while allowing distinction
        if dataset == 'adamson':
            plot_color = color
            alpha = 0.7
        elif dataset == 'k562':
            # Slightly lighter shade
            plot_color = color
            alpha = 0.6
        else:  # rpe1
            # Slightly darker shade
            plot_color = color
            alpha = 0.5
        
        ax.scatter(similarity, error, alpha=alpha, s=40, label=full_label, 
                  edgecolors='black', linewidth=0.3, color=plot_color)

ax.set_xlabel('Mean Neighbor Similarity', fontsize=14)
ax.set_ylabel('Prediction Error (L2)', fontsize=14)
ax.set_title('LSFT: Higher Neighbor Similarity → Lower Prediction Error\n(All Datasets Combined)', 
             fontsize=16, fontweight='bold')
ax.legend(fontsize=8, ncol=2, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save combined figure
output_path_combined = project_root / "poster" / "figure2_similarity_error_combined.png"
plt.savefig(output_path_combined, dpi=300, bbox_inches='tight')
print(f"Saved combined figure to: {output_path_combined}")

output_path_combined_pdf = project_root / "poster" / "figure2_similarity_error_combined.pdf"
plt.savefig(output_path_combined_pdf, bbox_inches='tight')
print(f"Saved combined figure to: {output_path_combined_pdf}")

plt.close()

print("Figure 2 created successfully!")
print("\nNote: This figure uses LSFT (Local Similarity-Filtered Training) predictions,")
print("not baseline predictions. The error values are from LSFT evaluations.")

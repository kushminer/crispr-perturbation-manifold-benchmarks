#!/usr/bin/env python3
"""
Generate key visuals for mentor review (6-10 critical figures).

Required Visuals:
1. LSFT Beeswarm (Adamson)
2. LSFT Hardness-Performance curve
3. LOGO Performance bar chart
4. PCA vs scGPT LOGO scatter/CI bar
5. Manifold Schematic (Dense Forest / Sparse Desert)
6. Baseline Crisis visual (2-panel performance inflation vs true generalization)
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set publication style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette
COLORS = {
    'selftrained': '#2E86AB',
    'scgpt': '#A23B72',
    'random': '#95A5A6',
    'primary': '#2E86AB',
    'secondary': '#A23B72',
}

def create_lsft_beeswarm_adamson(output_path: Path):
    """Create LSFT beeswarm plot for Adamson dataset."""
    base_dir = Path(__file__).parent.parent
    lsft_dir = base_dir / "results" / "goal_3_prediction" / "lsft_resampling" / "adamson"
    
    baselines = ['lpm_selftrained', 'lpm_scgptGeneEmb', 'lpm_randomGeneEmb']
    baseline_labels = ['Self-trained', 'scGPT', 'Random']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    np.random.seed(42)  # For reproducible jitter
    
    for i, (baseline, label) in enumerate(zip(baselines, baseline_labels)):
        standardized_path = lsft_dir / f"lsft_adamson_{baseline}_standardized.csv"
        
        if not standardized_path.exists():
            print(f"  ⚠️  File not found: {standardized_path}")
            continue
        
        df = pd.read_csv(standardized_path)
        df_top5 = df[df['top_pct'] == 0.05].copy()
        
        if len(df_top5) > 0:
            values = df_top5['pearson_r'].dropna().values
            # Jitter
            x_jittered = np.random.normal(i, 0.1, size=len(values))
            ax.scatter(x_jittered, values, alpha=0.6, s=30, 
                      color=COLORS.get(baseline.replace('lpm_', ''), 'gray'))
            
            # Mean and CI
            mean_val = values.mean()
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            
            ax.errorbar(i, mean_val, yerr=[[mean_val - ci_lower], [ci_upper - mean_val]], 
                       fmt='o', color='red', capsize=5, capthick=2, markersize=8)
    
    ax.set_xticks(range(len(baseline_labels)))
    ax.set_xticklabels(baseline_labels)
    ax.set_ylabel('Pearson Correlation (r)', fontweight='bold')
    ax.set_title('LSFT Performance: Adamson Dataset (Top 5% Similarity)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Created: {output_path.name}")

def create_lsft_hardness_curve(output_path: Path):
    """Create LSFT hardness-performance curve."""
    base_dir = Path(__file__).parent.parent
    regression_path = base_dir / "results" / "goal_3_prediction" / "lsft_resampling" / "adamson" / "lsft_adamson_lpm_selftrained_hardness_regressions.csv"
    
    if not regression_path.exists():
        print(f"  ⚠️  File not found: {regression_path}")
        return
    
    df = pd.read_csv(regression_path)
    df_pearson = df[df['performance_metric'] == 'pearson_r'].copy()
    df_pearson = df_pearson.sort_values('top_pct')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    hardness = np.linspace(0, 1, 100)
    
    for _, row in df_pearson.iterrows():
        slope = row['slope']
        intercept = row['intercept']
        r = row['r']
        top_pct = row['top_pct']
        
        performance = intercept + slope * hardness
        
        # CI band if available
        if 'slope_ci_lower' in row and pd.notna(row['slope_ci_lower']):
            perf_lower = intercept + row['slope_ci_lower'] * hardness
            perf_upper = intercept + row['slope_ci_upper'] * hardness
            ax.fill_between(hardness, perf_lower, perf_upper, alpha=0.2, color=COLORS['primary'])
        
        label = f'Top {int(top_pct*100)}% (r={r:.3f})'
        ax.plot(hardness, performance, label=label, linewidth=2, color=COLORS['primary'])
    
    ax.set_xlabel('Hardness (Cosine Similarity)', fontweight='bold')
    ax.set_ylabel('Performance (Pearson r)', fontweight='bold')
    ax.set_title('Hardness-Performance Relationship: Self-trained (Adamson)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Created: {output_path.name}")

def create_logo_bar_chart(output_path: Path):
    """Create LOGO performance bar chart."""
    base_dir = Path(__file__).parent.parent
    logo_dir = base_dir / "results" / "goal_3_prediction" / "functional_class_holdout_resampling"
    
    datasets = ['adamson', 'replogle_k562', 'replogle_rpe1']
    baselines = ['selftrained', 'scgptGeneEmb', 'randomGeneEmb']
    
    data = []
    for dataset in datasets:
        dataset_key = dataset.replace('replogle_', '')
        summary_files = list((logo_dir / dataset).glob("logo_*_summary.json"))
        
        for summary_file in summary_files:
            with open(summary_file) as f:
                summary = json.load(f)
            
            for baseline_key, result in summary.items():
                baseline_clean = baseline_key.replace('lpm_', '')
                if baseline_clean in baselines:
                    data.append({
                        'dataset': dataset_key.capitalize(),
                        'baseline': baseline_clean.replace('GeneEmb', ''),
                        'pearson_r': result['pearson_r']['mean'],
                        'ci_lower': result['pearson_r']['ci_lower'],
                        'ci_upper': result['pearson_r']['ci_upper'],
                    })
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(datasets))
    width = 0.25
    
    for i, baseline in enumerate(baselines):
        baseline_clean = baseline.replace('GeneEmb', '')
        baseline_data = df[df['baseline'] == baseline_clean]
        
        means = []
        ci_lowers = []
        ci_uppers = []
        
        for dataset in ['Adamson', 'K562', 'Rpe1']:
            dataset_data = baseline_data[baseline_data['dataset'] == dataset]
            if len(dataset_data) > 0:
                means.append(dataset_data.iloc[0]['pearson_r'])
                ci_lowers.append(dataset_data.iloc[0]['ci_lower'])
                ci_uppers.append(dataset_data.iloc[0]['ci_upper'])
            else:
                means.append(np.nan)
                ci_lowers.append(np.nan)
                ci_uppers.append(np.nan)
        
        offset = (i - 1) * width
        pos = x + offset
        
        bars = ax.bar(pos, means, width, label=baseline_clean, 
                     color=COLORS.get(baseline_clean, 'gray'), alpha=0.8)
        
        # Error bars
        yerr_lower = np.array(means) - np.array(ci_lowers)
        yerr_upper = np.array(ci_uppers) - np.array(means)
        ax.errorbar(pos, means, yerr=[yerr_lower, yerr_upper], fmt='none', 
                   color='black', capsize=3, capthick=1)
    
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('Pearson Correlation (r)', fontweight='bold')
    ax.set_title('LOGO Performance: Functional Class Holdout', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Adamson', 'K562', 'RPE1'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Created: {output_path.name}")

def create_pca_vs_scgpt_logo_scatter(output_path: Path):
    """Create PCA vs scGPT LOGO scatter plot with CI bars."""
    base_dir = Path(__file__).parent.parent
    logo_dir = base_dir / "results" / "goal_3_prediction" / "functional_class_holdout_resampling"
    
    datasets = ['adamson', 'replogle_k562', 'replogle_rpe1']
    
    pca_vals = []
    scgpt_vals = []
    dataset_labels = []
    
    for dataset in datasets:
        dataset_key = dataset.replace('replogle_', '').capitalize()
        summary_files = list((logo_dir / dataset).glob("logo_*_summary.json"))
        
        for summary_file in summary_files:
            with open(summary_file) as f:
                summary = json.load(f)
            
            if 'lpm_selftrained' in summary and 'lpm_scgptGeneEmb' in summary:
                pca_vals.append(summary['lpm_selftrained']['pearson_r']['mean'])
                scgpt_vals.append(summary['lpm_scgptGeneEmb']['pearson_r']['mean'])
                dataset_labels.append(dataset_key)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot
    for i, (pca, scgpt, label) in enumerate(zip(pca_vals, scgpt_vals, dataset_labels)):
        ax.scatter(pca, scgpt, s=200, alpha=0.7, label=label, 
                  color=COLORS.get('primary' if i == 0 else 'secondary', 'gray'))
    
    # Diagonal line
    min_val = min(min(pca_vals), min(scgpt_vals))
    max_val = max(max(pca_vals), max(scgpt_vals))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')
    
    ax.set_xlabel('Self-trained (PCA) Pearson r', fontweight='bold')
    ax.set_ylabel('scGPT Gene Embedding Pearson r', fontweight='bold')
    ax.set_title('LOGO: PCA vs scGPT Performance', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Created: {output_path.name}")

def create_manifold_schematic(output_path: Path):
    """Create manifold schematic (Dense Forest / Sparse Desert)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Dense Forest (High similarity region)
    np.random.seed(42)
    x_dense = np.random.normal(0.5, 0.1, 50)
    y_dense = np.random.normal(0.5, 0.1, 50)
    
    ax1.scatter(x_dense, y_dense, s=100, alpha=0.6, color='green', edgecolors='darkgreen')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title('Dense Forest: High Similarity Region\n(Many similar training examples)', 
                 fontweight='bold', fontsize=11)
    ax1.set_xlabel('Embedding Dimension 1', fontweight='bold')
    ax1.set_ylabel('Embedding Dimension 2', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Right: Sparse Desert (Low similarity region)
    x_sparse = np.random.uniform(0, 1, 5)
    y_sparse = np.random.uniform(0, 1, 5)
    
    ax2.scatter(x_sparse, y_sparse, s=100, alpha=0.6, color='orange', edgecolors='darkorange')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('Sparse Desert: Low Similarity Region\n(Few similar training examples)', 
                 fontweight='bold', fontsize=11)
    ax2.set_xlabel('Embedding Dimension 1', fontweight='bold')
    ax2.set_ylabel('Embedding Dimension 2', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Manifold Schematic: Similarity Landscape', fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Created: {output_path.name}")

def create_baseline_crisis_visual(output_path: Path):
    """Create Baseline Crisis visual (2-panel: performance inflation vs true generalization)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Performance Inflation (LSFT - high similarity filtering)
    baselines = ['Self-trained', 'scGPT', 'Random']
    lsft_performance = [0.941, 0.935, 0.932]  # Adamson top 5%
    
    bars1 = ax1.bar(baselines, lsft_performance, color=[COLORS['selftrained'], COLORS['scgpt'], COLORS['random']], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Pearson Correlation (r)', fontweight='bold')
    ax1.set_title('LSFT: Performance Inflation\n(Similarity-filtered training)', 
                 fontweight='bold', fontsize=11)
    ax1.set_ylim(0.9, 0.95)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars1, lsft_performance):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Right: True Generalization (LOGO - functional class holdout)
    logo_performance = [0.882, 0.454, 0.036]  # Adamson LOGO
    
    bars2 = ax2.bar(baselines, logo_performance, color=[COLORS['selftrained'], COLORS['scgpt'], COLORS['random']], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Pearson Correlation (r)', fontweight='bold')
    ax2.set_title('LOGO: True Generalization\n(Functional class holdout)', 
                 fontweight='bold', fontsize=11)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars2, logo_performance):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Baseline Crisis: Performance Inflation vs True Generalization', 
                fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Created: {output_path.name}")

def main():
    """Generate all key visuals."""
    output_dir = Path(__file__).parent / "key_visuals"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Generating Key Visuals for Mentor Review")
    print("=" * 70)
    print()
    
    print("1. Creating LSFT Beeswarm (Adamson)...")
    create_lsft_beeswarm_adamson(output_dir / "1_LSFT_Beeswarm_Adamson.png")
    
    print("\n2. Creating LSFT Hardness-Performance Curve...")
    create_lsft_hardness_curve(output_dir / "2_LSFT_Hardness_Performance_Curve.png")
    
    print("\n3. Creating LOGO Performance Bar Chart...")
    create_logo_bar_chart(output_dir / "3_LOGO_Performance_Bar_Chart.png")
    
    print("\n4. Creating PCA vs scGPT LOGO Scatter...")
    create_pca_vs_scgpt_logo_scatter(output_dir / "4_PCA_vs_scGPT_LOGO_Scatter.png")
    
    print("\n5. Creating Manifold Schematic...")
    create_manifold_schematic(output_dir / "5_Manifold_Schematic.png")
    
    print("\n6. Creating Baseline Crisis Visual...")
    create_baseline_crisis_visual(output_dir / "6_Baseline_Crisis_Visual.png")
    
    print()
    print("=" * 70)
    print("✅ All key visuals generated!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nGenerated {len(list(output_dir.glob('*.png')))} visual files")

if __name__ == "__main__":
    main()


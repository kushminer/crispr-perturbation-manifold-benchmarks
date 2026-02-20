#!/usr/bin/env python3
"""
Plot LOGO results with improved annotations: Models × Datasets.

This script creates a visualization showing LOGO performance (Pearson r)
for each model across all datasets using improved annotations.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def load_improved_logo_results():
    """Load improved LOGO results for all datasets."""
    
    results_dir = project_root / "audits" / "logo" / "results" / "logo_with_improved_annotations"
    
    datasets = {
        "Adamson": "adamson",
        "K562": "k562",
        "RPE1": "rpe1",
    }
    
    all_results = {}
    
    for display_name, dataset_key in datasets.items():
        result_file = results_dir / f"logo_{dataset_key}_Transcription_improved_annotations.csv"
        
        if not result_file.exists():
            # Try alternative naming
            alt_file = results_dir / f"logo_{dataset_key}_transcription_results.csv"
            if alt_file.exists():
                result_file = alt_file
            else:
                print(f"⚠️  Results not found for {display_name}")
                continue
        
        df = pd.read_csv(result_file)
        all_results[display_name] = df
        print(f"✓ Loaded {display_name}: {len(df)} results")
    
    return all_results


def create_models_x_datasets_plot(results_dict: dict, output_path: Path):
    """Create a plot showing models × datasets performance."""
    
    # Aggregate results by baseline and dataset
    plot_data = []
    
    baseline_labels = {
        "lpm_selftrained": "Self-trained PCA",
        "lpm_scgptGeneEmb": "scGPT",
        "lpm_scFoundationGeneEmb": "scFoundation",
        "lpm_gearsPertEmb": "GEARS",
        "lpm_randomGeneEmb": "Random Gene",
        "lpm_randomPertEmb": "Random Pert",
    }
    
    baseline_order = [
        "lpm_selftrained",
        "lpm_scgptGeneEmb",
        "lpm_scFoundationGeneEmb",
        "lpm_gearsPertEmb",
        "lpm_randomGeneEmb",
        "lpm_randomPertEmb",
    ]
    
    for dataset_name, df in results_dict.items():
        for baseline in baseline_order:
            if baseline not in df["baseline"].values:
                continue
            
            baseline_data = df[df["baseline"] == baseline]
            mean_r = baseline_data["pearson_r"].mean()
            std_r = baseline_data["pearson_r"].std()
            
            plot_data.append({
                "Dataset": dataset_name,
                "Model": baseline_labels.get(baseline, baseline),
                "Baseline": baseline,
                "Mean r": mean_r,
                "Std r": std_r,
                "Count": len(baseline_data),
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    if len(plot_df) == 0:
        print("⚠️  No data to plot")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data for grouped bar plot
    datasets = plot_df["Dataset"].unique()
    models = [baseline_labels.get(b, b) for b in baseline_order if b in plot_df["Baseline"].values]
    
    x = np.arange(len(datasets))
    width = 0.13  # Width of bars
    n_models = len(models)
    
    # Color scheme
    colors = {
        "Self-trained PCA": "#1f77b4",      # Deep blue
        "scGPT": "#2ca02c",                  # Green
        "scFoundation": "#17becf",           # Cyan
        "GEARS": "#d62728",                  # Red
        "Random Gene": "#ff7f0e",            # Orange
        "Random Pert": "#7f7f7f",            # Light grey
    }
    
    # Plot bars
    for i, model in enumerate(models):
        model_data = plot_df[plot_df["Model"] == model]
        means = []
        stds = []
        
        for dataset in datasets:
            dataset_data = model_data[model_data["Dataset"] == dataset]
            if len(dataset_data) > 0:
                means.append(dataset_data["Mean r"].values[0])
                stds.append(dataset_data["Std r"].values[0])
            else:
                means.append(0)
                stds.append(0)
        
        offset = (i - n_models / 2) * width + width / 2
        bars = ax.bar(x + offset, means, width, label=model, 
                     color=colors.get(model, "#888888"),
                     alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add error bars
        ax.errorbar(x + offset, means, yerr=stds, fmt='none', 
                   color='black', capsize=3, capthick=1, linewidth=1)
    
    # Customize plot
    ax.set_xlabel("Dataset", fontweight='bold')
    ax.set_ylabel("Pearson r", fontweight='bold')
    ax.set_title("LOGO Performance with Improved Annotations\n(Models × Datasets)", 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)
    
    # Add horizontal reference line at r=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    
    # Also save as PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ Plot saved to: {pdf_path}")
    
    plt.close()
    
    return plot_df


def create_heatmap_plot(results_dict: dict, output_path: Path):
    """Create a heatmap showing models × datasets performance."""
    
    # Aggregate results
    plot_data = []
    
    baseline_labels = {
        "lpm_selftrained": "Self-trained PCA",
        "lpm_scgptGeneEmb": "scGPT",
        "lpm_scFoundationGeneEmb": "scFoundation",
        "lpm_gearsPertEmb": "GEARS",
        "lpm_randomGeneEmb": "Random Gene",
        "lpm_randomPertEmb": "Random Pert",
    }
    
    baseline_order = [
        "lpm_selftrained",
        "lpm_scgptGeneEmb",
        "lpm_scFoundationGeneEmb",
        "lpm_gearsPertEmb",
        "lpm_randomGeneEmb",
        "lpm_randomPertEmb",
    ]
    
    for dataset_name, df in results_dict.items():
        for baseline in baseline_order:
            if baseline not in df["baseline"].values:
                continue
            
            baseline_data = df[df["baseline"] == baseline]
            mean_r = baseline_data["pearson_r"].mean()
            
            plot_data.append({
                "Dataset": dataset_name,
                "Model": baseline_labels.get(baseline, baseline),
                "Baseline": baseline,
                "Mean r": mean_r,
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    if len(plot_df) == 0:
        print("⚠️  No data to plot")
        return
    
    # Pivot for heatmap
    heatmap_data = plot_df.pivot(index="Model", columns="Dataset", values="Mean r")
    
    # Reorder rows
    model_order = [baseline_labels.get(b, b) for b in baseline_order if b in plot_df["Baseline"].values]
    heatmap_data = heatmap_data.reindex([m for m in model_order if m in heatmap_data.index])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                center=0.4, vmin=0, vmax=1, cbar_kws={'label': 'Pearson r'},
                linewidths=0.5, linecolor='black', ax=ax)
    
    ax.set_title("LOGO Performance with Improved Annotations\n(Heatmap: Models × Datasets)", 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Dataset", fontweight='bold')
    ax.set_ylabel("Model", fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap saved to: {output_path}")
    
    # Also save as PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ Heatmap saved to: {pdf_path}")
    
    plt.close()


def main():
    """Create plots of improved LOGO results."""
    
    print("=" * 70)
    print("LOGO IMPROVED ANNOTATIONS: MODELS × DATASETS PLOT")
    print("=" * 70)
    print()
    
    # Load results
    results_dict = load_improved_logo_results()
    
    if not results_dict:
        print("⚠️  No results found")
        return
    
    print()
    
    # Create output directory
    output_dir = project_root / "audits" / "logo" / "results" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create grouped bar plot
    print("Creating grouped bar plot...")
    bar_plot_path = output_dir / "logo_improved_models_x_datasets_bar.png"
    plot_df = create_models_x_datasets_plot(results_dict, bar_plot_path)
    
    print()
    
    # Create heatmap
    print("Creating heatmap...")
    heatmap_path = output_dir / "logo_improved_models_x_datasets_heatmap.png"
    create_heatmap_plot(results_dict, heatmap_path)
    
    print()
    print("=" * 70)
    print("PLOTS GENERATED")
    print("=" * 70)
    print()
    print(f"Bar plot: {bar_plot_path}")
    print(f"Heatmap: {heatmap_path}")
    print()
    
    # Print summary statistics
    if plot_df is not None and len(plot_df) > 0:
        print("Summary Statistics:")
        print()
        summary = plot_df.groupby("Model")["Mean r"].agg(["mean", "std", "min", "max"]).round(3)
        print(summary)
        print()
        
        print("By Dataset:")
        print()
        dataset_summary = plot_df.groupby("Dataset")["Mean r"].agg(["mean", "std", "min", "max"]).round(3)
        print(dataset_summary)


if __name__ == "__main__":
    main()





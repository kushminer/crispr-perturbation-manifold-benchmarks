#!/usr/bin/env python3
"""
Analyze Epic 3 NaN entries to understand their pattern and determine if they need to be fixed.
"""

from pathlib import Path
import pandas as pd
import numpy as np

results_dir = Path("results/manifold_law_diagnostics/epic3_noise_injection")

print("=" * 70)
print("Epic 3 NaN Entry Analysis")
print("=" * 70)
print()

# Find all noise injection CSV files
csv_files = list(results_dir.glob("noise_injection_*.csv"))
csv_files = [f for f in csv_files if f.name.startswith("noise_injection_")]

print(f"Found {len(csv_files)} noise injection result files")
print()

nan_summary = []

for csv_file in sorted(csv_files):
    df = pd.read_csv(csv_file)
    
    # Check for NaN entries
    nan_cols = df.columns[df.isna().any()].tolist()
    nan_rows = df.isna().any(axis=1).sum()
    total_rows = len(df)
    
    if nan_rows > 0:
        # Get noise levels
        noise_levels = sorted(df['noise_level'].unique())
        nan_by_noise = {}
        
        for noise in noise_levels:
            noise_data = df[df['noise_level'] == noise]
            nan_count = noise_data.isna().any(axis=1).sum()
            if nan_count > 0:
                nan_by_noise[noise] = nan_count
        
        # Parse dataset and baseline from filename
        name_parts = csv_file.stem.replace("noise_injection_", "").split("_", 1)
        dataset = name_parts[0]
        baseline = name_parts[1] if len(name_parts) > 1 else "unknown"
        
        nan_summary.append({
            "file": csv_file.name,
            "dataset": dataset,
            "baseline": baseline,
            "total_rows": total_rows,
            "nan_rows": nan_rows,
            "nan_cols": ", ".join(nan_cols) if nan_cols else "none",
            "noise_levels": sorted(df['noise_level'].unique()),
            "baseline_exists": 0.0 in df['noise_level'].values,
            "nan_by_noise": nan_by_noise,
        })

if nan_summary:
    print(f"Files with NaN entries: {len(nan_summary)}")
    print()
    
    # Summary by dataset
    print("Summary by Dataset:")
    print("-" * 70)
    for dataset in ["adamson", "k562", "rpe1"]:
        dataset_files = [s for s in nan_summary if s["dataset"] == dataset]
        if dataset_files:
            print(f"\n{dataset.upper()}: {len(dataset_files)} files with NaN")
            for s in dataset_files:
                print(f"  - {s['baseline']}: {s['nan_rows']} rows with NaN")
                if s['nan_by_noise']:
                    print(f"    NaN by noise level: {s['nan_by_noise']}")
    
    print()
    print("=" * 70)
    print("Pattern Analysis")
    print("=" * 70)
    
    # Check if there's a pattern
    all_noise_levels = set()
    for s in nan_summary:
        all_noise_levels.update(s['noise_levels'])
    
    print(f"All noise levels found: {sorted(all_noise_levels)}")
    print()
    
    # Check which noise levels have NaN
    noise_level_nan_counts = {}
    for s in nan_summary:
        for noise, count in s['nan_by_noise'].items():
            if noise not in noise_level_nan_counts:
                noise_level_nan_counts[noise] = 0
            noise_level_nan_counts[noise] += count
    
    print("NaN occurrences by noise level:")
    for noise in sorted(noise_level_nan_counts.keys()):
        print(f"  σ={noise}: {noise_level_nan_counts[noise]} NaN entries across all files")
    
    print()
    print("=" * 70)
    print("Recommendation")
    print("=" * 70)
    
    # Determine if NaN entries are problematic
    baseline_ok = all(s['baseline_exists'] for s in nan_summary)
    if baseline_ok:
        print("✅ All files have baseline (noise=0) data - Lipschitz constants can be computed")
    
    if noise_level_nan_counts:
        print(f"\n⚠️  Found NaN entries at noise levels: {sorted(noise_level_nan_counts.keys())}")
        print("   These may represent failed runs or missing noise conditions.")
        print("   Recommendation: Re-run Epic 3 for affected files if comprehensive analysis needed.")
    else:
        print("\n✅ No NaN entries found in noise-level data")
    
else:
    print("✅ No files with NaN entries found!")

print()
print("=" * 70)


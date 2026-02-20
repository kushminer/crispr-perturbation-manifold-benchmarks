#!/usr/bin/env python3
"""
Generate minimum data tables for mentor review.

Creates three CSV files:
1. LSFT Summary Table
2. LOGO Summary Table
3. Hardness-Performance Regression Table
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List

def load_lsft_summaries(base_dir: Path) -> pd.DataFrame:
    """Load and aggregate LSFT summary data across all datasets and baselines."""
    datasets = ['adamson', 'k562', 'rpe1']
    rows = []
    
    for dataset in datasets:
        dataset_dir = base_dir / dataset
        
        # Find all baseline summary files
        summary_files = list(dataset_dir.glob(f"lsft_{dataset}_*_summary.json"))
        
        for summary_file in summary_files:
            baseline = summary_file.stem.replace(f"lsft_{dataset}_", "").replace("_summary", "")
            
            with open(summary_file) as f:
                data = json.load(f)
            
            # Extract top_pct=0.05 results (most representative)
            key = f"{baseline}_top5pct"
            if key in data:
                result = data[key]
                rows.append({
                    'baseline': baseline.replace('lpm_', ''),
                    'dataset': dataset,
                    'pearson_r': result['pearson_r']['mean'],
                    'pearson_ci_lower': result['pearson_r']['ci_lower'],
                    'pearson_ci_upper': result['pearson_r']['ci_upper'],
                    'l2': result['l2']['mean'],
                    'l2_ci_lower': result['l2']['ci_lower'],
                    'l2_ci_upper': result['l2']['ci_upper'],
                    'n_test': result['n_perturbations'],
                })
    
    return pd.DataFrame(rows)

def load_logo_summaries(base_dir: Path) -> pd.DataFrame:
    """Load and aggregate LOGO summary data across all datasets."""
    datasets = {
        'adamson': 'adamson',
        'k562': 'replogle_k562',
        'rpe1': 'replogle_rpe1',
    }
    
    rows = []
    
    for dataset_key, dataset_dir_name in datasets.items():
        dataset_dir = base_dir / dataset_dir_name
        
        # Find summary file
        summary_files = list(dataset_dir.glob("logo_*_summary.json"))
        
        for summary_file in summary_files:
            with open(summary_file) as f:
                data = json.load(f)
            
            for baseline, result in data.items():
                if isinstance(result, dict) and 'pearson_r' in result:
                    rows.append({
                        'baseline': baseline.replace('lpm_', ''),
                        'dataset': dataset_key,
                        'pearson_r': result['pearson_r']['mean'],
                        'pearson_ci_lower': result['pearson_r']['ci_lower'],
                        'pearson_ci_upper': result['pearson_r']['ci_upper'],
                        'l2': result['l2']['mean'],
                        'l2_ci_lower': result['l2']['ci_lower'],
                        'l2_ci_upper': result['l2']['ci_upper'],
                        'n_test': result['n_perturbations'],
                    })
    
    return pd.DataFrame(rows)

def load_hardness_regressions(base_dir: Path) -> pd.DataFrame:
    """Load and aggregate hardness-performance regression data."""
    datasets = ['adamson', 'k562', 'rpe1']
    rows = []
    
    for dataset in datasets:
        dataset_dir = base_dir / dataset
        
        # Find all regression files
        regression_files = list(dataset_dir.glob(f"lsft_{dataset}_*_hardness_regressions.csv"))
        
        for reg_file in regression_files:
            baseline = reg_file.stem.replace(f"lsft_{dataset}_", "").replace("_hardness_regressions", "")
            
            df = pd.read_csv(reg_file)
            
            # Filter to pearson_r metric and top baselines
            df_pearson = df[df['performance_metric'] == 'pearson_r'].copy()
            
            for _, row in df_pearson.iterrows():
                rows.append({
                    'baseline': baseline.replace('lpm_', ''),
                    'dataset': dataset,
                    'top_pct': row['top_pct'],
                    'slope': row['slope'],
                    'slope_ci_lower': row.get('slope_ci_lower', None),
                    'slope_ci_upper': row.get('slope_ci_upper', None),
                    'r': row['r'],  # Correlation between hardness and performance
                    'sample_size': row['n_points'],
                })
    
    return pd.DataFrame(rows)

def create_lsft_summary_table(df: pd.DataFrame, output_path: Path):
    """Create LSFT summary table for mentor review."""
    # Keep only key columns and top baselines
    key_baselines = ['selftrained', 'scgptGeneEmb', 'randomGeneEmb']
    
    df_filtered = df[df['baseline'].isin(key_baselines)].copy()
    
    # Create concise table
    summary = df_filtered[['baseline', 'dataset', 'pearson_r', 'pearson_ci_lower', 'pearson_ci_upper', 
                           'l2', 'n_test']].copy()
    
    # Sort by dataset, then by baseline
    summary = summary.sort_values(['dataset', 'baseline'])
    
    summary.to_csv(output_path, index=False)
    print(f"✅ Created LSFT summary table: {output_path}")
    print(f"   Rows: {len(summary)}")
    return summary

def create_logo_summary_table(df: pd.DataFrame, output_path: Path):
    """Create LOGO summary table for mentor review."""
    # Keep only key baselines
    key_baselines = ['selftrained', 'scgptGeneEmb', 'randomGeneEmb']
    
    df_filtered = df[df['baseline'].isin(key_baselines)].copy()
    
    # Create concise table
    summary = df_filtered[['baseline', 'dataset', 'pearson_r', 'pearson_ci_lower', 'pearson_ci_upper',
                           'l2', 'n_test']].copy()
    
    # Sort by dataset, then by baseline
    summary = summary.sort_values(['dataset', 'baseline'])
    
    summary.to_csv(output_path, index=False)
    print(f"✅ Created LOGO summary table: {output_path}")
    print(f"   Rows: {len(summary)}")
    return summary

def create_hardness_regression_table(df: pd.DataFrame, output_path: Path):
    """Create hardness-performance regression table for mentor review."""
    # Keep only key baselines and top_pct=0.05 (most representative)
    key_baselines = ['selftrained', 'scgptGeneEmb', 'randomGeneEmb']
    
    df_filtered = df[
        (df['baseline'].isin(key_baselines)) & 
        (df['top_pct'] == 0.05)
    ].copy()
    
    # Create concise table
    summary = df_filtered[['baseline', 'dataset', 'top_pct', 'slope', 'slope_ci_lower', 'slope_ci_upper',
                           'r', 'sample_size']].copy()
    
    # Sort by dataset, then by baseline
    summary = summary.sort_values(['dataset', 'baseline'])
    
    summary.to_csv(output_path, index=False)
    print(f"✅ Created hardness regression table: {output_path}")
    print(f"   Rows: {len(summary)}")
    return summary

def main():
    """Generate all summary tables."""
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "mentor_review" / "data_tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lsft_base = base_dir / "results" / "goal_3_prediction" / "lsft_resampling"
    logo_base = base_dir / "results" / "goal_3_prediction" / "functional_class_holdout_resampling"
    
    print("=" * 70)
    print("Generating Minimum Data Tables for Mentor Review")
    print("=" * 70)
    print()
    
    # 1. LSFT Summary Table
    print("1. Loading LSFT summaries...")
    lsft_df = load_lsft_summaries(lsft_base)
    print(f"   Loaded {len(lsft_df)} baseline-dataset combinations")
    lsft_summary = create_lsft_summary_table(
        lsft_df, 
        output_dir / "A_LSFT_Summary_Table.csv"
    )
    
    print()
    
    # 2. LOGO Summary Table
    print("2. Loading LOGO summaries...")
    logo_df = load_logo_summaries(logo_base)
    print(f"   Loaded {len(logo_df)} baseline-dataset combinations")
    logo_summary = create_logo_summary_table(
        logo_df,
        output_dir / "B_LOGO_Summary_Table.csv"
    )
    
    print()
    
    # 3. Hardness-Performance Regression Table
    print("3. Loading hardness regressions...")
    hardness_df = load_hardness_regressions(lsft_base)
    print(f"   Loaded {len(hardness_df)} regression results")
    hardness_summary = create_hardness_regression_table(
        hardness_df,
        output_dir / "C_Hardness_Performance_Regression_Table.csv"
    )
    
    print()
    print("=" * 70)
    print("✅ All summary tables generated!")
    print("=" * 70)
    print()
    print(f"Output directory: {output_dir}")
    print()
    print("Tables created:")
    print(f"  A. LSFT Summary: {len(lsft_summary)} rows")
    print(f"  B. LOGO Summary: {len(logo_summary)} rows")
    print(f"  C. Hardness Regression: {len(hardness_summary)} rows")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Compare our implementation results with paper's published results.

Usage:
    python -m goal_2_baselines.compare_with_paper_results \
        --paper_results data/paper_results/single_perturbation_jobs_stats.tsv \
        --our_results results/goal_2_baselines/adamson_reproduced/baseline_results_reproduced.csv \
        --output_dir results/comparison_with_paper
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def load_paper_results(paper_results_path: Path) -> pd.DataFrame:
    """Load paper's published results."""
    LOGGER.info(f"Loading paper results from {paper_results_path}")
    
    # Try different separators and encodings
    try:
        df = pd.read_csv(paper_results_path, sep='\t')
    except:
        try:
            df = pd.read_csv(paper_results_path, sep=',')
        except:
            df = pd.read_csv(paper_results_path, sep='\t', encoding='latin-1')
    
    LOGGER.info(f"Loaded {len(df)} rows, columns: {df.columns.tolist()}")
    return df


def load_our_results(results_path: Path) -> pd.DataFrame:
    """Load our implementation results."""
    LOGGER.info(f"Loading our results from {results_path}")
    df = pd.read_csv(results_path)
    LOGGER.info(f"Loaded {len(df)} rows, columns: {df.columns.tolist()}")
    return df


def normalize_baseline_name(name: str) -> str:
    """Normalize baseline names for comparison."""
    # Map variations to standard names
    name_lower = name.lower()
    
    mapping = {
        'lpm_selftrained': 'lpm_selftrained',
        'selftrained': 'lpm_selftrained',
        'self_trained': 'lpm_selftrained',
        'lpm_k562pertemb': 'lpm_k562PertEmb',
        'k562pertemb': 'lpm_k562PertEmb',
        'k562_pert_emb': 'lpm_k562PertEmb',
        'lpm_rpe1pertemb': 'lpm_rpe1PertEmb',
        'rpe1pertemb': 'lpm_rpe1PertEmb',
        'rpe1_pert_emb': 'lpm_rpe1PertEmb',
        'lpm_randompertemb': 'lpm_randomPertEmb',
        'randompertemb': 'lpm_randomPertEmb',
        'lpm_randomgeneemb': 'lpm_randomGeneEmb',
        'randomgeneemb': 'lpm_randomGeneEmb',
        'lpm_scgptgeneemb': 'lpm_scgptGeneEmb',
        'scgptgeneemb': 'lpm_scgptGeneEmb',
        'lpm_scfoundationgeneemb': 'lpm_scFoundationGeneEmb',
        'scfoundationgeneemb': 'lpm_scFoundationGeneEmb',
        'lpm_gearspertemb': 'lpm_gearsPertEmb',
        'gearspertemb': 'lpm_gearsPertEmb',
        'mean_response': 'mean_response',
        'meanresponse': 'mean_response',
    }
    
    return mapping.get(name_lower, name)


def extract_metric_from_paper(df: pd.DataFrame, metric_name: str = 'pearson_r') -> Dict[str, float]:
    """Extract metric values from paper results."""
    LOGGER.info(f"DataFrame shape: {df.shape}")
    LOGGER.info(f"Columns: {df.columns.tolist()}")
    
    # Check if this is a job stats file (has 'name', 'metric', 'value' columns)
    if 'name' in df.columns and 'metric' in df.columns and 'value' in df.columns:
        LOGGER.info("Detected job stats format (name, metric, value)")
        
        # Filter for performance metrics only
        perf_metric_keywords = ['pearson', 'correlation', 'r2', 'r_squared', 'mean_pearson', 'pearson_r', 
                                'spearman', 'mse', 'mae', 'l2', 'accuracy', 'f1']
        
        perf_df = df[df['metric'].str.contains('|'.join(perf_metric_keywords), case=False, na=False)]
        
        if len(perf_df) == 0:
            LOGGER.warning("No performance metrics found in file. This might be a job statistics file.")
            LOGGER.info("Available metrics: " + str(df['metric'].unique()[:10].tolist()))
            LOGGER.info("Trying to extract baseline names from 'name' column...")
            
            # Try to extract baseline info from names
            # Format might be like "adamson-1-scgpt" or "adamson-1-lpm_selftrained"
            results = {}
            unique_names = df['name'].unique()
            
            for name in unique_names:
                # Extract baseline name from job name
                # Pattern: dataset-seed-baseline
                parts = str(name).split('-')
                if len(parts) >= 3:
                    baseline_candidate = '-'.join(parts[2:])  # Everything after dataset-seed
                    normalized = normalize_baseline_name(baseline_candidate)
                    
                    # If we can't find performance metrics, return empty dict with a note
                    LOGGER.warning(f"Could not find performance metrics. Found job names like: {name}")
                    LOGGER.warning("This file appears to contain job statistics, not performance results.")
                    LOGGER.warning("Please check if there's a different file with performance metrics.")
            
            return {}  # Return empty - we'll need a different file
        
        # Process performance metrics
        results = {}
        for _, row in perf_df.iterrows():
            name = str(row['name'])
            metric = str(row['metric']).lower()
            value = row['value']
            
            # Extract baseline from name (format: dataset-seed-baseline)
            parts = name.split('-')
            if len(parts) >= 3:
                baseline_candidate = '-'.join(parts[2:])
                normalized = normalize_baseline_name(baseline_candidate)
                
                # Only store if it's the metric we're looking for
                if metric_name.lower() in metric or 'pearson' in metric or 'correlation' in metric:
                    if pd.notna(value):
                        try:
                            results[normalized] = float(value)
                        except (ValueError, TypeError):
                            pass
        
        return results
    
    # Try to find the metric column
    metric_cols = [col for col in df.columns if metric_name.lower() in col.lower() or 'r' in col.lower() or 'correlation' in col.lower()]
    
    # Also look for baseline name column
    baseline_cols = [col for col in df.columns if 'baseline' in col.lower() or 'model' in col.lower() or 'method' in col.lower() or 'name' in col.lower()]
    
    LOGGER.info(f"Found metric columns: {metric_cols}")
    LOGGER.info(f"Found baseline columns: {baseline_cols}")
    
    results = {}
    
    if metric_cols and baseline_cols:
        metric_col = metric_cols[0]
        baseline_col = baseline_cols[0]
        
        for _, row in df.iterrows():
            baseline_name = str(row[baseline_col])
            normalized_name = normalize_baseline_name(baseline_name)
            metric_value = row[metric_col]
            
            if pd.notna(metric_value):
                try:
                    results[normalized_name] = float(metric_value)
                except (ValueError, TypeError):
                    pass
    elif len(df.columns) >= 2:
        # Assume first column is baseline name, second is metric
        for _, row in df.iterrows():
            baseline_name = str(row.iloc[0])
            normalized_name = normalize_baseline_name(baseline_name)
            metric_value = row.iloc[1]
            
            if pd.notna(metric_value):
                try:
                    results[normalized_name] = float(metric_value)
                except (ValueError, TypeError):
                    pass
    
    return results


def compare_results(
    paper_results: Dict[str, float],
    our_results: pd.DataFrame,
    metric_col: str = 'mean_pearson_r',
) -> pd.DataFrame:
    """Compare our results with paper's published results."""
    comparison = []
    
    # Get our results
    our_dict = {}
    for _, row in our_results.iterrows():
        baseline = normalize_baseline_name(row['baseline'])
        if metric_col in row:
            our_dict[baseline] = row[metric_col]
    
    # Compare all baselines
    all_baselines = set(list(paper_results.keys()) + list(our_dict.keys()))
    
    for baseline in all_baselines:
        paper_val = paper_results.get(baseline, np.nan)
        our_val = our_dict.get(baseline, np.nan)
        
        # Compute differences
        our_diff = our_val - paper_val if pd.notna(our_val) and pd.notna(paper_val) else np.nan
        
        comparison.append({
            'baseline': baseline,
            'paper_published': paper_val,
            'our_implementation': our_val,
            'our_vs_paper_diff': our_diff,
        })
    
    return pd.DataFrame(comparison)


def generate_report(
    comparison_df: pd.DataFrame,
    output_dir: Path,
    metric_name: str = 'Pearson r',
) -> None:
    """Generate comparison report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save comparison table
    comparison_path = output_dir / 'comparison_with_paper.csv'
    comparison_df.to_csv(comparison_path, index=False)
    LOGGER.info(f"Saved comparison to {comparison_path}")
    
    # Generate summary statistics
    summary = {
        'metric': metric_name,
        'n_baselines_compared': len(comparison_df),
        'n_baselines_in_paper': comparison_df['paper_published'].notna().sum(),
        'n_baselines_in_our_impl': comparison_df['our_implementation'].notna().sum(),
    }
    
    # Compute mean absolute differences
    our_diffs = comparison_df['our_vs_paper_diff'].abs()
    
    summary['mean_abs_diff_our_vs_paper'] = our_diffs.mean() if our_diffs.notna().any() else np.nan
    summary['max_abs_diff_our_vs_paper'] = our_diffs.max() if our_diffs.notna().any() else np.nan
    
    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_path = output_dir / 'comparison_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    LOGGER.info(f"Saved summary to {summary_path}")
    
    # Generate markdown report
    report_path = output_dir / 'COMPARISON_REPORT.md'
    with open(report_path, 'w') as f:
        f.write("# Comparison with Paper Results\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- **Metric**: {metric_name}\n")
        f.write(f"- **Baselines Compared**: {summary['n_baselines_compared']}\n")
        f.write(f"- **Baselines in Paper**: {summary['n_baselines_in_paper']}\n")
        f.write(f"- **Baselines in Our Implementation**: {summary['n_baselines_in_our_impl']}\n\n")
        
        f.write(f"## Differences\n\n")
        f.write(f"- **Mean Absolute Difference (Our vs Paper)**: {summary['mean_abs_diff_our_vs_paper']:.6f}\n")
        f.write(f"- **Max Absolute Difference (Our vs Paper)**: {summary['max_abs_diff_our_vs_paper']:.6f}\n\n")
        
        f.write(f"## Detailed Comparison\n\n")
        f.write(comparison_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write(f"## Notes\n\n")
        f.write(f"- NaN values indicate that a baseline was not found in that implementation.\n")
        f.write(f"- Differences are computed as: implementation_value - paper_published_value\n")
        f.write(f"- Positive differences indicate our implementation performs better than paper.\n")
    
    LOGGER.info(f"Saved report to {report_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Comparison with Paper Results")
    print("=" * 80)
    print(f"\nMetric: {metric_name}")
    print(f"\nSummary:")
    print(f"  Baselines compared: {summary['n_baselines_compared']}")
    print(f"  Baselines in paper: {summary['n_baselines_in_paper']}")
    print(f"  Baselines in our implementation: {summary['n_baselines_in_our_impl']}")
    print(f"\nDifferences:")
    print(f"  Mean abs diff (Our vs Paper): {summary['mean_abs_diff_our_vs_paper']:.6f}")
    print(f"  Max abs diff (Our vs Paper): {summary['max_abs_diff_our_vs_paper']:.6f}")
    print("\n" + "=" * 80)
    print("\nDetailed Comparison:")
    print(comparison_df.to_string(index=False))
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare our implementation results with paper's published results"
    )
    parser.add_argument(
        "--paper_results",
        type=Path,
        required=True,
        help="Path to paper's published results TSV file",
    )
    parser.add_argument(
        "--our_results",
        type=Path,
        required=True,
        help="Path to our implementation results CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/comparison_with_paper"),
        help="Output directory for comparison results",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mean_pearson_r",
        help="Metric column name to compare (default: mean_pearson_r)",
    )
    
    args = parser.parse_args()
    
    # Load results
    paper_df = load_paper_results(args.paper_results)
    our_df = load_our_results(args.our_results)
    
    # Extract paper results
    paper_results = extract_metric_from_paper(paper_df, metric_name='pearson_r')
    
    if not paper_results:
        LOGGER.warning("Could not extract results from paper file. Trying alternative approach...")
        # Try to use the dataframe directly
        if len(paper_df.columns) >= 2:
            paper_results = {}
            for _, row in paper_df.iterrows():
                baseline = normalize_baseline_name(str(row.iloc[0]))
                value = row.iloc[1]
                if pd.notna(value):
                    try:
                        paper_results[baseline] = float(value)
                    except:
                        pass
    
    LOGGER.info(f"Extracted {len(paper_results)} baselines from paper results: {list(paper_results.keys())}")
    
    # Compare results
    comparison_df = compare_results(
        paper_results=paper_results,
        our_results=our_df,
        metric_col=args.metric,
    )
    
    # Generate report
    generate_report(
        comparison_df=comparison_df,
        output_dir=args.output_dir,
        metric_name=args.metric.replace('_', ' ').title(),
    )
    
    return 0


if __name__ == "__main__":
    exit(main())


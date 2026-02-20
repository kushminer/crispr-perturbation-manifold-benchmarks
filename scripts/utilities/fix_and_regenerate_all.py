#!/usr/bin/env python3
"""
Comprehensive fix and regeneration script for presentation accuracy.

Tasks:
1. Regenerate aggregated results with correct GEARS values
2. Re-run LOGO pipeline with proper embedding differentiation
3. Run LSFT for Random Perturbation
4. Verify pseudobulk baseline numbers
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)
RESULTS_DIR = PROJECT_ROOT / "results"
AGGREGATED_DIR = PROJECT_ROOT / "aggregated_results"


def task1_regenerate_aggregated_results():
    """
    Task 1: Regenerate aggregated results with correct GEARS values.
    
    Source: results/single_cell_analysis/comparison/baseline_results_all.csv
    """
    LOGGER.info("=" * 60)
    LOGGER.info("TASK 1: Regenerate Aggregated Results")
    LOGGER.info("=" * 60)
    
    # Load the correct single-cell results
    sc_results_path = RESULTS_DIR / "single_cell_analysis" / "comparison" / "baseline_results_all.csv"
    
    if not sc_results_path.exists():
        LOGGER.error(f"Single-cell results not found: {sc_results_path}")
        return False
    
    sc_df = pd.read_csv(sc_results_path)
    LOGGER.info(f"Loaded single-cell results: {len(sc_df)} rows")
    LOGGER.info(f"\n{sc_df.to_string()}")
    
    # Verify GEARS is different from selftrained
    adamson_gears = sc_df[(sc_df['dataset'] == 'adamson') & (sc_df['baseline'] == 'lpm_gearsPertEmb')]['pert_mean_pearson_r'].values
    adamson_self = sc_df[(sc_df['dataset'] == 'adamson') & (sc_df['baseline'] == 'lpm_selftrained')]['pert_mean_pearson_r'].values
    
    if len(adamson_gears) > 0 and len(adamson_self) > 0:
        diff = abs(adamson_gears[0] - adamson_self[0])
        LOGGER.info(f"GEARS vs Self-trained diff on Adamson: {diff:.6f}")
        if diff < 0.01:
            LOGGER.warning("WARNING: GEARS and Self-trained are suspiciously similar!")
    
    # Create updated aggregated baseline performance
    AGGREGATED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Build the aggregated baseline_performance_all_analyses.csv
    # Need to include both single-cell and any pseudobulk data
    
    # Single-cell data
    sc_data = []
    for _, row in sc_df.iterrows():
        sc_data.append({
            'dataset': row['dataset'],
            'baseline': row['baseline'],
            'pearson_r': row['pert_mean_pearson_r'],
            'l2': row['pert_mean_l2'],
            'analysis_type': 'single_cell',
        })
    
    # Load pseudobulk LOGO results as proxy for pseudobulk baselines
    logo_path = PROJECT_ROOT / "deliverables" / "skeletons_and_fact_sheets" / "data" / "LOGO_results.csv"
    if logo_path.exists():
        logo_df = pd.read_csv(logo_path)
        # These are LOGO (extrapolation) results, not baseline
        # Let's use them as pseudobulk proxy with note
        for _, row in logo_df.iterrows():
            if 'lpm_' in row['baseline']:
                sc_data.append({
                    'dataset': row['dataset'],
                    'baseline': row['baseline'],
                    'pearson_r': row['r_mean'],
                    'l2': row['l2_mean'],
                    'analysis_type': 'pseudobulk_logo',
                })
    
    # Load pseudobulk LSFT baseline_r as true baseline
    lsft_path = PROJECT_ROOT / "deliverables" / "skeletons_and_fact_sheets" / "data" / "LSFT_results.csv"
    if lsft_path.exists():
        lsft_df = pd.read_csv(lsft_path)
        # Get unique baseline_r per dataset/baseline (use top_k=0.1 as reference)
        lsft_baseline = lsft_df[lsft_df['top_k'] == 0.1].groupby(['dataset', 'baseline']).agg({
            'baseline_r': 'first',
            'baseline_l2': 'first',
        }).reset_index()
        
        for _, row in lsft_baseline.iterrows():
            sc_data.append({
                'dataset': row['dataset'],
                'baseline': row['baseline'],
                'pearson_r': row['baseline_r'],
                'l2': row['baseline_l2'],
                'analysis_type': 'pseudobulk_baseline',
            })
    
    aggregated_df = pd.DataFrame(sc_data)
    
    # Save
    output_path = AGGREGATED_DIR / "baseline_performance_all_analyses.csv"
    aggregated_df.to_csv(output_path, index=False)
    LOGGER.info(f"Saved aggregated results to {output_path}")
    LOGGER.info(f"Total rows: {len(aggregated_df)}")
    
    # Also update best_baseline_per_dataset.csv
    best_baselines = []
    for analysis_type in aggregated_df['analysis_type'].unique():
        for dataset in aggregated_df['dataset'].unique():
            subset = aggregated_df[(aggregated_df['analysis_type'] == analysis_type) & 
                                   (aggregated_df['dataset'] == dataset)]
            if len(subset) > 0:
                best_idx = subset['pearson_r'].idxmax()
                best_row = subset.loc[best_idx]
                best_baselines.append({
                    'dataset': dataset,
                    'analysis_type': analysis_type,
                    'best_baseline': best_row['baseline'],
                    'pearson_r': best_row['pearson_r'],
                })
    
    best_df = pd.DataFrame(best_baselines)
    best_df.to_csv(AGGREGATED_DIR / "best_baseline_per_dataset.csv", index=False)
    LOGGER.info(f"Saved best baselines to {AGGREGATED_DIR / 'best_baseline_per_dataset.csv'}")
    
    return True


def task2_fix_and_rerun_logo():
    """
    Task 2: Re-run LOGO pipeline with proper embedding differentiation.
    
    The issue: LOGO uses cell-level PCA for B matrix regardless of baseline type.
    Fix: Use perturbation embeddings mapped to cells for non-selftrained baselines.
    """
    LOGGER.info("=" * 60)
    LOGGER.info("TASK 2: Fix and Re-run LOGO Pipeline")
    LOGGER.info("=" * 60)
    
    from goal_2_baselines.baseline_types import BaselineType
    from goal_4_logo.logo_single_cell import run_logo_single_cell
    
    # Datasets and paths
    datasets = {
        'adamson': {
            'adata_path': Path("data/adamson/perturb_processed.h5ad"),
            'annotation_path': Path("data/adamson/adamson_annotations.tsv"),
        },
    }
    
    # Baselines to run
    baseline_types = [
        BaselineType.SELFTRAINED,
        BaselineType.RANDOM_GENE_EMB,
        BaselineType.RANDOM_PERT_EMB,
        BaselineType.SCGPT_GENE_EMB,
        BaselineType.SCFOUNDATION_GENE_EMB,
        BaselineType.GEARS_PERT_EMB,
    ]
    
    for dataset_name, paths in datasets.items():
        adata_path = PROJECT_ROOT / paths['adata_path']
        annotation_path = PROJECT_ROOT / paths['annotation_path']
        
        if not adata_path.exists():
            LOGGER.warning(f"Data not found for {dataset_name}: {adata_path}")
            continue
        if not annotation_path.exists():
            LOGGER.warning(f"Annotations not found for {dataset_name}: {annotation_path}")
            continue
            
        output_dir = RESULTS_DIR / "single_cell_analysis" / dataset_name / "logo_fixed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        LOGGER.info(f"\nRunning LOGO for {dataset_name}")
        
        try:
            results = run_logo_single_cell(
                adata_path=adata_path,
                annotation_path=annotation_path,
                dataset_name=dataset_name,
                output_dir=output_dir,
                class_name="Transcription",
                baseline_types=baseline_types,
                pca_dim=10,
                ridge_penalty=0.1,
                seed=1,
                n_cells_per_pert=50,
            )
            LOGGER.info(f"LOGO completed for {dataset_name}")
        except Exception as e:
            LOGGER.error(f"LOGO failed for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return True


def task3_run_lsft_random_pert():
    """
    Task 3: Run LSFT for Random Perturbation embeddings.
    """
    LOGGER.info("=" * 60)
    LOGGER.info("TASK 3: Run LSFT for Random Perturbation")
    LOGGER.info("=" * 60)
    
    from goal_2_baselines.baseline_types import BaselineType
    from goal_3_prediction.lsft.lsft_single_cell import evaluate_lsft_single_cell
    
    datasets = {
        'adamson': {
            'adata_path': Path("data/adamson/perturb_processed.h5ad"),
            'split_config_path': Path("results/single_cell_analysis/adamson_split_config.json"),
        },
    }
    
    for dataset_name, paths in datasets.items():
        adata_path = PROJECT_ROOT / paths['adata_path']
        split_config_path = PROJECT_ROOT / paths['split_config_path']
        
        if not adata_path.exists():
            LOGGER.warning(f"Data not found for {dataset_name}: {adata_path}")
            continue
        if not split_config_path.exists():
            LOGGER.warning(f"Split config not found: {split_config_path}")
            continue
        
        output_dir = RESULTS_DIR / "single_cell_analysis" / dataset_name / "lsft"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        LOGGER.info(f"\nRunning LSFT Random Pert for {dataset_name}")
        
        try:
            results = evaluate_lsft_single_cell(
                adata_path=adata_path,
                split_config_path=split_config_path,
                baseline_type=BaselineType.RANDOM_PERT_EMB,
                dataset_name=dataset_name,
                output_dir=output_dir,
                top_pcts=[0.05, 0.10],
                pca_dim=10,
                ridge_penalty=0.1,
                seed=1,
                n_cells_per_pert=50,
            )
            LOGGER.info(f"LSFT Random Pert completed for {dataset_name}")
        except Exception as e:
            LOGGER.error(f"LSFT Random Pert failed for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return True


def task4_verify_pseudobulk_baselines():
    """
    Task 4: Verify pseudobulk baseline numbers against actual CSV files.
    """
    LOGGER.info("=" * 60)
    LOGGER.info("TASK 4: Verify Pseudobulk Baseline Numbers")
    LOGGER.info("=" * 60)
    
    # Sources of pseudobulk baseline data
    sources = [
        ("LSFT_results.csv (baseline_r column)", 
         PROJECT_ROOT / "deliverables" / "skeletons_and_fact_sheets" / "data" / "LSFT_results.csv"),
        ("LOGO_results.csv", 
         PROJECT_ROOT / "deliverables" / "skeletons_and_fact_sheets" / "data" / "LOGO_results.csv"),
        ("Epic2 mechanism ablation (original_r)", 
         PROJECT_ROOT / "results" / "manifold_law_diagnostics" / "epic2_mechanism_ablation" / "mechanism_ablation_adamson_lpm_selftrained.csv"),
    ]
    
    pseudobulk_summary = []
    
    for name, path in sources:
        if not path.exists():
            LOGGER.warning(f"Not found: {path}")
            continue
        
        LOGGER.info(f"\n{name}")
        LOGGER.info(f"Path: {path}")
        
        df = pd.read_csv(path)
        LOGGER.info(f"Columns: {df.columns.tolist()}")
        
        if 'baseline_r' in df.columns:
            # LSFT format
            for dataset in df['dataset'].unique():
                for baseline in df['baseline'].unique():
                    subset = df[(df['dataset'] == dataset) & (df['baseline'] == baseline)]
                    if len(subset) > 0:
                        r_val = subset['baseline_r'].iloc[0]
                        pseudobulk_summary.append({
                            'source': name,
                            'dataset': dataset,
                            'baseline': baseline,
                            'pearson_r': r_val,
                            'metric_type': 'baseline_r',
                        })
        
        if 'r_mean' in df.columns:
            # LOGO format
            for dataset in df['dataset'].unique():
                for baseline in df['baseline'].unique():
                    subset = df[(df['dataset'] == dataset) & (df['baseline'] == baseline)]
                    if len(subset) > 0:
                        r_val = subset['r_mean'].iloc[0]
                        pseudobulk_summary.append({
                            'source': name,
                            'dataset': dataset,
                            'baseline': baseline,
                            'pearson_r': r_val,
                            'metric_type': 'logo_r',
                        })
        
        if 'original_pearson_r' in df.columns:
            # Epic2 format
            r_val = df['original_pearson_r'].mean()
            pseudobulk_summary.append({
                'source': name,
                'dataset': 'adamson',
                'baseline': 'lpm_selftrained',
                'pearson_r': r_val,
                'metric_type': 'epic2_original',
            })
    
    summary_df = pd.DataFrame(pseudobulk_summary)
    
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("PSEUDOBULK BASELINE SUMMARY")
    LOGGER.info("=" * 60)
    
    # Focus on key baselines
    for baseline in ['lpm_selftrained', 'lpm_randomGeneEmb', 'lpm_gearsPertEmb']:
        LOGGER.info(f"\n{baseline}:")
        subset = summary_df[summary_df['baseline'] == baseline]
        if len(subset) > 0:
            for _, row in subset.iterrows():
                LOGGER.info(f"  {row['dataset']}: r={row['pearson_r']:.4f} ({row['metric_type']}, {row['source'][:30]}...)")
    
    # Save verification report
    report_path = AGGREGATED_DIR / "pseudobulk_verification_report.csv"
    summary_df.to_csv(report_path, index=False)
    LOGGER.info(f"\nSaved verification report to {report_path}")
    
    # Key numbers for presentation
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("KEY PSEUDOBULK NUMBERS FOR PRESENTATION")
    LOGGER.info("=" * 60)
    
    # Adamson self-trained baseline
    adamson_self = summary_df[(summary_df['dataset'] == 'adamson') & 
                               (summary_df['baseline'] == 'lpm_selftrained') &
                               (summary_df['metric_type'] == 'baseline_r')]
    if len(adamson_self) > 0:
        LOGGER.info(f"Adamson Self-trained Pseudobulk Baseline: r = {adamson_self['pearson_r'].iloc[0]:.4f}")
    
    # K562 self-trained baseline
    k562_self = summary_df[(summary_df['dataset'] == 'k562') & 
                            (summary_df['baseline'] == 'lpm_selftrained') &
                            (summary_df['metric_type'] == 'baseline_r')]
    if len(k562_self) > 0:
        LOGGER.info(f"K562 Self-trained Pseudobulk Baseline: r = {k562_self['pearson_r'].iloc[0]:.4f}")
    
    return True


def main():
    """Run all tasks."""
    LOGGER.info("=" * 60)
    LOGGER.info("COMPREHENSIVE FIX AND REGENERATION")
    LOGGER.info("=" * 60)
    
    # Change to project root
    os.chdir(PROJECT_ROOT)
    
    # Task 1: Regenerate aggregated results
    task1_regenerate_aggregated_results()
    
    # Task 4: Verify pseudobulk (do this before running new analyses)
    task4_verify_pseudobulk_baselines()
    
    # Task 2 and 3 require imports that may fail if data is missing
    # Run them conditionally
    try:
        task2_fix_and_rerun_logo()
    except Exception as e:
        LOGGER.error(f"Task 2 failed: {e}")
    
    try:
        task3_run_lsft_random_pert()
    except Exception as e:
        LOGGER.error(f"Task 3 failed: {e}")
    
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("ALL TASKS COMPLETE")
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()

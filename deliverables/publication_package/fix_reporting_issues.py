#!/usr/bin/env python3
"""
Fix Reporting Issues - Comprehensive Fix for Publication Package

Addresses:
1. Baseline naming inconsistency
2. Missing data in summary tables
3. Incomplete unified metrics
4. Tangent alignment interpretation
5. Direction flip threshold issues
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "manifold_law_diagnostics"
PUB_DIR = Path(__file__).parent
TABLES_DIR = PUB_DIR / "final_tables"

# Canonical baseline names (without dataset prefixes)
CANONICAL_BASELINES = {
    'lpm_selftrained',
    'lpm_randomGeneEmb',
    'lpm_randomPertEmb',
    'lpm_scgptGeneEmb',
    'lpm_scFoundationGeneEmb',
    'lpm_gearsPertEmb',
    'lpm_k562PertEmb',
    'lpm_rpe1PertEmb',
}

def normalize_baseline_name(name: str) -> str:
    """Normalize baseline name to canonical form."""
    if not name or pd.isna(name):
        return None
    
    name = str(name).strip()
    
    # Remove dataset prefix if present (e.g., "adamson_lpm_selftrained" -> "lpm_selftrained")
    if '_lpm_' in name:
        parts = name.split('_lpm_', 1)
        if len(parts) == 2:
            # Check if first part is a dataset name
            if parts[0] in ['adamson', 'k562', 'rpe1']:
                return f"lpm_{parts[1]}"
    
    # Remove 'lpm_' prefix if baseline doesn't have it but should
    if name.startswith('selftrained'):
        return 'lpm_selftrained'
    if name.startswith('randomGeneEmb'):
        return 'lpm_randomGeneEmb'
    if name.startswith('randomPertEmb'):
        return 'lpm_randomPertEmb'
    if name.startswith('scgptGeneEmb'):
        return 'lpm_scgptGeneEmb'
    if name.startswith('scFoundationGeneEmb'):
        return 'lpm_scFoundationGeneEmb'
    if name.startswith('gearsPertEmb'):
        return 'lpm_gearsPertEmb'
    if name.startswith('k562PertEmb'):
        return 'lpm_k562PertEmb'
    if name.startswith('rpe1PertEmb'):
        return 'lpm_rpe1PertEmb'
    
    # Return as-is if already canonical
    if name in CANONICAL_BASELINES:
        return name
    
    # Remove 'lpm_' prefix if it exists but check canonical
    if name.startswith('lpm_'):
        return name
    
    # Try adding lpm_ prefix
    if not name.startswith('lpm_'):
        candidate = f"lpm_{name}"
        if candidate in CANONICAL_BASELINES:
            return candidate
    
    return name


def fix_epic1_curvature_metrics() -> pd.DataFrame:
    """Fix and normalize Epic 1 metrics."""
    file_path = TABLES_DIR / "epic1_curvature_metrics.csv"
    if not file_path.exists():
        print(f"⚠️  {file_path} not found")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    
    # Normalize baseline names
    df['baseline'] = df['baseline'].apply(normalize_baseline_name)
    
    # Filter out invalid baselines
    df = df[df['baseline'].isin(CANONICAL_BASELINES)]
    
    # Sort for consistency
    df = df.sort_values(['dataset', 'baseline']).reset_index(drop=True)
    
    return df


def fix_epic2_alignment_summary() -> pd.DataFrame:
    """Fix and normalize Epic 2 metrics."""
    file_path = TABLES_DIR / "epic2_alignment_summary.csv"
    if not file_path.exists():
        print(f"⚠️  {file_path} not found")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    
    # Normalize baseline names
    if 'baseline' in df.columns:
        df['baseline'] = df['baseline'].apply(normalize_baseline_name)
        df = df[df['baseline'].isin(CANONICAL_BASELINES)]
    
    # Ensure dataset column exists
    if 'dataset' not in df.columns and 'baseline_type' in df.columns:
        # Try to extract dataset from baseline_type if it has dataset prefix
        df['dataset'] = df['baseline_type'].apply(lambda x: x.split('_')[0] if '_' in str(x) else None)
    
    df = df.sort_values(['dataset', 'baseline']).reset_index(drop=True) if 'dataset' in df.columns else df.sort_values('baseline').reset_index(drop=True)
    
    return df


def fix_epic3_lipschitz_summary() -> pd.DataFrame:
    """Fix and normalize Epic 3 metrics."""
    file_path = TABLES_DIR / "epic3_lipschitz_summary.csv"
    if not file_path.exists():
        print(f"⚠️  {file_path} not found")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    
    # Normalize baseline names
    if 'baseline' in df.columns:
        df['baseline'] = df['baseline'].apply(normalize_baseline_name)
        df = df[df['baseline'].isin(CANONICAL_BASELINES)]
    elif 'baseline_type' in df.columns:
        df['baseline'] = df['baseline_type'].apply(normalize_baseline_name)
        df = df[df['baseline'].isin(CANONICAL_BASELINES)]
        df = df.drop(columns=['baseline_type'], errors='ignore')
    
    # Ensure dataset column
    if 'dataset' not in df.columns and 'baseline_type' in df.columns:
        df['dataset'] = df['baseline_type'].apply(lambda x: x.split('_')[0] if '_' in str(x) else None)
    
    df = df.sort_values(['dataset', 'baseline']).reset_index(drop=True) if 'dataset' in df.columns else df.sort_values('baseline').reset_index(drop=True)
    
    return df


def fix_epic4_flip_summary() -> pd.DataFrame:
    """Fix and normalize Epic 4 metrics."""
    file_path = TABLES_DIR / "epic4_flip_summary.csv"
    if not file_path.exists():
        print(f"⚠️  {file_path} not found")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    
    # Normalize baseline names
    if 'baseline' in df.columns:
        df['baseline'] = df['baseline'].apply(normalize_baseline_name)
        df = df[df['baseline'].isin(CANONICAL_BASELINES)]
    
    # Filter out invalid entries
    if 'dataset' in df.columns:
        df = df[df['dataset'].isin(['adamson', 'k562', 'rpe1', 'probe'])]
    
    df = df.sort_values(['dataset', 'baseline']).reset_index(drop=True) if 'dataset' in df.columns else df.sort_values('baseline').reset_index(drop=True)
    
    return df


def fix_epic5_alignment_summary() -> pd.DataFrame:
    """Fix and normalize Epic 5 metrics."""
    file_path = TABLES_DIR / "epic5_alignment_summary.csv"
    if not file_path.exists():
        print(f"⚠️  {file_path} not found")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    
    # Normalize baseline names
    if 'baseline' in df.columns:
        df['baseline'] = df['baseline'].apply(normalize_baseline_name)
        df = df[df['baseline'].isin(CANONICAL_BASELINES)]
    
    df = df.sort_values(['dataset', 'baseline']).reset_index(drop=True) if 'dataset' in df.columns else df.sort_values('baseline').reset_index(drop=True)
    
    return df


def create_unified_baseline_summary(
    epic1: pd.DataFrame,
    epic2: pd.DataFrame,
    epic3: pd.DataFrame,
    epic4: pd.DataFrame,
    epic5: pd.DataFrame,
) -> pd.DataFrame:
    """Create properly unified baseline summary across all epics."""
    
    # Collect all unique canonical baselines
    all_baselines = set()
    for df in [epic1, epic2, epic3, epic4, epic5]:
        if len(df) > 0 and 'baseline' in df.columns:
            all_baselines.update(df['baseline'].dropna().unique())
    
    all_baselines = sorted([b for b in all_baselines if b in CANONICAL_BASELINES])
    
    results = []
    
    for baseline in all_baselines:
        row = {'baseline': baseline}
        
        # Epic 1: Peak r and curvature (aggregate across datasets)
        if len(epic1) > 0:
            e1_subset = epic1[epic1['baseline'] == baseline]
            if len(e1_subset) > 0:
                row['epic1_peak_r'] = e1_subset['peak_r'].mean()
                row['epic1_curvature'] = e1_subset['curvature_index'].mean()
                row['epic1_mean_r'] = e1_subset['mean_r'].mean()
                row['epic1_n_datasets'] = e1_subset['dataset'].nunique() if 'dataset' in e1_subset.columns else len(e1_subset)
            else:
                row['epic1_peak_r'] = np.nan
                row['epic1_curvature'] = np.nan
                row['epic1_mean_r'] = np.nan
                row['epic1_n_datasets'] = 0
        else:
            row['epic1_peak_r'] = np.nan
            row['epic1_curvature'] = np.nan
            row['epic1_mean_r'] = np.nan
            row['epic1_n_datasets'] = 0
        
        # Epic 2: Original r and delta_r
        if len(epic2) > 0:
            e2_subset = epic2[epic2['baseline'] == baseline]
            if len(e2_subset) > 0:
                # Try different column names
                if 'mean_original_r' in e2_subset.columns:
                    row['epic2_original_r'] = e2_subset['mean_original_r'].mean()
                elif 'original_r' in e2_subset.columns:
                    row['epic2_original_r'] = e2_subset['original_r'].mean()
                
                if 'mean_delta_r' in e2_subset.columns:
                    row['epic2_delta_r'] = e2_subset['mean_delta_r'].mean()
                elif 'delta_r' in e2_subset.columns:
                    row['epic2_delta_r'] = e2_subset['delta_r'].mean()
                
                row['epic2_n_datasets'] = e2_subset['dataset'].nunique() if 'dataset' in e2_subset.columns else len(e2_subset)
            else:
                row['epic2_original_r'] = np.nan
                row['epic2_delta_r'] = np.nan
                row['epic2_n_datasets'] = 0
        else:
            row['epic2_original_r'] = np.nan
            row['epic2_delta_r'] = np.nan
            row['epic2_n_datasets'] = 0
        
        # Epic 3: Lipschitz and baseline r
        if len(epic3) > 0:
            e3_subset = epic3[epic3['baseline'] == baseline]
            if len(e3_subset) > 0:
                if 'lipschitz_constant' in e3_subset.columns:
                    row['epic3_lipschitz'] = e3_subset['lipschitz_constant'].mean()
                if 'baseline_r' in e3_subset.columns:
                    row['epic3_baseline_r'] = e3_subset['baseline_r'].mean()
                row['epic3_n_datasets'] = e3_subset['dataset'].nunique() if 'dataset' in e3_subset.columns else len(e3_subset)
            else:
                row['epic3_lipschitz'] = np.nan
                row['epic3_baseline_r'] = np.nan
                row['epic3_n_datasets'] = 0
        else:
            row['epic3_lipschitz'] = np.nan
            row['epic3_baseline_r'] = np.nan
            row['epic3_n_datasets'] = 0
        
        # Epic 4: Flip rate
        if len(epic4) > 0:
            e4_subset = epic4[epic4['baseline'] == baseline]
            # Filter out 'probe' dataset rows if they exist
            if 'dataset' in e4_subset.columns:
                e4_subset = e4_subset[e4_subset['dataset'] != 'probe']
            if len(e4_subset) > 0:
                if 'mean_adversarial_rate' in e4_subset.columns:
                    row['epic4_flip_rate'] = e4_subset['mean_adversarial_rate'].mean()
                elif 'adversarial_rate' in e4_subset.columns:
                    row['epic4_flip_rate'] = e4_subset['adversarial_rate'].mean()
                row['epic4_n_datasets'] = e4_subset['dataset'].nunique() if 'dataset' in e4_subset.columns else len(e4_subset)
            else:
                row['epic4_flip_rate'] = np.nan
                row['epic4_n_datasets'] = 0
        else:
            row['epic4_flip_rate'] = np.nan
            row['epic4_n_datasets'] = 0
        
        # Epic 5: Tangent alignment
        if len(epic5) > 0:
            e5_subset = epic5[epic5['baseline'] == baseline]
            if len(e5_subset) > 0:
                if 'mean_tas' in e5_subset.columns:
                    row['epic5_tas'] = e5_subset['mean_tas'].mean()
                elif 'tas' in e5_subset.columns:
                    row['epic5_tas'] = e5_subset['tas'].mean()
                row['epic5_n_datasets'] = e5_subset['dataset'].nunique() if 'dataset' in e5_subset.columns else len(e5_subset)
            else:
                row['epic5_tas'] = np.nan
                row['epic5_n_datasets'] = 0
        else:
            row['epic5_tas'] = np.nan
            row['epic5_n_datasets'] = 0
        
        results.append(row)
    
    return pd.DataFrame(results)


def create_unified_metrics(epic1: pd.DataFrame, epic2: pd.DataFrame, epic3: pd.DataFrame,
                           epic4: pd.DataFrame, epic5: pd.DataFrame) -> pd.DataFrame:
    """Create truly unified metrics table with all epics."""
    
    all_baselines = set()
    for df in [epic1, epic2, epic3, epic4, epic5]:
        if len(df) > 0 and 'baseline' in df.columns:
            all_baselines.update(df['baseline'].dropna().unique())
    
    all_baselines = sorted([b for b in all_baselines if b in CANONICAL_BASELINES])
    
    results = []
    
    for baseline in all_baselines:
        row = {'baseline': baseline}
        
        # Epic 1
        if len(epic1) > 0:
            e1_subset = epic1[epic1['baseline'] == baseline]
            row['epic1_mean_r'] = e1_subset['mean_r'].mean() if len(e1_subset) > 0 else np.nan
            row['epic1_peak_r'] = e1_subset['peak_r'].mean() if len(e1_subset) > 0 else np.nan
        
        # Epic 2
        if len(epic2) > 0:
            e2_subset = epic2[epic2['baseline'] == baseline]
            if len(e2_subset) > 0:
                if 'mean_original_r' in e2_subset.columns:
                    row['epic2_original_r'] = e2_subset['mean_original_r'].mean()
                if 'mean_delta_r' in e2_subset.columns:
                    row['epic2_delta_r'] = e2_subset['mean_delta_r'].mean()
        
        # Epic 3
        if len(epic3) > 0:
            e3_subset = epic3[epic3['baseline'] == baseline]
            if len(e3_subset) > 0:
                if 'lipschitz_constant' in e3_subset.columns:
                    row['epic3_lipschitz'] = e3_subset['lipschitz_constant'].mean()
                if 'baseline_r' in e3_subset.columns:
                    row['epic3_baseline_r'] = e3_subset['baseline_r'].mean()
        
        # Epic 4
        if len(epic4) > 0:
            e4_subset = epic4[epic4['baseline'] == baseline]
            if 'dataset' in e4_subset.columns:
                e4_subset = e4_subset[e4_subset['dataset'] != 'probe']
            if len(e4_subset) > 0:
                if 'mean_adversarial_rate' in e4_subset.columns:
                    row['epic4_flip_rate'] = e4_subset['mean_adversarial_rate'].mean()
        
        # Epic 5
        if len(epic5) > 0:
            e5_subset = epic5[epic5['baseline'] == baseline]
            if len(e5_subset) > 0:
                if 'mean_tas' in e5_subset.columns:
                    row['epic5_tas'] = e5_subset['mean_tas'].mean()
        
        results.append(row)
    
    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("FIXING REPORTING ISSUES - COMPREHENSIVE FIX")
    print("=" * 70)
    print()
    
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load and fix each epic
    print("1. Fixing Epic 1 (Curvature)...")
    epic1 = fix_epic1_curvature_metrics()
    if len(epic1) > 0:
        epic1.to_csv(TABLES_DIR / "epic1_curvature_metrics.csv", index=False)
        print(f"   ✅ Fixed: {len(epic1)} rows")
    else:
        print("   ⚠️  No data")
    
    print("\n2. Fixing Epic 2 (Mechanism Ablation)...")
    epic2 = fix_epic2_alignment_summary()
    if len(epic2) > 0:
        epic2.to_csv(TABLES_DIR / "epic2_alignment_summary.csv", index=False)
        print(f"   ✅ Fixed: {len(epic2)} rows")
    else:
        print("   ⚠️  No data")
    
    print("\n3. Fixing Epic 3 (Noise/Lipschitz)...")
    epic3 = fix_epic3_lipschitz_summary()
    if len(epic3) > 0:
        epic3.to_csv(TABLES_DIR / "epic3_lipschitz_summary.csv", index=False)
        print(f"   ✅ Fixed: {len(epic3)} rows")
    else:
        print("   ⚠️  No data")
    
    print("\n4. Fixing Epic 4 (Direction-Flip)...")
    epic4 = fix_epic4_flip_summary()
    if len(epic4) > 0:
        epic4.to_csv(TABLES_DIR / "epic4_flip_summary.csv", index=False)
        print(f"   ✅ Fixed: {len(epic4)} rows")
    else:
        print("   ⚠️  No data")
    
    print("\n5. Fixing Epic 5 (Tangent Alignment)...")
    epic5 = fix_epic5_alignment_summary()
    if len(epic5) > 0:
        epic5.to_csv(TABLES_DIR / "epic5_alignment_summary.csv", index=False)
        print(f"   ✅ Fixed: {len(epic5)} rows")
    else:
        print("   ⚠️  No data")
    
    # Create unified summaries
    print("\n6. Creating unified baseline summary...")
    baseline_summary = create_unified_baseline_summary(epic1, epic2, epic3, epic4, epic5)
    baseline_summary.to_csv(TABLES_DIR / "baseline_summary.csv", index=False)
    print(f"   ✅ Created: {len(baseline_summary)} baselines")
    print(f"   Columns: {', '.join(baseline_summary.columns)}")
    
    print("\n7. Creating unified metrics table...")
    unified_metrics = create_unified_metrics(epic1, epic2, epic3, epic4, epic5)
    unified_metrics.to_csv(TABLES_DIR / "unified_metrics.csv", index=False)
    print(f"   ✅ Created: {len(unified_metrics)} baselines")
    print(f"   Columns: {', '.join(unified_metrics.columns)}")
    
    print("\n" + "=" * 70)
    print("✅ ALL FIXES COMPLETE!")
    print("=" * 70)
    print(f"\nFixed tables saved to: {TABLES_DIR}")
    print("\nNext steps:")
    print("  1. Review baseline_summary.csv for completeness")
    print("  2. Review unified_metrics.csv includes all epics")
    print("  3. Regenerate reports using generate_publication_reports.py")


if __name__ == "__main__":
    main()


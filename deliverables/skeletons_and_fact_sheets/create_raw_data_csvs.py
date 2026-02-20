#!/usr/bin/env python3
"""
Concatenate all raw per-perturbation data from LSFT and LOGO evaluations into single CSV files.
"""

import pandas as pd
from pathlib import Path
import re

def extract_dataset_and_baseline_from_filename(filename):
    """Extract dataset and baseline from standardized filename.
    
    Examples:
    - lsft_adamson_lpm_selftrained_standardized.csv -> (adamson, lpm_selftrained)
    - logo_k562_transcription_results.csv -> (k562, None)
    """
    filename = Path(filename).stem
    
    # LSFT pattern: lsft_{dataset}_{baseline}_standardized
    lsft_match = re.match(r'lsft_([^_]+)_(.+)_standardized', filename)
    if lsft_match:
        return lsft_match.group(1), lsft_match.group(2)
    
    # LOGO pattern: logo_{dataset}_transcription_results
    logo_match = re.match(r'logo_([^_]+)_transcription_results', filename)
    if logo_match:
        return logo_match.group(1), None
    
    return None, None

def concatenate_lsft_raw_data(base_dir: Path):
    """Concatenate all LSFT standardized CSV files."""
    lsft_dir = base_dir / "lpm-evaluation-framework-v2" / "results" / "goal_3_prediction" / "lsft_resampling"
    
    all_data = []
    
    for dataset_dir in ["adamson", "k562", "rpe1"]:
        dataset_path = lsft_dir / dataset_dir
        if not dataset_path.exists():
            print(f"Warning: {dataset_path} does not exist")
            continue
        
        # Find all standardized CSV files
        standardized_files = list(dataset_path.glob("*_standardized.csv"))
        
        for file_path in standardized_files:
            try:
                df = pd.read_csv(file_path)
                
                # Extract dataset and baseline from filename
                dataset, baseline = extract_dataset_and_baseline_from_filename(file_path.name)
                if dataset is None:
                    # Try to infer from path
                    dataset = dataset_dir
                    baseline = file_path.stem.replace(f"lsft_{dataset}_", "").replace("_standardized", "")
                
                # Add dataset and baseline columns if not present
                if "dataset" not in df.columns:
                    df["dataset"] = dataset
                if "baseline" not in df.columns:
                    df["baseline"] = baseline
                
                all_data.append(df)
                print(f"Loaded {file_path.name}: {len(df)} rows")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    if not all_data:
        print("No LSFT data found!")
        return None
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Ensure consistent column order
    priority_cols = ["dataset", "baseline", "test_perturbation", "top_pct"]
    other_cols = [c for c in combined_df.columns if c not in priority_cols]
    combined_df = combined_df[priority_cols + other_cols]
    
    return combined_df

def concatenate_logo_raw_data(base_dir: Path):
    """Concatenate all LOGO result CSV files."""
    logo_dir = base_dir / "lpm-evaluation-framework-v2" / "results" / "goal_3_prediction" / "functional_class_holdout_resampling"
    
    all_data = []
    
    # Try different possible directory structures
    possible_dirs = [
        logo_dir,
        base_dir / "evaluation_framework" / "results" / "goal_3_prediction" / "functional_class_holdout"
    ]
    
    for logo_base in possible_dirs:
        if not logo_base.exists():
            continue
        
        # Look for dataset subdirectories
        for dataset_dir in ["adamson", "k562", "rpe1", "replogle_k562", "replogle_rpe1"]:
            dataset_path = logo_base / dataset_dir
            if not dataset_path.exists():
                continue
            
            # Find LOGO result CSV files
            logo_files = list(dataset_path.glob("logo_*.csv"))
            if not logo_files:
                # Try alternative naming
                logo_files = list(dataset_path.glob("*_results.csv"))
            
            for file_path in logo_files:
                try:
                    df = pd.read_csv(file_path)
                    
                    # Extract dataset from filename or path
                    dataset, baseline = extract_dataset_and_baseline_from_filename(file_path.name)
                    if dataset is None:
                        dataset = dataset_dir.replace("replogle_", "")
                    
                    # Add dataset column if not present
                    if "dataset" not in df.columns:
                        df["dataset"] = dataset
                    
                    # LOGO files might have baseline column, or we need to infer it
                    if "baseline" not in df.columns and "baseline_type" in df.columns:
                        df["baseline"] = df["baseline_type"]
                    
                    all_data.append(df)
                    print(f"Loaded {file_path.name}: {len(df)} rows")
                    
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    if not all_data:
        print("No LOGO data found!")
        return None
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Ensure consistent column order
    priority_cols = ["dataset", "baseline"]
    if "test_perturbation" in combined_df.columns:
        priority_cols.append("test_perturbation")
    other_cols = [c for c in combined_df.columns if c not in priority_cols]
    combined_df = combined_df[priority_cols + other_cols]
    
    return combined_df

def main():
    base_dir = Path(__file__).parent.parent.parent
    
    print("=" * 60)
    print("Concatenating LSFT raw per-perturbation data...")
    print("=" * 60)
    lsft_raw = concatenate_lsft_raw_data(base_dir)
    
    if lsft_raw is not None:
        output_dir = Path(__file__).parent / "data"
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / "LSFT_raw_per_perturbation.csv"
        lsft_raw.to_csv(output_path, index=False)
        print(f"\nSaved LSFT raw data to {output_path}")
        print(f"  Total rows: {len(lsft_raw)}")
        print(f"  Columns: {list(lsft_raw.columns)}")
        print(f"  Datasets: {lsft_raw['dataset'].unique()}")
        print(f"  Baselines: {lsft_raw['baseline'].nunique()}")
    else:
        print("Failed to create LSFT raw data file")
    
    print("\n" + "=" * 60)
    print("Concatenating LOGO raw per-perturbation data...")
    print("=" * 60)
    logo_raw = concatenate_logo_raw_data(base_dir)
    
    if logo_raw is not None:
        output_dir = Path(__file__).parent / "data"
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / "LOGO_raw_per_perturbation.csv"
        logo_raw.to_csv(output_path, index=False)
        print(f"\nSaved LOGO raw data to {output_path}")
        print(f"  Total rows: {len(logo_raw)}")
        print(f"  Columns: {list(logo_raw.columns)}")
        print(f"  Datasets: {logo_raw['dataset'].unique()}")
        if "baseline" in logo_raw.columns:
            print(f"  Baselines: {logo_raw['baseline'].nunique()}")
    else:
        print("Failed to create LOGO raw data file")

if __name__ == "__main__":
    main()


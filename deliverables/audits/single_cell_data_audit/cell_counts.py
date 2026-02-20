#!/usr/bin/env python3
"""
Single-Cell Data Audit

Audits the single-cell data quality before running single-cell analysis:
- Cells per perturbation distribution (min/max/mean)
- Expression sparsity at single-cell level
- Control cell count
- Memory requirements estimate
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import anndata as ad
except ImportError:
    print("ERROR: anndata not installed. Run: pip install anndata")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


# Default data paths
DATA_PATHS = {
    "adamson": "/Users/samuelminer/Documents/classes/nih_research/data_adamson/perturb_processed.h5ad",
    "k562": "/Users/samuelminer/Documents/classes/nih_research/data_replogle_k562_essential/perturb_processed.h5ad",
    "rpe1": "/Users/samuelminer/Documents/classes/nih_research/data_replogle_rpe1_essential/perturb_processed.h5ad",
}


def audit_single_dataset(
    adata_path: Path,
    dataset_name: str,
) -> Dict:
    """
    Audit a single dataset for single-cell analysis readiness.
    
    Args:
        adata_path: Path to h5ad file
        dataset_name: Name for reporting
        
    Returns:
        Dictionary with audit results
    """
    LOGGER.info(f"Auditing {dataset_name}: {adata_path}")
    
    # Load data
    adata = ad.read_h5ad(adata_path)
    
    results = {
        "dataset": dataset_name,
        "path": str(adata_path),
    }
    
    # Basic dimensions
    n_cells, n_genes = adata.shape
    results["n_cells_total"] = n_cells
    results["n_genes"] = n_genes
    
    LOGGER.info(f"  Total cells: {n_cells:,}")
    LOGGER.info(f"  Total genes: {n_genes:,}")
    
    # Check for condition column
    if "condition" not in adata.obs.columns:
        LOGGER.error(f"  ERROR: 'condition' column not found in obs")
        results["error"] = "No condition column"
        return results
    
    # Unique conditions
    conditions = adata.obs["condition"].unique()
    n_conditions = len(conditions)
    results["n_conditions"] = n_conditions
    LOGGER.info(f"  Unique conditions: {n_conditions}")
    
    # Control cells
    ctrl_mask = adata.obs["condition"] == "ctrl"
    n_ctrl_cells = ctrl_mask.sum()
    results["n_control_cells"] = int(n_ctrl_cells)
    LOGGER.info(f"  Control cells: {n_ctrl_cells:,}")
    
    # Non-control conditions (perturbations)
    non_ctrl_mask = ~ctrl_mask
    perturbation_conditions = adata.obs.loc[non_ctrl_mask, "condition"].unique()
    n_perturbations = len(perturbation_conditions)
    results["n_perturbations"] = n_perturbations
    LOGGER.info(f"  Perturbation conditions: {n_perturbations}")
    
    # Cells per perturbation distribution
    cells_per_pert = adata.obs.loc[non_ctrl_mask, "condition"].value_counts()
    
    results["cells_per_pert_min"] = int(cells_per_pert.min())
    results["cells_per_pert_max"] = int(cells_per_pert.max())
    results["cells_per_pert_mean"] = float(cells_per_pert.mean())
    results["cells_per_pert_median"] = float(cells_per_pert.median())
    results["cells_per_pert_std"] = float(cells_per_pert.std())
    
    LOGGER.info(f"  Cells per perturbation:")
    LOGGER.info(f"    Min: {cells_per_pert.min()}")
    LOGGER.info(f"    Max: {cells_per_pert.max()}")
    LOGGER.info(f"    Mean: {cells_per_pert.mean():.1f}")
    LOGGER.info(f"    Median: {cells_per_pert.median():.1f}")
    LOGGER.info(f"    Std: {cells_per_pert.std():.1f}")
    
    # Perturbations with < 50 cells (sampling threshold)
    perts_below_50 = (cells_per_pert < 50).sum()
    perts_below_20 = (cells_per_pert < 20).sum()
    results["perts_below_50_cells"] = int(perts_below_50)
    results["perts_below_20_cells"] = int(perts_below_20)
    LOGGER.info(f"  Perturbations with <50 cells: {perts_below_50} ({100*perts_below_50/n_perturbations:.1f}%)")
    LOGGER.info(f"  Perturbations with <20 cells: {perts_below_20} ({100*perts_below_20/n_perturbations:.1f}%)")
    
    # Expression sparsity
    X = adata.X
    if hasattr(X, "toarray"):
        # Sparse matrix - sample cells to estimate sparsity
        sample_size = min(1000, n_cells)
        np.random.seed(42)
        sample_idx = np.random.choice(n_cells, sample_size, replace=False)
        X_sample = X[sample_idx].toarray()
        sparsity = (X_sample == 0).mean()
    else:
        sparsity = (X == 0).mean()
    
    results["expression_sparsity"] = float(sparsity)
    LOGGER.info(f"  Expression sparsity: {100*sparsity:.1f}% zeros")
    
    # Memory estimate for single-cell analysis
    # Y matrix: genes x cells (float64)
    bytes_per_float = 8
    
    # For 50 cells per perturbation
    n_sampled_cells = 50 * n_perturbations
    y_matrix_mb = (n_genes * n_sampled_cells * bytes_per_float) / (1024**2)
    results["estimated_y_matrix_mb_50cells"] = float(y_matrix_mb)
    
    # For all cells
    y_matrix_full_mb = (n_genes * n_cells * bytes_per_float) / (1024**2)
    results["full_y_matrix_mb"] = float(y_matrix_full_mb)
    
    LOGGER.info(f"  Memory estimates:")
    LOGGER.info(f"    Y matrix (50 cells/pert): {y_matrix_mb:.1f} MB")
    LOGGER.info(f"    Y matrix (all cells): {y_matrix_full_mb:.1f} MB")
    
    # Gene name info
    if "gene_name" in adata.var.columns:
        n_named = adata.var["gene_name"].notna().sum()
        results["n_genes_with_names"] = int(n_named)
    
    return results


def audit_all_datasets(
    datasets: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Audit all datasets.
    
    Args:
        datasets: List of dataset names to audit (default: all)
        output_dir: Directory to save audit results
        
    Returns:
        DataFrame with audit results
    """
    if datasets is None:
        datasets = list(DATA_PATHS.keys())
    
    results = []
    for dataset in datasets:
        if dataset not in DATA_PATHS:
            LOGGER.warning(f"Unknown dataset: {dataset}")
            continue
        
        path = Path(DATA_PATHS[dataset])
        if not path.exists():
            LOGGER.warning(f"Data file not found: {path}")
            continue
        
        result = audit_single_dataset(path, dataset)
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*80)
    print("SINGLE-CELL DATA AUDIT SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = output_dir / "single_cell_data_audit.csv"
        results_df.to_csv(csv_path, index=False)
        LOGGER.info(f"\nSaved audit results to: {csv_path}")
        
        # Save detailed report
        report_path = output_dir / "SINGLE_CELL_AUDIT_REPORT.md"
        with open(report_path, "w") as f:
            f.write("# Single-Cell Data Audit Report\n\n")
            f.write("## Summary\n\n")
            f.write(f"Datasets audited: {', '.join(datasets)}\n\n")
            
            f.write("## Key Findings\n\n")
            for _, row in results_df.iterrows():
                f.write(f"### {row['dataset'].upper()}\n\n")
                f.write(f"- **Total cells:** {row['n_cells_total']:,}\n")
                f.write(f"- **Total genes:** {row['n_genes']:,}\n")
                f.write(f"- **Perturbations:** {row['n_perturbations']}\n")
                f.write(f"- **Control cells:** {row['n_control_cells']:,}\n")
                f.write(f"- **Cells per perturbation:** {row['cells_per_pert_min']}-{row['cells_per_pert_max']} (mean: {row['cells_per_pert_mean']:.1f})\n")
                f.write(f"- **Expression sparsity:** {100*row['expression_sparsity']:.1f}% zeros\n")
                f.write(f"- **Memory (50 cells/pert):** {row['estimated_y_matrix_mb_50cells']:.1f} MB\n")
                f.write(f"- **Perturbations with <50 cells:** {row['perts_below_50_cells']}\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("1. **Sampling Strategy:** Sample 50 cells per perturbation (balance of compute and representativeness)\n")
            f.write("2. **Handling Low-Cell Perturbations:** For perturbations with <50 cells, use all available cells\n")
            f.write("3. **Memory Management:** Y matrices are manageable (~100-500 MB per dataset)\n")
            f.write("4. **Sparsity Handling:** Consider log-normalization before analysis\n")
        
        LOGGER.info(f"Saved detailed report to: {report_path}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Audit single-cell data for analysis")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATA_PATHS.keys()),
        help="Datasets to audit (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory to save audit results",
    )
    
    args = parser.parse_args()
    
    audit_all_datasets(
        datasets=args.datasets,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()


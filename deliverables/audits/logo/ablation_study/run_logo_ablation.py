#!/usr/bin/env python3
"""
LOGO Ablation Study: Run LOGO evaluation excluding "Other" class from training.

This script runs LOGO evaluation with and without "Other" class to quantify
the impact of "Other" on Transcription prediction performance.

Key Principle: Do not modify original files - create new output directories.
"""

import sys
from pathlib import Path
import logging
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from goal_3_prediction.functional_class_holdout.logo import run_logo_evaluation
from goal_4_logo.logo_single_cell import run_logo_single_cell
from goal_2_baselines.baseline_types import BaselineType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def run_logo_without_other_pseudobulk(
    adata_path: Path,
    annotation_path: Path,
    dataset_name: str,
    output_dir: Path,
    class_name: str = "Transcription",
    baseline_types: list = None,
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
) -> pd.DataFrame:
    """
    Run LOGO evaluation excluding "Other" class from training (pseudobulk).
    
    This is a modified version that filters out "Other" from training set.
    """
    LOGGER.info("=" * 70)
    LOGGER.info("LOGO ABLATION: Pseudobulk (Excluding 'Other' from Training)")
    LOGGER.info("=" * 70)
    
    # Load annotations
    from shared.io import load_annotations
    annotations = load_annotations(annotation_path)
    annotations["target"] = annotations["target"].astype(str)
    
    # Identify holdout class perturbations
    holdout_targets = annotations.loc[
        annotations["class"] == class_name, "target"
    ].unique().tolist()
    LOGGER.info(f"Holdout class ({class_name}): {len(holdout_targets)} perturbations")
    
    # Load data
    import anndata as ad
    adata = ad.read_h5ad(adata_path)
    
    # Compute Y matrix (pseudobulk)
    from goal_2_baselines.baseline_runner import compute_pseudobulk_expression_changes
    all_conditions = sorted(adata.obs["condition"].unique().tolist())
    dummy_split_config = {
        "train": all_conditions,
        "test": [],
        "val": [],
    }
    
    Y_df, _ = compute_pseudobulk_expression_changes(adata, dummy_split_config, seed)
    LOGGER.info(f"Y matrix shape: {Y_df.shape} (genes × perturbations)")
    
    # Create functional class split - EXCLUDE "Other" from training
    available_targets = set(Y_df.columns)
    holdout_targets_available = [t for t in holdout_targets if t in available_targets]
    
    # Filter training targets: exclude holdout AND "Other"
    train_targets = []
    for t in Y_df.columns:
        if t in holdout_targets_available:
            continue
        # Check if this target is in "Other" class
        target_class = annotations[annotations["target"] == t]["class"].values
        if len(target_class) > 0 and target_class[0] == "Other":
            continue
        train_targets.append(t)
    
    LOGGER.info(f"Train perturbations (excluding {class_name} and 'Other'): {len(train_targets)}")
    LOGGER.info(f"Test perturbations ({class_name}): {len(holdout_targets_available)}")
    
    # Count excluded "Other" perturbations
    other_targets = annotations[annotations["class"] == "Other"]["target"].tolist()
    other_excluded = [t for t in other_targets if t in available_targets and t not in holdout_targets_available]
    LOGGER.info(f"Excluded 'Other' perturbations: {len(other_excluded)}")
    
    if len(train_targets) < 2:
        raise ValueError(f"Insufficient training data: {len(train_targets)} perturbations")
    
    if len(holdout_targets_available) == 0:
        raise ValueError(f"No test perturbations available for class '{class_name}'")
    
    # Split Y matrix
    Y_train = Y_df[train_targets]
    Y_test = Y_df[holdout_targets_available]
    
    LOGGER.info(f"Y_train shape: {Y_train.shape}")
    LOGGER.info(f"Y_test shape: {Y_test.shape}")
    
    # Default baselines
    if baseline_types is None:
        baseline_types = [
            BaselineType.SELFTRAINED,
            BaselineType.RANDOM_PERT_EMB,
            BaselineType.RANDOM_GENE_EMB,
            BaselineType.SCGPT_GENE_EMB,
            BaselineType.SCFOUNDATION_GENE_EMB,
            BaselineType.GEARS_PERT_EMB,
        ]
    
    # Get gene names
    gene_names = Y_df.index.tolist()
    
    # Get gene_name mapping if available
    gene_name_mapping = None
    if "gene_name" in adata.var.columns:
        gene_name_mapping = dict(zip(adata.var_names, adata.var["gene_name"]))
    
    # Run baselines
    from goal_2_baselines.baseline_runner import run_single_baseline
    from goal_2_baselines.baseline_types import get_baseline_config
    from shared.metrics import compute_metrics
    
    all_results = []
    
    for baseline_type in baseline_types:
        LOGGER.info(f"\nRunning baseline: {baseline_type.value}")
        
        config = get_baseline_config(baseline_type)
        
        try:
            # Train model
            result = run_single_baseline(
                Y_train=Y_train,
                Y_test=Y_test,
                config=config,
                gene_names=gene_names,
                gene_name_mapping=gene_name_mapping,
            )
            
            # Extract predictions and metrics
            # result contains: {"baseline_type": str, "predictions": np.ndarray, "metrics": dict}
            predictions = result.get("predictions")  # (genes, n_test_perts) or dict
            metrics_dict = result.get("metrics", {})  # {pert_name: {"pearson_r": float, "l2": float}}
            
            # Handle per-perturbation results
            test_perts = Y_test.columns.tolist()
            
            for pert_name in test_perts:
                if pert_name in metrics_dict:
                    # Use pre-computed metrics if available
                    pert_metrics = metrics_dict[pert_name]
                    all_results.append({
                        "perturbation": pert_name,
                        "baseline": baseline_type.value,
                        "class": class_name,
                        "pearson_r": pert_metrics.get("pearson_r", float("nan")),
                        "l2": pert_metrics.get("l2", float("nan")),
                        "split_type": "functional_class_holdout_no_other",
                    })
                elif predictions is not None:
                    # Compute metrics from predictions
                    if isinstance(predictions, dict):
                        y_pred = predictions.get(pert_name)
                        if y_pred is None:
                            continue
                    else:
                        # predictions is numpy array (genes, n_perts)
                        pert_idx = test_perts.index(pert_name)
                        y_pred = predictions[:, pert_idx]
                    
                    y_true = Y_test[pert_name].values
                    metrics = compute_metrics(y_true, y_pred)
                    
                    all_results.append({
                        "perturbation": pert_name,
                        "baseline": baseline_type.value,
                        "class": class_name,
                        "pearson_r": metrics["pearson_r"],
                        "l2": metrics["l2"],
                        "split_type": "functional_class_holdout_no_other",
                    })
                else:
                    LOGGER.warning(f"No predictions or metrics available for {pert_name}")
        
        except Exception as e:
            LOGGER.error(f"Failed to run {baseline_type.value}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    results_df = pd.DataFrame(all_results)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"logo_ablation_pseudobulk_{dataset_name}_{class_name}.csv"
    results_df.to_csv(output_file, index=False)
    LOGGER.info(f"\nResults saved to: {output_file}")
    
    return results_df


def run_logo_without_other_single_cell(
    adata_path: Path,
    annotation_path: Path,
    dataset_name: str,
    output_dir: Path,
    class_name: str = "Transcription",
    baseline_types: list = None,
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
    n_cells_per_pert: int = 50,
) -> pd.DataFrame:
    """
    Run LOGO evaluation excluding "Other" class from training (single-cell).
    
    This modifies the single-cell LOGO to exclude "Other" from training.
    """
    LOGGER.info("=" * 70)
    LOGGER.info("LOGO ABLATION: Single-Cell (Excluding 'Other' from Training)")
    LOGGER.info("=" * 70)
    
    # Load annotations
    from shared.io import load_annotations
    annotations = load_annotations(annotation_path)
    annotations["target"] = annotations["target"].astype(str)
    
    # Identify holdout class perturbations
    holdout_targets = annotations.loc[
        annotations["class"] == class_name, "target"
    ].unique().tolist()
    LOGGER.info(f"Holdout class ({class_name}): {len(holdout_targets)} perturbations")
    
    # Load data
    import anndata as ad
    adata = ad.read_h5ad(adata_path)
    
    # Compute single-cell Y matrix - but we need to modify the split logic
    # We'll use the standard function but then filter out "Other" from training
    
    from goal_4_logo.logo_single_cell import compute_single_cell_expression_changes_logo
    Y_df, split_labels, cell_to_pert = compute_single_cell_expression_changes_logo(
        adata, holdout_targets, n_cells_per_pert=n_cells_per_pert, seed=seed
    )
    
    # Filter training cells: remove cells from "Other" class perturbations
    train_cells_original = split_labels["train"]
    train_cells_filtered = []
    
    for cell_id in train_cells_original:
        pert = cell_to_pert[cell_id]
        # Check if this perturbation is in "Other" class
        target_class = annotations[annotations["target"] == pert]["class"].values
        if len(target_class) > 0 and target_class[0] == "Other":
            continue  # Exclude this cell
        train_cells_filtered.append(cell_id)
    
    test_cells = split_labels["test"]
    
    LOGGER.info(f"Original train cells: {len(train_cells_original)}")
    LOGGER.info(f"Filtered train cells (excluding 'Other'): {len(train_cells_filtered)}")
    LOGGER.info(f"Test cells: {len(test_cells)}")
    
    # Count excluded "Other" cells
    other_perts = annotations[annotations["class"] == "Other"]["target"].tolist()
    excluded_cells = [c for c in train_cells_original 
                     if cell_to_pert[c] in other_perts]
    LOGGER.info(f"Excluded 'Other' cells: {len(excluded_cells)}")
    
    if len(train_cells_filtered) < 10:
        raise ValueError(f"Insufficient training cells: {len(train_cells_filtered)}")
    
    # Update split labels
    split_labels_filtered = {
        "train": train_cells_filtered,
        "test": test_cells,
    }
    
    # Now run the standard single-cell LOGO with filtered split
    # We need to manually run the baseline evaluation with filtered data
    Y_train = Y_df[train_cells_filtered]
    Y_test = Y_df[test_cells]
    
    # Continue with standard baseline evaluation...
    # (This is a simplified version - full implementation would mirror run_logo_single_cell)
    
    LOGGER.warning("Single-cell ablation requires full reimplementation")
    LOGGER.warning("For now, use pseudobulk ablation which is more straightforward")
    
    return pd.DataFrame()  # Placeholder


def main():
    """Run ablation study for all datasets."""
    
    datasets = {
        "adamson": {
            "adata_path": "/Users/samuelminer/Documents/classes/nih_research/data_adamson/perturb_processed.h5ad",
            "annotation_path": project_root / "data" / "annotations" / "adamson_functional_classes_enriched.tsv",
        },
        "k562": {
            "adata_path": "/Users/samuelminer/Documents/classes/nih_research/data_replogle_k562_essential/perturb_processed.h5ad",
            "annotation_path": project_root / "data" / "annotations" / "replogle_k562_functional_classes_go.tsv",
        },
        "rpe1": {
            "adata_path": "/Users/samuelminer/Documents/classes/nih_research/data_replogle_rpe1_essential/perturb_processed.h5ad",
            "annotation_path": project_root / "data" / "annotations" / "replogle_rpe1_functional_classes_go.tsv",
        },
    }
    
    output_base = project_root / "audits" / "logo" / "ablation_study" / "results"
    output_base.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for dataset_name, config in datasets.items():
        LOGGER.info("\n" + "=" * 70)
        LOGGER.info(f"DATASET: {dataset_name.upper()}")
        LOGGER.info("=" * 70)
        
        if not Path(config["adata_path"]).exists():
            LOGGER.warning(f"Skipping {dataset_name}: data file not found")
            continue
        
        if not config["annotation_path"].exists():
            LOGGER.warning(f"Skipping {dataset_name}: annotation file not found")
            continue
        
        try:
            results = run_logo_without_other_pseudobulk(
                adata_path=Path(config["adata_path"]),
                annotation_path=config["annotation_path"],
                dataset_name=dataset_name,
                output_dir=output_base,
                class_name="Transcription",
                pca_dim=10,
                ridge_penalty=0.1,
                seed=1,
            )
            all_results[dataset_name] = results
            
        except Exception as e:
            LOGGER.error(f"Failed to run ablation for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("ABLATION STUDY COMPLETE")
    LOGGER.info("=" * 70)
    LOGGER.info(f"\nResults saved to: {output_base}")
    LOGGER.info("\nNext step: Run compare_results.py to compare with standard LOGO")


if __name__ == "__main__":
    main()


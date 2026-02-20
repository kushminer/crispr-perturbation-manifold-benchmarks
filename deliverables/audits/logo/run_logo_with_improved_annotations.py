#!/usr/bin/env python3
"""
Re-run LOGO evaluation with improved annotations.

This script:
1. Uses combined improved annotations instead of original
2. Runs LOGO evaluation for all datasets
3. Compares results with original annotation results
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Add project root to path
# deliverables/audits/logo/run_logo_with_improved_annotations.py
# So: __file__ -> deliverables/audits/logo -> audits -> root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from goal_3_prediction.functional_class_holdout.logo import run_logo_evaluation
from goal_2_baselines.baseline_types import BaselineType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def run_logo_with_improved_annotations(
    adata_path: Path,
    improved_annotation_path: Path,
    dataset_name: str,
    output_dir: Path,
    class_name: str = "Transcription",
    baseline_types: list = None,
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
):
    """
    Run LOGO evaluation with improved annotations.
    
    This is a wrapper around the standard LOGO function that uses
    improved annotations instead of original.
    """
    
    LOGGER.info("=" * 70)
    LOGGER.info(f"LOGO WITH IMPROVED ANNOTATIONS: {dataset_name.upper()}")
    LOGGER.info("=" * 70)
    LOGGER.info(f"Using improved annotations: {improved_annotation_path}")
    LOGGER.info("")
    
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
    
    # Run LOGO evaluation
    results_df = run_logo_evaluation(
        adata_path=adata_path,
        annotation_path=improved_annotation_path,
        dataset_name=dataset_name,
        output_dir=output_dir,
        class_name=class_name,
        baseline_types=baseline_types,
        pca_dim=pca_dim,
        ridge_penalty=ridge_penalty,
        seed=seed,
    )
    
    # Add metadata to indicate improved annotations were used
    results_df["annotation_type"] = "improved_combined"
    
    # Save results
    output_file = output_dir / f"logo_{dataset_name}_{class_name}_improved_annotations.csv"
    results_df.to_csv(output_file, index=False)
    LOGGER.info(f"Results saved to: {output_file}")
    
    return results_df


def main():
    """Run LOGO with improved annotations for all datasets."""
    
    datasets = {
        "adamson": {
            "adata_path": "/Users/samuelminer/Documents/classes/nih_research/data_adamson/perturb_processed.h5ad",
            "improved_annotation": project_root / "audits" / "logo" / "annotation_improvement" / "improved_annotations" / "adamson_functional_classes_enriched_improved_combined.tsv",
        },
        "k562": {
            "adata_path": "/Users/samuelminer/Documents/classes/nih_research/data_replogle_k562_essential/perturb_processed.h5ad",
            "improved_annotation": project_root / "audits" / "logo" / "annotation_improvement" / "improved_annotations" / "replogle_k562_functional_classes_go_improved_combined.tsv",
        },
        "rpe1": {
            "adata_path": "/Users/samuelminer/Documents/classes/nih_research/data_replogle_rpe1_essential/perturb_processed.h5ad",
            "improved_annotation": project_root / "audits" / "logo" / "annotation_improvement" / "improved_annotations" / "replogle_rpe1_functional_classes_go_improved_combined.tsv",
        },
    }
    
    # Verify files exist
    for dataset_name, config in datasets.items():
        if not config["improved_annotation"].exists():
            LOGGER.warning(f"{dataset_name}: Improved annotation file not found at {config['improved_annotation']}")
            # Try to find it
            improved_dir = config["improved_annotation"].parent
            if improved_dir.exists():
                matching_files = list(improved_dir.glob(f"*{dataset_name}*combined*.tsv"))
                if matching_files:
                    LOGGER.info(f"  Found: {matching_files[0]}")
                    config["improved_annotation"] = matching_files[0]
    
    output_base = project_root / "audits" / "logo" / "results" / "logo_with_improved_annotations"
    output_base.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for dataset_name, config in datasets.items():
        if not Path(config["adata_path"]).exists():
            LOGGER.warning(f"Skipping {dataset_name}: data file not found")
            continue
        
        if not config["improved_annotation"].exists():
            LOGGER.warning(f"Skipping {dataset_name}: improved annotation file not found")
            continue
        
        LOGGER.info("")
        LOGGER.info("=" * 70)
        LOGGER.info(f"DATASET: {dataset_name.upper()}")
        LOGGER.info("=" * 70)
        LOGGER.info("")
        
        try:
            results = run_logo_with_improved_annotations(
                adata_path=Path(config["adata_path"]),
                improved_annotation_path=config["improved_annotation"],
                dataset_name=dataset_name,
                output_dir=output_base,
                class_name="Transcription",
                pca_dim=10,
                ridge_penalty=0.1,
                seed=1,
            )
            all_results[dataset_name] = results
            
        except Exception as e:
            LOGGER.error(f"Failed to run LOGO for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    LOGGER.info("")
    LOGGER.info("=" * 70)
    LOGGER.info("LOGO WITH IMPROVED ANNOTATIONS COMPLETE")
    LOGGER.info("=" * 70)
    LOGGER.info("")
    LOGGER.info(f"Results saved to: {output_base}")
    LOGGER.info("")
    LOGGER.info("Next step: Run compare_original_vs_improved_logo.py to compare results")


if __name__ == "__main__":
    main()


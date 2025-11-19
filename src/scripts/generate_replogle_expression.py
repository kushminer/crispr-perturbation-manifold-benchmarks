#!/usr/bin/env python3
"""
Generate expression data (predictions) for Replogle K562 dataset.

This script runs the linear pretrained model to generate predictions that
can be used by the evaluation framework.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("generate_replogle_expression")


def create_train_test_config(adata_path: Path, output_path: Path, seed: int = 1) -> str:
    """
    Create a train/test config with all perturbations in training set.
    
    For LOGO evaluation, we want all perturbations available for training,
    so we put everything in the training set.
    """
    import scanpy as sc
    import numpy as np
    
    np.random.seed(seed)
    
    # Load data to get all conditions
    LOGGER.info("Loading data from %s", adata_path)
    adata = sc.read_h5ad(adata_path)
    
    # Get all unique conditions
    all_conditions = sorted(adata.obs['condition'].unique().tolist())
    
    # For LOGO evaluation, put everything in training
    # (the evaluation framework will do leave-one-out internally)
    config = {
        "train": all_conditions,
        "test": [],
        "val": []
    }
    
    # Save config
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    config_id = output_path.name
    LOGGER.info("Created train/test config: %s", config_id)
    LOGGER.info("  Train: %d conditions", len(config["train"]))
    LOGGER.info("  Test: %d conditions", len(config["test"]))
    LOGGER.info("  Val: %d conditions", len(config["val"]))
    
    return config_id


def main():
    parser = argparse.ArgumentParser(
        description="Generate Replogle K562 expression data (predictions)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="replogle_k562_essential",
        help="Dataset name (default: replogle_k562_essential)",
    )
    parser.add_argument(
        "--working_dir",
        type=Path,
        default=Path("../../working_dir"),
        help="Working directory path",
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=10,
        help="PCA dimensions (default: 10)",
    )
    parser.add_argument(
        "--ridge_penalty",
        type=float,
        default=0.1,
        help="Ridge penalty (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: 1)",
    )
    parser.add_argument(
        "--pert_embedding_dataset",
        type=str,
        default="replogle_k562_essential",
        help="Dataset for perturbation embeddings (default: replogle_k562_essential)",
    )
    parser.add_argument(
        "--data_folder",
        type=Path,
        default=Path("../../paper/benchmark/data/gears_pert_data"),
        help="Path to GEARS perturbation data folder",
    )
    
    args = parser.parse_args()
    
    # Set up paths
    working_dir = Path(args.working_dir).resolve()
    data_folder = Path(args.data_folder).resolve()
    adata_path = data_folder / args.dataset_name / "perturb_processed.h5ad"
    
    # Check if data exists
    if not adata_path.exists():
        LOGGER.error("Data file not found: %s", adata_path)
        LOGGER.error("Please ensure the Replogle K562 data is available at this path.")
        return 1
    
    # Create train/test config
    config_id = f"{args.dataset_name}_all_train_seed{args.seed}"
    config_path = working_dir / "results" / config_id
    create_train_test_config(adata_path, config_path, seed=args.seed)
    
    # Set result ID
    result_id = f"py_comparison_{args.dataset_name}_seed{args.seed}"
    
    # Check if perturbation embeddings exist
    pert_emb_path = working_dir / "results" / f"{args.pert_embedding_dataset}_pert_emb_pca{args.pca_dim}_seed{args.seed}.tsv"
    
    if not pert_emb_path.exists():
        LOGGER.warning("Perturbation embeddings not found: %s", pert_emb_path)
        LOGGER.warning("Will use 'training_data' mode (PCA on training data)")
        pert_embedding = "training_data"
    else:
        LOGGER.info("Using perturbation embeddings from: %s", pert_emb_path)
        pert_embedding = str(pert_emb_path)
    
    # Run the linear pretrained model script
    script_path = Path(__file__).parent.parent.parent / "paper" / "benchmark" / "src" / "run_linear_pretrained_model.py"
    
    # Check if Python script exists (may not be in fresh repository)
    if not script_path.exists():
        LOGGER.error(
            f"Python script not found: {script_path}\n"
            f"This script may not be in the original repository.\n"
            f"Please check if the script exists or use an alternative approach."
        )
        return 1
    
    cmd = [
        sys.executable,
        str(script_path),
        "--dataset_name", args.dataset_name,
        "--test_train_config_id", config_id,
        "--pca_dim", str(args.pca_dim),
        "--ridge_penalty", str(args.ridge_penalty),
        "--seed", str(args.seed),
        "--gene_embedding", "training_data",
        "--pert_embedding", pert_embedding,
        "--working_dir", str(working_dir),
        "--result_id", result_id,
    ]
    
    LOGGER.info("Running linear pretrained model...")
    LOGGER.info("Command: %s", " ".join(cmd))
    
    result = subprocess.run(cmd, cwd=script_path.parent.parent)
    
    if result.returncode != 0:
        LOGGER.error("Script failed with return code %d", result.returncode)
        return 1
    
    # Verify output
    output_dir = working_dir / "results" / result_id
    predictions_path = output_dir / "all_predictions.json"
    gene_names_path = output_dir / "gene_names.json"
    
    if predictions_path.exists() and gene_names_path.exists():
        LOGGER.info("✅ Expression data generated successfully!")
        LOGGER.info("  Predictions: %s", predictions_path)
        LOGGER.info("  Gene names: %s", gene_names_path)
        
        # Load and show summary
        with open(predictions_path) as f:
            predictions = json.load(f)
        with open(gene_names_path) as f:
            gene_names = json.load(f)
        
        LOGGER.info("  Summary: %d perturbations, %d genes", 
                   len(predictions), len(gene_names))
        
        LOGGER.info("")
        LOGGER.info("Next steps:")
        LOGGER.info("1. Generate annotations: PYTHONPATH=src python src/generate_replogle_annotations.py")
        LOGGER.info("2. Run evaluation: PYTHONPATH=src python src/main.py --config configs/config_replogle.yaml")
    else:
        LOGGER.error("Output files not found. Check script output for errors.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


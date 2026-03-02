"""
Split logic for train/test/val splits.

This module replicates the split logic from prepare_perturbation_data.py.
For most datasets, it uses a simple random split. For norman, it uses
special logic (single vs double perturbations).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import anndata as ad
import numpy as np

LOGGER = logging.getLogger(__name__)


def create_simple_split(
    conditions: List[str],
    seed: int = 1,
    train_frac: float = 0.7,
    test_frac: float = 0.15,
) -> Dict[str, List[str]]:
    """
    Create a simple random train/test/val split.
    
    This replicates the behavior of `pert_data.prepare_split(split='simulation')`
    for most datasets (Adamson, Replogle, etc.).
    
    Args:
        conditions: List of all condition names
        seed: Random seed
        train_frac: Fraction for training (default: 0.7)
        test_frac: Fraction for testing (default: 0.15)
        # val_frac = 1 - train_frac - test_frac
    
    Returns:
        Dictionary with 'train', 'test', 'val' keys mapping to condition lists
    """
    # Ensure ctrl is always in training
    conditions = list(set(conditions))  # Remove duplicates
    # Only match exact "ctrl" condition, not conditions containing "ctrl" (like "CREB1+ctrl")
    ctrl_conditions = [c for c in conditions if c.lower() == "ctrl"]
    non_ctrl = [c for c in conditions if c not in ctrl_conditions]
    
    # Set random seed
    rng = np.random.default_rng(seed)
    
    # Shuffle non-ctrl conditions
    shuffled = rng.permutation(non_ctrl).tolist()
    
    # Split
    n_train = int(len(shuffled) * train_frac)
    n_test = int(len(shuffled) * test_frac)
    
    train_conditions = shuffled[:n_train]
    test_conditions = shuffled[n_train : n_train + n_test]
    val_conditions = shuffled[n_train + n_test :]
    
    # Add ctrl to training
    train_conditions.extend(ctrl_conditions)
    
    return {
        "train": sorted(train_conditions),
        "test": sorted(test_conditions),
        "val": sorted(val_conditions),
    }


def create_norman_split(
    conditions: List[str],
    seed: int = 1,
) -> Dict[str, List[str]]:
    """
    Create split for norman dataset (single vs double perturbations).
    
    Replicates the norman-specific split logic from prepare_perturbation_data.py:
    - Single perturbations (containing 'ctrl') go to training
    - Double perturbations split 50% train, 25% test, 25% val
    
    Args:
        conditions: List of all condition names
        seed: Random seed
    
    Returns:
        Dictionary with 'train', 'test', 'val' keys
    """
    conditions = list(set(conditions))
    
    # Separate single and double perturbations
    single_pert = [x for x in conditions if "ctrl" in x]
    double_pert = [x for x in conditions if x not in single_pert]
    
    # Set random seed
    rng = np.random.default_rng(seed)
    
    # Split double perturbations: 50% train, 25% test, 25% val
    shuffled_double = rng.permutation(double_pert).tolist()
    n_double = len(shuffled_double)
    
    double_training = shuffled_double[: n_double // 2]
    remaining = shuffled_double[n_double // 2 :]
    double_test = remaining[: len(remaining) // 2]
    double_holdout = remaining[len(remaining) // 2 :]
    
    return {
        "train": sorted(single_pert + double_training),
        "test": sorted(double_test),
        "val": sorted(double_holdout),
    }


def create_split_from_adata(
    adata: ad.AnnData,
    dataset_name: str,
    seed: int = 1,
    use_gears: bool = True,
    pert_data_folder: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """
    Create train/test/val split from AnnData object.
    
    This matches the paper's prepare_perturbation_data.py logic:
    - For norman datasets: uses custom split logic
    - For all other datasets (Adamson, Replogle): uses GEARS prepare_split(split='simulation')
    
    Args:
        adata: AnnData object with condition information
        dataset_name: Name of dataset (e.g., 'adamson', 'replogle_k562_essential')
        seed: Random seed
        use_gears: If True, try to use GEARS prepare_split (default: True, matching paper)
        pert_data_folder: Path to GEARS perturbation data folder. If None, uses paper's structure.
    
    Returns:
        Dictionary with 'train', 'test', 'val' keys
    """
    # Get all unique conditions
    if "condition" not in adata.obs.columns:
        raise ValueError("AnnData must have 'condition' column in obs")
    
    conditions = sorted(adata.obs["condition"].unique().tolist())
    LOGGER.info(f"Found {len(conditions)} unique conditions")
    
    # Use dataset-specific logic for norman
    if dataset_name == "norman" or dataset_name == "norman_from_scfoundation":
        return create_norman_split(conditions, seed=seed)
    
    # For all other datasets (Adamson, Replogle), use GEARS prepare_split by default
    if use_gears:
        try:
            from gears import PertData
            import gears.version
            
            # Determine GEARS data folder path.
            # Prefer this repo's canonical data path, then fall back to the paper-style layout.
            if pert_data_folder is None:
                repo_root = Path(__file__).resolve().parents[2]
                candidates = [
                    repo_root / "data" / "gears_pert_data",
                    repo_root / "paper" / "benchmark" / "data" / "gears_pert_data",
                    Path("data/gears_pert_data"),
                    Path("paper/benchmark/data/gears_pert_data"),
                ]
                pert_data_folder = next((path for path in candidates if path.exists()), candidates[0])
            
            if not pert_data_folder.exists():
                raise FileNotFoundError(
                    f"GEARS perturbation data folder not found: {pert_data_folder}. "
                    f"Expected structure: data/gears_pert_data/"
                )
            
            LOGGER.info(f"Using GEARS data folder: {pert_data_folder}")
            pert_data = PertData(pert_data_folder)
            pert_data.load(dataset_name)
            pert_data.prepare_split(split="simulation", seed=seed)
            LOGGER.info("Successfully generated split using GEARS prepare_split")
            return pert_data.set2conditions
        except ImportError:
            LOGGER.warning("GEARS not available (ImportError), falling back to simple split")
            LOGGER.warning("To use GEARS splits (matching paper), install GEARS: pip install gears")
        except Exception as e:
            LOGGER.warning(f"GEARS split failed: {e}, falling back to simple split")
            LOGGER.warning("This may result in different splits than the paper")
    
    # Fallback: simple random split (replicates GEARS behavior if GEARS unavailable)
    LOGGER.info("Using simple random split (fallback)")
    return create_simple_split(conditions, seed=seed)


def load_split_config(split_path: Path) -> Dict[str, List[str]]:
    """Load split configuration from JSON file."""
    with open(split_path, "r") as f:
        return json.load(f)


def save_split_config(
    split_config: Dict[str, List[str]],
    output_path: Path,
) -> None:
    """Save split configuration to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(split_config, f, indent=2)
    LOGGER.info(f"Saved split config to {output_path}")


def prepare_perturbation_splits(
    adata_path: Path,
    dataset_name: str,
    output_path: Path,
    seed: int = 1,
    use_gears: bool = True,
    pert_data_folder: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """
    Prepare train/test/val splits for a dataset.
    
    This is the main entry point that replicates prepare_perturbation_data.py.
    By default, uses GEARS prepare_split(split='simulation') to match the paper's splits.
    
    Args:
        adata_path: Path to perturb_processed.h5ad
        dataset_name: Name of dataset
        output_path: Path to save split JSON file
        seed: Random seed
        use_gears: Try to use GEARS if available (default: True, matching paper)
        pert_data_folder: Path to GEARS perturbation data folder. If None, auto-detects.
    
    Returns:
        Dictionary with 'train', 'test', 'val' keys
    """
    LOGGER.info(f"Preparing splits for {dataset_name} (seed={seed}, use_gears={use_gears})")
    
    # Load data
    adata = ad.read_h5ad(adata_path)
    
    # Create split
    split_config = create_split_from_adata(
        adata=adata,
        dataset_name=dataset_name,
        seed=seed,
        use_gears=use_gears,
        pert_data_folder=pert_data_folder,
    )
    
    # Save
    save_split_config(split_config, output_path)
    
    LOGGER.info(f"Train: {len(split_config['train'])} conditions")
    LOGGER.info(f"Test: {len(split_config['test'])} conditions")
    LOGGER.info(f"Val: {len(split_config['val'])} conditions")
    
    return split_config

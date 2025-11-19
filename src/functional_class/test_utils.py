"""
Utilities for testing and validation, including synthetic data generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def generate_synthetic_class_annotations(
    perturbation_names: List[str],
    n_classes: int = 5,
    min_class_size: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic functional class annotations for testing.
    
    Creates balanced classes with at least min_class_size members each.
    
    Args:
        perturbation_names: List of perturbation/gene names
        n_classes: Number of functional classes to create
        min_class_size: Minimum perturbations per class
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with 'target' and 'class' columns
    """
    rng = np.random.RandomState(seed)
    n_perts = len(perturbation_names)
    
    if n_perts < n_classes * min_class_size:
        raise ValueError(
            f"Need at least {n_classes * min_class_size} perturbations "
            f"for {n_classes} classes of size {min_class_size}, got {n_perts}"
        )
    
    # Create balanced assignment
    class_names = [f"Class_{i+1}" for i in range(n_classes)]
    assignments = []
    
    # Assign minimum required per class
    remaining = list(perturbation_names)
    for cls in class_names:
        n_assign = min_class_size
        selected = rng.choice(remaining, size=n_assign, replace=False)
        assignments.extend([(pert, cls) for pert in selected])
        remaining = [p for p in remaining if p not in selected]
    
    # Distribute remaining perturbations randomly
    for pert in remaining:
        cls = rng.choice(class_names)
        assignments.append((pert, cls))
    
    df = pd.DataFrame(assignments, columns=["target", "class"])
    LOGGER.info("Generated synthetic annotations: %d classes, %d total perturbations", 
                n_classes, len(df))
    
    return df


def save_synthetic_annotations(
    perturbation_names: List[str],
    output_path: Path,
    n_classes: int = 5,
    min_class_size: int = 5,
    seed: int = 42,
) -> None:
    """
    Generate and save synthetic class annotations to TSV file.
    
    Args:
        perturbation_names: List of perturbation names
        output_path: Path to save TSV file
        n_classes: Number of classes to create
        min_class_size: Minimum perturbations per class
        seed: Random seed
    """
    df = generate_synthetic_class_annotations(
        perturbation_names=perturbation_names,
        n_classes=n_classes,
        min_class_size=min_class_size,
        seed=seed,
    )
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    LOGGER.info("Saved synthetic annotations to %s", output_path)


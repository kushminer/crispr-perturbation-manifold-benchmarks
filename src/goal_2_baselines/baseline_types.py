"""
Baseline type definitions and configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class BaselineType(str, Enum):
    """Enumeration of all 8 linear baseline types."""
    
    SELFTRAINED = "lpm_selftrained"
    RANDOM_PERT_EMB = "lpm_randomPertEmb"
    RANDOM_GENE_EMB = "lpm_randomGeneEmb"
    SCGPT_GENE_EMB = "lpm_scgptGeneEmb"
    SCFOUNDATION_GENE_EMB = "lpm_scFoundationGeneEmb"
    GEARS_PERT_EMB = "lpm_gearsPertEmb"
    K562_PERT_EMB = "lpm_k562PertEmb"
    RPE1_PERT_EMB = "lpm_rpe1PertEmb"
    
    # Additional baselines
    MEAN_RESPONSE = "mean_response"


@dataclass
class BaselineConfig:
    """Configuration for a single baseline."""
    
    baseline_type: BaselineType
    gene_embedding_source: str  # "training_data", "random", or path to TSV
    pert_embedding_source: str  # "training_data", "random", or path to TSV
    pca_dim: int = 10
    ridge_penalty: float = 0.1
    seed: int = 1
    
    # Optional: specific embedding loader args
    gene_embedding_args: Optional[dict] = None
    pert_embedding_args: Optional[dict] = None


def get_baseline_config(baseline_type: BaselineType, **kwargs) -> BaselineConfig:
    """
    Get configuration for a baseline type.
    
    Args:
        baseline_type: The baseline type
        **kwargs: Override default parameters (pca_dim, ridge_penalty, seed, etc.)
    
    Returns:
        BaselineConfig for the specified baseline type
    """
    defaults = {
        "pca_dim": 10,
        "ridge_penalty": 0.1,
        "seed": 1,
    }
    defaults.update(kwargs)
    
    configs = {
        BaselineType.SELFTRAINED: BaselineConfig(
            baseline_type=BaselineType.SELFTRAINED,
            gene_embedding_source="training_data",
            pert_embedding_source="training_data",
            **defaults,
        ),
        BaselineType.RANDOM_PERT_EMB: BaselineConfig(
            baseline_type=BaselineType.RANDOM_PERT_EMB,
            gene_embedding_source="training_data",
            pert_embedding_source="random",
            **defaults,
        ),
        BaselineType.RANDOM_GENE_EMB: BaselineConfig(
            baseline_type=BaselineType.RANDOM_GENE_EMB,
            gene_embedding_source="random",
            pert_embedding_source="training_data",
            **defaults,
        ),
        BaselineType.SCGPT_GENE_EMB: BaselineConfig(
            baseline_type=BaselineType.SCGPT_GENE_EMB,
            gene_embedding_source="scgpt",  # Will use embedding loader
            pert_embedding_source="training_data",
            gene_embedding_args={"checkpoint_dir": "data/models/scgpt/scGPT_human"},
            **defaults,
        ),
        BaselineType.SCFOUNDATION_GENE_EMB: BaselineConfig(
            baseline_type=BaselineType.SCFOUNDATION_GENE_EMB,
            gene_embedding_source="scfoundation",  # Will use embedding loader
            pert_embedding_source="training_data",
            gene_embedding_args={
                "checkpoint_path": "data/models/scfoundation/models.ckpt",
                "demo_h5ad": "data/models/scfoundation/demo.h5ad",
            },
            **defaults,
        ),
        BaselineType.GEARS_PERT_EMB: BaselineConfig(
            baseline_type=BaselineType.GEARS_PERT_EMB,
            gene_embedding_source="training_data",
            pert_embedding_source="gears",  # Will use embedding loader
            pert_embedding_args={
                "source_csv": "../paper/benchmark/data/gears_pert_data/go_essential_all/go_essential_all.csv",
            },
            **defaults,
        ),
        BaselineType.K562_PERT_EMB: BaselineConfig(
            baseline_type=BaselineType.K562_PERT_EMB,
            gene_embedding_source="training_data",
            pert_embedding_source="k562_pca",  # Cross-dataset: PCA on K562, transform target
            pert_embedding_args={
                "source_adata_path": "/Users/samuelminer/Documents/classes/nih_research/data_replogle_k562_essential/perturb_processed.h5ad",
            },
            **defaults,
        ),
        BaselineType.RPE1_PERT_EMB: BaselineConfig(
            baseline_type=BaselineType.RPE1_PERT_EMB,
            gene_embedding_source="training_data",
            pert_embedding_source="rpe1_pca",  # Cross-dataset: PCA on RPE1, transform target
            pert_embedding_args={
                "source_adata_path": "/Users/samuelminer/Documents/classes/nih_research/data_replogle_rpe1_essential/perturb_processed.h5ad",
            },
            **defaults,
        ),
        BaselineType.MEAN_RESPONSE: BaselineConfig(
            baseline_type=BaselineType.MEAN_RESPONSE,
            gene_embedding_source="mean",  # Special case
            pert_embedding_source="mean",  # Special case
            **defaults,
        ),
    }
    
    return configs[baseline_type]


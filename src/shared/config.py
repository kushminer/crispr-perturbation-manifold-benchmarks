"""
Configuration utilities for the evaluation framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DatasetConfig:
    name: str
    expression_path: Path
    adata_path: Optional[Path] = None
    gene_names_path: Optional[Path] = None
    annotation_path: Optional[Path] = None
    results_dir: Path = Path("results")
    cache_dir: Optional[Path] = None
    seed: int = 1
    hardness_bins: Optional[List[float]] = field(default_factory=lambda: [0.33, 0.66])
    hardness_method: str = "mean"  # Options: "mean", "min", "median", "k_farthest"
    k_farthest: int = 10  # Number of farthest neighbors for k_farthest method
    logo_cluster_block_size: Optional[int] = None
    functional_min_class_size: int = 3


@dataclass
class ModelConfig:
    type: str = "linear"
    gene_embedding: str = "training_data"
    pert_embedding: str = "external"
    pca_dim: int = 10
    ridge_penalty: float = 0.1


@dataclass
class EmbeddingSourcesConfig:
    """Optional paths to external pretrained checkpoint folders."""

    scgpt_model_dir: Optional[Path] = None
    scfoundation_model_dir: Optional[Path] = None


@dataclass
class EmbeddingValidationConfig:
    config_path: Optional[Path] = None
    legacy_dir: Path = Path("validation/legacy_runs")
    plots_dir: Path = Path("validation/embedding_parity_plots")


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    model: ModelConfig
    tasks: List[str]
    output_root: Path = Path("results")
    embedding_sources: EmbeddingSourcesConfig = field(
        default_factory=EmbeddingSourcesConfig
    )
    embedding_validation: EmbeddingValidationConfig = field(
        default_factory=EmbeddingValidationConfig
    )


def _expand_path(base_dir: Path, path_like: Optional[str]) -> Optional[Path]:
    if path_like is None:
        return None
    path = Path(path_like)
    return (base_dir / path).expanduser().resolve() if not path.is_absolute() else path


def load_config(path: Path | str) -> ExperimentConfig:
    """
    Load an experiment configuration file.
    """
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw_cfg: Dict[str, Any] = yaml.safe_load(fh)

    base_dir = cfg_path.parent

    dataset_cfg = raw_cfg.get("dataset", {})
    model_cfg = raw_cfg.get("model", {})
    tasks = raw_cfg.get("tasks", [])
    embedding_sources_cfg = raw_cfg.get("embedding_sources", {})
    embedding_validation_cfg = raw_cfg.get("embedding_validation", {})

    dataset = DatasetConfig(
        name=dataset_cfg["name"],
        expression_path=_expand_path(base_dir, dataset_cfg["expression_path"]),
        adata_path=_expand_path(base_dir, dataset_cfg.get("adata_path")),
        gene_names_path=_expand_path(base_dir, dataset_cfg.get("gene_names_path")),
        annotation_path=_expand_path(base_dir, dataset_cfg.get("annotation_path")),
        results_dir=_expand_path(base_dir, dataset_cfg.get("results_dir", "results")),
        cache_dir=_expand_path(base_dir, dataset_cfg.get("cache_dir")),
        seed=int(dataset_cfg.get("seed", 1)),
        hardness_bins=list(dataset_cfg.get("hardness_bins", [0.33, 0.66])),
        hardness_method=str(dataset_cfg.get("hardness_method", "mean")),
        k_farthest=int(dataset_cfg.get("k_farthest", 10)),
        logo_cluster_block_size=dataset_cfg.get("logo_cluster_block_size"),
        functional_min_class_size=int(dataset_cfg.get("functional_min_class_size", 3)),
    )

    model = ModelConfig(
        type=model_cfg.get("type", "linear"),
        gene_embedding=model_cfg.get("gene_embedding", "training_data"),
        pert_embedding=model_cfg.get("pert_embedding", "external"),
        pca_dim=int(model_cfg.get("pca_dim", 10)),
        ridge_penalty=float(model_cfg.get("ridge_penalty", 0.1)),
    )

    embedding_sources = EmbeddingSourcesConfig(
        scgpt_model_dir=_expand_path(
            base_dir, embedding_sources_cfg.get("scgpt_model_dir")
        ),
        scfoundation_model_dir=_expand_path(
            base_dir, embedding_sources_cfg.get("scfoundation_model_dir")
        ),
    )
    embedding_validation = EmbeddingValidationConfig(
        config_path=_expand_path(base_dir, embedding_validation_cfg.get("config_path")),
        legacy_dir=_expand_path(
            base_dir, embedding_validation_cfg.get("legacy_dir", "validation/legacy_runs")
        )
        or Path("validation/legacy_runs"),
        plots_dir=_expand_path(
            base_dir,
            embedding_validation_cfg.get(
                "plots_dir", "validation/embedding_parity_plots"
            ),
        )
        or Path("validation/embedding_parity_plots"),
    )

    return ExperimentConfig(
        dataset=dataset,
        model=model,
        tasks=[str(task) for task in tasks],
        output_root=_expand_path(base_dir, raw_cfg.get("output_root", dataset.results_dir)) or dataset.results_dir,
        embedding_sources=embedding_sources,
        embedding_validation=embedding_validation,
    )

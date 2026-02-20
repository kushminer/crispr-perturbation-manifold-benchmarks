"""
Command line entry point for the linear perturbation evaluation framework.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Dict

import pandas as pd

from shared.config import ExperimentConfig, load_config
from shared.io import (
    load_annotations,
    load_expression_dataset,
    align_expression_with_annotations,
)
from functional_class.functional_class import (
    run_class_holdout,
    class_results_to_dataframe,
)
from shared.validation import (
    validate_annotation_quality,
    run_comprehensive_validation,
    save_validation_report,
)
from shared.embedding_parity import run_embedding_parity


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("eval_framework")


TaskFn = Callable[[ExperimentConfig], None]


def task_logo(cfg: ExperimentConfig) -> None:
    """
    LOGO (Leave One Gene Out) evaluation: Functional Class Holdout.
    
    This task isolates a specific functional class (e.g., Transcription genes) as
    the test set and trains on all other classes. Runs all 8 baselines for comparison
    to evaluate biological extrapolation.
    
    This is different from task_class (multi-class holdout), which iterates over
    all classes one at a time.
    """
    from goal_3_prediction.functional_class_holdout.logo import run_logo_evaluation
    from goal_2_baselines.baseline_types import BaselineType
    
    if cfg.dataset.annotation_path is None:
        LOGGER.error("Annotation path is required for LOGO evaluation.")
        return
    
    if cfg.dataset.adata_path is None:
        LOGGER.error("AnnData path is required for LOGO evaluation.")
        return
    
    LOGGER.info("Running LOGO (Functional Class Holdout) evaluation.")
    
    # Default to Transcription class if not specified
    class_name = getattr(cfg.dataset, "logo_class_name", "Transcription")
    
    # Determine output directory
    output_dir = Path(cfg.output_dir) / "functional_class_holdout" / cfg.dataset.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run LOGO evaluation with all baselines
    results_df = run_logo_evaluation(
        adata_path=cfg.dataset.adata_path,
        annotation_path=cfg.dataset.annotation_path,
        dataset_name=cfg.dataset.name,
        output_dir=output_dir,
        class_name=class_name,
        baseline_types=None,  # All 8 baselines + mean_response
        pca_dim=cfg.model.pca_dim,
        ridge_penalty=cfg.model.ridge_penalty,
        seed=cfg.dataset.seed,
    )
    
    LOGGER.info("LOGO evaluation complete. Results saved to %s", output_dir)
    
    # Optionally run comparison
    from goal_3_prediction.functional_class_holdout.compare_baselines import compare_baselines
    
    results_csv = output_dir / f"logo_{cfg.dataset.name}_{class_name.lower()}_results.csv"
    if results_csv.exists():
        LOGGER.info("Running baseline comparison (scGPT vs Random)...")
        compare_baselines(
            results_csv=results_csv,
            output_dir=output_dir,
            dataset_name=cfg.dataset.name,
            class_name=class_name,
            focus_comparison="scgpt_vs_random",
        )
        LOGGER.info("Baseline comparison complete.")


def task_class(cfg: ExperimentConfig) -> None:
    if cfg.dataset.annotation_path is None:
        LOGGER.error("Annotation path is required for functional class holdout.")
        return

    LOGGER.info("Running functional class holdout evaluation.")
    expression = load_expression_dataset(
        cfg.dataset.expression_path, cfg.dataset.gene_names_path
    )
    
    # Validate config: warn if threshold > dataset size
    n_perturbations = len(expression.index)
    if cfg.dataset.functional_min_class_size > n_perturbations:
        LOGGER.warning(
            "Config validation: functional_min_class_size (%d) exceeds dataset size (%d perturbations). "
            "This will result in no eligible classes. Consider lowering the threshold.",
            cfg.dataset.functional_min_class_size, n_perturbations
        )
    elif cfg.dataset.functional_min_class_size > n_perturbations / 2:
        LOGGER.warning(
            "Config validation: functional_min_class_size (%d) is >50%% of dataset size (%d). "
            "This may result in very few eligible classes.",
            cfg.dataset.functional_min_class_size, n_perturbations
        )
    
    annotations = load_annotations(cfg.dataset.annotation_path)
    
    # Validate annotation quality
    LOGGER.info("Validating annotation quality...")
    validation = validate_annotation_quality(
        annotations=annotations,
        expression_targets=expression.index.tolist(),
        min_class_size=cfg.dataset.functional_min_class_size,
    )
    if not validation.get("is_valid", False):
        LOGGER.warning("Annotation validation failed. Proceeding with warnings.")
        LOGGER.warning("Validation results: %s", validation)
    else:
        LOGGER.info("Annotation validation passed: %d classes, %d valid classes",
                   validation.get("n_classes", 0), validation.get("n_valid_classes", 0))
    
    expression, annotations = align_expression_with_annotations(expression, annotations)

    results = run_class_holdout(
        expression=expression,
        annotations=annotations,
        ridge_penalty=cfg.model.ridge_penalty,
        pca_dim=cfg.model.pca_dim,
        seed=cfg.dataset.seed,
        min_class_size=cfg.dataset.functional_min_class_size,
    )
    df = class_results_to_dataframe(results)
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    output_path = cfg.output_root / "results_class.csv"
    df.to_csv(output_path, index=False)
    LOGGER.info("Saved functional class results to %s", output_path)


def task_combined(cfg: ExperimentConfig) -> None:
    """Combined analysis has been archived."""
    LOGGER.warning(
        "Combined analysis (LOGO + class) has been archived. "
        "This functionality is no longer available in the core framework."
    )
    raise NotImplementedError(
        "Combined analysis has been moved to deliverables/archive/eval_framework_logo/"
    )


def task_visualize(cfg: ExperimentConfig) -> None:
    """Visualization has been archived."""
    LOGGER.warning(
        "LOGO-specific visualizations have been archived. "
        "This functionality is no longer available in the core framework."
    )
    raise NotImplementedError(
        "LOGO visualizations have been moved to deliverables/archive/eval_framework_logo/"
    )


def task_validate(cfg: ExperimentConfig) -> None:
    """Run comprehensive validation suite and generate report."""
    LOGGER.info("Running comprehensive validation suite...")
    
    validation_results = run_comprehensive_validation(cfg)
    
    # Save validation report
    report_path = cfg.output_root / "validation_report.json"
    save_validation_report(validation_results, report_path)
    
    LOGGER.info("Validation complete. Report saved to %s", report_path)


def task_validate_embeddings(cfg: ExperimentConfig) -> None:
    """Run embedding parity validation."""
    LOGGER.info("Running embedding parity validation...")
    report_path = run_embedding_parity(cfg)
    LOGGER.info("Embedding parity report saved to %s", report_path)


TASKS: Dict[str, TaskFn] = {
    "logo": task_logo,  # LOGO: Functional Class Holdout with all baselines
    "class": task_class,  # Multi-class holdout: iterate over all classes
    # "combined": task_combined,  # Archived
    # "visualize": task_visualize,  # Archived
    "validate": task_validate,
    "validate-embeddings": task_validate_embeddings,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Linear perturbation evaluation framework CLI"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment configuration YAML file.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=sorted(TASKS),
        default=None,
        help="Specific task to run. If omitted, runs tasks listed in config.",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    tasks_to_run = [args.task] if args.task else config.tasks
    if not tasks_to_run:
        LOGGER.info("No tasks requested. Exiting.")
        return 0

    for task_name in tasks_to_run:
        task_fn = TASKS.get(task_name)
        if task_fn is None:
            LOGGER.error("Unknown task '%s'. Skipping.", task_name)
            continue
        LOGGER.info("Starting task: %s", task_name)
        task_fn(config)
        LOGGER.info("Finished task: %s", task_name)

    return 0


if __name__ == "__main__":
    sys.exit(run())

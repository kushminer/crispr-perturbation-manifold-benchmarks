"""
Validation utilities to ensure parity with R implementation and paper results.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def validate_logo_integrity(logo_df: pd.DataFrame, all_perturbations: List[str]) -> Dict[str, bool]:
    """
    Validate that LOGO evaluation covers all perturbations without overlap.
    
    Returns:
        Dictionary with validation checks (all True if valid)
    """
    logo_perturbations = set(logo_df["perturbation"].unique())
    all_set = set(all_perturbations)
    
    checks = {
        "all_covered": logo_perturbations == all_set,
        "no_duplicates": len(logo_df["perturbation"]) == len(logo_perturbations),
        "complete_coverage": len(logo_perturbations) == len(all_set),
    }
    
    if not checks["all_covered"]:
        missing = all_set - logo_perturbations
        extra = logo_perturbations - all_set
        LOGGER.warning("LOGO coverage mismatch: missing %d, extra %d", len(missing), len(extra))
    
    return checks


def validate_hardness_bins(logo_df: pd.DataFrame, expected_bins: List[str] = ["near", "mid", "far"]) -> Dict[str, bool]:
    """
    Validate that hardness bins have approximately equal sizes.
    
    Returns:
        Dictionary with validation checks
    """
    bin_counts = logo_df["hardness_bin"].value_counts()
    total = len(logo_df)
    expected_per_bin = total / len(expected_bins)
    tolerance = max(1, int(0.05 * total))  # 5% tolerance
    
    checks = {
        "all_bins_present": set(bin_counts.index) == set(expected_bins),
        "balanced_sizes": all(
            abs(count - expected_per_bin) <= tolerance
            for count in bin_counts.values
        ),
    }
    
    if not checks["balanced_sizes"]:
        LOGGER.warning("Hardness bins unbalanced: %s", bin_counts.to_dict())
    
    return checks


def validate_class_holdout(class_df: pd.DataFrame, annotations: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate that each class appears exactly once as test set.
    
    Returns:
        Dictionary with validation checks
    """
    if class_df.empty:
        return {"has_results": False}
    
    # Each perturbation should appear once
    pert_counts = class_df["perturbation"].value_counts()
    checks = {
        "has_results": True,
        "no_duplicate_perturbations": (pert_counts == 1).all(),
        "all_classes_represented": set(class_df["class"].unique()) == set(annotations["class"].unique()),
    }
    
    return checks


def validate_model_parity(
    python_predictions: Dict[str, List[float]],
    r_predictions: Dict[str, List[float]],
    threshold: float = 0.99,
) -> Dict[str, float | bool]:
    """
    Validate that Python predictions match R implementation.
    
    Args:
        python_predictions: Dictionary mapping perturbation → predicted DE vector
        r_predictions: Dictionary mapping perturbation → R predicted DE vector
        threshold: Minimum mean Pearson r for validation to pass
    
    Returns:
        Dictionary with correlation metrics and validation status
    """
    common_perts = set(python_predictions.keys()) & set(r_predictions.keys())
    if not common_perts:
        return {"valid": False, "error": "No common perturbations"}
    
    correlations = []
    for pert in common_perts:
        py_vec = np.array(python_predictions[pert])
        r_vec = np.array(r_predictions[pert])
        
        if len(py_vec) != len(r_vec):
            continue
        
        # Remove NaN values
        mask = ~(np.isnan(py_vec) | np.isnan(r_vec))
        if mask.sum() < 2:
            continue
        
        corr = np.corrcoef(py_vec[mask], r_vec[mask])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    
    if not correlations:
        return {"valid": False, "error": "No valid correlations computed"}
    
    mean_corr = np.mean(correlations)
    min_corr = np.min(correlations)
    
    return {
        "valid": mean_corr >= threshold,
        "mean_pearson_r": float(mean_corr),
        "min_pearson_r": float(min_corr),
        "n_perturbations": len(correlations),
        "threshold": threshold,
    }


def run_full_validation(
    results_dir: Path,
    python_predictions_path: Path | None = None,
    r_predictions_path: Path | None = None,
    annotations_path: Path | None = None,
) -> Dict:
    """
    Run complete validation suite.
    
    Returns:
        Dictionary with all validation results
    """
    results_dir = Path(results_dir)
    validation_results = {}
    
    # Validate LOGO results
    logo_path = results_dir / "results_logo.csv"
    if logo_path.exists():
        logo_df = pd.read_csv(logo_path)
        if not logo_df.empty:
            # Get all perturbations from predictions if available
            all_perts = None
            if python_predictions_path and python_predictions_path.exists():
                with open(python_predictions_path) as f:
                    preds = json.load(f)
                    all_perts = list(preds.keys())
            
            if all_perts:
                validation_results["logo_integrity"] = validate_logo_integrity(logo_df, all_perts)
            validation_results["hardness_bins"] = validate_hardness_bins(logo_df)
    
    # Validate class holdout results
    class_path = results_dir / "results_class.csv"
    if class_path.exists() and annotations_path and annotations_path.exists():
        class_df = pd.read_csv(class_path)
        annotations = pd.read_csv(annotations_path, sep="\t")
        validation_results["class_holdout"] = validate_class_holdout(class_df, annotations)
    
    # Validate model parity if R predictions available
    if python_predictions_path and r_predictions_path:
        if python_predictions_path.exists() and r_predictions_path.exists():
            with open(python_predictions_path) as f:
                py_preds = json.load(f)
            with open(r_predictions_path) as f:
                r_preds = json.load(f)
            validation_results["model_parity"] = validate_model_parity(py_preds, r_preds)
    
    return validation_results


def validate_annotation_quality(
    annotations: pd.DataFrame,
    expression_targets: List[str] | None = None,
    min_class_size: int = 3,
) -> Dict[str, bool | int | float]:
    """
    Validate annotation file quality and provide diagnostics.
    
    Args:
        annotations: DataFrame with 'target' and 'class' columns
        expression_targets: Optional list of available perturbation names from expression data
        min_class_size: Minimum perturbations per class to be considered valid
    
    Returns:
        Dictionary with validation checks and statistics
    """
    checks = {}
    
    # Check required columns
    required_cols = {"target", "class"}
    has_required = required_cols.issubset(annotations.columns)
    checks["has_required_columns"] = has_required
    
    if not has_required:
        missing = required_cols - set(annotations.columns)
        LOGGER.error("Missing required columns: %s", missing)
        return checks
    
    # Check for empty annotations
    checks["non_empty"] = len(annotations) > 0
    if not checks["non_empty"]:
        LOGGER.error("Annotation file is empty")
        return checks
    
    # Check for duplicates
    duplicates = annotations["target"].duplicated()
    checks["no_duplicate_targets"] = not duplicates.any()
    if duplicates.any():
        n_dup = duplicates.sum()
        LOGGER.warning("Found %d duplicate targets in annotations", n_dup)
        checks["n_duplicates"] = int(n_dup)
    
    # Check for missing values
    missing_targets = annotations["target"].isna().sum()
    missing_classes = annotations["class"].isna().sum()
    checks["no_missing_targets"] = missing_targets == 0
    checks["no_missing_classes"] = missing_classes == 0
    if missing_targets > 0:
        LOGGER.warning("Found %d missing target values", missing_targets)
    if missing_classes > 0:
        LOGGER.warning("Found %d missing class values", missing_classes)
    
    # Class distribution
    class_counts = annotations.groupby("class")["target"].nunique()
    checks["n_classes"] = int(len(class_counts))
    checks["min_class_size"] = int(class_counts.min())
    checks["median_class_size"] = float(class_counts.median())
    checks["max_class_size"] = int(class_counts.max())
    
    # Classes meeting minimum size
    valid_classes = class_counts.loc[class_counts >= min_class_size]
    checks["n_valid_classes"] = int(len(valid_classes))
    checks["all_classes_valid"] = len(valid_classes) == len(class_counts)
    
    if not checks["all_classes_valid"]:
        small_classes = class_counts.loc[class_counts < min_class_size]
        LOGGER.warning(
            "Found %d classes below minimum size (%d): %s",
            len(small_classes),
            min_class_size,
            small_classes.to_dict(),
        )
    
    # Check alignment with expression data
    if expression_targets:
        expr_set = set(str(t) for t in expression_targets)
        annot_set = set(annotations["target"].astype(str))
        
        overlap = expr_set & annot_set
        missing_in_expr = annot_set - expr_set
        missing_in_annot = expr_set - annot_set
        
        checks["n_overlapping_targets"] = len(overlap)
        checks["n_missing_in_expression"] = len(missing_in_expr)
        checks["n_missing_in_annotations"] = len(missing_in_annot)
        checks["coverage_ratio"] = float(len(overlap) / len(expr_set)) if expr_set else 0.0
        
        if missing_in_expr:
            LOGGER.warning(
                "%d annotation targets not found in expression data (first 5: %s)",
                len(missing_in_expr),
                list(missing_in_expr)[:5],
            )
        if missing_in_annot:
            LOGGER.warning(
                "%d expression targets missing from annotations (first 5: %s)",
                len(missing_in_annot),
                list(missing_in_annot)[:5],
            )
    
    # Summary
    checks["is_valid"] = (
        checks.get("has_required_columns", False)
        and checks.get("non_empty", False)
        and checks.get("no_duplicate_targets", False)
        and checks.get("no_missing_targets", False)
        and checks.get("no_missing_classes", False)
        and checks.get("n_valid_classes", 0) > 0
    )
    
    return checks


def format_validation_summary(validation_results: Dict) -> str:
    """
    Format validation results as a human-readable summary.
    
    Args:
        validation_results: Dictionary of validation results
        
    Returns:
        Formatted summary string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("VALIDATION SUMMARY")
    lines.append("=" * 60)
    
    all_passed = True
    
    # Check each validation category
    if "logo_integrity" in validation_results:
        logo_checks = validation_results["logo_integrity"]
        lines.append("\nLOGO Integrity:")
        for check_name, check_result in logo_checks.items():
            status = "✅ PASS" if check_result else "❌ FAIL"
            lines.append(f"  {check_name}: {status}")
            if not check_result:
                all_passed = False
    
    if "hardness_bins" in validation_results:
        bin_checks = validation_results["hardness_bins"]
        lines.append("\nHardness Bins:")
        for check_name, check_result in bin_checks.items():
            status = "✅ PASS" if check_result else "❌ FAIL"
            lines.append(f"  {check_name}: {status}")
            if not check_result:
                all_passed = False
    
    if "class_holdout" in validation_results:
        class_checks = validation_results["class_holdout"]
        lines.append("\nFunctional-Class Holdout:")
        for check_name, check_result in class_checks.items():
            if isinstance(check_result, bool):
                status = "✅ PASS" if check_result else "❌ FAIL"
                lines.append(f"  {check_name}: {status}")
                if not check_result:
                    all_passed = False
            else:
                lines.append(f"  {check_name}: {check_result}")
    
    if "annotation_quality" in validation_results:
        annot_checks = validation_results["annotation_quality"]
        lines.append("\nAnnotation Quality:")
        is_valid = annot_checks.get("is_valid", False)
        status = "✅ PASS" if is_valid else "❌ FAIL"
        lines.append(f"  Overall: {status}")
        if not is_valid:
            all_passed = False
        
        # Show key statistics
        if "n_classes" in annot_checks:
            lines.append(f"  Classes: {annot_checks['n_classes']}")
        if "n_valid_classes" in annot_checks:
            lines.append(f"  Valid classes: {annot_checks['n_valid_classes']}")
        if "coverage_ratio" in annot_checks:
            lines.append(f"  Coverage: {annot_checks['coverage_ratio']:.2%}")
    
    if "model_parity" in validation_results:
        parity = validation_results["model_parity"]
        lines.append("\nModel Parity (R vs Python):")
        is_valid = parity.get("valid", False)
        status = "✅ PASS" if is_valid else "❌ FAIL"
        lines.append(f"  Overall: {status}")
        if not is_valid:
            all_passed = False
        
        if "mean_pearson_r" in parity:
            lines.append(f"  Mean Pearson r: {parity['mean_pearson_r']:.4f}")
        if "threshold" in parity:
            lines.append(f"  Threshold: {parity['threshold']:.2f}")
    
    lines.append("\n" + "=" * 60)
    if all_passed:
        lines.append("✅ ALL VALIDATION CHECKS PASSED")
    else:
        lines.append("❌ SOME VALIDATION CHECKS FAILED")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def run_comprehensive_validation(
    config,
    python_predictions_path: Path | None = None,
    r_predictions_path: Path | None = None,
) -> Dict:
    """
    Run comprehensive validation suite for a dataset configuration.
    
    Args:
        config: ExperimentConfig object
        python_predictions_path: Optional path to Python predictions JSON
        r_predictions_path: Optional path to R predictions JSON
        
    Returns:
        Dictionary with all validation results
    """
    from shared.io import load_annotations, load_expression_dataset
    
    validation_results = {}
    
    # Validate annotations if available
    if config.dataset.annotation_path and config.dataset.annotation_path.exists():
        LOGGER.info("Validating annotation quality...")
        annotations = load_annotations(config.dataset.annotation_path)
        
        # Load expression data to check alignment
        expression_targets = None
        try:
            expression = load_expression_dataset(
                config.dataset.expression_path,
                config.dataset.gene_names_path,
            )
            expression_targets = expression.index.tolist()
        except Exception as e:
            LOGGER.warning("Could not load expression data for validation: %s", e)
        
        validation_results["annotation_quality"] = validate_annotation_quality(
            annotations=annotations,
            expression_targets=expression_targets,
            min_class_size=config.dataset.functional_min_class_size,
        )
    
    # Validate results if they exist
    if config.output_root.exists():
        validation_results.update(
            run_full_validation(
                results_dir=config.output_root,
                python_predictions_path=python_predictions_path or config.dataset.expression_path,
                r_predictions_path=r_predictions_path,
                annotations_path=config.dataset.annotation_path,
            )
        )
    else:
        LOGGER.warning("Results directory does not exist: %s", config.output_root)
        LOGGER.warning("Skipping results validation. Run evaluation tasks first.")
    
    return validation_results


def _convert_to_json_serializable(obj):
    """Convert numpy types and other non-serializable objects to JSON-compatible types."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def save_validation_report(validation_results: Dict, output_path: Path) -> None:
    """Save validation results to JSON file and print summary."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to JSON-serializable types
    serializable_results = _convert_to_json_serializable(validation_results)
    
    # Save JSON
    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    LOGGER.info("Validation report saved to %s", output_path)
    
    # Print summary
    summary = format_validation_summary(validation_results)
    LOGGER.info("\n%s", summary)


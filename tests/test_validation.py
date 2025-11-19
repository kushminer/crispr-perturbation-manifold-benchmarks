"""
Unit tests for validation functions.
"""

import pandas as pd
import pytest
from shared.validation import (
    validate_annotation_quality,
    validate_logo_integrity,
    validate_hardness_bins,
    validate_class_holdout,
    format_validation_summary,
)


def test_validate_annotation_quality_basic(synthetic_annotations):
    """Test basic annotation quality validation."""
    checks = validate_annotation_quality(
        annotations=synthetic_annotations,
        min_class_size=5,
    )
    
    assert "has_required_columns" in checks
    assert "non_empty" in checks
    assert "n_classes" in checks
    assert checks["has_required_columns"] is True
    assert checks["non_empty"] is True


def test_validate_annotation_quality_missing_columns():
    """Test validation with missing required columns."""
    annotations = pd.DataFrame({
        "target": ["pert_0", "pert_1"],
        "wrong_column": ["Class_1", "Class_2"],
    })
    
    checks = validate_annotation_quality(annotations)
    
    assert checks["has_required_columns"] is False


def test_validate_annotation_quality_duplicates():
    """Test validation detects duplicate targets."""
    annotations = pd.DataFrame({
        "target": ["pert_0", "pert_0", "pert_1"],  # Duplicate
        "class": ["Class_1", "Class_1", "Class_2"],
    })
    
    checks = validate_annotation_quality(annotations)
    
    assert checks["no_duplicate_targets"] is False
    assert "n_duplicates" in checks


def test_validate_annotation_quality_alignment():
    """Test annotation alignment with expression data."""
    annotations = pd.DataFrame({
        "target": ["pert_0", "pert_1", "pert_2"],
        "class": ["Class_1", "Class_1", "Class_2"],
    })
    
    expression_targets = ["pert_0", "pert_1", "pert_3"]  # pert_2 missing, pert_3 extra
    
    checks = validate_annotation_quality(
        annotations=annotations,
        expression_targets=expression_targets,
    )
    
    assert "n_overlapping_targets" in checks
    assert "n_missing_in_expression" in checks
    assert "n_missing_in_annotations" in checks
    assert checks["n_overlapping_targets"] == 2
    assert checks["n_missing_in_expression"] == 1
    assert checks["n_missing_in_annotations"] == 1


def test_validate_logo_integrity():
    """Test LOGO integrity validation."""
    logo_df = pd.DataFrame({
        "perturbation": [f"pert_{i}" for i in range(10)],
        "pearson_r": [0.5] * 10,
    })
    all_perts = [f"pert_{i}" for i in range(10)]
    
    checks = validate_logo_integrity(logo_df, all_perts)
    
    assert checks["all_covered"] is True
    assert checks["no_duplicates"] is True
    assert checks["complete_coverage"] is True


def test_validate_logo_integrity_missing():
    """Test LOGO integrity with missing perturbations."""
    logo_df = pd.DataFrame({
        "perturbation": ["pert_0", "pert_1"],
        "pearson_r": [0.5, 0.6],
    })
    all_perts = ["pert_0", "pert_1", "pert_2", "pert_3"]
    
    checks = validate_logo_integrity(logo_df, all_perts)
    
    assert checks["all_covered"] is False


def test_validate_hardness_bins():
    """Test hardness bin validation."""
    logo_df = pd.DataFrame({
        "perturbation": [f"pert_{i}" for i in range(30)],
        "hardness_bin": ["near"] * 10 + ["mid"] * 10 + ["far"] * 10,
        "pearson_r": [0.5] * 30,
    })
    
    checks = validate_hardness_bins(logo_df)
    
    assert checks["all_bins_present"] is True
    assert checks["balanced_sizes"] is True


def test_validate_class_holdout(synthetic_annotations):
    """Test class holdout validation."""
    class_df = pd.DataFrame({
        "perturbation": ["pert_0", "pert_1", "pert_2"],
        "class": ["Class_1", "Class_1", "Class_2"],
        "pearson_r": [0.5, 0.6, 0.7],
    })
    
    checks = validate_class_holdout(class_df, synthetic_annotations)
    
    assert checks["has_results"] is True
    # Check that no_duplicate_perturbations exists and is True
    assert "no_duplicate_perturbations" in checks
    assert bool(checks["no_duplicate_perturbations"]) is True


def test_format_validation_summary():
    """Test validation summary formatting."""
    validation_results = {
        "logo_integrity": {
            "all_covered": True,
            "no_duplicates": True,
        },
        "hardness_bins": {
            "all_bins_present": True,
            "balanced_sizes": False,
        },
        "annotation_quality": {
            "is_valid": True,
            "n_classes": 5,
        },
    }
    
    summary = format_validation_summary(validation_results)
    
    assert isinstance(summary, str)
    assert "VALIDATION SUMMARY" in summary
    assert "✅" in summary or "❌" in summary


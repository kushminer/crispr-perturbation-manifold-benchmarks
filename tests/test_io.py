"""
Unit tests for I/O functions.
"""

import json
import pandas as pd
import pytest
from pathlib import Path
from shared.io import (
    load_expression_matrix,
    load_annotations,
    load_expression_dataset,
)


def test_load_expression_matrix_json(temp_dir, sample_predictions_json):
    """Test loading expression matrix from JSON."""
    pred_path, _ = sample_predictions_json
    
    df = load_expression_matrix(pred_path)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # 3 perturbations
    assert len(df.columns) == 5  # 5 genes
    assert "pert_0" in df.index
    assert "pert_1" in df.index
    assert "ctrl" in df.index


def test_load_expression_matrix_csv(temp_dir):
    """Test loading expression matrix from CSV."""
    csv_path = temp_dir / "expression.csv"
    df_data = pd.DataFrame({
        "gene_0": [0.1, 0.2, 0.3],
        "gene_1": [0.4, 0.5, 0.6],
    }, index=["pert_0", "pert_1", "pert_2"])
    df_data.to_csv(csv_path)
    
    df = load_expression_matrix(csv_path)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert len(df.columns) == 2


def test_load_expression_matrix_tsv(temp_dir):
    """Test loading expression matrix from TSV."""
    tsv_path = temp_dir / "expression.tsv"
    df_data = pd.DataFrame({
        "gene_0": [0.1, 0.2],
        "gene_1": [0.3, 0.4],
    }, index=["pert_0", "pert_1"])
    df_data.to_csv(tsv_path, sep="\t")
    
    df = load_expression_matrix(tsv_path)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_load_expression_dataset_with_gene_names(temp_dir, sample_predictions_json):
    """Test loading expression dataset with separate gene names file."""
    pred_path, gene_path = sample_predictions_json
    
    df = load_expression_dataset(pred_path, gene_path)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    # Gene names should be used as column names
    assert "gene_0" in df.columns


def test_load_annotations(temp_dir):
    """Test loading annotations from TSV."""
    annot_path = temp_dir / "annotations.tsv"
    df_data = pd.DataFrame({
        "target": ["pert_0", "pert_1", "pert_2"],
        "class": ["Class_1", "Class_1", "Class_2"],
    })
    df_data.to_csv(annot_path, sep="\t", index=False)
    
    df = load_annotations(annot_path)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "target" in df.columns
    assert "class" in df.columns


def test_load_expression_dataset(temp_dir, sample_predictions_json):
    """Test loading expression dataset with gene names."""
    pred_path, gene_path = sample_predictions_json
    
    df = load_expression_dataset(pred_path, gene_path)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert len(df.columns) == 5
    # Check that gene names are correct
    assert "gene_0" in df.columns


def test_load_expression_matrix_nonexistent():
    """Test that loading nonexistent file raises error."""
    with pytest.raises(FileNotFoundError):
        load_expression_matrix(Path("nonexistent_file.json"))


def test_load_expression_matrix_invalid_format(temp_dir):
    """Test that invalid format raises error."""
    invalid_path = temp_dir / "invalid.xyz"
    invalid_path.write_text("invalid content")
    
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_expression_matrix(invalid_path)


"""
Input/output helpers for loading differential expression data and annotations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pandas as pd


def load_expression_matrix(path: Path) -> pd.DataFrame:
    """
    Load a differential expression matrix with perturbations on rows and genes on columns.

    The loader infers the format from the file suffix:
        - .csv / .tsv / .txt via pandas read_csv
        - .parquet via pandas read_parquet
        - .feather via pandas read_feather
        - .json mapping perturbation -> vector (list)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Expression matrix not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        sep = "," if suffix == ".csv" else "\t"
        return pd.read_csv(path, sep=sep, index_col=0)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path).set_index("target")
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            return pd.DataFrame.from_dict(payload, orient="index")
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        raise ValueError("JSON expression file must be a dict or list of rows.")

    raise ValueError(f"Unsupported file format for expression matrix: {path}")


def load_annotations(path: Path) -> pd.DataFrame:
    """
    Load functional annotations mapping perturbations to classes.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "," if suffix == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported annotation format: {path}")

    expected_cols = {"target", "class"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Annotation file missing required columns: {missing}")

    return df[["target", "class"]]


def align_expression_with_annotations(
    expression: pd.DataFrame, annotations: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align expression matrix and annotation data frames on the target column/index.
    """
    annotations = annotations.copy()
    annotations["target"] = annotations["target"].astype(str)

    expression = expression.copy()
    expression.index = expression.index.astype(str)

    common = expression.index.intersection(annotations["target"])
    if common.empty:
        raise ValueError("No overlap between expression targets and annotations.")

    expression = expression.loc[common]
    annotations = annotations.set_index("target").loc[common].reset_index()
    return expression, annotations


def load_expression_dataset(
    expression_path: Path, gene_names_path: Path | None = None
) -> pd.DataFrame:
    """
    Load expression data and return a perturbation x gene matrix.

    If `gene_names_path` is provided and the expression file lacks column names,
    they will be applied from the JSON or text file provided.
    """
    expression = load_expression_matrix(expression_path)

    if gene_names_path is not None:
        gene_names_path = Path(gene_names_path)
        if not gene_names_path.exists():
            raise FileNotFoundError(f"Gene names file not found: {gene_names_path}")
        suffix = gene_names_path.suffix.lower()
        if suffix == ".json":
            with gene_names_path.open("r", encoding="utf-8") as fh:
                gene_names = json.load(fh)
        elif suffix in {".csv", ".tsv", ".txt"}:
            sep = "," if suffix == ".csv" else "\t"
            gene_names = pd.read_csv(gene_names_path, sep=sep, header=None).iloc[:, 0].tolist()
        else:
            raise ValueError(f"Unsupported gene names format: {gene_names_path}")

        if expression.shape[1] != len(gene_names):
            raise ValueError(
                f"Gene name count ({len(gene_names)}) does not match expression columns ({expression.shape[1]})."
            )
        expression.columns = gene_names

    expression.index.name = "target"
    expression = expression.sort_index()
    expression = expression.apply(pd.to_numeric, errors="coerce")
    return expression


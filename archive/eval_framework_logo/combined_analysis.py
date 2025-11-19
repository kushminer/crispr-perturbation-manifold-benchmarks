"""
Utilities to merge LOGO and functional-class evaluation outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_result_tables(output_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load previously generated LOGO and class-holdout results.
    """
    output_root = Path(output_root)
    logo_path = output_root / "results_logo.csv"
    class_path = output_root / "results_class.csv"

    if not logo_path.exists():
        raise FileNotFoundError(f"LOGO results not found at {logo_path}")
    if not class_path.exists():
        raise FileNotFoundError(f"Class holdout results not found at {class_path}")

    try:
        logo_df = pd.read_csv(logo_path)
    except pd.errors.EmptyDataError:
        logo_df = pd.DataFrame(columns=["perturbation", "hardness_bin", "cluster_blocked", "split_type", "pearson_r", "spearman_rho", "mse", "mae"])

    try:
        class_df = pd.read_csv(class_path)
    except pd.errors.EmptyDataError:
        class_df = pd.DataFrame(columns=["perturbation", "class", "split_type", "pearson_r", "spearman_rho", "mse", "mae"])
    return logo_df, class_df


def compute_combined_summary(
    logo_df: pd.DataFrame, class_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a summary table of median metrics across hardness bins and classes.
    """
    summaries = []

    if not logo_df.empty:
        logo_summary = (
            logo_df.groupby("hardness_bin")[["pearson_r", "spearman_rho", "mse", "mae"]]
            .median()
            .reset_index()
        )
        logo_summary.insert(0, "split_type", "logo")
        logo_summary.rename(columns={"hardness_bin": "group"}, inplace=True)
        summaries.append(logo_summary)

    if not class_df.empty:
        class_summary = (
            class_df.groupby("class")[["pearson_r", "spearman_rho", "mse", "mae"]]
            .median()
            .reset_index()
        )
        class_summary.insert(0, "split_type", "class")
        class_summary.rename(columns={"class": "group"}, inplace=True)
        summaries.append(class_summary)

    if not summaries:
        return pd.DataFrame(
            columns=["split_type", "group", "pearson_r", "spearman_rho", "mse", "mae"]
        )

    return pd.concat(summaries, ignore_index=True)


def compute_heatmap_matrix(
    logo_df: pd.DataFrame, class_df: pd.DataFrame, metric: str = "pearson_r"
) -> pd.DataFrame:
    """
    Construct a matrix for heatmap visualization with hardness bins as rows
    and functional classes as columns.
    """
    if logo_df.empty or class_df.empty:
        return pd.DataFrame()

    merged = logo_df.merge(
        class_df,
        on="perturbation",
        suffixes=("_logo", "_class"),
        how="inner",
    )

    if merged.empty or f"{metric}_class" not in merged.columns:
        return pd.DataFrame()

    pivot = (
        merged.pivot_table(
            index="hardness_bin",
            columns="class",
            values=f"{metric}_class",
            aggfunc="median",
        )
        .sort_index()
        .sort_index(axis=1)
    )
    return pivot


def save_combined_outputs(
    summary_df: pd.DataFrame,
    heatmap_df: pd.DataFrame,
    output_root: Path,
) -> None:
    """
    Persist combined analysis tables.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_path = output_root / "combined_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    heatmap_path = output_root / "combined_heatmap.csv"
    heatmap_df.to_csv(heatmap_path)


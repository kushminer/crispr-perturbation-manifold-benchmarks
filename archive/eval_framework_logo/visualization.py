"""
Visualization utilities for the evaluation framework.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from umap import UMAP


def plot_umap(
    expression: pd.DataFrame,
    annotations: Optional[pd.DataFrame],
    output_path: Path,
    seed: int = 1,
) -> None:
    reducer = UMAP(n_neighbors=10, min_dist=0.5, random_state=seed)
    embedding = reducer.fit_transform(expression.to_numpy())

    fig, ax = plt.subplots(figsize=(6, 5))
    if annotations is not None and "class" in annotations.columns:
        merged = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
        merged["class"] = annotations.set_index("target").loc[expression.index, "class"].values
        sns.scatterplot(
            data=merged,
            x="UMAP1",
            y="UMAP2",
            hue="class",
            ax=ax,
            s=20,
            palette="tab10",
            linewidth=0,
        )
        ax.legend(loc="best", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=15, alpha=0.7)
    ax.set_title("Perturbation response UMAP")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_logo_lines(logo_df: pd.DataFrame, output_path: Path) -> None:
    if logo_df.empty:
        return
    summary = (
        logo_df.groupby("hardness_bin")[["pearson_r", "spearman_rho", "mse", "mae"]]
        .median()
        .reset_index()
        .sort_values("hardness_bin")
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=summary, x="hardness_bin", y="pearson_r", marker="o", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("LOGO performance vs. hardness")
    ax.set_xlabel("Hardness bin")
    ax.set_ylabel("Median Pearson r")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_class_bars(class_df: pd.DataFrame, output_path: Path) -> None:
    if class_df.empty:
        return
    summary = (
        class_df.groupby("class")[["pearson_r", "spearman_rho", "mse", "mae"]]
        .median()
        .reset_index()
        .sort_values("pearson_r", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=summary, x="pearson_r", y="class", palette="viridis", ax=ax)
    ax.set_xlim(0, 1)
    ax.set_title("Functional-class holdout (median Pearson r)")
    ax.set_xlabel("Median Pearson r")
    ax.set_ylabel("Class")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_combined_heatmap(heatmap_df: pd.DataFrame, output_path: Path) -> None:
    if heatmap_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="crest",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Median Pearson r across hardness × class")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


#!/usr/bin/env python3
"""
Visualization helpers for the single-cell baseline audit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


plt.style.use("seaborn-v0_8-whitegrid")


def _scatter_plot(df: pd.DataFrame, dataset: str, baseline_a: str, baseline_b: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(
        data=df,
        x=f"{baseline_a}_pearson_r",
        y=f"{baseline_b}_pearson_r",
        ax=ax,
        s=25,
    )
    lims = [
        min(df[f"{baseline_a}_pearson_r"].min(), df[f"{baseline_b}_pearson_r"].min()) - 0.02,
        max(df[f"{baseline_a}_pearson_r"].max(), df[f"{baseline_b}_pearson_r"].max()) + 0.02,
    ]
    ax.plot(lims, lims, linestyle="--", color="gray", label="y=x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(f"{dataset}: {baseline_b} vs {baseline_a}")
    ax.set_xlabel(f"{baseline_a} Pearson r")
    ax.set_ylabel(f"{baseline_b} Pearson r")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _delta_plot(df: pd.DataFrame, dataset: str, baseline_a: str, baseline_b: str, out_path: Path) -> None:
    sorted_df = df.sort_values("delta_pearson_r")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=sorted_df,
        x="perturbation",
        y="delta_pearson_r",
        color="#2ecc71",
        ax=ax,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(f"{dataset}: Δr = {baseline_b} - {baseline_a}")
    ax.set_ylabel("Pearson r difference")
    ax.set_xlabel("Perturbation (sorted)")
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _integrity_dashboard(scatter_path: Path, delta_path: Path, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, img_path, title in zip(
        axes,
        [scatter_path, delta_path],
        ["Correlation", "Difference distribution"],
    ):
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_visuals(audit_output_dir: Path, datasets: List[str]) -> None:
    for dataset in datasets:
        csv_path = audit_output_dir / f"{dataset}_lpm_gearsPertEmb_vs_lpm_selftrained.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        scatter_path = audit_output_dir / f"{dataset}_gears_vs_selftrained_scatter.png"
        delta_path = audit_output_dir / f"{dataset}_gears_vs_selftrained_delta.png"
        dashboard_path = audit_output_dir / f"{dataset}_gears_vs_selftrained_dashboard.png"
        
        _scatter_plot(df, dataset, "lpm_selftrained", "lpm_gearsPertEmb", scatter_path)
        _delta_plot(df, dataset, "lpm_selftrained", "lpm_gearsPertEmb", delta_path)
        _integrity_dashboard(scatter_path, delta_path, dashboard_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visuals for single-cell audit.")
    parser.add_argument(
        "--audit_dir",
        type=Path,
        default=Path("deliverables/audits/single_cell_data_audit/output"),
        help="Directory containing audit CSVs (from validate_embeddings.py).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["adamson", "k562", "rpe1"],
        help="Datasets to visualize.",
    )
    args = parser.parse_args()
    
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    build_visuals(args.audit_dir, args.datasets)


if __name__ == "__main__":
    main()


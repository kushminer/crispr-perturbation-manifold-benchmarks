#!/usr/bin/env python3
"""
Audit utilities for comparing single-cell baseline embeddings.

Focus: determine whether GEARS perturbation embeddings diverge from
self-trained PCA embeddings at the single-cell level.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


BASELINE_PAIRS = [
    ("lpm_selftrained", "lpm_gearsPertEmb"),
    ("lpm_selftrained", "lpm_scgptGeneEmb"),
]


def _resolve_dataset_dir(results_root: Path, dataset: str) -> Optional[Path]:
    """Prefer *_expanded directories (if present) for richer baselines."""
    expanded = results_root / f"{dataset}_expanded"
    if expanded.exists():
        return expanded
    regular = results_root / dataset
    if regular.exists():
        return regular
    return None


def _load_pert_metrics(dataset_dir: Path, baseline: str) -> Optional[pd.DataFrame]:
    """Load perturbation-level metrics for a baseline."""
    csv_path = dataset_dir / baseline / "pert_metrics.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, index_col=0)
    df = df.rename(columns={col: f"{baseline}_{col}" for col in df.columns})
    df["perturbation"] = df.index
    return df


def compare_baselines(dataset_dir: Path, baseline_a: str, baseline_b: str) -> pd.DataFrame:
    """Join perturbation metrics for two baselines and compute deltas."""
    df_a = _load_pert_metrics(dataset_dir, baseline_a)
    df_b = _load_pert_metrics(dataset_dir, baseline_b)
    if df_a is None or df_b is None:
        raise FileNotFoundError(f"Missing pert_metrics for {baseline_a} or {baseline_b} in {dataset_dir}")
    
    merged = pd.merge(df_a, df_b, on="perturbation", how="inner")
    merged["delta_pearson_r"] = merged[f"{baseline_b}_pearson_r"] - merged[f"{baseline_a}_pearson_r"]
    merged["delta_l2"] = merged[f"{baseline_b}_l2"] - merged[f"{baseline_a}_l2"]
    return merged


def audit_dataset(results_root: Path, dataset: str, output_dir: Path) -> Dict[str, float]:
    """Audit a single dataset; returns summary statistics."""
    dataset_dir = _resolve_dataset_dir(results_root, dataset)
    if dataset_dir is None:
        raise FileNotFoundError(f"No results found for dataset '{dataset}' under {results_root}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: Dict[str, float] = {}
    
    for baseline_a, baseline_b in BASELINE_PAIRS:
        try:
            comparison = compare_baselines(dataset_dir, baseline_a, baseline_b)
        except FileNotFoundError:
            continue
        
        pair_name = f"{baseline_b}_vs_{baseline_a}"
        comparison.to_csv(output_dir / f"{dataset}_{pair_name}.csv", index=False)
        
        summaries[f"{pair_name}_mean_delta_r"] = comparison["delta_pearson_r"].mean()
        summaries[f"{pair_name}_std_delta_r"] = comparison["delta_pearson_r"].std()
        identical = (comparison["delta_pearson_r"].abs() < 1e-9).all()
        summaries[f"{pair_name}_identical"] = float(identical)
    
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate single-cell baseline embeddings.")
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path("results/single_cell_analysis"),
        help="Root directory containing single-cell baseline outputs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["adamson", "k562", "rpe1"],
        help="Datasets to audit.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("deliverables/audits/single_cell_data_audit/output"),
        help="Directory to store audit artifacts.",
    )
    args = parser.parse_args()
    
    all_summaries = []
    for dataset in args.datasets:
        try:
            summary = audit_dataset(args.results_root, dataset, args.output_dir)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}")
            continue
        summary["dataset"] = dataset
        all_summaries.append(summary)
        print(f"[INFO] {dataset} summary: {summary}")
    
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(args.output_dir / "baseline_audit_summary.csv", index=False)
        print(f"[INFO] Wrote summary to {args.output_dir / 'baseline_audit_summary.csv'}")
    else:
        print("[WARN] No datasets were successfully audited.")


if __name__ == "__main__":
    main()


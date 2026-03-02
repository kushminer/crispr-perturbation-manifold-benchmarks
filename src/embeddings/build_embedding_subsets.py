#!/usr/bin/env python3
"""
Utility script to build lightweight subsets of legacy embedding inputs for parity
validation (Sprint 5 / Issue #18).

Outputs are written under validation/embedding_subsets/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
FRAMEWORK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GO_SOURCE = (
    REPO_ROOT / "paper/benchmark/data/gears_pert_data/go_essential_all/go_essential_all.csv"
)
DEFAULT_H5AD_SOURCE = (
    REPO_ROOT
    / "paper/benchmark/data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad"
)
DEFAULT_OUTPUT_DIR = FRAMEWORK_ROOT / "validation/embedding_subsets"


def sha256_file(path: Path) -> str:
    """Compute SHA256 for the given file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def subset_go_graph(
    source_csv: Path,
    output_csv: Path,
    max_nodes: int = 150,
    seed: int = 123,
) -> Dict[str, int]:
    """Create a small induced subgraph from the GO similarity CSV."""
    if not source_csv.exists():
        raise FileNotFoundError(f"GO source file not found: {source_csv}")

    df = pd.read_csv(source_csv)
    nodes = sorted(set(df["source"]).union(set(df["target"])))
    rng = np.random.default_rng(seed)

    if len(nodes) <= max_nodes:
        selected = nodes
    else:
        selected = sorted(rng.choice(nodes, size=max_nodes, replace=False))

    mask = df["source"].isin(selected) & df["target"].isin(selected)
    subset_df = df.loc[mask].copy()
    subset_df.to_csv(output_csv, index=False)

    return {
        "total_nodes": len(nodes),
        "selected_nodes": len(selected),
        "total_edges": len(df),
        "selected_edges": len(subset_df),
    }


def subset_h5ad(
    input_h5ad: Path,
    output_h5ad: Path,
    max_perturbations: int = 100,
    max_genes: int = 512,
    seed: int = 123,
) -> Dict[str, int]:
    """Trim an h5ad file to a manageable number of perturbations and genes."""
    import anndata as ad

    if not input_h5ad.exists():
        raise FileNotFoundError(f"h5ad source file not found: {input_h5ad}")

    rng = np.random.default_rng(seed)
    adata = ad.read_h5ad(input_h5ad)

    condition_col = "clean_condition" if "clean_condition" in adata.obs.columns else None
    if condition_col is None:
        if "condition" in adata.obs.columns:
            condition_col = "condition"
        else:
            raise ValueError(
                "AnnData is missing both 'clean_condition' and 'condition' in obs."
            )

    unique_perts: List[str] = sorted(adata.obs[condition_col].astype(str).unique().tolist())
    if len(unique_perts) <= max_perturbations:
        keep_perts = unique_perts
    else:
        keep_perts = sorted(
            rng.choice(unique_perts, size=max_perturbations, replace=False).tolist()
        )

    mask = adata.obs[condition_col].astype(str).isin(keep_perts)
    subset = adata[mask].copy()

    if subset.n_obs == 0:
        raise ValueError("No cells left after perturbation filtering.")

    if subset.n_vars > max_genes:
        gene_indices = np.arange(max_genes)
        subset = subset[:, gene_indices].copy()

    subset.write_h5ad(output_h5ad, compression="gzip")

    return {
        "total_cells": int(adata.n_obs),
        "selected_cells": int(subset.n_obs),
        "total_genes": int(adata.n_vars),
        "selected_genes": int(subset.n_vars),
        "total_perturbations": len(unique_perts),
        "selected_perturbations": len(keep_perts),
        "condition_column": condition_col,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build subsets for embedding parity validation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to store subset artifacts (default: validation/embedding_subsets).",
    )
    parser.add_argument(
        "--go-source",
        type=Path,
        default=DEFAULT_GO_SOURCE,
        help="Path to GO similarity CSV.",
    )
    parser.add_argument(
        "--h5ad-source",
        type=Path,
        default=DEFAULT_H5AD_SOURCE,
        help="Path to perturb_processed.h5ad (default: Replogle K562).",
    )
    parser.add_argument("--go-nodes", type=int, default=150, help="GO nodes to sample.")
    parser.add_argument(
        "--perturbations",
        type=int,
        default=120,
        help="Perturbations to sample from h5ad.",
    )
    parser.add_argument(
        "--genes",
        type=int,
        default=512,
        help="Maximum genes to keep in h5ad subset.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}

    go_output = output_dir / "go_subset.csv"
    go_stats = subset_go_graph(
        source_csv=args.go_source.resolve(),
        output_csv=go_output,
        max_nodes=args.go_nodes,
        seed=args.seed,
    )
    manifest["go_subset.csv"] = {
        "source": str(args.go_source.resolve()),
        "sha256": sha256_file(go_output),
        "stats": go_stats,
    }

    h5ad_output = output_dir / "replogle_subset.h5ad"
    h5ad_stats = subset_h5ad(
        input_h5ad=args.h5ad_source.resolve(),
        output_h5ad=h5ad_output,
        max_perturbations=args.perturbations,
        max_genes=args.genes,
        seed=args.seed,
    )
    manifest["replogle_subset.h5ad"] = {
        "source": str(args.h5ad_source.resolve()),
        "sha256": sha256_file(h5ad_output),
        "stats": h5ad_stats,
    }

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"Subset artifacts written to {output_dir}")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()

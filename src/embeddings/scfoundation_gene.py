"""
Loader for scFoundation (maeautobin) gene embeddings.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import anndata as ad
import numpy as np

from .base import EmbeddingResult
from .registry import register


@register("scfoundation_gene")
def load_scfoundation_gene_embeddings(
    checkpoint_path: Path,
    demo_h5ad: Path,
    subset_genes: Optional[List[str]] = None,
    subset_genes_path: Optional[Path] = None,
) -> EmbeddingResult:
    """
    Load scFoundation gene embeddings using the maeautobin checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Path to `models.ckpt` (PyTorch checkpoint containing `'gene'` weights).
    demo_h5ad:
        Path to `demo.h5ad` providing gene ordering.
    subset_genes:
        Optional list of gene symbols to select (order preserved).
    """

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    demo_path = Path(demo_h5ad).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"scFoundation checkpoint not found: {checkpoint_path}")
    if not demo_path.exists():
        raise FileNotFoundError(f"scFoundation demo file not found: {demo_path}")

    import torch  # defer heavy import

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    try:
        pos_emb = ckpt["gene"]["state_dict"]["model.pos_emb.weight"].numpy()
    except KeyError as exc:
        raise KeyError("Could not find gene positional embeddings in checkpoint") from exc

    demo = ad.read_h5ad(demo_path)
    gene_names = demo.var["gene_name"].tolist()
    gene_names = gene_names + ["log10TotalCount1", "log10TotalCount2", "<pad>"]

    if pos_emb.shape[0] != len(gene_names):
        raise ValueError("Mismatch between embedding rows and gene names.")

    if subset_genes is None and subset_genes_path is not None:
        subset_genes = [
            gene.strip()
            for gene in Path(subset_genes_path).read_text(encoding="utf-8").splitlines()
            if gene.strip()
        ]

    if subset_genes:
        name_to_idx = {g: i for i, g in enumerate(gene_names)}
        missing = [g for g in subset_genes if g not in name_to_idx]
        if missing:
            raise ValueError(f"Genes not found in scFoundation demo set: {missing[:5]}...")
        indices = [name_to_idx[g] for g in subset_genes]
        values = pos_emb[indices, :].T
        labels = subset_genes
    else:
        values = pos_emb.T
        labels = gene_names

    dim_labels = [f"dim_{i+1}" for i in range(values.shape[0])]
    return EmbeddingResult(
        values=values,
        item_labels=labels,
        dim_labels=dim_labels,
        metadata={
            "checkpoint_path": str(checkpoint_path),
            "demo_h5ad": str(demo_path),
            "n_genes": len(labels),
            "embedding_dim": values.shape[0],
        },
    )


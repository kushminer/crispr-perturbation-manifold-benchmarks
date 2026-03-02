"""
Loader for scGPT gene embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import EmbeddingResult
from .registry import register


def _load_vocab(vocab_path: Path) -> List[str]:
    with vocab_path.open("r", encoding="utf-8") as fh:
        vocab_json = json.load(fh)
    if isinstance(vocab_json, dict) and "itos" in vocab_json:
        return vocab_json["itos"]
    if isinstance(vocab_json, dict) and "stoi" in vocab_json:
        stoi = vocab_json["stoi"]
        size = max(int(idx) for idx in stoi.values()) + 1
        vocab_list: List[Optional[str]] = [None] * size  # type: ignore[name-defined]
        for gene, idx in stoi.items():
            vocab_list[int(idx)] = gene
        return [gene for gene in vocab_list if gene is not None]
    if isinstance(vocab_json, dict) and all(
        isinstance(idx, int) for idx in vocab_json.values()
    ):
        size = max(vocab_json.values()) + 1
        vocab_list = [None] * size
        for gene, idx in vocab_json.items():
            vocab_list[int(idx)] = gene
        return [gene for gene in vocab_list if gene is not None]
    if isinstance(vocab_json, list):
        return vocab_json
    raise ValueError(f"Unexpected vocab format in {vocab_path}")


@register("scgpt_gene")
def load_scgpt_gene_embeddings(
    checkpoint_dir: Path,
    subset_genes: Optional[List[str]] = None,
    subset_genes_path: Optional[Path] = None,
) -> EmbeddingResult:
    """
    Load scGPT encoder embeddings from a checkpoint directory.

    Parameters
    ----------
    checkpoint_dir:
        Directory containing `best_model.pt` (or shard) and `vocab.json`.
    subset_genes:
        Optional list of gene symbols to extract (order preserved).
    """

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    model_path = checkpoint_dir / "best_model.pt"
    vocab_path = checkpoint_dir / "vocab.json"

    if not model_path.exists():
        raise FileNotFoundError(f"scGPT checkpoint missing: {model_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"scGPT vocab missing: {vocab_path}")

    vocab_list = _load_vocab(vocab_path)
    vocab_index = {gene: idx for idx, gene in enumerate(vocab_list)}

    import torch  # defer heavy import

    state_dict = torch.load(model_path, map_location="cpu")
    # Find embedding layer weight
    weight_key = None
    for key in state_dict.keys():
        if key.endswith("encoder.embedding.weight") or key.endswith("encoder.weight"):
            weight_key = key
            break
    if weight_key is None:
        raise KeyError("Could not locate encoder embedding weights in scGPT checkpoint.")

    embedding_matrix = state_dict[weight_key].cpu().numpy()

    if subset_genes is None and subset_genes_path is not None:
        subset_genes = [
            gene.strip()
            for gene in Path(subset_genes_path).read_text(encoding="utf-8").splitlines()
            if gene.strip()
        ]

    if subset_genes:
        missing = [g for g in subset_genes if g not in vocab_index]
        if missing:
            raise ValueError(f"Genes not in scGPT vocab: {missing[:5]}...")
        indices = [vocab_index[g] for g in subset_genes]
        values = embedding_matrix[indices, :].T
        labels = subset_genes
    else:
        values = embedding_matrix.T
        labels = vocab_list

    dim_labels = [f"dim_{i+1}" for i in range(values.shape[0])]
    return EmbeddingResult(
        values=values,
        item_labels=labels,
        dim_labels=dim_labels,
        metadata={
            "checkpoint_dir": str(checkpoint_dir),
            "n_genes": len(labels),
            "embedding_dim": values.shape[0],
            "state_dict_key": weight_key,
        },
    )

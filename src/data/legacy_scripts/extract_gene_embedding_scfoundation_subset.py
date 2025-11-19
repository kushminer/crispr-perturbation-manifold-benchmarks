#!/usr/bin/env python3
"""
Subset extraction of scFoundation gene embeddings for parity validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

parser = argparse.ArgumentParser(description="Extract subset of scFoundation gene embeddings")
parser.add_argument("--checkpoint", required=True, help="Path to scFoundation models.ckpt")
parser.add_argument("--demo_h5ad", required=True, help="Path to scFoundation demo.h5ad")
parser.add_argument("--gene_list", required=True, help="Path to subset gene list (one gene per line)")
parser.add_argument("--output_tsv", required=True, help="Output TSV path")
args = parser.parse_args()

checkpoint_path = Path(args.checkpoint).expanduser().resolve()
demo_path = Path(args.demo_h5ad).expanduser().resolve()
gene_list_path = Path(args.gene_list).expanduser().resolve()
output_path = Path(args.output_tsv).expanduser().resolve()

if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
if not demo_path.exists():
    raise FileNotFoundError(f"demo.h5ad not found: {demo_path}")
if not gene_list_path.exists():
    raise FileNotFoundError(f"Gene list not found: {gene_list_path}")

ckpt = torch.load(checkpoint_path, map_location="cpu")
try:
    pos_emb = ckpt["gene"]["state_dict"]["model.pos_emb.weight"].cpu().numpy()
except KeyError as exc:
    raise KeyError("Could not locate gene positional embeddings in checkpoint") from exc

demo = ad.read_h5ad(demo_path)
gene_names = demo.var["gene_name"].astype(str).tolist()
gene_names = gene_names + ["log10TotalCount1", "log10TotalCount2", "<pad>"]
name_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}

subset_genes = [
    line.strip()
    for line in gene_list_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
missing = [gene for gene in subset_genes if gene not in name_to_idx]
if missing:
    raise ValueError(f"Subset genes missing from demo list: {missing[:5]}...")

indices = [name_to_idx[gene] for gene in subset_genes]
subset_embeddings = pos_emb[indices, :].T  # dims x genes

df = pd.DataFrame(subset_embeddings, columns=subset_genes)
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, sep="\t", index=False)



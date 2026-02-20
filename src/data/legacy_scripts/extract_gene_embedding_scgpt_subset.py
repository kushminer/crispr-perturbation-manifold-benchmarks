#!/usr/bin/env python3
"""
Subset version of extract_gene_embedding_scgpt.py for parity validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

parser = argparse.ArgumentParser(description="Extract subset of scGPT gene embeddings")
parser.add_argument("--model_dir", type=str, required=True, help="Path to scgpt_human checkpoint directory")
parser.add_argument("--gene_list", type=str, required=True, help="Path to text file listing subset genes (one per line)")
parser.add_argument("--output_tsv", type=str, required=True, help="Where to write the legacy-format TSV output")
args = parser.parse_args()

model_dir = Path(args.model_dir).expanduser().resolve()
model_path = model_dir / "best_model.pt"
vocab_path = model_dir / "vocab.json"

if not model_path.exists():
    raise FileNotFoundError(f"Missing scGPT checkpoint: {model_path}")
if not vocab_path.exists():
    raise FileNotFoundError(f"Missing vocab: {vocab_path}")

with vocab_path.open("r", encoding="utf-8") as fh:
    vocab_json = json.load(fh)

gene_to_idx = {}
if isinstance(vocab_json, dict):
    if "itos" in vocab_json:
        vocab_list = vocab_json["itos"]
        gene_to_idx = {gene: idx for idx, gene in enumerate(vocab_list)}
    elif all(isinstance(v, int) for v in vocab_json.values()):
        gene_to_idx = {gene: int(idx) for gene, idx in vocab_json.items()}
    elif "stoi" in vocab_json:
        stoi = vocab_json["stoi"]
        gene_to_idx = {gene: int(idx) for gene, idx in stoi.items()}
elif isinstance(vocab_json, list):
    gene_to_idx = {gene: idx for idx, gene in enumerate(vocab_json)}

if not gene_to_idx:
    raise ValueError("Unexpected vocab format")

checkpoint = torch.load(model_path, map_location="cpu")
weight_key = None
for key in checkpoint:
    if key.endswith("encoder.embedding.weight") or key.endswith("encoder.weight"):
        weight_key = key
        break
if weight_key is None:
    raise KeyError("Could not locate encoder embedding weights in checkpoint.")

embedding_matrix = checkpoint[weight_key].cpu().numpy()

with open(args.gene_list, "r", encoding="utf-8") as fh:
    subset_genes = [line.strip() for line in fh if line.strip()]

missing = [g for g in subset_genes if g not in gene_to_idx]
if missing:
    raise ValueError(f"Subset genes missing from vocab: {missing[:5]}...")

indices = [gene_to_idx[g] for g in subset_genes]
subset_embeddings = embedding_matrix[indices, :].T  # dim x genes

df = pd.DataFrame(subset_embeddings, columns=subset_genes)
Path(args.output_tsv).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.output_tsv, sep="\t", index=False)



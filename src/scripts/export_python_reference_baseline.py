#!/usr/bin/env python3
"""Export synthetic baseline outputs for Python reference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPR_PATH = Path("reference_data/synthetic_baseline/expression.csv")
OUTPUT_ROOT = Path("reference_outputs/python")
BASELINE_NAME = "selftrained_synthetic"
RIDGE = 0.01
PCA_DIM = 2


def solve_y_axb(Y: np.ndarray, A: np.ndarray, B: np.ndarray, ridge: float):
    center = Y.mean(axis=1, keepdims=True)
    Yc = Y - center
    AtA = A.T @ A + ridge * np.eye(A.shape[1])
    BBt = B @ B.T + ridge * np.eye(B.shape[0])
    tmp = np.linalg.solve(AtA, A.T @ Yc @ B.T)
    K = np.linalg.solve(BBt, tmp.T).T
    return K, center[:, 0]


def main() -> None:
    expr = pd.read_csv(EXPR_PATH, index_col=0)
    expr_mat = expr.to_numpy(dtype=float)
    pert_names = expr.index.tolist()
    gene_names = expr.columns.tolist()

    U, s, Vt = np.linalg.svd(expr_mat.T, full_matrices=True)
    gene_emb = U[:, :PCA_DIM]
    pert_emb = Vt[:PCA_DIM, :]

    Y = expr_mat.T
    K, center = solve_y_axb(Y, gene_emb, pert_emb, RIDGE)
    pred = gene_emb @ K @ pert_emb + center[:, None]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for sub in ("predictions", "embeddings", "coefficients"):
        (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

    pred_dict = {pert_names[i]: pred[:, i].tolist() for i in range(len(pert_names))}
    with open(OUTPUT_ROOT / "predictions" / f"{BASELINE_NAME}.json", "w", encoding="utf-8") as fh:
        json.dump(pred_dict, fh, indent=2)

    gene_df = pd.DataFrame(gene_emb, columns=[f"PC{i+1}" for i in range(PCA_DIM)])
    gene_df.insert(0, "gene", gene_names)
    gene_df.to_csv(OUTPUT_ROOT / "embeddings" / f"{BASELINE_NAME}.tsv", sep="\t", index=False)

    pert_df = pd.DataFrame(pert_emb.T, columns=[f"PC{i+1}" for i in range(PCA_DIM)])
    pert_df.insert(0, "perturbation", pert_names)
    pert_df.to_csv(OUTPUT_ROOT / "embeddings" / f"{BASELINE_NAME}_pert.tsv", sep="\t", index=False)

    pd.DataFrame(K).to_csv(
        OUTPUT_ROOT / "coefficients" / f"{BASELINE_NAME}.tsv",
        sep="\t",
        index=False,
        header=False,
    )

    print("Python reference baseline exported to", OUTPUT_ROOT)


if __name__ == "__main__":
    main()

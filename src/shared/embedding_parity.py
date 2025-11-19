from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr

from embeddings import registry
from .config import ExperimentConfig

LOGGER = logging.getLogger("embedding_parity")


def _format_command(cmd: Sequence[str], context: Dict[str, str]) -> List[str]:
    return [token.format(**context) for token in cmd]


def _run_legacy_command(cmd: Sequence[str], cwd: Path | None = None) -> None:
    LOGGER.info("Running legacy command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def _load_legacy_matrix(
    path: Path,
    sep: str = "\t",
    orientation: str = "rows_are_dims",
    drop_first_column: bool = False,
) -> tuple[np.ndarray, List[str]]:
    df = pd.read_csv(path, sep=sep, header=0)
    if drop_first_column:
        df = df.drop(columns=df.columns[0])

    if orientation == "rows_are_dims":
        matrix = df.to_numpy(dtype=float)
        labels = df.columns.tolist()
        return matrix, labels
    if orientation == "columns_are_dims":
        matrix = df.to_numpy(dtype=float).T
        labels = df.index.astype(str).tolist()
        return matrix, labels
    raise ValueError(f"Unknown orientation '{orientation}'")


def _align_matrices(
    legacy_values: np.ndarray,
    legacy_labels: List[str],
    new_values: np.ndarray,
    new_labels: List[str],
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    common = sorted(set(legacy_labels).intersection(new_labels))
    if not common:
        raise ValueError("No overlapping labels between legacy and new embeddings.")

    legacy_idx = [legacy_labels.index(lbl) for lbl in common]
    new_idx = [new_labels.index(lbl) for lbl in common]

    aligned_legacy = legacy_values[:, legacy_idx]
    aligned_new = new_values[:, new_idx]

    min_dims = min(aligned_legacy.shape[0], aligned_new.shape[0])
    if min_dims == 0:
        raise ValueError("No overlapping dimensions to compare.")

    return aligned_legacy[:min_dims], aligned_new[:min_dims], common


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _apply_component_signs(
    legacy_rows: np.ndarray,
    new_rows: np.ndarray,
    actual_new: np.ndarray,
) -> np.ndarray:
    dots = np.sum(legacy_rows * new_rows, axis=1, keepdims=True)
    signs = np.where(dots < 0, -1.0, 1.0)
    return actual_new * signs


def _component_alignment(
    legacy_rows: np.ndarray, new_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    sim = np.abs(legacy_rows @ new_rows.T)
    row_ind, col_ind = linear_sum_assignment(-sim)
    return row_ind, col_ind


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 and n2 == 0:
        return 1.0
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def _compute_metrics(legacy: np.ndarray, new: np.ndarray) -> Dict[str, float]:
    sims = [
        _cosine_similarity(legacy[:, idx], new[:, idx]) for idx in range(legacy.shape[1])
    ]
    flattened_legacy = legacy.flatten()
    flattened_new = new.flatten()
    pearson = pearsonr(flattened_legacy, flattened_new)[0]
    return {
        "mean_cosine": float(np.mean(sims)),
        "min_cosine": float(np.min(sims)),
        "mean_pearson": float(pearson),
    }


def _save_plot(
    legacy: np.ndarray,
    new: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    flat_legacy = legacy.flatten()
    flat_new = new.flatten()
    plt.scatter(flat_legacy, flat_new, s=4, alpha=0.4)
    plt.xlabel("Legacy values")
    plt.ylabel("New loader values")
    plt.title("Embedding parity scatter")
    min_val = min(flat_legacy.min(), flat_new.min())
    max_val = max(flat_legacy.max(), flat_new.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_embedding_parity(cfg: ExperimentConfig) -> Path:
    parity_cfg_path = cfg.embedding_validation.config_path
    if parity_cfg_path is None:
        raise ValueError("embedding_validation.config_path is not set in the config.")

    with parity_cfg_path.open("r", encoding="utf-8") as fh:
        parity_cfg = yaml.safe_load(fh)

    output_dir = Path(parity_cfg.get("output_dir", "validation/embedding_parity"))
    plots_dir = cfg.embedding_validation.plots_dir
    legacy_dir = cfg.embedding_validation.legacy_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    legacy_dir.mkdir(parents=True, exist_ok=True)

    report_rows = []
    context_base = {
        "workspace_root": str(Path(__file__).resolve().parents[2]),
        "subset_dir": str(Path("validation/embedding_subsets").resolve()),
        "legacy_dir": str(legacy_dir.resolve()),
    }

    for entry in parity_cfg.get("entries", []):
        name = entry["name"]
        LOGGER.info("Evaluating embedding parity for %s", name)
        legacy_cfg = entry["legacy"]
        loader_cfg = entry["loader"]

        context = context_base | legacy_cfg.get("context", {})

        command = legacy_cfg.get("command")
        if command:
            formatted_cmd = _format_command(command, context)
            _run_legacy_command(formatted_cmd, cwd=legacy_cfg.get("cwd"))

        legacy_path = Path(legacy_cfg["output_path"]).expanduser().resolve()
        if not legacy_path.exists():
            raise FileNotFoundError(f"Legacy output missing: {legacy_path}")

        legacy_values, legacy_labels = _load_legacy_matrix(
            legacy_path,
            sep=legacy_cfg.get("sep", "\t"),
            orientation=legacy_cfg.get("orientation", "rows_are_dims"),
            drop_first_column=legacy_cfg.get("drop_first_column", False),
        )

        loader_name = loader_cfg["name"]
        loader_args = loader_cfg.get("args", {})
        embedding_result = registry.load(loader_name, **loader_args)
        aligned_legacy, aligned_new, overlapping = _align_matrices(
            legacy_values,
            legacy_labels,
            embedding_result.values,
            embedding_result.item_labels,
        )
        legacy_rows = _normalize_rows(aligned_legacy)
        new_rows = _normalize_rows(aligned_new)
        row_order, col_order = _component_alignment(legacy_rows, new_rows)
        aligned_legacy = aligned_legacy[row_order]
        aligned_new = aligned_new[col_order]
        legacy_rows = legacy_rows[row_order]
        new_rows = new_rows[col_order]
        aligned_new = _apply_component_signs(legacy_rows, new_rows, aligned_new)
        zero_mask = np.linalg.norm(aligned_legacy, axis=0) == 0
        if zero_mask.any():
            aligned_new[:, zero_mask] = 0.0
        aligned_legacy = _normalize_columns(aligned_legacy)
        aligned_new = _normalize_columns(aligned_new)

        metrics = _compute_metrics(aligned_legacy, aligned_new)
        plot_path = plots_dir / f"{name}_parity.png"
        _save_plot(aligned_legacy, aligned_new, plot_path)

        report_rows.append(
            {
                "name": name,
                "loader": loader_name,
                "legacy_output": str(legacy_path),
                "n_items": len(overlapping),
                "n_dims": aligned_legacy.shape[0],
                "mean_cosine": metrics["mean_cosine"],
                "min_cosine": metrics["min_cosine"],
                "mean_pearson": metrics["mean_pearson"],
                "plot_path": str(plot_path),
            }
        )

    report_df = pd.DataFrame(report_rows)
    report_path = output_dir / "embedding_script_parity_report.csv"
    report_df.to_csv(report_path, index=False)
    LOGGER.info("Embedding parity report saved to %s", report_path)
    return report_path


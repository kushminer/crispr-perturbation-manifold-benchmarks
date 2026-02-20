#!/bin/bash
#
# Run single-cell baseline evaluation for Adamson and K562 datasets.
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Canonical paths (override via env vars if needed)
ADAMSON_DATA_PATH="${ADAMSON_DATA_PATH:-data/gears_pert_data/adamson/perturb_processed.h5ad}"
K562_DATA_PATH="${K562_DATA_PATH:-data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad}"

RESULTS_DIR="$REPO_ROOT/results/single_cell_analysis"
mkdir -p "$RESULTS_DIR"

if [[ ! -f "$ADAMSON_DATA_PATH" ]]; then
  echo "Missing Adamson data: $ADAMSON_DATA_PATH"
  exit 1
fi
if [[ ! -f "$K562_DATA_PATH" ]]; then
  echo "Missing K562 data: $K562_DATA_PATH"
  exit 1
fi

echo "=============================================="
echo "SINGLE-CELL BASELINE EVALUATION"
echo "=============================================="
echo "Adamson data: $ADAMSON_DATA_PATH"
echo "K562 data: $K562_DATA_PATH"

echo ""

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export ADAMSON_DATA_PATH
export K562_DATA_PATH

"${PYTHON_BIN}" <<'PY'
import json
import logging
import os
from pathlib import Path

import anndata as ad
import numpy as np

from goal_2_baselines.baseline_runner_single_cell import run_all_baselines_single_cell
from goal_2_baselines.baseline_types import BaselineType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

datasets = {
    "adamson": Path(os.environ["ADAMSON_DATA_PATH"]),
    "k562": Path(os.environ["K562_DATA_PATH"]),
}

n_cells_per_pert = 50
pca_dim = 10
ridge_penalty = 0.1
seed = 1

results_dir = Path("results/single_cell_analysis")
results_dir.mkdir(parents=True, exist_ok=True)

baselines = [
    BaselineType.SELFTRAINED,
    BaselineType.RANDOM_GENE_EMB,
    BaselineType.RANDOM_PERT_EMB,
]

for dataset_name, adata_path in datasets.items():
    LOGGER.info("\n%s", "=" * 60)
    LOGGER.info("Dataset: %s", dataset_name)
    LOGGER.info("%s", "=" * 60)

    if not adata_path.exists():
        LOGGER.warning("Data file not found: %s", adata_path)
        continue

    split_config_path = results_dir / f"{dataset_name}_split_config.json"

    if not split_config_path.exists():
        LOGGER.info("Creating train/test split...")
        adata = ad.read_h5ad(adata_path)
        conditions = [c for c in adata.obs["condition"].unique() if c != "ctrl"]
        np.random.seed(seed)
        np.random.shuffle(conditions)

        n_train = int(0.8 * len(conditions))
        train_conditions = conditions[:n_train]
        test_conditions = conditions[n_train:]

        split_config = {
            "train": list(train_conditions) + ["ctrl"],
            "test": list(test_conditions),
        }
        with open(split_config_path, "w", encoding="utf-8") as handle:
            json.dump(split_config, handle, indent=2)

        LOGGER.info("Created split: %s train, %s test", len(train_conditions), len(test_conditions))
    else:
        LOGGER.info("Using existing split config: %s", split_config_path)

    output_dir = results_dir / dataset_name

    try:
        results_df = run_all_baselines_single_cell(
            adata_path=adata_path,
            split_config_path=split_config_path,
            output_dir=output_dir,
            baseline_types=baselines,
            pca_dim=pca_dim,
            ridge_penalty=ridge_penalty,
            seed=seed,
            n_cells_per_pert=n_cells_per_pert,
            cell_embedding_method="cell_pca",
        )
        LOGGER.info("\n%s Results:\n%s", dataset_name, results_df.to_string())
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to run %s: %s", dataset_name, exc)
        raise

LOGGER.info("\n%s", "=" * 60)
LOGGER.info("SINGLE-CELL BASELINE EVALUATION COMPLETE")
LOGGER.info("%s", "=" * 60)
PY

echo ""
echo "Done. Results saved to $RESULTS_DIR"

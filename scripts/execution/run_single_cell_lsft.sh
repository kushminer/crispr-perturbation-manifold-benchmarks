#!/bin/bash
#
# Run single-cell LSFT evaluation for Adamson and K562 datasets.
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

ADAMSON_DATA_PATH="${ADAMSON_DATA_PATH:-data/gears_pert_data/adamson/perturb_processed.h5ad}"
K562_DATA_PATH="${K562_DATA_PATH:-data/gears_pert_data/replogle_k562_essential/perturb_processed.h5ad}"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export ADAMSON_DATA_PATH
export K562_DATA_PATH

echo "=============================================="
echo "SINGLE-CELL LSFT EVALUATION"
echo "=============================================="

echo "Adamson data: $ADAMSON_DATA_PATH"
echo "K562 data: $K562_DATA_PATH"

"${PYTHON_BIN}" <<'PY'
import logging
import os
from pathlib import Path

from goal_2_baselines.baseline_types import BaselineType
from goal_3_prediction.lsft.lsft_single_cell import run_lsft_single_cell_all_baselines

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

results_dir = Path("results/single_cell_analysis")

for path_label, path_value in {
    "adamson": Path(os.environ["ADAMSON_DATA_PATH"]),
    "k562": Path(os.environ["K562_DATA_PATH"]),
}.items():
    if not path_value.exists():
        raise FileNotFoundError(f"Missing {path_label} data: {path_value}")

datasets = {
    "adamson": {
        "adata_path": Path(os.environ["ADAMSON_DATA_PATH"]),
        "split_config": results_dir / "adamson_split_config.json",
    },
    "k562": {
        "adata_path": Path(os.environ["K562_DATA_PATH"]),
        "split_config": results_dir / "k562_split_config.json",
    },
}

baselines = [
    BaselineType.SELFTRAINED,
    BaselineType.RANDOM_GENE_EMB,
]

top_pcts = [0.05, 0.10]

for dataset_name, config in datasets.items():
    LOGGER.info("\n%s", "=" * 60)
    LOGGER.info("Dataset: %s", dataset_name)
    LOGGER.info("%s", "=" * 60)

    output_dir = results_dir / dataset_name / "lsft"

    run_lsft_single_cell_all_baselines(
        adata_path=config["adata_path"],
        split_config_path=config["split_config"],
        output_dir=output_dir,
        dataset_name=dataset_name,
        baseline_types=baselines,
        top_pcts=top_pcts,
        pca_dim=10,
        ridge_penalty=0.1,
        seed=1,
        n_cells_per_pert=50,
    )

    LOGGER.info("%s LSFT results saved to %s", dataset_name, output_dir)

LOGGER.info("\n%s", "=" * 60)
LOGGER.info("SINGLE-CELL LSFT EVALUATION COMPLETE")
LOGGER.info("%s", "=" * 60)
PY

echo "Done"

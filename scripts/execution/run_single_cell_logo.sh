#!/bin/bash
#
# Run single-cell LOGO evaluation for Adamson dataset.
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

ADAMSON_DATA_PATH="${ADAMSON_DATA_PATH:-data/gears_pert_data/adamson/perturb_processed.h5ad}"
ADAMSON_ANNOTATION_PATH="${ADAMSON_ANNOTATION_PATH:-data/annotations/adamson_functional_classes_enriched.tsv}"

if [[ ! -f "$ADAMSON_DATA_PATH" ]]; then
  echo "Missing Adamson data: $ADAMSON_DATA_PATH"
  exit 1
fi
if [[ ! -f "$ADAMSON_ANNOTATION_PATH" ]]; then
  echo "Missing Adamson annotation file: $ADAMSON_ANNOTATION_PATH"
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export ADAMSON_DATA_PATH
export ADAMSON_ANNOTATION_PATH

echo "=============================================="
echo "SINGLE-CELL LOGO EVALUATION"
echo "=============================================="

echo "Adamson data: $ADAMSON_DATA_PATH"
echo "Adamson annotations: $ADAMSON_ANNOTATION_PATH"

"${PYTHON_BIN}" <<'PY'
import logging
import os
from pathlib import Path

from goal_2_baselines.baseline_types import BaselineType
from goal_4_logo.logo_single_cell import run_logo_single_cell

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

results_dir = Path("results/single_cell_analysis")

baselines = [
    BaselineType.SELFTRAINED,
    BaselineType.RANDOM_GENE_EMB,
]

run_logo_single_cell(
    adata_path=Path(os.environ["ADAMSON_DATA_PATH"]),
    annotation_path=Path(os.environ["ADAMSON_ANNOTATION_PATH"]),
    dataset_name="adamson",
    output_dir=results_dir / "adamson" / "logo",
    class_name="Transcription",
    baseline_types=baselines,
    pca_dim=10,
    ridge_penalty=0.1,
    seed=1,
    n_cells_per_pert=50,
)

LOGGER.info("LOGO results saved to %s", results_dir / "adamson" / "logo")
PY

echo "Done"

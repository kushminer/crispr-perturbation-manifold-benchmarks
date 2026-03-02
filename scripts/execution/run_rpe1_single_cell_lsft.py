#!/usr/bin/env python3
"""
Run complete single-cell LSFT evaluation for RPE1 dataset.
This will generate all missing RPE1 single-cell LSFT results.
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
LOGGER = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from goal_3_prediction.lsft.lsft_single_cell import run_lsft_single_cell_all_baselines
from goal_2_baselines.baseline_types import BaselineType

project_root = Path(__file__).parent.parent.parent
results_dir = project_root / 'results/single_cell_analysis'

# RPE1 configuration
dataset_name = 'rpe1'
config = {
    'adata_path': 'data/gears_pert_data/replogle_rpe1_essential/perturb_processed.h5ad',
    'split_config': results_dir / 'rpe1_split_config.json',
}

# All baselines to run (skip SELFTRAINED as it's already complete)
all_baselines = [
    BaselineType.RANDOM_GENE_EMB,
    BaselineType.RANDOM_PERT_EMB,
    BaselineType.SCGPT_GENE_EMB,
    BaselineType.SCFOUNDATION_GENE_EMB,
    BaselineType.GEARS_PERT_EMB,
]

top_pcts = [0.01, 0.05, 0.10]  # Include all percentages

LOGGER.info('='*70)
LOGGER.info('RPE1 SINGLE-CELL LSFT EVALUATION')
LOGGER.info('='*70)
LOGGER.info('')
LOGGER.info('This will run LSFT for remaining 5 baselines:')
LOGGER.info('  (Skipping lpm_selftrained - already complete)')
for baseline in all_baselines:
    LOGGER.info(f'  - {baseline.value}')
LOGGER.info('')
LOGGER.info('Parameters:')
LOGGER.info(f'  - Top percentages: {top_pcts}')
LOGGER.info('  - PCA dimension: 10')
LOGGER.info('  - Ridge penalty: 0.1')
LOGGER.info('  - Seed: 1')
LOGGER.info('  - Cells per perturbation: 50')
LOGGER.info('')

output_dir = results_dir / dataset_name / 'lsft'
output_dir.mkdir(parents=True, exist_ok=True)

LOGGER.info(f'Output directory: {output_dir}')
LOGGER.info('')

try:
    results = run_lsft_single_cell_all_baselines(
        adata_path=Path(config['adata_path']),
        split_config_path=config['split_config'],
        output_dir=output_dir,
        dataset_name=dataset_name,
        baseline_types=all_baselines,
        top_pcts=top_pcts,
        pca_dim=10,
        ridge_penalty=0.1,
        seed=1,
        n_cells_per_pert=50,
    )
    
    LOGGER.info('')
    LOGGER.info('='*70)
    LOGGER.info('✅ RPE1 SINGLE-CELL LSFT EVALUATION COMPLETE')
    LOGGER.info('='*70)
    LOGGER.info('')
    LOGGER.info(f'Results saved to: {output_dir}')
    
    # List generated files
    summary_files = list(output_dir.glob(f'lsft_single_cell_summary_{dataset_name}_*.csv'))
    if summary_files:
        LOGGER.info('')
        LOGGER.info(f'Generated {len(summary_files)} summary file(s):')
        for summary_file in sorted(summary_files):
            LOGGER.info(f'  - {summary_file.name}')
    
    LOGGER.info('')
    LOGGER.info('Next steps:')
    LOGGER.info('  1. Verify all summary files were generated')
    LOGGER.info('  2. Regenerate Figure 3 to include RPE1 single-cell LSFT data')
    LOGGER.info('')
    
except Exception as e:
    LOGGER.error(f'\n❌ Failed to run LSFT for RPE1: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)


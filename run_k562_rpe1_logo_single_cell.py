#!/usr/bin/env python3
"""
Run single-cell LOGO evaluation for K562 and RPE1 datasets.
Includes all baselines, especially random pert which is missing.
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
LOGGER = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from goal_4_logo.logo_single_cell import run_logo_single_cell
from goal_2_baselines.baseline_types import BaselineType

results_dir = Path('results/single_cell_analysis')

datasets = {
    'k562': {
        'adata_path': '/Users/samuelminer/Documents/classes/nih_research/data_replogle_k562_essential/perturb_processed.h5ad',
        'annotation_path': 'data/annotations/replogle_k562_functional_classes_go.tsv',
    },
    'rpe1': {
        'adata_path': '/Users/samuelminer/Documents/classes/nih_research/data_replogle_rpe1_essential/perturb_processed.h5ad',
        'annotation_path': 'data/annotations/replogle_rpe1_functional_classes_go.tsv',
    },
}

# All baselines including random pert
all_baselines = [
    BaselineType.SELFTRAINED,
    BaselineType.RANDOM_GENE_EMB,
    BaselineType.RANDOM_PERT_EMB,  # This is the missing one!
    BaselineType.SCGPT_GENE_EMB,
    BaselineType.SCFOUNDATION_GENE_EMB,
    BaselineType.GEARS_PERT_EMB,
]

for dataset_name, config in datasets.items():
    LOGGER.info(f'\n{"="*60}')
    LOGGER.info(f'Dataset: {dataset_name}')
    LOGGER.info(f'{"="*60}')
    
    adata_path = Path(config['adata_path'])
    annotation_path = Path(config['annotation_path'])
    
    if not adata_path.exists():
        LOGGER.error(f'Data file not found: {adata_path}')
        continue
    
    if not annotation_path.exists():
        LOGGER.error(f'Annotation file not found: {annotation_path}')
        continue
    
    output_dir = results_dir / dataset_name / 'logo'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        LOGGER.info(f'Running LOGO for {dataset_name} with all baselines...')
        results = run_logo_single_cell(
            adata_path=adata_path,
            annotation_path=annotation_path,
            dataset_name=dataset_name,
            output_dir=output_dir,
            class_name='Transcription',
            baseline_types=all_baselines,
            pca_dim=10,
            ridge_penalty=0.1,
            seed=1,
            n_cells_per_pert=50,
        )
        
        if results is not None and not results.empty:
            LOGGER.info(f'\n{dataset_name} LOGO Results:')
            LOGGER.info(f'\n{results.to_string()}')
        
        LOGGER.info(f'\n{dataset_name} LOGO Results saved to {output_dir}')
        
    except Exception as e:
        LOGGER.error(f'Failed {dataset_name}: {e}')
        import traceback
        traceback.print_exc()

LOGGER.info('\n' + '='*60)
LOGGER.info('SINGLE-CELL LOGO EVALUATION COMPLETE')
LOGGER.info('='*60)


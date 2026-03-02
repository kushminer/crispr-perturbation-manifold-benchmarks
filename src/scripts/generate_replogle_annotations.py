#!/usr/bin/env python3
"""
Generate functional class annotations for Replogle K562 dataset.

This script extracts perturbation names from expression data and generates
functional class annotations using the annotation script.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from shared.config import load_config
from shared.io import load_expression_dataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("generate_replogle_annotations")


def _load_annotation_mapper(name: str):
    """Import annotation helpers for both direct-script and PYTHONPATH=src usage."""
    try:
        from scripts import annotate_classes as module
    except ImportError:
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))
        import annotate_classes as module  # type: ignore
    return getattr(module, name)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Replogle K562 functional class annotations"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_replogle.yaml",
        help="Path to Replogle config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output annotation file path (default: from config)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["manual", "go", "reactome"],
        default="manual",
        help="Annotation method (default: manual)",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=10,
        help="Number of classes for manual method (default: 10)",
    )
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(Path(args.config))
    
    # Load expression data to get perturbation list
    LOGGER.info("Loading expression data from %s", cfg.dataset.expression_path)
    try:
        expression = load_expression_dataset(
            cfg.dataset.expression_path,
            cfg.dataset.gene_names_path,
        )
        perturbations = expression.index.tolist()
        LOGGER.info("Found %d perturbations in expression data", len(perturbations))
    except FileNotFoundError as e:
        LOGGER.error("Could not load expression data: %s", e)
        LOGGER.error("Please ensure Replogle K562 predictions have been generated first.")
        LOGGER.error("Expected path: %s", cfg.dataset.expression_path)
        return 1
    
    # Determine output path
    output_path = Path(args.output) if args.output else cfg.dataset.annotation_path
    if output_path is None:
        output_path = Path("data/annotations/replogle_k562_functional_classes.tsv")
    
    LOGGER.info("Generating annotations using method: %s", args.method)
    
    if args.method == "manual":
        # For Replogle, we'll use a simpler approach: create balanced classes
        # This is a fallback when GO/Reactome data is not available
        LOGGER.info("Creating balanced functional classes (synthetic)...")
        LOGGER.warning(
            "Using synthetic classes. For real biological annotations, use --method go or --method reactome."
        )
        
        from functional_class.test_utils import generate_synthetic_class_annotations
        
        # Use synthetic generation for balanced classes
        annotations_df = generate_synthetic_class_annotations(
            perturbation_names=perturbations,
            n_classes=args.n_classes,
            min_class_size=5,
            seed=cfg.dataset.seed,
        )
        
        # Rename classes to be more meaningful
        class_mapping = {
            f"Class_{i+1}": f"Functional_Group_{i+1}"
            for i in range(args.n_classes)
        }
        annotations_df["class"] = annotations_df["class"].map(class_mapping)
    
    elif args.method == "go":
        LOGGER.info("Using GO term mapping...")
        map_genes_to_classes_go = _load_annotation_mapper("map_genes_to_classes_go")
        
        # Extract gene symbols from perturbation names
        # Perturbation names might be gene symbols or have suffixes like "+ctrl"
        gene_symbols = []
        for pert in perturbations:
            # Remove common suffixes
            gene = pert.replace("+ctrl", "").strip()
            gene_symbols.append(gene)
        
        annotations_df = map_genes_to_classes_go(
            genes=gene_symbols,
            organism="human",
            min_class_size=5,
        )
    
    elif args.method == "reactome":
        LOGGER.info("Using Reactome pathway mapping...")
        map_genes_to_classes_reactome = _load_annotation_mapper("map_genes_to_classes_reactome")
        
        # Extract gene symbols from perturbation names
        gene_symbols = []
        for pert in perturbations:
            gene = pert.replace("+ctrl", "").strip()
            gene_symbols.append(gene)
        
        annotations_df = map_genes_to_classes_reactome(
            genes=gene_symbols,
            min_class_size=5,
        )
    
    # Save annotations
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotations_df.to_csv(output_path, sep="\t", index=False)
    
    LOGGER.info("Saved annotations to %s", output_path)
    LOGGER.info("Total: %d perturbations, %d classes", 
                len(annotations_df), annotations_df["class"].nunique())
    
    # Show class distribution
    class_counts = annotations_df["class"].value_counts()
    LOGGER.info("Class distribution:")
    for cls, count in class_counts.items():
        LOGGER.info("  %s: %d perturbations", cls, count)
    
    LOGGER.info("✅ Annotation generation complete!")
    LOGGER.info("Next step: Run evaluation with --config %s --task class", args.config)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

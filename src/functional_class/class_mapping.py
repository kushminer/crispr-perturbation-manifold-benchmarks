"""
Cross-dataset class mapping utilities.

This module provides functions to map functional classes across different datasets,
enabling comparison and alignment of annotations between Adamson and Replogle K562.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_replogle_classes(annotation_path: Path) -> Dict[str, List[str]]:
    """
    Load Replogle K562 functional classes and return as a mapping.
    
    Args:
        annotation_path: Path to Replogle annotation TSV file
        
    Returns:
        Dictionary mapping class names to lists of gene symbols
    """
    df = pd.read_csv(annotation_path, sep="\t")
    mapping = {}
    for _, row in df.iterrows():
        cls = row["class"]
        target = row["target"]
        mapping.setdefault(cls, []).append(target)
    return mapping


def map_adamson_to_replogle_classes(
    adamson_genes: List[str],
    replogle_annotation_path: Path,
    min_overlap: int = 1,
) -> pd.DataFrame:
    """
    Map Adamson genes to Replogle K562 functional classes via symbol overlap.
    
    For each Adamson gene, finds the Replogle class with the most overlapping
    members. If no overlap is found, assigns to "Other".
    
    Args:
        adamson_genes: List of Adamson gene symbols (perturbation targets)
        replogle_annotation_path: Path to Replogle annotation TSV
        min_overlap: Minimum number of overlapping genes required for mapping
        
    Returns:
        DataFrame with columns: target, class
    """
    replogle_classes = load_replogle_classes(replogle_annotation_path)
    
    # Create reverse mapping: gene -> classes it belongs to
    gene_to_classes: Dict[str, List[str]] = {}
    for cls, genes in replogle_classes.items():
        for gene in genes:
            gene_to_classes.setdefault(gene, []).append(cls)
    
    results = []
    unmapped = []
    
    for adamson_gene in adamson_genes:
        # Find classes containing this gene
        matching_classes = gene_to_classes.get(adamson_gene, [])
        
        if matching_classes:
            # If gene appears in multiple classes, use the first one
            # (could be enhanced to use class with most overlap)
            assigned_class = matching_classes[0]
            results.append({"target": adamson_gene, "class": assigned_class})
        else:
            unmapped.append(adamson_gene)
    
    # Assign unmapped genes to "Other"
    for gene in unmapped:
        results.append({"target": gene, "class": "Other"})
    
    df = pd.DataFrame(results)
    
    LOGGER.info(
        "Mapped %d Adamson genes to Replogle classes: %d mapped, %d assigned to 'Other'",
        len(adamson_genes), len(adamson_genes) - len(unmapped), len(unmapped)
    )
    
    # Log class distribution
    class_counts = df["class"].value_counts()
    LOGGER.info("Class distribution:")
    for cls, count in class_counts.items():
        LOGGER.info("  %s: %d genes", cls, count)
    
    return df


def create_canonical_class_mapping(
    replogle_classes: Dict[str, List[str]],
    canonical_modules: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, str]:
    """
    Map Replogle K562 classes to canonical functional modules.
    
    Args:
        replogle_classes: Dictionary of Replogle class names to gene lists
        canonical_modules: Optional dictionary of canonical module names to gene lists.
                          If None, uses built-in mapping.
    
    Returns:
        Dictionary mapping Replogle class names to canonical module names
    """
    # Built-in canonical modules (can be extended)
    if canonical_modules is None:
        canonical_modules = {
            "Protein_Folding": ["HSP90", "HSP70", "DNAJB", "HSPA"],
            "ER_Stress": ["XBP1", "ATF6", "IRE1", "ERN1"],
            "Translation": ["EIF", "RPS", "RPL"],
            "Transcription": ["POLR", "TBP", "GTF"],
            "Metabolism": ["GAPDH", "PKM", "LDHA"],
        }
    
    mapping = {}
    
    for replogle_class, genes in replogle_classes.items():
        best_match = None
        best_score = 0
        
        for canonical_module, module_genes in canonical_modules.items():
            # Simple overlap scoring (could be enhanced)
            overlap = len(set(genes) & set(module_genes))
            if overlap > best_score:
                best_score = overlap
                best_match = canonical_module
        
        if best_match and best_score > 0:
            mapping[replogle_class] = best_match
        else:
            mapping[replogle_class] = "Other"
    
    return mapping


def main():
    """CLI entry point for cross-dataset class mapping."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Map functional classes across datasets"
    )
    parser.add_argument(
        "--adamson-genes",
        type=Path,
        required=True,
        help="Path to Adamson gene list (one per line) or TSV with gene column",
    )
    parser.add_argument(
        "--replogle-annotations",
        type=Path,
        required=True,
        help="Path to Replogle K562 annotation TSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output TSV file path",
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=1,
        help="Minimum gene overlap for mapping (default: 1)",
    )
    
    args = parser.parse_args()
    
    # Load Adamson genes
    if args.adamson_genes.suffix == ".tsv":
        df_genes = pd.read_csv(args.adamson_genes, sep="\t")
        if "gene" in df_genes.columns:
            adamson_genes = df_genes["gene"].tolist()
        elif "target" in df_genes.columns:
            adamson_genes = df_genes["target"].tolist()
        else:
            raise ValueError("TSV must have 'gene' or 'target' column")
    else:
        with open(args.adamson_genes) as f:
            adamson_genes = [line.strip() for line in f if line.strip()]
    
    # Map to Replogle classes
    df_mapped = map_adamson_to_replogle_classes(
        adamson_genes=adamson_genes,
        replogle_annotation_path=args.replogle_annotations,
        min_overlap=args.min_overlap,
    )
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_mapped.to_csv(args.output, sep="\t", index=False)
    LOGGER.info("Saved mapped annotations to %s", args.output)
    
    return 0


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    sys.exit(main())


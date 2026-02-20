#!/usr/bin/env python3
"""
Validation Framework for Functional Class Annotations.

This script provides methods to validate annotation quality and ensure
appropriate class assignments with certainty.

Validation strategies:
1. GO term consistency check
2. Expression similarity validation
3. Cross-dataset consistency
4. Literature/domain knowledge validation
5. Statistical validation
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from shared.io import load_annotations


def validate_go_consistency(
    annotation_path: Path,
    gene_symbols: List[str],
    organism: str = "human"
) -> Dict[str, List[str]]:
    """
    Validate that gene assignments are consistent with GO terms.
    
    For each gene, check:
    1. Does it have GO annotations?
    2. Are GO terms consistent with assigned functional class?
    3. Are there conflicting GO terms?
    
    Returns:
        Dict mapping gene -> list of validation issues
    """
    print("=" * 70)
    print("GO CONSISTENCY VALIDATION")
    print("=" * 70)
    print()
    
    annotations = load_annotations(annotation_path)
    
    # Map functional classes to expected GO terms
    # This is a simplified mapping - should be expanded
    class_to_go_terms = {
        "Transcription": [
            "GO:0006351",  # DNA-templated transcription
            "GO:0006355",  # regulation of transcription
            "GO:0003700",  # DNA-binding transcription factor
        ],
        "Translation": [
            "GO:0006412",  # translation
            "GO:0003743",  # translation initiation factor
            "GO:0003746",  # translation elongation factor
        ],
        "Metabolism": [
            "GO:0008152",  # metabolic process
            "GO:0044237",  # cellular metabolic process
        ],
        # Add more mappings as needed
    }
    
    validation_results = {}
    
    print("⚠️  GO consistency validation requires GO API or database")
    print("   This is a framework - implement with actual GO queries")
    print()
    
    return validation_results


def validate_expression_similarity(
    annotation_path: Path,
    adata_path: Path,
    class_name: str
) -> pd.DataFrame:
    """
    Validate class assignments using expression similarity.
    
    Hypothesis: Genes in the same functional class should have
    similar expression patterns across perturbations.
    
    For each class:
    1. Compute mean expression profile for class
    2. Compute correlation of each gene with class mean
    3. Flag genes with low correlation (potential misclassification)
    """
    print("=" * 70)
    print(f"EXPRESSION SIMILARITY VALIDATION: {class_name}")
    print("=" * 70)
    print()
    
    try:
        import anndata as ad
        adata = ad.read_h5ad(adata_path)
    except Exception as e:
        print(f"⚠️  Could not load data: {e}")
        return pd.DataFrame()
    
    annotations = load_annotations(annotation_path)
    
    # Get genes in this class
    class_genes = annotations[annotations["class"] == class_name]["target"].tolist()
    
    if len(class_genes) < 2:
        print(f"⚠️  Too few genes in class '{class_name}' for validation")
        return pd.DataFrame()
    
    print(f"Validating {len(class_genes)} genes in class '{class_name}'")
    print()
    
    # Compute pseudobulk expression changes
    # (Simplified - would use actual perturbation data)
    print("⚠️  Expression similarity validation requires:")
    print("   1. Compute perturbation expression profiles")
    print("   2. Compute class mean profile")
    print("   3. Correlate each gene with class mean")
    print("   4. Flag outliers (low correlation)")
    
    return pd.DataFrame()


def validate_cross_dataset_consistency(
    annotation_paths: Dict[str, Path],
    common_genes: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Validate consistency across datasets.
    
    For genes present in multiple datasets:
    1. Check if they have consistent class assignments
    2. Flag inconsistencies
    3. Suggest most common assignment
    """
    print("=" * 70)
    print("CROSS-DATASET CONSISTENCY VALIDATION")
    print("=" * 70)
    print()
    
    all_annotations = {}
    for dataset_name, path in annotation_paths.items():
        if path.exists():
            all_annotations[dataset_name] = load_annotations(path)
            print(f"Loaded {dataset_name}: {len(all_annotations[dataset_name])} annotations")
    
    if len(all_annotations) < 2:
        print("⚠️  Need at least 2 datasets for cross-dataset validation")
        return pd.DataFrame()
    
    # Find common genes
    if common_genes is None:
        all_genes = set()
        for ann in all_annotations.values():
            all_genes.update(ann["target"].unique())
        common_genes = list(all_genes)
    
    print(f"\nChecking consistency for {len(common_genes)} genes...")
    print()
    
    inconsistencies = []
    
    for gene in common_genes:
        assignments = {}
        for dataset_name, ann in all_annotations.items():
            gene_ann = ann[ann["target"] == gene]
            if len(gene_ann) > 0:
                assignments[dataset_name] = gene_ann["class"].values[0]
        
        if len(set(assignments.values())) > 1:
            # Inconsistency found
            inconsistencies.append({
                "gene": gene,
                "assignments": assignments,
                "most_common": max(set(assignments.values()), key=list(assignments.values()).count),
            })
    
    if inconsistencies:
        print(f"Found {len(inconsistencies)} inconsistent assignments:")
        for inc in inconsistencies[:10]:  # Show first 10
            print(f"  {inc['gene']}: {inc['assignments']} → suggest {inc['most_common']}")
        if len(inconsistencies) > 10:
            print(f"  ... and {len(inconsistencies) - 10} more")
    else:
        print("✓ No inconsistencies found")
    
    return pd.DataFrame(inconsistencies)


def validate_statistical_properties(
    annotation_path: Path
) -> Dict[str, float]:
    """
    Validate statistical properties of annotations.
    
    Checks:
    1. Class size distribution (are classes balanced?)
    2. Gene coverage (are all genes annotated?)
    3. Class diversity (are there too many/few classes?)
    """
    print("=" * 70)
    print("STATISTICAL VALIDATION")
    print("=" * 70)
    print()
    
    annotations = load_annotations(annotation_path)
    
    class_counts = annotations["class"].value_counts()
    
    stats = {
        "total_genes": len(annotations),
        "num_classes": len(class_counts),
        "mean_class_size": class_counts.mean(),
        "std_class_size": class_counts.std(),
        "min_class_size": class_counts.min(),
        "max_class_size": class_counts.max(),
        "other_pct": (class_counts.get("Other", 0) / len(annotations) * 100) if "Other" in class_counts.index else 0,
    }
    
    print(f"Total genes: {stats['total_genes']}")
    print(f"Number of classes: {stats['num_classes']}")
    print(f"Mean class size: {stats['mean_class_size']:.1f}")
    print(f"Class size range: {stats['min_class_size']} - {stats['max_class_size']}")
    print(f"'Other' class: {stats['other_pct']:.1f}%")
    print()
    
    # Validation checks
    issues = []
    
    if stats['other_pct'] > 30:
        issues.append(f"'Other' class is too large ({stats['other_pct']:.1f}%)")
    
    if stats['max_class_size'] / stats['min_class_size'] > 10:
        issues.append(f"High class imbalance ({stats['max_class_size'] / stats['min_class_size']:.1f}x)")
    
    if stats['min_class_size'] < 3:
        issues.append(f"Some classes are too small (min: {stats['min_class_size']})")
    
    if issues:
        print("⚠️  Validation Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ Statistical properties look good")
    
    print()
    
    return stats


def create_validation_report(
    annotation_path: Path,
    dataset_name: str,
    output_dir: Path
):
    """Create comprehensive validation report."""
    
    print("=" * 70)
    print(f"VALIDATION REPORT: {dataset_name.upper()}")
    print("=" * 70)
    print()
    
    # Run all validations
    stats = validate_statistical_properties(annotation_path)
    
    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / f"validation_report_{dataset_name}.md"
    
    with open(report_file, "w") as f:
        f.write(f"# Validation Report: {dataset_name}\n\n")
        f.write("## Statistical Properties\n\n")
        f.write(f"- Total genes: {stats['total_genes']}\n")
        f.write(f"- Number of classes: {stats['num_classes']}\n")
        f.write(f"- Mean class size: {stats['mean_class_size']:.1f}\n")
        f.write(f"- 'Other' class: {stats['other_pct']:.1f}%\n\n")
        f.write("## Validation Status\n\n")
        f.write("⚠️  Full validation requires:\n")
        f.write("1. GO term consistency check\n")
        f.write("2. Expression similarity validation\n")
        f.write("3. Cross-dataset consistency\n")
        f.write("4. Literature validation\n")
    
    print(f"Report saved to: {report_file}")


def main():
    """Run validation for all datasets."""
    
    datasets = {
        "adamson": project_root / "data" / "annotations" / "adamson_functional_classes_enriched.tsv",
        "k562": project_root / "data" / "annotations" / "replogle_k562_functional_classes_go.tsv",
        "rpe1": project_root / "data" / "annotations" / "replogle_rpe1_functional_classes_go.tsv",
    }
    
    output_dir = project_root / "audits" / "logo" / "annotation_improvement" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset_name, annotation_path in datasets.items():
        if not annotation_path.exists():
            print(f"⚠️  Skipping {dataset_name}: {annotation_path} not found")
            continue
        
        print("\n" + "=" * 70)
        print(f"DATASET: {dataset_name.upper()}")
        print("=" * 70)
        print()
        
        try:
            create_validation_report(annotation_path, dataset_name, output_dir)
        except Exception as e:
            print(f"Error validating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Cross-dataset validation
    print("\n" + "=" * 70)
    print("CROSS-DATASET VALIDATION")
    print("=" * 70)
    print()
    
    validate_cross_dataset_consistency(datasets)


if __name__ == "__main__":
    main()


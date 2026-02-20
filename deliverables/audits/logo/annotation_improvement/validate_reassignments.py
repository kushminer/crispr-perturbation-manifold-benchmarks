#!/usr/bin/env python3
"""
Validate reassignments from expression similarity and GO term analysis.

This script:
1. Loads original and improved annotations
2. Compares reassignments
3. Checks for consistency
4. Reports validation statistics
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from shared.io import load_annotations


def validate_reassignments(
    original_path: Path,
    improved_path: Path,
    reassignments_path: Path,
    dataset_name: str
):
    """Validate reassignments."""
    
    print("=" * 70)
    print(f"VALIDATING REASSIGNMENTS: {dataset_name.upper()}")
    print("=" * 70)
    print()
    
    # Load files
    original = load_annotations(original_path)
    improved = load_annotations(improved_path)
    reassignments = pd.read_csv(reassignments_path) if reassignments_path.exists() else pd.DataFrame()
    
    print(f"Original annotations: {len(original)} genes")
    print(f"Improved annotations: {len(improved)} genes")
    print(f"Reassignments logged: {len(reassignments)} genes")
    print()
    
    # Check consistency
    original["target"] = original["target"].astype(str)
    improved["target"] = improved["target"].astype(str)
    
    # Find genes that changed
    merged = original.merge(improved, on="target", suffixes=("_original", "_improved"))
    changed = merged[merged["class_original"] != merged["class_improved"]]
    
    print("=" * 70)
    print("REASSIGNMENT SUMMARY")
    print("=" * 70)
    print()
    
    print(f"Genes that changed class: {len(changed)}")
    print()
    
    # Count changes by original class
    if len(changed) > 0:
        print("Changes by original class:")
        changes_by_original = changed["class_original"].value_counts()
        for class_name, count in changes_by_original.items():
            print(f"  {class_name:30s}: {count:4d} genes")
        print()
        
        # Count changes by new class
        print("Changes by new class:")
        changes_by_new = changed["class_improved"].value_counts()
        for class_name, count in changes_by_new.items():
            print(f"  {class_name:30s}: {count:4d} genes")
        print()
    
    # Check "Other" reduction
    original_other = (original["class"] == "Other").sum()
    improved_other = (improved["class"] == "Other").sum()
    reduction = original_other - improved_other
    
    print("=" * 70)
    print("'OTHER' CLASS REDUCTION")
    print("=" * 70)
    print()
    print(f"Original 'Other': {original_other} genes ({original_other/len(original)*100:.1f}%)")
    print(f"Improved 'Other': {improved_other} genes ({improved_other/len(improved)*100:.1f}%)")
    print(f"Reduction: {reduction} genes ({reduction/original_other*100:.1f}%)")
    print()
    
    # Validate reassignment log matches actual changes
    if len(reassignments) > 0:
        print("=" * 70)
        print("REASSIGNMENT LOG VALIDATION")
        print("=" * 70)
        print()
        
        log_genes = set(reassignments["gene"].astype(str))
        changed_genes = set(changed["target"].astype(str))
        
        in_log_not_changed = log_genes - changed_genes
        changed_not_in_log = changed_genes - log_genes
        
        if in_log_not_changed:
            print(f"⚠️  {len(in_log_not_changed)} genes in log but not changed in annotations")
        if changed_not_in_log:
            print(f"⚠️  {len(changed_not_in_log)} genes changed but not in log")
        
        if not in_log_not_changed and not changed_not_in_log:
            print("✓ Reassignment log matches actual changes")
        print()
    
    # Class distribution comparison
    print("=" * 70)
    print("CLASS DISTRIBUTION COMPARISON")
    print("=" * 70)
    print()
    
    original_dist = original["class"].value_counts().sort_index()
    improved_dist = improved["class"].value_counts().sort_index()
    
    print("Original distribution:")
    for class_name, count in original_dist.items():
        print(f"  {class_name:30s}: {count:4d} ({count/len(original)*100:5.1f}%)")
    print()
    
    print("Improved distribution:")
    for class_name, count in improved_dist.items():
        print(f"  {class_name:30s}: {count:4d} ({count/len(improved)*100:5.1f}%)")
    print()
    
    return {
        "original_count": len(original),
        "improved_count": len(improved),
        "reassignments_count": len(changed),
        "original_other": original_other,
        "improved_other": improved_other,
        "reduction": reduction,
        "reduction_pct": reduction / original_other * 100 if original_other > 0 else 0,
    }


def main():
    """Validate reassignments for all datasets."""
    
    datasets = {
        "adamson": {
            "original": project_root / "data" / "annotations" / "adamson_functional_classes_enriched.tsv",
            "improved": project_root / "audits" / "logo" / "annotation_improvement" / "improved_annotations" / "adamson_functional_classes_enriched_improved_expression.tsv",
            "reassignments": project_root / "audits" / "logo" / "annotation_improvement" / "improved_annotations" / "reassignments_expression_adamson_*.csv",
        },
        "k562": {
            "original": project_root / "data" / "annotations" / "replogle_k562_functional_classes_go.tsv",
            "improved": project_root / "audits" / "logo" / "annotation_improvement" / "improved_annotations" / "replogle_k562_functional_classes_go_improved_expression.tsv",
            "reassignments": project_root / "audits" / "logo" / "annotation_improvement" / "improved_annotations" / "reassignments_expression_k562_*.csv",
        },
        "rpe1": {
            "original": project_root / "data" / "annotations" / "replogle_rpe1_functional_classes_go.tsv",
            "improved": project_root / "audits" / "logo" / "annotation_improvement" / "improved_annotations" / "replogle_rpe1_functional_classes_go_improved_expression.tsv",
            "reassignments": project_root / "audits" / "logo" / "annotation_improvement" / "improved_annotations" / "reassignments_expression_rpe1_*.csv",
        },
    }
    
    all_validations = {}
    
    for dataset_name, config in datasets.items():
        if not config["original"].exists():
            print(f"⚠️  Skipping {dataset_name}: original file not found")
            continue
        
        if not config["improved"].exists():
            print(f"⚠️  Skipping {dataset_name}: improved file not found")
            continue
        
        # Find reassignment file
        reassignments_pattern = str(config["reassignments"]).replace("*", "*")
        import glob
        reassignments_files = glob.glob(reassignments_pattern)
        reassignments_path = Path(reassignments_files[0]) if reassignments_files else None
        
        print("\n" + "=" * 70)
        print(f"DATASET: {dataset_name.upper()}")
        print("=" * 70)
        print()
        
        try:
            validation = validate_reassignments(
                config["original"],
                config["improved"],
                reassignments_path,
                dataset_name
            )
            all_validations[dataset_name] = validation
        except Exception as e:
            print(f"Error validating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL VALIDATION SUMMARY")
    print("=" * 70)
    print()
    
    for dataset_name, validation in all_validations.items():
        print(f"{dataset_name:10s}: {validation['reduction']:4d} genes reassigned ({validation['reduction_pct']:5.1f}% reduction)")
        print(f"           Original 'Other': {validation['original_other']:4d}, Improved 'Other': {validation['improved_other']:4d}")
    
    print()
    print("✓ Validation complete")


if __name__ == "__main__":
    main()


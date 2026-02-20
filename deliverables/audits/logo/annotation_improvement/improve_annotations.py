#!/usr/bin/env python3
"""
Improve functional class annotations by reducing "Other" class size.

This script:
1. Analyzes "Other" class genes
2. Attempts to assign them to appropriate classes using:
   - GO term enrichment
   - Expression similarity
   - Cross-dataset consistency
3. Creates improved annotation files (backup originals)

Key Principle: Never modify original files - always create new files.
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from shared.io import load_annotations


def backup_original_annotation(annotation_path: Path, backup_dir: Path):
    """Create backup of original annotation file."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"{annotation_path.stem}_backup_{timestamp}{annotation_path.suffix}"
    
    import shutil
    shutil.copy2(annotation_path, backup_file)
    
    print(f"✓ Original file backed up to: {backup_file}")
    return backup_file


def assign_genes_from_go_terms(
    other_genes: List[str],
    organism: str = "human"
) -> Dict[str, str]:
    """
    Assign "Other" genes to classes based on GO terms.
    
    This function would:
    1. Query GO database for each gene
    2. Map GO terms to functional classes
    3. Assign gene to most appropriate class
    
    Returns:
        Dict mapping gene -> suggested class
    """
    print("=" * 70)
    print("GO-BASED CLASS ASSIGNMENT")
    print("=" * 70)
    print()
    
    print("⚠️  GO-based assignment requires:")
    print("   1. GO API access or local GO database")
    print("   2. Gene symbol to GO term mapping")
    print("   3. GO term to functional class mapping")
    print()
    
    # Placeholder - would implement actual GO queries
    assignments = {}
    
    return assignments


def assign_genes_from_expression(
    other_genes: List[str],
    annotation_path: Path,
    adata_path: Path
) -> Dict[str, str]:
    """
    Assign "Other" genes to classes based on expression similarity.
    
    For each "Other" gene:
    1. Compute expression profile across perturbations
    2. Correlate with mean expression of each class
    3. Assign to class with highest correlation
    """
    print("=" * 70)
    print("EXPRESSION-BASED CLASS ASSIGNMENT")
    print("=" * 70)
    print()
    
    print("⚠️  Expression-based assignment requires:")
    print("   1. Load perturbation expression data")
    print("   2. Compute class mean expression profiles")
    print("   3. Correlate 'Other' genes with class profiles")
    print("   4. Assign to class with highest correlation")
    print()
    
    # Placeholder
    assignments = {}
    
    return assignments


def create_improved_annotations(
    annotation_path: Path,
    new_assignments: Dict[str, str],
    output_path: Path
) -> pd.DataFrame:
    """
    Create improved annotation file with new assignments.
    
    Args:
        annotation_path: Original annotation file
        new_assignments: Dict mapping gene -> new class
        output_path: Path for new annotation file
    
    Returns:
        Improved annotations DataFrame
    """
    print("=" * 70)
    print("CREATING IMPROVED ANNOTATIONS")
    print("=" * 70)
    print()
    
    # Load original
    annotations = load_annotations(annotation_path)
    
    # Apply new assignments
    improved = annotations.copy()
    
    reassigned_count = 0
    for gene, new_class in new_assignments.items():
        mask = improved["target"] == gene
        if mask.sum() > 0:
            old_class = improved.loc[mask, "class"].values[0]
            improved.loc[mask, "class"] = new_class
            print(f"  {gene}: {old_class} → {new_class}")
            reassigned_count += 1
    
    print(f"\nReassigned {reassigned_count} genes")
    
    # Check "Other" reduction
    original_other = (annotations["class"] == "Other").sum()
    improved_other = (improved["class"] == "Other").sum()
    reduction = original_other - improved_other
    
    print(f"\n'Other' class reduction: {original_other} → {improved_other} ({reduction} genes)")
    print(f"Reduction: {reduction / original_other * 100:.1f}%")
    
    # Save improved annotations
    improved.to_csv(output_path, sep="\t", index=False)
    print(f"\n✓ Improved annotations saved to: {output_path}")
    
    return improved


def main():
    """Improve annotations for all datasets."""
    
    datasets = {
        "adamson": project_root / "data" / "annotations" / "adamson_functional_classes_enriched.tsv",
        "k562": project_root / "data" / "annotations" / "replogle_k562_functional_classes_go.tsv",
        "rpe1": project_root / "data" / "annotations" / "replogle_rpe1_functional_classes_go.tsv",
    }
    
    improved_dir = project_root / "audits" / "logo" / "annotation_improvement" / "improved_annotations"
    backup_dir = improved_dir / "backups"
    improved_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("ANNOTATION IMPROVEMENT")
    print("=" * 70)
    print()
    print("⚠️  This is a framework - implement with:")
    print("   1. GO term queries")
    print("   2. Expression similarity analysis")
    print("   3. Manual curation for ambiguous cases")
    print()
    
    for dataset_name, annotation_path in datasets.items():
        if not annotation_path.exists():
            print(f"⚠️  Skipping {dataset_name}: {annotation_path} not found")
            continue
        
        print("\n" + "=" * 70)
        print(f"DATASET: {dataset_name.upper()}")
        print("=" * 70)
        print()
        
        # Backup original
        backup_file = backup_original_annotation(annotation_path, backup_dir)
        
        # Load annotations
        annotations = load_annotations(annotation_path)
        other_genes = annotations[annotations["class"] == "Other"]["target"].tolist()
        
        print(f"Found {len(other_genes)} genes in 'Other' class")
        print()
        
        # Get new assignments (placeholder - implement actual logic)
        new_assignments = {}
        
        # For now, just create framework
        output_path = improved_dir / f"{annotation_path.stem}_improved{annotation_path.suffix}"
        
        if new_assignments:
            create_improved_annotations(annotation_path, new_assignments, output_path)
        else:
            print("⚠️  No new assignments generated (implement assignment logic)")
            print(f"   Framework ready at: {output_path}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    print("1. Implement GO term queries")
    print("2. Implement expression similarity analysis")
    print("3. Generate new assignments")
    print("4. Validate improved annotations")
    print("5. Re-run LOGO with improved annotations")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Analyze "Other" class composition to understand what genes are unannotated.

This script:
1. Loads functional class annotations
2. Identifies genes in "Other" class
3. Attempts to find GO/Reactome annotations for these genes
4. Suggests appropriate class assignments

Key Principle: Do not modify original annotation files - create new improved versions.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from shared.io import load_annotations

def analyze_other_class(annotation_path: Path, dataset_name: str):
    """Analyze what's in the "Other" class."""
    
    print("=" * 70)
    print(f"ANALYZING 'OTHER' CLASS: {dataset_name.upper()}")
    print("=" * 70)
    print()
    
    # Load annotations
    annotations = load_annotations(annotation_path)
    annotations["target"] = annotations["target"].astype(str)
    
    # Get "Other" class genes
    other_genes = annotations[annotations["class"] == "Other"]["target"].tolist()
    
    print(f"Total annotations: {len(annotations)}")
    print(f"'Other' class genes: {len(other_genes)}")
    print(f"'Other' as % of total: {len(other_genes) / len(annotations) * 100:.1f}%")
    print()
    
    # Class distribution
    print("=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)
    class_counts = annotations["class"].value_counts().sort_index()
    print(class_counts.to_string())
    print()
    
    # Analyze "Other" genes
    print("=" * 70)
    print("'OTHER' CLASS GENES")
    print("=" * 70)
    print(f"\nGenes in 'Other' class ({len(other_genes)}):")
    for i, gene in enumerate(sorted(other_genes), 1):
        print(f"  {i:3d}. {gene}")
    print()
    
    return {
        "other_genes": other_genes,
        "total_annotations": len(annotations),
        "other_count": len(other_genes),
        "class_distribution": class_counts,
    }


def check_go_annotations(genes: list, organism: str = "human"):
    """
    Check if "Other" genes have GO annotations.
    
    This function attempts to find GO terms for genes in "Other" class.
    """
    print("=" * 70)
    print("CHECKING GO ANNOTATIONS")
    print("=" * 70)
    print()
    
    try:
        import gseapy as gp
    except ImportError:
        print("⚠️  gseapy not available. Install with: pip install gseapy")
        print("   Skipping GO annotation check.")
        return None
    
    print(f"Checking GO annotations for {len(genes)} genes...")
    print("(This may take a few minutes)")
    print()
    
    # Try to get GO annotations
    # Note: This is a placeholder - actual implementation would use GO API
    # or gseapy to query GO terms
    
    print("⚠️  GO annotation check not yet implemented")
    print("   This requires:")
    print("   1. GO API access or local GO database")
    print("   2. Gene symbol to GO term mapping")
    print("   3. Functional class assignment logic")
    
    return None


def suggest_class_assignments(other_genes: list, dataset_name: str):
    """
    Suggest class assignments for "Other" genes based on:
    1. GO term enrichment
    2. Expression similarity to known classes
    3. Literature/domain knowledge
    """
    print("=" * 70)
    print("SUGGESTED CLASS ASSIGNMENTS")
    print("=" * 70)
    print()
    
    print("⚠️  Class assignment suggestions require:")
    print("   1. GO term analysis for each gene")
    print("   2. Expression correlation with known classes")
    print("   3. Manual curation for ambiguous cases")
    print()
    
    print("Recommended approach:")
    print("  1. Use GO enrichment analysis to identify overrepresented terms")
    print("  2. Map GO terms to functional classes")
    print("  3. Validate assignments with expression similarity")
    print("  4. Manual review of ambiguous assignments")
    
    return None


def main():
    """Analyze "Other" class for all datasets."""
    
    datasets = {
        "adamson": project_root / "data" / "annotations" / "adamson_functional_classes_enriched.tsv",
        "k562": project_root / "data" / "annotations" / "replogle_k562_functional_classes_go.tsv",
        "rpe1": project_root / "data" / "annotations" / "replogle_rpe1_functional_classes_go.tsv",
    }
    
    output_dir = project_root / "audits" / "logo" / "annotation_improvement"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_analyses = {}
    
    for dataset_name, annotation_path in datasets.items():
        if not annotation_path.exists():
            print(f"⚠️  Skipping {dataset_name}: {annotation_path} not found")
            continue
        
        print("\n" + "=" * 70)
        print(f"DATASET: {dataset_name.upper()}")
        print("=" * 70)
        print()
        
        try:
            analysis = analyze_other_class(annotation_path, dataset_name)
            all_analyses[dataset_name] = analysis
            
            # Save analysis
            output_file = output_dir / f"other_class_analysis_{dataset_name}.txt"
            with open(output_file, "w") as f:
                f.write(f"Other Class Analysis: {dataset_name}\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Total annotations: {analysis['total_annotations']}\n")
                f.write(f"'Other' class genes: {analysis['other_count']}\n")
                f.write(f"'Other' as %: {analysis['other_count'] / analysis['total_annotations'] * 100:.1f}%\n\n")
                f.write("Other genes:\n")
                for gene in sorted(analysis['other_genes']):
                    f.write(f"  {gene}\n")
            
            print(f"\nAnalysis saved to: {output_file}")
            
        except Exception as e:
            print(f"Error analyzing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    for dataset_name, analysis in all_analyses.items():
        pct = analysis['other_count'] / analysis['total_annotations'] * 100
        print(f"{dataset_name:10s}: {analysis['other_count']:4d} 'Other' genes ({pct:5.1f}%)")
    
    print()
    print("Next steps:")
    print("  1. Run GO annotation check for 'Other' genes")
    print("  2. Map GO terms to functional classes")
    print("  3. Create improved annotation files")
    print("  4. Validate improved annotations")


if __name__ == "__main__":
    main()


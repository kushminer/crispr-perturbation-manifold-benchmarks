#!/usr/bin/env python3
"""
Query GO terms for genes in "Other" class to suggest appropriate functional classes.

This script uses GO API or local database to:
1. Query GO terms for each "Other" gene
2. Map GO terms to functional classes
3. Suggest class assignments

Key Principle: Do not modify original files - create new improved versions.
"""

import sys
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from shared.io import load_annotations


def query_go_terms_gseapy(genes: List[str], organism: str = "human") -> Dict[str, List[str]]:
    """
    Query GO terms using gseapy library.
    
    Args:
        genes: List of gene symbols
        organism: Organism (human, mouse, etc.)
    
    Returns:
        Dict mapping gene -> list of GO terms
    """
    try:
        import gseapy as gp
    except ImportError:
        print("⚠️  gseapy not available. Install with: pip install gseapy")
        return {}
    
    print(f"Querying GO terms for {len(genes)} genes using gseapy...")
    
    # Use gseapy to query GO database
    # This is a placeholder - actual implementation would use GO API
    go_results = {}
    
    for gene in genes:
        try:
            # Query GO terms for this gene
            # gseapy doesn't have direct GO query, would need GO API
            # For now, return empty - implement with actual GO API
            go_results[gene] = []
        except Exception as e:
            print(f"  Error querying {gene}: {e}")
            go_results[gene] = []
    
    return go_results


def query_go_terms_goatools(genes: List[str], organism: str = "human") -> Dict[str, List[str]]:
    """
    Query GO terms using goatools library.
    
    Args:
        genes: List of gene symbols
        organism: Organism
    
    Returns:
        Dict mapping gene -> list of GO terms
    """
    try:
        from goatools import GOEnrichmentStudy
        from goatools.associations import read_ncbi_gene_file
    except ImportError:
        print("⚠️  goatools not available. Install with: pip install goatools")
        return {}
    
    print(f"Querying GO terms for {len(genes)} genes using goatools...")
    
    # Placeholder - would need GO database files
    go_results = {}
    
    return go_results


def map_go_terms_to_classes(go_terms: List[str]) -> Optional[str]:
    """
    Map GO terms to functional classes.
    
    This function uses a mapping from GO terms to functional classes.
    
    Args:
        go_terms: List of GO term IDs
    
    Returns:
        Suggested functional class name, or None
    """
    # GO term to functional class mapping
    # This is a simplified mapping - should be expanded
    go_to_class = {
        # Transcription-related GO terms
        "GO:0006351": "Transcription",  # DNA-templated transcription
        "GO:0006355": "Transcription",  # regulation of transcription
        "GO:0003700": "Transcription",  # DNA-binding transcription factor
        "GO:0003677": "Transcription",  # DNA binding
        "GO:0043565": "Transcription",  # sequence-specific DNA binding
        
        # Translation-related GO terms
        "GO:0006412": "Translation",  # translation
        "GO:0003743": "Translation",  # translation initiation factor
        "GO:0003746": "Translation",  # translation elongation factor
        "GO:0006414": "Translation",  # translational elongation
        
        # Metabolism-related GO terms
        "GO:0008152": "Metabolism",  # metabolic process
        "GO:0044237": "Metabolism",  # cellular metabolic process
        "GO:0044238": "Metabolism",  # primary metabolic process
        
        # Add more mappings as needed
    }
    
    # Find matching class
    for go_term in go_terms:
        if go_term in go_to_class:
            return go_to_class[go_term]
    
    return None


def suggest_classes_from_go(
    annotation_path: Path,
    dataset_name: str,
    output_dir: Path
) -> pd.DataFrame:
    """
    Suggest class assignments for "Other" genes based on GO terms.
    
    Args:
        annotation_path: Original annotation file
        dataset_name: Dataset name
        output_dir: Output directory for suggestions
    
    Returns:
        DataFrame with suggested assignments
    """
    print("=" * 70)
    print(f"GO-BASED CLASS SUGGESTIONS: {dataset_name.upper()}")
    print("=" * 70)
    print()
    
    # Load annotations
    annotations = load_annotations(annotation_path)
    annotations["target"] = annotations["target"].astype(str)
    
    # Get "Other" class genes
    other_genes = annotations[annotations["class"] == "Other"]["target"].tolist()
    
    print(f"Found {len(other_genes)} genes in 'Other' class")
    print()
    
    # Query GO terms (placeholder - implement actual GO queries)
    print("⚠️  GO term queries require:")
    print("   1. GO API access (http://geneontology.org/docs/api)")
    print("   2. Or local GO database (go.obo, gene2go files)")
    print("   3. Gene symbol to GO term mapping")
    print()
    
    # For now, create framework
    suggestions = []
    
    for gene in other_genes:
        # Placeholder - would query actual GO terms
        go_terms = []  # Would be populated by GO query
        suggested_class = map_go_terms_to_classes(go_terms)
        
        suggestions.append({
            "gene": gene,
            "current_class": "Other",
            "suggested_class": suggested_class or "Other",
            "go_terms": ",".join(go_terms) if go_terms else "",
            "confidence": "low" if not go_terms else "medium",
        })
    
    suggestions_df = pd.DataFrame(suggestions)
    
    # Save suggestions
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"go_suggestions_{dataset_name}.csv"
    suggestions_df.to_csv(output_file, index=False)
    
    print(f"Suggestions saved to: {output_file}")
    print()
    print(f"Genes with suggestions: {(suggestions_df['suggested_class'] != 'Other').sum()}")
    print(f"Genes remaining in 'Other': {(suggestions_df['suggested_class'] == 'Other').sum()}")
    
    return suggestions_df


def main():
    """Generate GO-based class suggestions for all datasets."""
    
    datasets = {
        "adamson": project_root / "data" / "annotations" / "adamson_functional_classes_enriched.tsv",
        "k562": project_root / "data" / "annotations" / "replogle_k562_functional_classes_go.tsv",
        "rpe1": project_root / "data" / "annotations" / "replogle_rpe1_functional_classes_go.tsv",
    }
    
    output_dir = project_root / "audits" / "logo" / "annotation_improvement" / "improved_annotations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_suggestions = {}
    
    for dataset_name, annotation_path in datasets.items():
        if not annotation_path.exists():
            print(f"⚠️  Skipping {dataset_name}: {annotation_path} not found")
            continue
        
        print("\n" + "=" * 70)
        print(f"DATASET: {dataset_name.upper()}")
        print("=" * 70)
        print()
        
        try:
            suggestions = suggest_classes_from_go(annotation_path, dataset_name, output_dir)
            all_suggestions[dataset_name] = suggestions
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("⚠️  GO term queries not yet implemented")
    print("   To implement:")
    print("   1. Set up GO API access or local GO database")
    print("   2. Map gene symbols to GO terms")
    print("   3. Map GO terms to functional classes")
    print("   4. Generate suggestions")


if __name__ == "__main__":
    main()


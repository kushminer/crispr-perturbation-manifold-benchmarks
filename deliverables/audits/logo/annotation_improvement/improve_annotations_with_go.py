#!/usr/bin/env python3
"""
Improve functional class annotations using GO terms and expression similarity.

This script:
1. Queries GO terms for "Other" class genes using mygene API
2. Maps GO terms to functional classes
3. Optionally uses expression similarity as additional evidence
4. Creates improved annotation files (backups originals)

Key Principle: Never modify original files - always create backups and new files.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from shared.io import load_annotations


def query_go_terms_mygene(genes: List[str], organism: str = "human") -> Dict[str, List[Dict]]:
    """
    Query GO terms for genes using mygene.info API.
    
    Args:
        genes: List of gene symbols
        organism: Organism (default: "human")
    
    Returns:
        Dict mapping gene -> list of GO term dicts
    """
    try:
        import mygene
    except ImportError:
        print("⚠️  mygene not available. Install with: pip install mygene")
        return {}
    
    print(f"Querying GO terms for {len(genes)} genes using mygene.info API...")
    
    mg = mygene.MyGeneInfo()
    
    # Query in batches
    batch_size = 1000
    gene_to_go = {}
    
    for batch_start in range(0, len(genes), batch_size):
        batch_end = min(batch_start + batch_size, len(genes))
        batch_genes = genes[batch_start:batch_end]
        
        if batch_start > 0:
            print(f"  Processing batch {batch_start + 1}-{batch_end} of {len(genes)}...")
        
        try:
            gene_info = mg.querymany(
                batch_genes,
                scopes="symbol",
                fields="go",
                species=organism,
                returnall=True,
            )
            
            for result in gene_info["out"]:
                if "query" in result:
                    gene = result["query"]
                    go_data = result.get("go", {})
                    
                    # Extract GO Biological Process terms
                    bp_terms = go_data.get("BP", [])
                    if bp_terms:
                        gene_to_go[gene] = bp_terms
                        print(f"    {gene}: {len(bp_terms)} GO terms")
        
        except Exception as e:
            print(f"  Error querying batch: {e}")
            continue
    
    print(f"\nFound GO terms for {len(gene_to_go)}/{len(genes)} genes")
    return gene_to_go


def map_go_term_to_class(go_term) -> Optional[str]:
    """
    Map a GO term to a functional class.
    
    Args:
        go_term: GO term dict with 'id', 'term', 'category' OR string
    
    Returns:
        Functional class name, or None
    """
    # Handle both dict and string formats
    if isinstance(go_term, dict):
        go_id = go_term.get("id", "")
        go_term_name = go_term.get("term", go_term.get("name", "")).lower()
    elif isinstance(go_term, str):
        # If it's a string, try to extract GO ID or use as term name
        if go_term.startswith("GO:"):
            go_id = go_term
            go_term_name = go_term.lower()
        else:
            go_id = ""
            go_term_name = go_term.lower()
    else:
        return None
    
    # GO ID to class mapping (key terms)
    go_id_keywords = {
        "GO:0006351": "Transcription",  # DNA-templated transcription
        "GO:0006355": "Transcription",  # regulation of transcription
        "GO:0003700": "Transcription",  # DNA-binding transcription factor
        "GO:0003677": "Transcription",  # DNA binding
        "GO:0043565": "Transcription",  # sequence-specific DNA binding
        "GO:0006412": "Translation",  # translation
        "GO:0003743": "Translation",  # translation initiation factor
        "GO:0003746": "Translation",  # translation elongation factor
        "GO:0006414": "Translation",  # translational elongation
        "GO:0008152": "Metabolism",  # metabolic process
        "GO:0044237": "Metabolism",  # cellular metabolic process
        "GO:0044238": "Metabolism",  # primary metabolic process
    }
    
    # Check GO ID first
    if go_id in go_id_keywords:
        return go_id_keywords[go_id]
    
    # Check term name keywords
    transcription_keywords = ["transcription", "rna polymerase", "transcription factor", "dna binding"]
    translation_keywords = ["translation", "ribosome", "trna", "translation factor"]
    metabolism_keywords = ["metabolic", "metabolism", "biosynthetic", "catabolic"]
    
    if any(kw in go_term_name for kw in transcription_keywords):
        return "Transcription"
    if any(kw in go_term_name for kw in translation_keywords):
        return "Translation"
    if any(kw in go_term_name for kw in metabolism_keywords):
        return "Metabolism"
    
    return None


def assign_class_from_go_terms(gene: str, go_terms: List[Dict]) -> Optional[str]:
    """
    Assign functional class to a gene based on its GO terms.
    
    Args:
        gene: Gene symbol
        go_terms: List of GO term dicts
    
    Returns:
        Suggested functional class, or None
    """
    if not go_terms:
        return None
    
    # Count class assignments from GO terms
    class_counts = {}
    
    for go_term in go_terms:
        suggested_class = map_go_term_to_class(go_term)
        if suggested_class:
            class_counts[suggested_class] = class_counts.get(suggested_class, 0) + 1
    
    if not class_counts:
        return None
    
    # Return most common class
    best_class = max(class_counts, key=class_counts.get)
    
    # Only suggest if we have strong evidence (multiple GO terms pointing to same class)
    if class_counts[best_class] >= 2:
        return best_class
    
    return None


def improve_annotations_with_go(
    annotation_path: Path,
    dataset_name: str,
    output_dir: Path,
    organism: str = "human"
) -> pd.DataFrame:
    """
    Improve annotations by querying GO terms for "Other" class genes.
    
    Args:
        annotation_path: Original annotation file
        dataset_name: Dataset name
        output_dir: Output directory for improved annotations
        organism: Organism (default: "human")
    
    Returns:
        Improved annotations DataFrame
    """
    print("=" * 70)
    print(f"IMPROVING ANNOTATIONS WITH GO: {dataset_name.upper()}")
    print("=" * 70)
    print()
    
    # Backup original
    backup_dir = output_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"{annotation_path.stem}_backup_{timestamp}{annotation_path.suffix}"
    
    import shutil
    shutil.copy2(annotation_path, backup_file)
    print(f"✓ Original file backed up to: {backup_file}")
    print()
    
    # Load original annotations
    annotations = load_annotations(annotation_path)
    annotations["target"] = annotations["target"].astype(str)
    
    # Get "Other" class genes
    other_genes = annotations[annotations["class"] == "Other"]["target"].tolist()
    
    print(f"Found {len(other_genes)} genes in 'Other' class")
    print()
    
    if len(other_genes) == 0:
        print("No 'Other' genes to improve")
        return annotations
    
    # Query GO terms
    gene_to_go = query_go_terms_mygene(other_genes, organism=organism)
    
    if not gene_to_go:
        print("⚠️  No GO terms found. Cannot improve annotations.")
        return annotations
    
    print()
    
    # Assign classes based on GO terms
    improved = annotations.copy()
    reassignments = []
    
    print("Assigning classes based on GO terms...")
    print()
    
    for gene in other_genes:
        if gene not in gene_to_go:
            continue
        
        go_terms = gene_to_go[gene]
        suggested_class = assign_class_from_go_terms(gene, go_terms)
        
        if suggested_class:
            mask = improved["target"] == gene
            if mask.sum() > 0:
                old_class = improved.loc[mask, "class"].values[0]
                improved.loc[mask, "class"] = suggested_class
                reassignments.append({
                    "gene": gene,
                    "old_class": old_class,
                    "new_class": suggested_class,
                    "go_terms_count": len(go_terms),
                })
                print(f"  {gene}: {old_class} → {suggested_class} ({len(go_terms)} GO terms)")
    
    print()
    
    # Summary
    original_other = (annotations["class"] == "Other").sum()
    improved_other = (improved["class"] == "Other").sum()
    reduction = original_other - improved_other
    
    print("=" * 70)
    print("IMPROVEMENT SUMMARY")
    print("=" * 70)
    print(f"Genes reassigned: {len(reassignments)}")
    print(f"'Other' class reduction: {original_other} → {improved_other} ({reduction} genes)")
    if original_other > 0:
        print(f"Reduction: {reduction / original_other * 100:.1f}%")
    print()
    
    # Save improved annotations
    output_file = output_dir / f"{annotation_path.stem}_improved_go{annotation_path.suffix}"
    improved.to_csv(output_file, sep="\t", index=False)
    print(f"✓ Improved annotations saved to: {output_file}")
    
    # Save reassignment log
    if reassignments:
        reassignments_df = pd.DataFrame(reassignments)
        log_file = output_dir / f"reassignments_{dataset_name}_{timestamp}.csv"
        reassignments_df.to_csv(log_file, index=False)
        print(f"✓ Reassignment log saved to: {log_file}")
    
    return improved


def main():
    """Improve annotations for all datasets."""
    
    datasets = {
        "adamson": {
            "annotation_path": project_root / "data" / "annotations" / "adamson_functional_classes_enriched.tsv",
            "organism": "human",
        },
        "k562": {
            "annotation_path": project_root / "data" / "annotations" / "replogle_k562_functional_classes_go.tsv",
            "organism": "human",
        },
        "rpe1": {
            "annotation_path": project_root / "data" / "annotations" / "replogle_rpe1_functional_classes_go.tsv",
            "organism": "human",
        },
    }
    
    output_dir = project_root / "audits" / "logo" / "annotation_improvement" / "improved_annotations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_improved = {}
    
    for dataset_name, config in datasets.items():
        if not config["annotation_path"].exists():
            print(f"⚠️  Skipping {dataset_name}: {config['annotation_path']} not found")
            continue
        
        print("\n" + "=" * 70)
        print(f"DATASET: {dataset_name.upper()}")
        print("=" * 70)
        print()
        
        try:
            improved = improve_annotations_with_go(
                config["annotation_path"],
                dataset_name,
                output_dir,
                organism=config["organism"]
            )
            all_improved[dataset_name] = improved
        except Exception as e:
            print(f"Error improving {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print()
    
    for dataset_name, improved in all_improved.items():
        other_count = (improved["class"] == "Other").sum()
        total = len(improved)
        pct = other_count / total * 100
        print(f"{dataset_name:10s}: {other_count:4d} 'Other' genes ({pct:5.1f}%)")
    
    print()
    print("Next steps:")
    print("  1. Review improved annotations")
    print("  2. Validate reassignments")
    print("  3. Re-run LOGO with improved annotations")
    print("  4. Compare results with original annotations")


if __name__ == "__main__":
    main()


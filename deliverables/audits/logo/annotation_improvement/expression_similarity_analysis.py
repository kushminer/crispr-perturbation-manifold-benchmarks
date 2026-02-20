#!/usr/bin/env python3
"""
Expression Similarity Analysis for "Other" Class Assignment.

This script:
1. Computes mean expression profiles for each functional class
2. Correlates "Other" genes with class profiles
3. Suggests class assignments based on expression similarity

Key Principle: Do not modify original files - create new improved versions.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from shared.io import load_annotations


def compute_class_expression_profiles(
    adata_path: Path,
    annotation_path: Path,
    class_name: str
) -> np.ndarray:
    """
    Compute mean expression profile for a functional class.
    
    Args:
        adata_path: Path to AnnData file
        annotation_path: Path to annotations
        class_name: Functional class name
    
    Returns:
        Mean expression profile (genes,)
    """
    try:
        import anndata as ad
        adata = ad.read_h5ad(adata_path)
    except Exception as e:
        print(f"⚠️  Could not load data: {e}")
        return None
    
    annotations = load_annotations(annotation_path)
    
    # Get genes in this class
    class_genes = annotations[annotations["class"] == class_name]["target"].tolist()
    
    if len(class_genes) == 0:
        return None
    
    # Compute pseudobulk for each perturbation in this class
    # (Simplified - would use actual perturbation data)
    print(f"Computing expression profile for class '{class_name}' ({len(class_genes)} genes)")
    
    # Placeholder - would compute actual mean expression
    # This requires loading perturbation data and computing mean
    return None


def correlate_with_classes(
    adata_path: Path,
    annotation_path: Path,
    other_genes: List[str],
    min_correlation: float = 0.3
) -> Dict[str, Tuple[str, float]]:
    """
    Correlate "Other" genes with class expression profiles.
    
    Args:
        adata_path: Path to AnnData file
        annotation_path: Path to annotations
        other_genes: List of genes in "Other" class
        min_correlation: Minimum correlation to suggest assignment
    
    Returns:
        Dict mapping gene -> (suggested_class, correlation)
    """
    print("=" * 70)
    print("EXPRESSION SIMILARITY ANALYSIS")
    print("=" * 70)
    print()
    
    annotations = load_annotations(annotation_path)
    
    # Get all classes (excluding "Other")
    all_classes = [c for c in annotations["class"].unique() if c != "Other"]
    
    print(f"Analyzing {len(other_genes)} 'Other' genes")
    print(f"Comparing against {len(all_classes)} classes")
    print()
    
    print("⚠️  Expression similarity analysis requires:")
    print("   1. Load perturbation expression data")
    print("   2. Compute class mean expression profiles")
    print("   3. Correlate 'Other' genes with each class profile")
    print("   4. Assign to class with highest correlation")
    print()
    
    # Placeholder - would implement actual correlation
    suggestions = {}
    
    for gene in other_genes:
        # Placeholder - would compute actual correlations
        best_class = None
        best_corr = 0.0
        
        # Would iterate over classes and compute correlation
        # For now, return None
        suggestions[gene] = (best_class, best_corr)
    
    return suggestions


def suggest_classes_from_expression(
    adata_path: Path,
    annotation_path: Path,
    dataset_name: str,
    output_dir: Path,
    min_correlation: float = 0.3
) -> pd.DataFrame:
    """
    Suggest class assignments based on expression similarity.
    
    Args:
        adata_path: Path to AnnData file
        annotation_path: Path to annotations
        dataset_name: Dataset name
        output_dir: Output directory
        min_correlation: Minimum correlation threshold
    
    Returns:
        DataFrame with suggested assignments
    """
    print("=" * 70)
    print(f"EXPRESSION-BASED CLASS SUGGESTIONS: {dataset_name.upper()}")
    print("=" * 70)
    print()
    
    if not Path(adata_path).exists():
        print(f"⚠️  Data file not found: {adata_path}")
        return pd.DataFrame()
    
    # Load annotations
    annotations = load_annotations(annotation_path)
    annotations["target"] = annotations["target"].astype(str)
    
    # Get "Other" class genes
    other_genes = annotations[annotations["class"] == "Other"]["target"].tolist()
    
    print(f"Found {len(other_genes)} genes in 'Other' class")
    print()
    
    # Correlate with classes
    suggestions_dict = correlate_with_classes(
        Path(adata_path),
        annotation_path,
        other_genes,
        min_correlation=min_correlation
    )
    
    # Create suggestions DataFrame
    suggestions = []
    for gene, (suggested_class, correlation) in suggestions_dict.items():
        suggestions.append({
            "gene": gene,
            "current_class": "Other",
            "suggested_class": suggested_class or "Other",
            "correlation": correlation,
            "confidence": "high" if correlation > 0.5 else "medium" if correlation > min_correlation else "low",
        })
    
    suggestions_df = pd.DataFrame(suggestions)
    
    # Save suggestions
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"expression_suggestions_{dataset_name}.csv"
    suggestions_df.to_csv(output_file, index=False)
    
    print(f"Suggestions saved to: {output_file}")
    print()
    
    if len(suggestions_df) > 0:
        high_confidence = (suggestions_df["confidence"] == "high").sum()
        medium_confidence = (suggestions_df["confidence"] == "medium").sum()
        print(f"High confidence suggestions: {high_confidence}")
        print(f"Medium confidence suggestions: {medium_confidence}")
        print(f"Genes remaining in 'Other': {(suggestions_df['suggested_class'] == 'Other').sum()}")
    
    return suggestions_df


def main():
    """Generate expression-based class suggestions for all datasets."""
    
    datasets = {
        "adamson": {
            "adata_path": "/Users/samuelminer/Documents/classes/nih_research/data_adamson/perturb_processed.h5ad",
            "annotation_path": project_root / "data" / "annotations" / "adamson_functional_classes_enriched.tsv",
        },
        "k562": {
            "adata_path": "/Users/samuelminer/Documents/classes/nih_research/data_replogle_k562_essential/perturb_processed.h5ad",
            "annotation_path": project_root / "data" / "annotations" / "replogle_k562_functional_classes_go.tsv",
        },
        "rpe1": {
            "adata_path": "/Users/samuelminer/Documents/classes/nih_research/data_replogle_rpe1_essential/perturb_processed.h5ad",
            "annotation_path": project_root / "data" / "annotations" / "replogle_rpe1_functional_classes_go.tsv",
        },
    }
    
    output_dir = project_root / "audits" / "logo" / "annotation_improvement" / "improved_annotations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_suggestions = {}
    
    for dataset_name, config in datasets.items():
        if not Path(config["adata_path"]).exists():
            print(f"⚠️  Skipping {dataset_name}: data file not found")
            continue
        
        print("\n" + "=" * 70)
        print(f"DATASET: {dataset_name.upper()}")
        print("=" * 70)
        print()
        
        try:
            suggestions = suggest_classes_from_expression(
                Path(config["adata_path"]),
                config["annotation_path"],
                dataset_name,
                output_dir,
                min_correlation=0.3
            )
            all_suggestions[dataset_name] = suggestions
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("⚠️  Expression similarity analysis not yet fully implemented")
    print("   Framework is ready - implement:")
    print("   1. Load perturbation expression data")
    print("   2. Compute class mean profiles")
    print("   3. Correlate 'Other' genes with profiles")
    print("   4. Generate suggestions")


if __name__ == "__main__":
    main()


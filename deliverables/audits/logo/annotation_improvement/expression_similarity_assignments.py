#!/usr/bin/env python3
"""
Expression Similarity-Based Class Assignment for "Other" Genes.

This script:
1. Computes mean expression profiles for each functional class from perturbation data
2. Correlates "Other" genes with each class profile
3. Assigns "Other" genes to class with highest correlation
4. Creates improved annotation files (backups originals)

Key Principle: Never modify original files - always create backups and new files.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from scipy.stats import pearsonr

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from shared.io import load_annotations
from goal_2_baselines.baseline_runner import compute_pseudobulk_expression_changes


def compute_class_profiles(
    adata_path: Path,
    annotation_path: Path,
    min_perturbations: int = 2
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Compute mean expression profile for each functional class.
    
    Args:
        adata_path: Path to AnnData file
        annotation_path: Path to annotations
        min_perturbations: Minimum perturbations per class to compute profile
    
    Returns:
        Tuple of (Dict mapping class_name -> mean expression profile (genes,), List of gene names)
    """
    print("=" * 70)
    print("COMPUTING CLASS EXPRESSION PROFILES")
    print("=" * 70)
    print()
    
    try:
        import anndata as ad
        adata = ad.read_h5ad(adata_path)
    except Exception as e:
        print(f"⚠️  Could not load data: {e}")
        return {}, []
    
    annotations = load_annotations(annotation_path)
    annotations["target"] = annotations["target"].astype(str)
    
    # Compute pseudobulk for all perturbations
    all_conditions = sorted(adata.obs["condition"].unique().tolist())
    dummy_split_config = {
        "train": all_conditions,
        "test": [],
        "val": [],
    }
    
    print("Computing pseudobulk expression changes...")
    Y_df, _ = compute_pseudobulk_expression_changes(adata, dummy_split_config, seed=1)
    print(f"Y matrix shape: {Y_df.shape} (genes × perturbations)")
    print()
    
    # Group perturbations by class
    class_to_perts = {}
    for _, row in annotations.iterrows():
        pert = row["target"]
        class_name = row["class"]
        
        if pert in Y_df.columns:
            if class_name not in class_to_perts:
                class_to_perts[class_name] = []
            class_to_perts[class_name].append(pert)
    
    # Compute mean profile for each class
    class_profiles = {}
    gene_names = Y_df.index.tolist()
    
    print("Computing mean expression profiles per class...")
    print()
    
    for class_name, perts in class_to_perts.items():
        if len(perts) < min_perturbations:
            print(f"  {class_name:30s}: {len(perts):3d} perturbations (skipping - too few)")
            continue
        
        # Get expression for perturbations in this class
        class_perts_in_data = [p for p in perts if p in Y_df.columns]
        
        if len(class_perts_in_data) == 0:
            continue
        
        # Compute mean across perturbations in this class
        class_expr = Y_df[class_perts_in_data].mean(axis=1).values
        class_profiles[class_name] = class_expr
        
        print(f"  {class_name:30s}: {len(class_perts_in_data):3d} perturbations")
    
    print()
    print(f"Computed profiles for {len(class_profiles)} classes")
    print()
    
    return class_profiles, gene_names


def correlate_other_genes_with_classes(
    adata_path: Path,
    annotation_path: Path,
    class_profiles: Dict[str, np.ndarray],
    gene_names: List[str],
    min_correlation: float = 0.3
) -> Dict[str, Tuple[str, float]]:
    """
    Correlate "Other" genes with class expression profiles.
    
    Args:
        adata_path: Path to AnnData file
        annotation_path: Path to annotations
        class_profiles: Dict mapping class -> expression profile
        gene_names: List of gene names
        min_correlation: Minimum correlation to suggest assignment
    
    Returns:
        Dict mapping gene -> (suggested_class, correlation)
    """
    print("=" * 70)
    print("CORRELATING 'OTHER' GENES WITH CLASS PROFILES")
    print("=" * 70)
    print()
    
    annotations = load_annotations(annotation_path)
    annotations["target"] = annotations["target"].astype(str)
    
    # Get "Other" class genes
    other_genes = annotations[annotations["class"] == "Other"]["target"].tolist()
    
    # Compute pseudobulk for all perturbations
    import anndata as ad
    adata = ad.read_h5ad(adata_path)
    
    all_conditions = sorted(adata.obs["condition"].unique().tolist())
    dummy_split_config = {
        "train": all_conditions,
        "test": [],
        "val": [],
    }
    
    Y_df, _ = compute_pseudobulk_expression_changes(adata, dummy_split_config, seed=1)
    
    # Get perturbations for "Other" genes
    other_perts = annotations[annotations["class"] == "Other"]["target"].tolist()
    other_perts_in_data = [p for p in other_perts if p in Y_df.columns]
    
    print(f"Found {len(other_perts_in_data)} 'Other' perturbations in data")
    print(f"Comparing against {len(class_profiles)} class profiles")
    print()
    
    suggestions = {}
    
    # For each "Other" perturbation, compute correlation with each class profile
    for pert in other_perts_in_data:
        if pert not in Y_df.columns:
            continue
        
        pert_expr = Y_df[pert].values
        
        best_class = None
        best_corr = -1.0
        
        for class_name, class_profile in class_profiles.items():
            if class_name == "Other":
                continue
            
            # Compute correlation
            try:
                corr, pval = pearsonr(pert_expr, class_profile)
                
                if not np.isnan(corr) and corr > best_corr:
                    best_corr = corr
                    best_class = class_name
            except Exception as e:
                continue
        
        # Only suggest if correlation is above threshold
        if best_corr >= min_correlation:
            suggestions[pert] = (best_class, best_corr)
            print(f"  {pert:20s}: {best_class:20s} (r={best_corr:.3f})")
    
    print()
    print(f"Found {len(suggestions)} genes with correlation >= {min_correlation}")
    
    return suggestions


def improve_annotations_with_expression(
    adata_path: Path,
    annotation_path: Path,
    dataset_name: str,
    output_dir: Path,
    min_correlation: float = 0.3,
    min_perturbations: int = 2
) -> pd.DataFrame:
    """
    Improve annotations using expression similarity.
    
    Args:
        adata_path: Path to AnnData file
        annotation_path: Path to annotations
        dataset_name: Dataset name
        output_dir: Output directory
        min_correlation: Minimum correlation threshold
        min_perturbations: Minimum perturbations per class
    
    Returns:
        Improved annotations DataFrame
    """
    print("=" * 70)
    print(f"IMPROVING ANNOTATIONS WITH EXPRESSION: {dataset_name.upper()}")
    print("=" * 70)
    print()
    
    if not Path(adata_path).exists():
        print(f"⚠️  Data file not found: {adata_path}")
        return pd.DataFrame()
    
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
    
    # Compute class profiles
    class_profiles, gene_names = compute_class_profiles(
        Path(adata_path),
        annotation_path,
        min_perturbations=min_perturbations
    )
    
    if not class_profiles:
        print("⚠️  Could not compute class profiles")
        return annotations
    
    # Correlate "Other" genes with classes
    suggestions = correlate_other_genes_with_classes(
        Path(adata_path),
        annotation_path,
        class_profiles,
        gene_names,
        min_correlation=min_correlation
    )
    
    if not suggestions:
        print("⚠️  No suggestions found")
        return annotations
    
    # Apply suggestions
    improved = annotations.copy()
    reassignments = []
    
    print()
    print("Applying suggestions...")
    print()
    
    for gene, (suggested_class, correlation) in suggestions.items():
        mask = improved["target"] == gene
        if mask.sum() > 0:
            old_class = improved.loc[mask, "class"].values[0]
            improved.loc[mask, "class"] = suggested_class
            reassignments.append({
                "gene": gene,
                "old_class": old_class,
                "new_class": suggested_class,
                "correlation": correlation,
            })
            print(f"  {gene}: {old_class} → {suggested_class} (r={correlation:.3f})")
    
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
    output_file = output_dir / f"{annotation_path.stem}_improved_expression{annotation_path.suffix}"
    improved.to_csv(output_file, sep="\t", index=False)
    print(f"✓ Improved annotations saved to: {output_file}")
    
    # Save reassignment log
    if reassignments:
        reassignments_df = pd.DataFrame(reassignments)
        log_file = output_dir / f"reassignments_expression_{dataset_name}_{timestamp}.csv"
        reassignments_df.to_csv(log_file, index=False)
        print(f"✓ Reassignment log saved to: {log_file}")
    
    return improved


def main():
    """Improve annotations using expression similarity for all datasets."""
    
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
    
    all_improved = {}
    
    for dataset_name, config in datasets.items():
        if not Path(config["adata_path"]).exists():
            print(f"⚠️  Skipping {dataset_name}: data file not found")
            continue
        
        print("\n" + "=" * 70)
        print(f"DATASET: {dataset_name.upper()}")
        print("=" * 70)
        print()
        
        try:
            improved = improve_annotations_with_expression(
                Path(config["adata_path"]),
                config["annotation_path"],
                dataset_name,
                output_dir,
                min_correlation=0.3,
                min_perturbations=2
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
        if len(improved) > 0:
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


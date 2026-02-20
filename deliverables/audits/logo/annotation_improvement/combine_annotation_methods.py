#!/usr/bin/env python3
"""
Combine GO term and expression similarity annotation improvements.

This script:
1. Loads GO-based improved annotations
2. Loads expression similarity improved annotations
3. Combines them intelligently (prefer GO when available, use expression as fallback)
4. Creates final improved annotation files
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from shared.io import load_annotations


def combine_annotations(
    original_path: Path,
    go_improved_path: Path,
    expression_improved_path: Path,
    dataset_name: str,
    output_dir: Path
) -> pd.DataFrame:
    """
    Combine GO and expression similarity improvements.
    
    Strategy:
    1. Start with original annotations
    2. Apply GO-based reassignments (if available)
    3. Apply expression-based reassignments for genes not reassigned by GO
    4. Prefer GO when both methods suggest different classes
    """
    
    print("=" * 70)
    print(f"COMBINING ANNOTATION METHODS: {dataset_name.upper()}")
    print("=" * 70)
    print()
    
    # Load all annotation files
    original = load_annotations(original_path)
    original["target"] = original["target"].astype(str)
    
    # Try to load GO-improved (may not exist yet)
    if go_improved_path.exists():
        go_improved = load_annotations(go_improved_path)
        go_improved["target"] = go_improved["target"].astype(str)
        print(f"✓ Loaded GO-improved annotations: {len(go_improved)} genes")
    else:
        go_improved = None
        print("⚠️  GO-improved annotations not found, using expression only")
    
    # Load expression-improved (should always exist)
    if expression_improved_path.exists():
        expression_improved = load_annotations(expression_improved_path)
        expression_improved["target"] = expression_improved["target"].astype(str)
        print(f"✓ Loaded expression-improved annotations: {len(expression_improved)} genes")
    else:
        print("⚠️  Expression-improved annotations not found")
        return original
    
    print()
    
    # Start with original
    combined = original.copy()
    
    # Track reassignments
    reassignments = []
    
    # Find genes that changed in each method
    if go_improved is not None:
        go_changes = original.merge(go_improved, on="target", suffixes=("_orig", "_go"))
        go_changes = go_changes[go_changes["class_orig"] != go_changes["class_go"]]
        go_changes_dict = dict(zip(go_changes["target"], go_changes["class_go"]))
        print(f"GO-based reassignments: {len(go_changes_dict)} genes")
    else:
        go_changes_dict = {}
    
    expression_changes = original.merge(expression_improved, on="target", suffixes=("_orig", "_expr"))
    expression_changes = expression_changes[expression_changes["class_orig"] != expression_changes["class_expr"]]
    expression_changes_dict = dict(zip(expression_changes["target"], expression_changes["class_expr"]))
    print(f"Expression-based reassignments: {len(expression_changes_dict)} genes")
    print()
    
    # Apply reassignments: prefer GO, fallback to expression
    print("Applying reassignments (prefer GO, fallback to expression)...")
    print()
    
    for gene in original["target"]:
        original_class = original[original["target"] == gene]["class"].values[0]
        new_class = None
        method = None
        
        # Check GO first
        if gene in go_changes_dict:
            new_class = go_changes_dict[gene]
            method = "GO"
        # Fallback to expression
        elif gene in expression_changes_dict:
            new_class = expression_changes_dict[gene]
            method = "expression"
        
        # Apply if changed
        if new_class and new_class != original_class:
            mask = combined["target"] == gene
            combined.loc[mask, "class"] = new_class
            reassignments.append({
                "gene": gene,
                "original_class": original_class,
                "new_class": new_class,
                "method": method,
            })
    
    print(f"Total reassignments: {len(reassignments)}")
    
    # Count by method
    if reassignments:
        method_counts = pd.DataFrame(reassignments)["method"].value_counts()
        print("\nReassignments by method:")
        for method, count in method_counts.items():
            print(f"  {method:15s}: {count:4d} genes")
    
    print()
    
    # Summary
    original_other = (original["class"] == "Other").sum()
    combined_other = (combined["class"] == "Other").sum()
    reduction = original_other - combined_other
    
    print("=" * 70)
    print("COMBINED IMPROVEMENT SUMMARY")
    print("=" * 70)
    print()
    print(f"Original 'Other': {original_other} genes ({original_other/len(original)*100:.1f}%)")
    print(f"Combined 'Other': {combined_other} genes ({combined_other/len(combined)*100:.1f}%)")
    print(f"Reduction: {reduction} genes ({reduction/original_other*100:.1f}%)")
    print()
    
    # Save combined annotations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{original_path.stem}_improved_combined{original_path.suffix}"
    combined.to_csv(output_file, sep="\t", index=False)
    print(f"✓ Combined annotations saved to: {output_file}")
    
    # Save reassignment log
    if reassignments:
        reassignments_df = pd.DataFrame(reassignments)
        log_file = output_dir / f"reassignments_combined_{dataset_name}_{timestamp}.csv"
        reassignments_df.to_csv(log_file, index=False)
        print(f"✓ Reassignment log saved to: {log_file}")
    
    return combined


def main():
    """Combine annotation improvements for all datasets."""
    
    datasets = {
        "adamson": {
            "original": project_root / "data" / "annotations" / "adamson_functional_classes_enriched.tsv",
            "go_improved": project_root / "audits" / "logo" / "annotation_improvement" / "improved_annotations" / "adamson_functional_classes_enriched_improved_go.tsv",
            "expression_improved": project_root / "audits" / "logo" / "annotation_improvement" / "improved_annotations" / "adamson_functional_classes_enriched_improved_expression.tsv",
        },
        "k562": {
            "original": project_root / "data" / "annotations" / "replogle_k562_functional_classes_go.tsv",
            "go_improved": project_root / "audits" / "logo" / "annotation_improvement" / "improved_annotations" / "replogle_k562_functional_classes_go_improved_go.tsv",
            "expression_improved": project_root / "audits" / "logo" / "annotation_improvement" / "improved_annotations" / "replogle_k562_functional_classes_go_improved_expression.tsv",
        },
        "rpe1": {
            "original": project_root / "data" / "annotations" / "replogle_rpe1_functional_classes_go.tsv",
            "go_improved": project_root / "audits" / "logo" / "annotation_improvement" / "improved_annotations" / "replogle_rpe1_functional_classes_go_improved_go.tsv",
            "expression_improved": project_root / "audits" / "logo" / "annotation_improvement" / "improved_annotations" / "replogle_rpe1_functional_classes_go_improved_expression.tsv",
        },
    }
    
    output_dir = project_root / "audits" / "logo" / "annotation_improvement" / "improved_annotations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_combined = {}
    
    for dataset_name, config in datasets.items():
        print("\n" + "=" * 70)
        print(f"DATASET: {dataset_name.upper()}")
        print("=" * 70)
        print()
        
        try:
            combined = combine_annotations(
                config["original"],
                config["go_improved"],
                config["expression_improved"],
                dataset_name,
                output_dir
            )
            all_combined[dataset_name] = combined
        except Exception as e:
            print(f"Error combining {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print()
    
    for dataset_name, combined in all_combined.items():
        other_count = (combined["class"] == "Other").sum()
        total = len(combined)
        pct = other_count / total * 100
        print(f"{dataset_name:10s}: {other_count:4d} 'Other' genes ({pct:5.1f}%)")
    
    print()
    print("✓ Combined annotations ready for LOGO re-run")


if __name__ == "__main__":
    main()


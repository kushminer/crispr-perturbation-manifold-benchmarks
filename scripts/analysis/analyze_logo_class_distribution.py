#!/usr/bin/env python3
"""
Analyze class distribution in LOGO evaluation to assess impact of 'Other' class.

This script checks:
1. Class distribution (how many perturbations per class)
2. "Other" as percentage of training data
3. Expression similarity between "Other" and holdout class
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from shared.io import load_annotations

def analyze_class_distribution(annotation_path: Path, holdout_class: str = "Transcription"):
    """Analyze class distribution for LOGO evaluation."""
    
    print("=" * 70)
    print("LOGO CLASS DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print()
    
    # Load annotations
    annotations = load_annotations(annotation_path)
    annotations["target"] = annotations["target"].astype(str)
    
    print(f"Annotation file: {annotation_path}")
    print(f"Holdout class: {holdout_class}")
    print()
    
    # Class distribution
    class_counts = annotations["class"].value_counts().sort_index()
    print("=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)
    print(class_counts.to_string())
    print()
    
    total_perturbations = len(annotations)
    holdout_count = class_counts.get(holdout_class, 0)
    other_count = class_counts.get("Other", 0)
    
    print(f"Total perturbations: {total_perturbations}")
    print(f"Holdout class ({holdout_class}): {holdout_count} ({holdout_count/total_perturbations*100:.1f}%)")
    print(f"Other class: {other_count} ({other_count/total_perturbations*100:.1f}%)")
    print()
    
    # Training set composition
    train_classes = class_counts[class_counts.index != holdout_class]
    train_total = train_classes.sum()
    
    print("=" * 70)
    print("TRAINING SET COMPOSITION (excluding holdout)")
    print("=" * 70)
    print(f"Total training perturbations: {train_total}")
    print()
    
    for class_name, count in train_classes.items():
        pct = count / train_total * 100
        print(f"  {class_name:30s}: {count:4d} ({pct:5.1f}%)")
    
    print()
    
    # "Other" impact
    if "Other" in train_classes.index:
        other_pct = train_classes["Other"] / train_total * 100
        print("=" * 70)
        print("⚠️  'OTHER' CLASS IMPACT")
        print("=" * 70)
        print(f"'Other' represents {other_pct:.1f}% of training data")
        print()
        
        if other_pct > 30:
            print("⚠️  WARNING: 'Other' class dominates training set (>30%)")
            print("   This may inflate performance on holdout class.")
            print("   Recommendation: Run ablation study excluding 'Other'.")
        elif other_pct > 20:
            print("⚠️  CAUTION: 'Other' class is substantial (>20%)")
            print("   Consider analyzing impact via ablation study.")
        else:
            print("✓ 'Other' class is relatively small (<20%)")
            print("   Impact is likely minimal, but ablation study still recommended.")
    
    print()
    
    # Class diversity
    print("=" * 70)
    print("CLASS DIVERSITY")
    print("=" * 70)
    print(f"Number of training classes: {len(train_classes)}")
    print(f"Average perturbations per class: {train_total / len(train_classes):.1f}")
    print(f"Largest training class: {train_classes.idxmax()} ({train_classes.max()} perts)")
    print(f"Smallest training class: {train_classes.idxmin()} ({train_classes.min()} perts)")
    
    # Imbalance ratio
    if len(train_classes) > 1:
        imbalance_ratio = train_classes.max() / train_classes.min()
        print(f"Class imbalance ratio: {imbalance_ratio:.1f}x")
        if imbalance_ratio > 5:
            print("⚠️  WARNING: High class imbalance (>5x)")
            print("   Largest class may dominate model training.")
    
    print()
    
    return {
        "total": total_perturbations,
        "holdout_count": holdout_count,
        "other_count": other_count,
        "train_total": train_total,
        "other_pct": other_pct if "Other" in train_classes.index else 0,
        "class_counts": class_counts,
        "train_classes": train_classes,
    }


def main():
    """Run analysis for all datasets."""
    
    datasets = {
        "adamson": project_root / "data" / "annotations" / "adamson_functional_classes_enriched.tsv",
        "k562": project_root / "data" / "annotations" / "replogle_k562_functional_classes_go.tsv",
        "rpe1": project_root / "data" / "annotations" / "replogle_rpe1_functional_classes_go.tsv",
    }
    
    results = {}
    
    for dataset_name, annotation_path in datasets.items():
        if not annotation_path.exists():
            print(f"⚠️  Skipping {dataset_name}: {annotation_path} not found")
            continue
        
        print("\n" + "=" * 70)
        print(f"DATASET: {dataset_name.upper()}")
        print("=" * 70)
        print()
        
        try:
            stats = analyze_class_distribution(annotation_path, holdout_class="Transcription")
            results[dataset_name] = stats
        except Exception as e:
            print(f"❌ Error analyzing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    if results:
        print(f"{'Dataset':<15} {'Total':<8} {'Holdout':<8} {'Other':<8} {'Other%':<8} {'Train':<8}")
        print("-" * 70)
        for dataset_name, stats in results.items():
            print(f"{dataset_name:<15} {stats['total']:<8} {stats['holdout_count']:<8} "
                  f"{stats['other_count']:<8} {stats['other_pct']:<7.1f}% {stats['train_total']:<8}")
    
    print()
    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print()
    print("1. If 'Other' > 30%: Run ablation study excluding 'Other' from training")
    print("2. If 'Other' > 20%: Analyze expression similarity between 'Other' and holdout class")
    print("3. Consider reporting LOGO results with and without 'Other' for transparency")
    print("4. Improve annotations to reduce 'Other' class size in future work")
    print()


if __name__ == "__main__":
    main()


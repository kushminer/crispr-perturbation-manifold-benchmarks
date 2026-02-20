#!/usr/bin/env python3
"""
Fix Report Aggregation and Regenerate 5-Epic Winner Grid

This script:
1. Fixes baseline naming (removes dataset prefix)
2. Fixes column lookups in cross-epic summary
3. Properly aggregates all 5 epics
4. Regenerates the 5-epic winner grid
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

RESULTS_DIR = BASE_DIR / "results" / "manifold_law_diagnostics"
OUTPUT_DIR = Path(__file__).parent

# Canonical baseline names (without dataset prefix)
CANONICAL_BASELINES = [
    "lpm_selftrained",
    "lpm_randomGeneEmb",
    "lpm_randomPertEmb",
    "lpm_scgptGeneEmb",
    "lpm_scFoundationGeneEmb",
    "lpm_gearsPertEmb",
    "lpm_k562PertEmb",
    "lpm_rpe1PertEmb",
]

# Display names for plotting
BASELINE_DISPLAY_NAMES = {
    "lpm_selftrained": "PCA (Self-trained)",
    "lpm_randomGeneEmb": "Random Gene Emb.",
    "lpm_randomPertEmb": "Random Pert. Emb.",
    "lpm_scgptGeneEmb": "scGPT Gene Emb.",
    "lpm_scFoundationGeneEmb": "scFoundation Gene Emb.",
    "lpm_gearsPertEmb": "GEARS (GO Graph)",
    "lpm_k562PertEmb": "K562 Pert. Emb.",
    "lpm_rpe1PertEmb": "RPE1 Pert. Emb.",
}


def normalize_baseline_name(name: str) -> str:
    """Remove dataset prefix from baseline name."""
    if not name or pd.isna(name):
        return None
    
    name = str(name).strip()
    
    # Skip invalid entries
    if name in ["results", "baseline", ""]:
        return None
    
    # Remove dataset prefix if present
    for prefix in ["adamson_", "k562_", "rpe1_", "replogle_k562_essential_", "replogle_rpe1_essential_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    
    return name if name in CANONICAL_BASELINES else None


def load_epic1_curvature() -> pd.DataFrame:
    """Load Epic 1 curvature metrics."""
    epic1_dir = RESULTS_DIR / "epic1_curvature"
    all_data = []
    
    for csv_file in epic1_dir.glob("curvature_sweep_summary_*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                continue
            
            # Extract dataset and baseline from filename
            parts = csv_file.stem.replace("curvature_sweep_summary_", "").split("_", 1)
            if len(parts) == 2:
                dataset, baseline = parts
                df["dataset"] = dataset
                df["baseline"] = normalize_baseline_name(baseline)
                all_data.append(df)
        except Exception as e:
            print(f"  Warning: Error loading {csv_file.name}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined[combined["baseline"].notna()]
    
    return combined


def load_epic2_ablation() -> pd.DataFrame:
    """Load Epic 2 mechanism ablation metrics."""
    epic2_dir = RESULTS_DIR / "epic2_mechanism_ablation"
    all_data = []
    
    for csv_file in epic2_dir.glob("mechanism_ablation_*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                continue
            
            # Normalize baseline name
            if "baseline_type" in df.columns:
                df["baseline"] = df["baseline_type"].apply(normalize_baseline_name)
            elif "baseline" in df.columns:
                df["baseline"] = df["baseline"].apply(normalize_baseline_name)
            
            # Keep only rows with valid delta_r (ablation actually ran)
            if "delta_r" in df.columns:
                df = df[df["delta_r"].notna()]
            
            all_data.append(df)
        except Exception as e:
            print(f"  Warning: Error loading {csv_file.name}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined[combined["baseline"].notna()]
    
    return combined


def load_epic3_lipschitz() -> pd.DataFrame:
    """Load Epic 3 Lipschitz metrics."""
    epic3_dir = RESULTS_DIR / "epic3_noise_injection"
    epic1_dir = RESULTS_DIR / "epic1_curvature"
    
    # First, get baseline r from Epic 1 (k-sweep) for each baseline × dataset
    baseline_r_from_k_sweep = {}
    for csv_file in epic1_dir.glob("curvature_sweep_summary_*.csv"):
        parts = csv_file.stem.replace("curvature_sweep_summary_", "").split("_", 1)
        if len(parts) == 2:
            dataset, baseline = parts
            df = pd.read_csv(csv_file)
            # Get r at k=5 (typical baseline)
            k5_data = df[df["k"] == 5]
            if len(k5_data) > 0:
                baseline_r_from_k_sweep[(dataset, normalize_baseline_name(baseline))] = k5_data["mean_r"].iloc[0]
    
    # Try to load complete aggregated summary first
    summary_path = epic3_dir / "lipschitz_summary_complete.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        df["baseline"] = df["baseline"].apply(normalize_baseline_name)
        return df[df["baseline"].notna()]
    
    # Fallback to original summary
    summary_path = epic3_dir / "lipschitz_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        df["baseline"] = df["baseline"].apply(normalize_baseline_name)
        return df[df["baseline"].notna()]
    
    # Otherwise, aggregate from noise injection files
    all_data = []
    
    for csv_file in epic3_dir.glob("noise_injection_*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                continue
            
            parts = csv_file.stem.replace("noise_injection_", "").split("_", 1)
            if len(parts) != 2:
                continue
            dataset, baseline = parts
            
            # Compute Lipschitz per k
            for k in df["k"].unique():
                k_data = df[df["k"] == k].sort_values("noise_level")
                baseline_row = k_data[k_data["noise_level"] == 0.0]
                if len(baseline_row) == 0:
                    continue
                
                r_baseline = baseline_row["mean_r"].iloc[0]
                noisy_data = k_data[k_data["noise_level"] > 0]
                
                if len(noisy_data) == 0:
                    continue
                
                noise_levels = noisy_data["noise_level"].values
                r_values = noisy_data["mean_r"].values
                delta_r = r_baseline - r_values
                sensitivity = np.abs(delta_r) / noise_levels
                lipschitz = np.max(sensitivity) if len(sensitivity) > 0 else np.nan
                
                all_data.append({
                    "dataset": dataset,
                    "baseline": normalize_baseline_name(baseline),
                    "k": k,
                    "baseline_r": r_baseline,
                    "lipschitz_constant": lipschitz,
                })
        except Exception as e:
            print(f"  Warning: Error processing {csv_file.name}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    combined = pd.DataFrame(all_data)
    combined = combined[combined["baseline"].notna()]
    
    return combined


def load_epic4_flip() -> pd.DataFrame:
    """Load Epic 4 direction-flip metrics."""
    epic4_dir = RESULTS_DIR / "epic4_direction_flip"
    all_data = []
    
    for csv_file in epic4_dir.glob("direction_flip_*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                continue
            
            # Normalize baseline name
            if "baseline" in df.columns:
                df["baseline"] = df["baseline"].apply(normalize_baseline_name)
            
            all_data.append(df)
        except Exception as e:
            print(f"  Warning: Error loading {csv_file.name}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined[combined["baseline"].notna()]
    
    return combined


def load_epic5_tangent() -> pd.DataFrame:
    """Load Epic 5 tangent alignment metrics."""
    epic5_dir = RESULTS_DIR / "epic5_tangent_alignment"
    all_data = []
    
    for csv_file in epic5_dir.glob("tangent_alignment_*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                continue
            
            # Normalize baseline name
            if "baseline" in df.columns:
                df["baseline"] = df["baseline"].apply(normalize_baseline_name)
            
            all_data.append(df)
        except Exception as e:
            print(f"  Warning: Error loading {csv_file.name}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined[combined["baseline"].notna()]
    
    return combined


def create_cross_epic_summary(
    epic1: pd.DataFrame,
    epic2: pd.DataFrame,
    epic3: pd.DataFrame,
    epic4: pd.DataFrame,
    epic5: pd.DataFrame,
) -> pd.DataFrame:
    """Create unified baseline summary across all epics."""
    
    results = []
    
    for baseline in CANONICAL_BASELINES:
        row = {"baseline": baseline}
        
        # Epic 1: Peak r (find max mean_r across k values)
        if len(epic1) > 0:
            e1 = epic1[epic1["baseline"] == baseline]
            if len(e1) > 0:
                # Peak r is the maximum mean_r across all k values
                if "mean_r" in e1.columns:
                    row["epic1_peak_r"] = e1["mean_r"].max()
                elif "peak_r" in e1.columns:
                    row["epic1_peak_r"] = e1["peak_r"].mean()
                
                # Curvature: computed as correlation between k and r (negative = good)
                if "mean_r" in e1.columns and "k" in e1.columns:
                    r_values = e1.groupby("k")["mean_r"].mean()
                    if len(r_values) > 2:
                        # Fit linear trend: negative slope means r decreases as k increases
                        k_values = r_values.index.values.astype(float)
                        r_vals = r_values.values
                        if len(k_values) > 1:
                            slope = np.polyfit(k_values, r_vals, 1)[0]
                            row["epic1_curvature"] = slope
        
        # Epic 2: Delta r (ablation effect)
        if len(epic2) > 0:
            e2 = epic2[epic2["baseline"] == baseline]
            if len(e2) > 0:
                if "delta_r" in e2.columns:
                    row["epic2_delta_r"] = e2["delta_r"].mean()
                if "original_pearson_r" in e2.columns:
                    row["epic2_original_r"] = e2["original_pearson_r"].mean()
        
        # Epic 3: Lipschitz constant
        if len(epic3) > 0:
            e3 = epic3[epic3["baseline"] == baseline]
            if len(e3) > 0:
                row["epic3_lipschitz"] = e3["lipschitz_constant"].mean()
                if "baseline_r" in e3.columns:
                    row["epic3_baseline_r"] = e3["baseline_r"].mean()
        
        # Epic 4: Flip rate
        if len(epic4) > 0:
            e4 = epic4[epic4["baseline"] == baseline]
            if len(e4) > 0:
                if "adversarial_rate" in e4.columns:
                    row["epic4_flip_rate"] = e4["adversarial_rate"].mean()
                elif "mean_adversarial_rate" in e4.columns:
                    row["epic4_flip_rate"] = e4["mean_adversarial_rate"].mean()
        
        # Epic 5: Tangent alignment score
        if len(epic5) > 0:
            e5 = epic5[epic5["baseline"] == baseline]
            if len(e5) > 0:
                if "tangent_alignment_score" in e5.columns:
                    row["epic5_tas"] = e5["tangent_alignment_score"].mean()
                elif "mean_tas" in e5.columns:
                    row["epic5_tas"] = e5["mean_tas"].mean()
        
        results.append(row)
    
    return pd.DataFrame(results)


def create_5epic_winner_grid(summary: pd.DataFrame, output_path: Path):
    """Create the 5-epic winner grid visualization."""
    
    # Define metrics for each epic (higher is better for some, lower for others)
    epic_configs = [
        ("epic1_peak_r", "Epic 1: Curvature\n(Peak r)", True),   # Higher is better
        ("epic2_delta_r", "Epic 2: Ablation\n(Δr)", True),        # Higher is better (more drop = more biological)
        ("epic3_lipschitz", "Epic 3: Noise\n(Lipschitz)", False), # Lower is better (more robust)
        ("epic4_flip_rate", "Epic 4: Flip\n(Rate)", False),       # Lower is better
        ("epic5_tas", "Epic 5: Tangent\n(TAS)", True),            # Higher is better
    ]
    
    # Get baselines with data
    baselines_with_data = summary[summary[[col for col, _, _ in epic_configs if col in summary.columns]].notna().any(axis=1)]["baseline"].tolist()
    
    if not baselines_with_data:
        print("⚠️  No baselines with complete data for 5-epic grid")
        return
    
    # Create figure
    n_baselines = len(baselines_with_data)
    n_epics = len(epic_configs)
    
    fig, axes = plt.subplots(n_epics, 1, figsize=(12, 2 * n_epics))
    if n_epics == 1:
        axes = [axes]
    
    for ax, (metric, title, higher_is_better) in zip(axes, epic_configs):
        if metric not in summary.columns:
            ax.text(0.5, 0.5, f"{title}\n(No Data)", ha="center", va="center", fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            continue
        
        # Get data for this metric
        data = summary[summary["baseline"].isin(baselines_with_data)][["baseline", metric]].copy()
        data = data.dropna()
        
        if len(data) == 0:
            ax.text(0.5, 0.5, f"{title}\n(No Data)", ha="center", va="center", fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            continue
        
        # Sort by metric value
        data = data.sort_values(metric, ascending=not higher_is_better)
        
        # Determine winner
        winner = data.iloc[0]["baseline"]
        
        # Create bar plot
        colors = ["green" if b == winner else "steelblue" for b in data["baseline"]]
        
        x = range(len(data))
        ax.barh(x, data[metric].values, color=colors)
        
        # Add baseline labels
        display_names = [BASELINE_DISPLAY_NAMES.get(b, b) for b in data["baseline"]]
        ax.set_yticks(x)
        ax.set_yticklabels(display_names)
        
        # Add values on bars
        for i, (_, row) in enumerate(data.iterrows()):
            val = row[metric]
            if not np.isnan(val):
                ax.text(val + 0.01, i, f"{val:.3f}", va="center", fontsize=9)
        
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Value")
        
        # Mark winner
        ax.text(1.02, 0.5, f"Winner:\n{BASELINE_DISPLAY_NAMES.get(winner, winner)}", 
                transform=ax.transAxes, fontsize=9, va="center", color="green", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✅ Saved 5-epic winner grid: {output_path}")


def main():
    print("=" * 60)
    print("FIX AGGREGATION AND REGENERATE 5-EPIC GRID")
    print("=" * 60)
    print()
    
    # Load all epic data
    print("Loading Epic 1 (Curvature)...")
    epic1 = load_epic1_curvature()
    print(f"  Loaded {len(epic1)} rows")
    
    print("Loading Epic 2 (Ablation)...")
    epic2 = load_epic2_ablation()
    print(f"  Loaded {len(epic2)} rows")
    
    print("Loading Epic 3 (Lipschitz)...")
    epic3 = load_epic3_lipschitz()
    print(f"  Loaded {len(epic3)} rows")
    
    print("Loading Epic 4 (Flip)...")
    epic4 = load_epic4_flip()
    print(f"  Loaded {len(epic4)} rows")
    
    print("Loading Epic 5 (Tangent)...")
    epic5 = load_epic5_tangent()
    print(f"  Loaded {len(epic5)} rows")
    
    print()
    
    # Create cross-epic summary
    print("Creating cross-epic summary...")
    summary = create_cross_epic_summary(epic1, epic2, epic3, epic4, epic5)
    
    # Save summary
    summary_path = OUTPUT_DIR / "final_tables" / "baseline_summary_fixed.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"✅ Saved baseline summary: {summary_path}")
    
    # Show summary
    print()
    print("Cross-Epic Summary:")
    print(summary.to_string(index=False))
    print()
    
    # Create 5-epic winner grid
    print("Creating 5-epic winner grid...")
    grid_path = OUTPUT_DIR / "poster_figures" / "5epic_winner_grid_fixed.png"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    create_5epic_winner_grid(summary, grid_path)
    
    print()
    print("=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()


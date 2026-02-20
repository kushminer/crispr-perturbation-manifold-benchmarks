#!/usr/bin/env python3
"""
Extract LSFT and LOGO results from skeleton markdown files and create CSV files.
"""

import re
import csv
from pathlib import Path

def parse_lsft_table(lines, dataset_name):
    """Parse LSFT performance table from markdown."""
    results = []
    in_table = False
    
    for line in lines:
        # Check if we're in the performance table
        if "| Baseline | Top % | Local Pearson r |" in line:
            in_table = True
            continue
        
        # Stop at next section
        if in_table and line.startswith("####"):
            break
        
        if in_table and line.startswith("|") and "|" in line[1:]:
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 10 and parts[0] != "Baseline" and parts[0] != "---":
                baseline = parts[0].strip()
                top_pct_str = parts[1].strip().replace("%", "")
                try:
                    top_k = float(top_pct_str) / 100.0  # Convert percentage to decimal
                    local_r = float(parts[2].strip())
                    baseline_r = float(parts[3].strip())
                    local_l2 = float(parts[5].strip())
                    baseline_l2 = float(parts[6].strip())
                    mean_similarity = float(parts[9].strip())
                    
                    results.append({
                        "dataset": dataset_name,
                        "baseline": baseline,
                        "top_k": top_k,
                        "local_r": local_r,
                        "baseline_r": baseline_r,
                        "local_l2": local_l2,
                        "baseline_l2": baseline_l2,
                        "mean_similarity": mean_similarity
                    })
                except (ValueError, IndexError) as e:
                    continue
    
    return results

def parse_logo_table(lines, dataset_name):
    """Parse LOGO performance table from markdown."""
    results = []
    in_logo_section = False
    in_table = False
    dataset_found = False
    
    for i, line in enumerate(lines):
        # Check if we're in the LOGO Performance section
        if "### LOGO Performance" in line or "LOGO Performance (Transcription class held out)" in line:
            in_logo_section = True
            continue
        
        # Stop if we hit the next major section after LOGO
        if in_logo_section and line.startswith("### ") and "LOGO" not in line:
            break
        
        # Check if we're in the LOGO table for this dataset
        if in_logo_section:
            # Match dataset name (case-insensitive)
            dataset_patterns = [
                f"#### {dataset_name} Dataset",
                f"#### {dataset_name.capitalize()} Dataset",
                f"#### {dataset_name.upper()} Dataset"
            ]
            
            if any(pattern in line for pattern in dataset_patterns):
                dataset_found = True
                # Look for the table header in next few lines
                for j in range(i+1, min(i+10, len(lines))):
                    if "| Rank | Baseline | Pearson r (95% CI) |" in lines[j]:
                        in_table = True
                        start_idx = j + 1
                        break
                
                if in_table:
                    for j in range(start_idx, min(start_idx+20, len(lines))):
                        table_line = lines[j]
                        # Stop if we hit the next section
                        if table_line.startswith("####"):
                            break
                        if table_line.startswith("|") and "|" in table_line[1:]:
                            parts = [p.strip() for p in table_line.split("|")[1:-1]]
                            if len(parts) >= 4 and parts[0] != "Rank" and parts[0] != "---" and not parts[0].startswith("------"):
                                baseline = parts[1].strip()
                                
                                # Parse Pearson r with CI: "0.882 [0.842, 0.924]"
                                r_str = parts[2].strip()
                                r_match = re.match(r'([-\d.]+)\s*\[([-\d.]+),\s*([-\d.]+)\]', r_str)
                                if r_match:
                                    r_mean = float(r_match.group(1))
                                    r_ci_low = float(r_match.group(2))
                                    r_ci_high = float(r_match.group(3))
                                else:
                                    continue
                                
                                # Parse L2 with CI: "4.36 [2.88, 5.45]"
                                l2_str = parts[3].strip()
                                l2_match = re.match(r'([-\d.]+)\s*\[([-\d.]+),\s*([-\d.]+)\]', l2_str)
                                if l2_match:
                                    l2_mean = float(l2_match.group(1))
                                    l2_ci_low = float(l2_match.group(2))
                                    l2_ci_high = float(l2_match.group(3))
                                else:
                                    continue
                                
                                # Map baseline names (remove "lpm_" prefix if present, handle "selftrained" vs "lpm_selftrained")
                                if baseline == "selftrained":
                                    baseline = "lpm_selftrained"
                                elif baseline == "mean_response":
                                    baseline = "mean_response"
                                elif not baseline.startswith("lpm_") and baseline != "mean_response":
                                    baseline = f"lpm_{baseline}"
                                
                                results.append({
                                    "dataset": dataset_name.lower(),
                                    "baseline": baseline,
                                    "r_mean": r_mean,
                                    "r_ci_low": r_ci_low,
                                    "r_ci_high": r_ci_high,
                                    "l2_mean": l2_mean,
                                    "l2_ci_low": l2_ci_low,
                                    "l2_ci_high": l2_ci_high
                                })
                    break
    
    return results

def main():
    base_dir = Path(__file__).parent
    
    # Read LSFT skeleton
    lsft_file = base_dir / "lsft_analysis_skeleton.md"
    with open(lsft_file, 'r') as f:
        lsft_lines = f.readlines()
    
    # Extract LSFT data for all datasets
    lsft_results = []
    lsft_results.extend(parse_lsft_table(lsft_lines, "adamson"))
    lsft_results.extend(parse_lsft_table(lsft_lines, "k562"))
    lsft_results.extend(parse_lsft_table(lsft_lines, "rpe1"))
    
    # Write LSFT CSV
    lsft_csv = base_dir / "LSFT_results.csv"
    with open(lsft_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "baseline", "top_k", "local_r", "baseline_r", 
            "local_l2", "baseline_l2", "mean_similarity"
        ])
        writer.writeheader()
        writer.writerows(lsft_results)
    
    print(f"Created {lsft_csv} with {len(lsft_results)} rows")
    
    # Read LOGO skeleton
    logo_file = base_dir / "RESAMPLING_FINDINGS_REPORT_SKELETON.md"
    with open(logo_file, 'r') as f:
        logo_lines = f.readlines()
    
    # Extract LOGO data for all datasets
    logo_results = []
    logo_results.extend(parse_logo_table(logo_lines, "Adamson"))
    logo_results.extend(parse_logo_table(logo_lines, "K562"))
    logo_results.extend(parse_logo_table(logo_lines, "RPE1"))
    
    # Normalize dataset names to lowercase
    for r in logo_results:
        r["dataset"] = r["dataset"].lower()
    
    # Write LOGO CSV
    logo_csv = base_dir / "LOGO_results.csv"
    with open(logo_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "baseline", "r_mean", "r_ci_low", "r_ci_high",
            "l2_mean", "l2_ci_low", "l2_ci_high"
        ])
        writer.writeheader()
        writer.writerows(logo_results)
    
    print(f"Created {logo_csv} with {len(logo_results)} rows")

if __name__ == "__main__":
    main()


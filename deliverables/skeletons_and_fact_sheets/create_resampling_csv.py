#!/usr/bin/env python3
"""
Extract LSFT and LOGO resampling results (with confidence intervals) from skeleton markdown files and create CSV files.
"""

import re
import csv
from pathlib import Path

def parse_lsft_resampling_table(lines):
    """Parse LSFT resampling performance table (with CIs) from markdown."""
    results = []
    in_lsft_section = False
    current_dataset = None
    
    for i, line in enumerate(lines):
        # Check if we're in the LSFT Performance section
        if "### LSFT Performance (top_pct=0.05)" in line:
            in_lsft_section = True
            continue
        
        # Stop if we hit the next major section
        if in_lsft_section and line.startswith("### ") and "LSFT" not in line:
            break
        
        # Check for dataset headers
        if in_lsft_section:
            if "#### Adamson Dataset" in line:
                current_dataset = "adamson"
                continue
            elif "#### K562 Dataset" in line:
                current_dataset = "k562"
                continue
            elif "#### RPE1 Dataset" in line:
                current_dataset = "rpe1"
                continue
            
            # Look for table rows
            if current_dataset and line.startswith("|") and "|" in line[1:]:
                parts = [p.strip() for p in line.split("|")[1:-1]]
                if len(parts) >= 4 and parts[0] not in ["Rank", "---", ""] and not parts[0].startswith("------"):
                    baseline = parts[1].strip()
                    
                    # Parse Pearson r with CI: "0.941 [0.900, 0.966]"
                    r_str = parts[2].strip()
                    r_match = re.match(r'([-\d.]+)\s*\[([-\d.]+),\s*([-\d.]+)\]', r_str)
                    if r_match:
                        r_mean = float(r_match.group(1))
                        r_ci_low = float(r_match.group(2))
                        r_ci_high = float(r_match.group(3))
                    else:
                        continue
                    
                    # Parse L2 with CI: "2.09 [1.66, 2.70]"
                    l2_str = parts[3].strip()
                    l2_match = re.match(r'([-\d.]+)\s*\[([-\d.]+),\s*([-\d.]+)\]', l2_str)
                    if l2_match:
                        l2_mean = float(l2_match.group(1))
                        l2_ci_low = float(l2_match.group(2))
                        l2_ci_high = float(l2_match.group(3))
                    else:
                        continue
                    
                    # Map baseline names
                    if baseline == "selftrained":
                        baseline = "lpm_selftrained"
                    elif baseline == "mean_response":
                        baseline = "mean_response"
                    elif not baseline.startswith("lpm_"):
                        baseline = f"lpm_{baseline}"
                    
                    results.append({
                        "dataset": current_dataset.lower(),
                        "baseline": baseline,
                        "top_k": 0.05,  # Resampling report only has top_pct=0.05
                        "r_mean": r_mean,
                        "r_ci_low": r_ci_low,
                        "r_ci_high": r_ci_high,
                        "l2_mean": l2_mean,
                        "l2_ci_low": l2_ci_low,
                        "l2_ci_high": l2_ci_high
                    })
    
    return results

def parse_logo_resampling_table(lines):
    """Parse LOGO resampling performance table (with CIs) from markdown."""
    results = []
    in_logo_section = False
    current_dataset = None
    
    for i, line in enumerate(lines):
        # Check if we're in the LOGO Performance section
        if "### LOGO Performance (Transcription class held out)" in line:
            in_logo_section = True
            continue
        
        # Stop if we hit the next major section
        if in_logo_section and line.startswith("### ") and "LOGO" not in line:
            break
        
        # Check for dataset headers
        if in_logo_section:
            if "#### Adamson Dataset" in line:
                current_dataset = "adamson"
                continue
            elif "#### K562 Dataset" in line:
                current_dataset = "k562"
                continue
            elif "#### RPE1 Dataset" in line:
                current_dataset = "rpe1"
                continue
            
            # Look for table rows
            if current_dataset and line.startswith("|") and "|" in line[1:]:
                parts = [p.strip() for p in line.split("|")[1:-1]]
                if len(parts) >= 4 and parts[0] not in ["Rank", "---", ""] and not parts[0].startswith("------"):
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
                    
                    # Map baseline names
                    if baseline == "selftrained":
                        baseline = "lpm_selftrained"
                    elif baseline == "mean_response":
                        baseline = "mean_response"
                    elif not baseline.startswith("lpm_"):
                        baseline = f"lpm_{baseline}"
                    
                    results.append({
                        "dataset": current_dataset.lower(),
                        "baseline": baseline,
                        "r_mean": r_mean,
                        "r_ci_low": r_ci_low,
                        "r_ci_high": r_ci_high,
                        "l2_mean": l2_mean,
                        "l2_ci_low": l2_ci_low,
                        "l2_ci_high": l2_ci_high
                    })
    
    return results

def main():
    base_dir = Path(__file__).parent
    
    # Read resampling skeleton
    resampling_file = base_dir / "RESAMPLING_FINDINGS_REPORT_SKELETON.md"
    with open(resampling_file, 'r') as f:
        resampling_lines = f.readlines()
    
    # Extract LSFT resampling data
    lsft_resampling_results = parse_lsft_resampling_table(resampling_lines)
    
    # Write LSFT resampling CSV
    lsft_resampling_csv = base_dir / "LSFT_resampling.csv"
    with open(lsft_resampling_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "baseline", "top_k", "r_mean", "r_ci_low", "r_ci_high",
            "l2_mean", "l2_ci_low", "l2_ci_high"
        ])
        writer.writeheader()
        writer.writerows(lsft_resampling_results)
    
    print(f"Created {lsft_resampling_csv} with {len(lsft_resampling_results)} rows")
    
    # Extract LOGO resampling data
    logo_resampling_results = parse_logo_resampling_table(resampling_lines)
    
    # Write LOGO resampling CSV
    logo_resampling_csv = base_dir / "LOGO_resampling.csv"
    with open(logo_resampling_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "baseline", "r_mean", "r_ci_low", "r_ci_high",
            "l2_mean", "l2_ci_low", "l2_ci_high"
        ])
        writer.writeheader()
        writer.writerows(logo_resampling_results)
    
    print(f"Created {logo_resampling_csv} with {len(logo_resampling_results)} rows")

if __name__ == "__main__":
    main()


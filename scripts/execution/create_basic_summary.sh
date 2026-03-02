#!/bin/bash
# Create a basic summary of results without Python dependencies

RESULTS_DIR="results/manifold_law_diagnostics"
OUTPUT_FILE="results/manifold_law_diagnostics/summary_reports/BASIC_SUMMARY.txt"

echo "Manifold Law Diagnostic Suite - Basic Summary" > "$OUTPUT_FILE"
echo "Generated: $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "=== Epic 1: Curvature Sweep ===" >> "$OUTPUT_FILE"
echo "Files: $(find $RESULTS_DIR/epic1_curvature -name "*.csv" | wc -l | tr -d ' ')" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "=== Epic 2: Mechanism Ablation ===" >> "$OUTPUT_FILE"
echo "Files: $(find $RESULTS_DIR/epic2_mechanism_ablation -name "mechanism_ablation_*.csv" | wc -l | tr -d ' ')" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "=== Epic 3: Noise Injection ===" >> "$OUTPUT_FILE"
echo "Files: $(find $RESULTS_DIR/epic3_noise_injection -name "noise_injection_*.csv" | wc -l | tr -d ' ')" >> "$OUTPUT_FILE"
echo "Analysis file: $([ -f $RESULTS_DIR/epic3_noise_injection/noise_sensitivity_analysis.csv ] && echo "YES" || echo "NO")" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "=== Epic 4: Direction-Flip Probe ===" >> "$OUTPUT_FILE"
echo "Files: $(find $RESULTS_DIR/epic4_direction_flip -name "*.csv" | wc -l | tr -d ' ')" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "=== Epic 5: Tangent Alignment ===" >> "$OUTPUT_FILE"
echo "Files: $(find $RESULTS_DIR/epic5_tangent_alignment -name "*.csv" | wc -l | tr -d ' ')" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "=== Total Files ===" >> "$OUTPUT_FILE"
echo "Total CSV files: $(find $RESULTS_DIR -name "*.csv" | wc -l | tr -d ' ')" >> "$OUTPUT_FILE"

echo "✅ Basic summary created: $OUTPUT_FILE"

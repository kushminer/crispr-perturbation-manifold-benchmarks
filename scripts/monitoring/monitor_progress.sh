#!/bin/bash
# Monitor progress of full diagnostic suite run

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RESULTS_DIR="results/manifold_law_diagnostics"
LOG_FILE="full_diagnostic_suite_run.log"

echo "============================================================"
echo "Diagnostic Suite Progress Monitor"
echo "============================================================"
echo ""

# Count completed results
echo "=== Epic 1: Curvature Sweep ==="
epic1_count=$(find "$RESULTS_DIR/epic1_curvature" -name "curvature_sweep_summary_*.csv" 2>/dev/null | wc -l | tr -d ' ')
echo "  Completed: $epic1_count / 24 (8 baselines × 3 datasets)"
echo ""

echo "=== Epic 2: Mechanism Ablation ==="
epic2_count=$(find "$RESULTS_DIR/epic2_mechanism_ablation" -name "mechanism_ablation_*.csv" 2>/dev/null | grep -v "/original/" | wc -l | tr -d ' ')
echo "  Completed: $epic2_count / 24"
echo ""

echo "=== Epic 3: Noise Injection ==="
epic3_count=$(find "$RESULTS_DIR/epic3_noise_injection" -name "noise_injection_*.csv" 2>/dev/null | wc -l | tr -d ' ')
echo "  Completed: $epic3_count / 24"
echo "  Baseline entries: $(find "$RESULTS_DIR/epic3_noise_injection/baseline" -name "*.csv" 2>/dev/null | wc -l | tr -d ' ')"
echo ""

echo "=== Epic 4: Direction-Flip Probe ==="
epic4_count=$(find "$RESULTS_DIR/epic4_direction_flip" -name "direction_flip_probe_*.csv" 2>/dev/null | grep -v "_results.csv" | wc -l | tr -d ' ')
echo "  Completed: $epic4_count / 24"
echo ""

echo "=== Epic 5: Tangent Alignment ==="
epic5_count=$(find "$RESULTS_DIR/epic5_tangent_alignment" -name "tangent_alignment_*.csv" 2>/dev/null | wc -l | tr -d ' ')
echo "  Completed: $epic5_count / 24"
echo ""

total_completed=$((epic1_count + epic2_count + epic3_count + epic4_count + epic5_count))
total_expected=120

echo "============================================================"
echo "Overall Progress: $total_completed / $total_expected ($(echo "scale=1; $total_completed * 100 / $total_expected" | bc)%)"
echo "============================================================"
echo ""

# Check if process is running
if [ -f "$LOG_FILE" ]; then
    echo "=== Recent Log Activity ==="
    tail -5 "$LOG_FILE" | grep -E "(Running|Completed|ERROR|WARNING)" | tail -5
    echo ""
fi

# Check for empty files
echo "=== Empty Files Check ==="
empty_epic1=$(find "$RESULTS_DIR/epic1_curvature" -name "*gearsPertEmb*.csv" -o -name "*k562PertEmb*.csv" -o -name "*rpe1PertEmb*.csv" 2>/dev/null | xargs -I {} sh -c 'test $(wc -l < {}) -le 1 && echo {}' | wc -l | tr -d ' ')
if [ "$empty_epic1" -gt 0 ]; then
    echo "  ⚠️  Epic 1: $empty_epic1 empty cross-dataset/GEARS files"
else
    echo "  ✅ Epic 1: All cross-dataset/GEARS files have data"
fi

empty_epic3=$(find "$RESULTS_DIR/epic3_noise_injection" -name "noise_injection_*.csv" 2>/dev/null | xargs -I {} sh -c 'grep -q ",," {} && echo {}' | wc -l | tr -d ' ')
if [ "$empty_epic3" -gt 0 ]; then
    echo "  ⚠️  Epic 3: $empty_epic3 files with NaN entries"
else
    echo "  ✅ Epic 3: All noise injection files filled"
fi
echo ""


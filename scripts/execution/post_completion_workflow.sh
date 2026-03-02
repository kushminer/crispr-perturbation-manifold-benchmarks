#!/bin/bash
# Post-completion workflow: Run after diagnostic suite finishes

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "Post-Completion Workflow"
echo "Diagnostic Suite Results Processing"
echo "============================================================"
echo ""

# Step 1: Verify all results
echo "Step 1: Verifying all results..."
./verify_fixes.sh

echo ""
echo "Step 2: Generating comprehensive summaries..."
python3 generate_diagnostic_summary.py

echo ""
echo "Step 3: Checking for any remaining issues..."

# Check Epic 3 for NaN entries
epic3_nan_count=$(find results/manifold_law_diagnostics/epic3_noise_injection -name "noise_injection_*.csv" -exec grep -l ",," {} \; 2>/dev/null | wc -l | tr -d ' ')
if [ "$epic3_nan_count" -gt 0 ]; then
    echo "  ⚠️  Epic 3: $epic3_nan_count files still have NaN entries"
    echo "     These may need to be re-run with noise injection"
else
    echo "  ✅ Epic 3: All noise injection files complete"
fi

# Check for empty files
empty_count=$(find results/manifold_law_diagnostics/epic* -name "*.csv" -exec sh -c 'test $(wc -l < {}) -le 1 && echo {}' \; 2>/dev/null | wc -l | tr -d ' ')
if [ "$empty_count" -gt 0 ]; then
    echo "  ⚠️  Found $empty_count empty result files"
else
    echo "  ✅ All result files have data"
fi

echo ""
echo "Step 4: Generating final status report..."
./monitor_progress.sh > results/manifold_law_diagnostics/FINAL_PROGRESS_REPORT.txt

echo ""
echo "============================================================"
echo "✅ Post-completion workflow finished"
echo "============================================================"
echo ""
echo "Next: Review summaries in results/manifold_law_diagnostics/summary_reports/"


#!/bin/bash
# Master script to run all resampling evaluations (LSFT + LOGO)

set -e  # Exit on error

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "Complete Resampling Evaluation"
echo "============================================================"
echo ""
echo "This will run:"
echo "  1. LSFT with resampling (all datasets × all baselines)"
echo "  2. LOGO with resampling (all datasets)"
echo ""
echo "Estimated total runtime: 5-15 hours"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "============================================================"
echo "Step 1: LSFT with Resampling"
echo "============================================================"
echo ""

./run_lsft_resampling_all.sh

echo ""
echo "============================================================"
echo "Step 2: LOGO with Resampling"
echo "============================================================"
echo ""

./run_logo_resampling_all.sh

echo ""
echo "============================================================"
echo "✅ All Resampling Evaluations Complete!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - results/goal_3_prediction/lsft_resampling/"
echo "  - results/goal_3_prediction/functional_class_holdout_resampling/"
echo ""
echo "Next steps:"
echo "  1. Review summary JSON files for bootstrap CIs"
echo "  2. Generate visualizations with CI overlays"
echo "  3. Analyze baseline comparisons"
echo "  4. Prepare results for publication"
echo ""


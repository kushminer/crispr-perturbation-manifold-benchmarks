#!/bin/bash
# Generate all diagnostic suite visualizations

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT}/src"

echo "============================================================"
echo "Generating Diagnostic Suite Visualizations"
echo "============================================================"
echo ""

# Try to find Python with required packages
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

echo "Using Python: $PYTHON_CMD"
echo ""

# Check if visualization script exists
if [ ! -f "create_diagnostic_visualizations.py" ]; then
    echo "❌ Error: create_diagnostic_visualizations.py not found"
    exit 1
fi

# Run visualization script
echo "Running visualization script..."
$PYTHON_CMD create_diagnostic_visualizations.py 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Visualization generation complete!"
    echo "============================================================"
    echo ""
    echo "Figures saved to: results/manifold_law_diagnostics/summary_reports/figures/"
    echo ""
    ls -lh results/manifold_law_diagnostics/summary_reports/figures/*.png 2>/dev/null | awk '{print "  -", $9, "(" $5 ")"}'
else
    echo ""
    echo "❌ Visualization generation failed"
    exit 1
fi


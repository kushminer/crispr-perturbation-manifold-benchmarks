#!/bin/bash
# =============================================================================
# GENERATE ALL PUBLICATION REPORTS
# =============================================================================
# This script generates the complete publication package for the Manifold Law
# Diagnostic Suite.
#
# Outputs:
#   - Per-epic figures and summary tables
#   - Cross-epic meta-analysis
#   - Poster-ready figures
#   - Final CSV data tables
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT}/src"

echo "============================================================"
echo "  MANIFOLD LAW PUBLICATION PACKAGE GENERATOR"
echo "============================================================"
echo ""
echo "Repository root: $REPO_ROOT"
echo "Output directory: $SCRIPT_DIR"
echo ""

# Check for required Python packages
echo "Checking Python environment..."
python3 -c "import pandas, matplotlib, seaborn, numpy" 2>/dev/null || {
    echo "❌ Missing required packages. Please install:"
    echo "   pip install pandas matplotlib seaborn numpy"
    exit 1
}
echo "✅ Python environment OK"
echo ""

# Step 1: Generate main publication reports
echo "============================================================"
echo "Step 1: Generating Publication Reports"
echo "============================================================"
python3 "$SCRIPT_DIR/generate_publication_reports.py"
echo ""

# Step 2: Generate poster figures
echo "============================================================"
echo "Step 2: Generating Poster Figures"
echo "============================================================"
python3 "$SCRIPT_DIR/generate_poster_figures.py"
echo ""

# Step 3: Summary
echo "============================================================"
echo "  PUBLICATION PACKAGE COMPLETE!"
echo "============================================================"
echo ""
echo "Generated outputs:"
echo ""

# Count files in each directory
for dir in epic1_curvature epic2_mechanism_ablation epic3_noise_injection epic4_direction_flip epic5_tangent_alignment cross_epic_analysis poster_figures final_tables; do
    count=$(find "$SCRIPT_DIR/$dir" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "  📁 $dir/: $count files"
done

echo ""
echo "Key deliverables:"
echo "  📄 MANIFOLD_LAW_SUMMARY.md - Executive summary"
echo "  🖼️  poster_figures/ - Publication-ready figures"
echo "  📊 final_tables/ - CSV summary tables"
echo ""
echo "============================================================"


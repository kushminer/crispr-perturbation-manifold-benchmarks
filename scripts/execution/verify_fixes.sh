#!/bin/bash
# Quick verification of fixes

echo "=== Checking GEARS Results ==="
gears_count=0
for file in results/manifold_law_diagnostics/epic1_curvature/*gearsPertEmb*.csv; do
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file" 2>/dev/null || echo "0")
        if [ "$lines" -gt 1 ]; then
            echo "✅ $(basename $file): $lines lines"
            gears_count=$((gears_count + 1))
        else
            echo "❌ $(basename $file): EMPTY ($lines lines)"
        fi
    fi
done

if [ $gears_count -eq 0 ]; then
    echo "⚠️  No GEARS files found in Epic 1"
fi

echo ""
echo "=== Checking Epic 3 Noise Injection ==="
epic3_count=0
for file in results/manifold_law_diagnostics/epic3_noise_injection/noise_injection_*.csv; do
    if [ -f "$file" ]; then
        nan_count=$(grep -c ",," "$file" 2>/dev/null || echo "0")
        total_lines=$(wc -l < "$file" 2>/dev/null || echo "0")
        if [ "$total_lines" -gt 1 ]; then
            if [ "$nan_count" -eq 0 ]; then
                echo "✅ $(basename $file): All values filled ($total_lines lines)"
            else
                echo "⚠️  $(basename $file): $nan_count NaN entries ($total_lines lines)"
            fi
            epic3_count=$((epic3_count + 1))
        fi
    fi
done

if [ $epic3_count -eq 0 ]; then
    echo "⚠️  No Epic 3 noise injection files found"
fi

echo ""
echo "=== Summary ==="
echo "GEARS files with data: $gears_count"
echo "Epic 3 files: $epic3_count"

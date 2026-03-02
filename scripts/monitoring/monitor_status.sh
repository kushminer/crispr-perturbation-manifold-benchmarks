#!/bin/bash
# Status monitoring script for LSFT resampling evaluation
# Run this periodically (e.g., every 30 minutes) to check progress

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "LSFT Resampling Evaluation - Status Update"
echo "============================================================"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check if processes are running
ACTIVE_PROCESSES=$(ps aux | grep -E "run_lsft_resampling|python.*run_lsft_with_resampling" | grep -v grep | wc -l | tr -d ' ')
if [ "$ACTIVE_PROCESSES" -gt 0 ]; then
    echo "✅ Evaluation is RUNNING ($ACTIVE_PROCESSES active process(es))"
else
    echo "❌ No active processes found (evaluation may have completed or failed)"
fi
echo ""

# Count completed evaluations
echo "--- Progress by Dataset ---"
echo ""

# Adamson
ADAMSON_COMPLETED=$(ls -1 results/goal_3_prediction/lsft_resampling/adamson/*_standardized.csv 2>/dev/null | wc -l | tr -d ' ')
ADAMSON_TOTAL=8
ADAMSON_PCT=$((ADAMSON_COMPLETED * 100 / ADAMSON_TOTAL))
echo "Adamson: $ADAMSON_COMPLETED/$ADAMSON_TOTAL baselines ($ADAMSON_PCT%)"
if [ "$ADAMSON_COMPLETED" -gt 0 ]; then
    echo "  Completed:"
    ls -1 results/goal_3_prediction/lsft_resampling/adamson/*_standardized.csv 2>/dev/null | \
        sed 's|.*/lsft_adamson_||; s|_standardized.csv||' | \
        sed 's/^/    - /' || echo ""
fi

# K562
K562_COMPLETED=$(ls -1 results/goal_3_prediction/lsft_resampling/k562/*_standardized.csv 2>/dev/null | wc -l | tr -d ' ')
K562_TOTAL=8
if [ "$K562_COMPLETED" -gt 0 ]; then
    K562_PCT=$((K562_COMPLETED * 100 / K562_TOTAL))
    echo "K562: $K562_COMPLETED/$K562_TOTAL baselines ($K562_PCT%)"
    echo "  Completed:"
    ls -1 results/goal_3_prediction/lsft_resampling/k562/*_standardized.csv 2>/dev/null | \
        sed 's|.*/lsft_k562_||; s|_standardized.csv||' | \
        sed 's/^/    - /' || echo ""
else
    echo "K562: 0/$K562_TOTAL baselines (0%) - Not started"
fi

# RPE1
RPE1_COMPLETED=$(ls -1 results/goal_3_prediction/lsft_resampling/rpe1/*_standardized.csv 2>/dev/null | wc -l | tr -d ' ')
RPE1_TOTAL=8
if [ "$RPE1_COMPLETED" -gt 0 ]; then
    RPE1_PCT=$((RPE1_COMPLETED * 100 / RPE1_TOTAL))
    echo "RPE1: $RPE1_COMPLETED/$RPE1_TOTAL baselines ($RPE1_PCT%)"
    echo "  Completed:"
    ls -1 results/goal_3_prediction/lsft_resampling/rpe1/*_standardized.csv 2>/dev/null | \
        sed 's|.*/lsft_rpe1_||; s|_standardized.csv||' | \
        sed 's/^/    - /' || echo ""
else
    echo "RPE1: 0/$RPE1_TOTAL baselines (0%) - Not started"
fi

echo ""

# Overall progress
TOTAL_COMPLETED=$((ADAMSON_COMPLETED + K562_COMPLETED + RPE1_COMPLETED))
TOTAL_TARGET=$((ADAMSON_TOTAL + K562_TOTAL + RPE1_TOTAL))
OVERALL_PCT=$((TOTAL_COMPLETED * 100 / TOTAL_TARGET))

echo "--- Overall Progress ---"
echo "Total: $TOTAL_COMPLETED/$TOTAL_TARGET evaluations ($OVERALL_PCT%)"
echo ""

# Check log file
if [ -f "run_lsft_resampling.log" ]; then
    echo "--- Latest Activity (last 5 lines) ---"
    tail -5 run_lsft_resampling.log | sed 's/^/  /'
    echo ""
    
    # Check for errors
    ERROR_COUNT=$(grep -i "error\|failed\|❌" run_lsft_resampling.log | tail -5 | wc -l | tr -d ' ')
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo "⚠️  Recent errors detected. Last error:"
        grep -i "error\|failed\|❌" run_lsft_resampling.log | tail -1 | sed 's/^/  /'
        echo ""
    fi
else
    echo "⚠️  Log file not found"
    echo ""
fi

# Estimate time remaining (rough estimate)
if [ "$ACTIVE_PROCESSES" -gt 0 ] && [ "$TOTAL_COMPLETED" -gt 0 ]; then
    # Very rough: assume ~20-30 min per evaluation on average
    REMAINING=$((TOTAL_TARGET - TOTAL_COMPLETED))
    EST_MINUTES=$((REMAINING * 25))
    EST_HOURS=$((EST_MINUTES / 60))
    EST_MINS=$((EST_MINUTES % 60))
    echo "--- Time Estimate ---"
    echo "Estimated remaining: ~$EST_HOURS hours $EST_MINS minutes"
    echo "(Based on $TOTAL_COMPLETED completed evaluations)"
fi

echo ""
echo "============================================================"

